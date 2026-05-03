import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers.models.opt.modeling_opt import OPTAttention


class H2OAttention(OPTAttention):
    heavy_ratio:  float = 0.10
    recent_ratio: float = 0.10
    decay_lambda: float = 1.0  

    _kv_k:      Optional[torch.Tensor] = None  
    _kv_v:      Optional[torch.Tensor] = None  
    _kv_scores: Optional[torch.Tensor] = None  
    _kv_pos:    Optional[torch.Tensor] = None  

    _h2o_budget:   Optional[int] = None
    _h2o_heavy_k:  Optional[int] = None
    _h2o_recent_k: Optional[int] = None
    _h2o_step:     int = 0   
    _prefill_len:  int = 0   

    _evict_scores: Optional[torch.Tensor] = None 
    _b_idx:        Optional[torch.Tensor] = None  

    def _prefill_forward(self, hidden_states, attention_mask):
        bsz, tgt_len, _ = hidden_states.shape
        dev = hidden_states.device

        q = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        k = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        v = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        q_2d = q.view(*proj_shape)
        k_2d = k.view(*proj_shape)
        v_2d = v.view(*proj_shape)

        attn_w = torch.bmm(q_2d, k_2d.transpose(1, 2)) 

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :tgt_len, -tgt_len:]
            attn_w = attn_w.view(bsz, self.num_heads, tgt_len, tgt_len) + causal_mask
            attn_w = attn_w.view(bsz * self.num_heads, tgt_len, tgt_len)

        heavy_k  = self._h2o_heavy_k
        recent_k = self._h2o_recent_k

        positions   = torch.arange(tgt_len, device=dev)
        key_pos     = positions.unsqueeze(0).expand(tgt_len, tgt_len)  
        query_pos   = positions.unsqueeze(1).expand(tgt_len, tgt_len)  
        causal_band = key_pos <= query_pos                         
        recent_band = (key_pos <= query_pos) & (key_pos > query_pos - recent_k) \
                      if recent_k > 0 else torch.zeros(tgt_len, tgt_len, dtype=torch.bool, device=dev)

        keep_mask = recent_band.unsqueeze(0).expand(bsz, tgt_len, tgt_len).clone()

        if heavy_k > 0:
            attn_float = attn_w.view(bsz, self.num_heads, tgt_len, tgt_len).float()
            if self.decay_lambda == 1.0:
                cum_scores = attn_float.cumsum(dim=2).mean(dim=1)
            else:
                cum_scores_4d = torch.zeros(bsz, self.num_heads, tgt_len, tgt_len,
                                            device=attn_float.device, dtype=torch.float32)
                running = torch.zeros(bsz, self.num_heads, tgt_len,
                                      device=attn_float.device, dtype=torch.float32)
                for q_idx in range(tgt_len):
                    running = running * self.decay_lambda + attn_float[:, :, q_idx, :]
                    cum_scores_4d[:, :, q_idx, :] = running
                cum_scores = cum_scores_4d.mean(dim=1)  
            NEG_INF = torch.finfo(torch.float32).min

            heavy_scores = cum_scores.masked_fill(
                (recent_band | ~causal_band).unsqueeze(0), NEG_INF
            )  

            n_avail_per_q = (~recent_band & causal_band).long().sum(dim=1) 
            actual_heavy_per_q = n_avail_per_q.clamp(max=heavy_k)    

            max_heavy = int(actual_heavy_per_q.max().item())
            if max_heavy > 0:
                _, heavy_idx = heavy_scores.topk(max_heavy, dim=2) 
                valid = (torch.arange(max_heavy, device=dev).unsqueeze(0)
                         < actual_heavy_per_q.unsqueeze(1))      
                valid = valid.unsqueeze(0).expand(bsz, -1, -1)    
                heavy_idx = heavy_idx.masked_fill(~valid, 0)         

                heavy_keep = torch.zeros(bsz, tgt_len, tgt_len, dtype=torch.bool, device=dev)
                heavy_keep.scatter_(2, heavy_idx, valid)
                keep_mask = keep_mask | heavy_keep

        final_mask = (keep_mask & causal_band.unsqueeze(0)) \
                         .unsqueeze(1) \
                         .expand(bsz, self.num_heads, tgt_len, tgt_len) \
                         .reshape(bsz * self.num_heads, tgt_len, tgt_len)
        attn_w = attn_w.masked_fill(~final_mask, torch.finfo(attn_w.dtype).min)

        if attn_w.dtype == torch.float16:
            attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_w = F.softmax(attn_w, dim=-1)

        attn_output = torch.bmm(attn_w, v_2d)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if heavy_k > 0:
            final_scores = cum_scores[:, -1, :]     
        else:
            final_scores = torch.zeros(bsz, tgt_len, device=dev, dtype=torch.float32)

        full_k = k.view(bsz, self.num_heads, tgt_len, self.head_dim)
        full_v = v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        self._init_fixed_cache(full_k, full_v, final_scores)
        self._prefill_len = tgt_len

        return attn_output

    def _init_fixed_cache(self, full_k, full_v, scores):
        bsz, n_heads, seq_len, head_dim = full_k.shape
        heavy_k  = self._h2o_heavy_k
        recent_k = self._h2o_recent_k
        budget   = self._h2o_budget
        dev      = full_k.device

        if seq_len <= budget:
            keep_idx = torch.arange(seq_len, device=dev).unsqueeze(0).expand(bsz, -1)
            actual   = seq_len
        else:
            recent_start = seq_len - recent_k
            recent_idx   = torch.arange(recent_start, seq_len, device=dev) \
                               .unsqueeze(0).expand(bsz, -1)     
            n_hvy        = min(heavy_k, recent_start)
            _, hvy       = scores[:, :recent_start].topk(n_hvy, dim=-1)
            hvy, _       = hvy.sort(dim=-1)
            keep_idx     = torch.cat([hvy, recent_idx], dim=-1)    
            actual       = keep_idx.shape[1]

        buf_k = torch.empty(bsz, n_heads, budget, head_dim, dtype=full_k.dtype, device=dev)
        buf_v = torch.empty_like(buf_k)
        buf_s = torch.zeros(bsz, budget, dtype=torch.float32, device=dev)

        buf_p = torch.zeros(bsz, budget, dtype=torch.int32, device=dev)
        buf_p[:, :actual] = torch.arange(actual, device=dev, dtype=torch.int32) \
                                 .unsqueeze(0).expand(bsz, -1)

        idx4 = keep_idx.unsqueeze(1).unsqueeze(-1).expand(bsz, n_heads, actual, head_dim)
        buf_k[:, :, :actual].copy_(full_k.gather(2, idx4))
        buf_v[:, :, :actual].copy_(full_v.gather(2, idx4))
        buf_s[:, :actual].copy_(scores.gather(1, keep_idx))

        self._kv_k      = buf_k
        self._kv_v      = buf_v
        self._kv_scores = buf_s
        self._kv_pos    = buf_p

        self._evict_scores  = torch.empty(bsz, budget, dtype=torch.float32, device=dev)
        self._b_idx         = torch.arange(bsz, device=dev)
        self._zero_score    = torch.zeros(bsz, 1, dtype=torch.float32, device=dev)
        self._step_pos_buf  = torch.zeros(bsz, 1, dtype=torch.int32,   device=dev)

        self._h2o_step = actual

    def _find_evict_slot(self):
        step     = self._h2o_step
        recent_k = self._h2o_recent_k
        recent_mask = self._kv_pos >= (step - recent_k)
        self._evict_scores.copy_(self._kv_scores)
        self._evict_scores.masked_fill_(recent_mask, float('inf'))
        _, slot = self._evict_scores.min(dim=-1)
        return slot

    def _decode_step(self, hidden_states, attention_mask):
        bsz = hidden_states.shape[0]
        dev = hidden_states.device

        q = self.q_proj(hidden_states) * self.scaling  
        k = self.k_proj(hidden_states)                  
        v = self.v_proj(hidden_states)

        q = q.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)

        step  = self._h2o_step
        slot  = self._find_evict_slot()  

        slot_k = slot.view(bsz, 1, 1, 1).expand(bsz, self.num_heads, 1, self.head_dim)
        self._kv_k.scatter_(2, slot_k, k)
        self._kv_v.scatter_(2, slot_k, v)

        self._zero_score.zero_()
        self._step_pos_buf.fill_(step)
        self._kv_scores.scatter_(1, slot.unsqueeze(1), self._zero_score)
        self._kv_pos.scatter_(1, slot.unsqueeze(1), self._step_pos_buf)

        self._h2o_step = step + 1  

        attn = torch.matmul(q, self._kv_k.transpose(-1, -2)) 
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, self._kv_v)                  
        out = out.transpose(1, 2).reshape(bsz, 1, self.embed_dim)
        out = self.out_proj(out)

        if self.decay_lambda != 1.0:
            self._kv_scores.mul_(self.decay_lambda)
        self._kv_scores.add_(attn[:, :, 0].mean(dim=1).float())

        return out

    def forward(
        self,
        hidden_states:     torch.Tensor,
        key_value_states:  Optional[torch.Tensor] = None,
        past_key_value:    Optional[Tuple[torch.Tensor]] = None,
        attention_mask:    Optional[torch.Tensor] = None,
        layer_head_mask:   Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, tgt_len, _ = hidden_states.shape

        if tgt_len > 1:
            full_cache = (self.heavy_ratio + self.recent_ratio) >= 1.0

            if full_cache:
                return super().forward(
                    hidden_states, key_value_states, past_key_value,
                    attention_mask, layer_head_mask, output_attentions,
                )

            self._h2o_heavy_k  = max(1, int(tgt_len * self.heavy_ratio))  if self.heavy_ratio  > 0 else 0
            self._h2o_recent_k = max(1, int(tgt_len * self.recent_ratio)) if self.recent_ratio > 0 else 0
            self._h2o_budget   = self._h2o_heavy_k + self._h2o_recent_k

            attn_out = self._prefill_forward(hidden_states, attention_mask)

            stub = hidden_states.new_empty(bsz, self.num_heads, self._prefill_len, self.head_dim)
            return attn_out, None, (stub, stub)

        # Decode step (generation / autoregressive sampling).
        if self._h2o_budget is None:
            return super().forward(
                hidden_states, key_value_states, past_key_value,
                attention_mask, layer_head_mask, output_attentions,
            )
        attn_out = self._decode_step(hidden_states, attention_mask)
        bsz = hidden_states.shape[0]
        stub = hidden_states.new_empty(bsz, self.num_heads, self._h2o_step, self.head_dim)
        return attn_out, None, (stub, stub)

def patch_model_with_h2o(model, heavy_ratio: float = 0.10,
                          recent_ratio: float = 0.10,
                          decay_lambda: float = 1.0) -> None:
    patched = 0
    for module in model.modules():
        if type(module) is OPTAttention:
            module.__class__     = H2OAttention
            module.heavy_ratio   = heavy_ratio
            module.recent_ratio  = recent_ratio
            module.decay_lambda  = decay_lambda
            module._kv_k         = None
            module._kv_v         = None
            module._kv_scores    = None
            module._kv_pos       = None
            module._h2o_budget   = None
            module._h2o_heavy_k  = None
            module._h2o_recent_k = None
            module._h2o_step     = 0
            module._evict_scores = None
            module._b_idx        = None
            module._zero_score   = None
            module._step_pos_buf = None
            module._prefill_len  = 0
            module._prefill_len  = 0
            patched += 1
    print(f"[H2O] Patched {patched} OPTAttention layers "
          f"(heavy={heavy_ratio:.0%}, recent={recent_ratio:.0%}, "
          f"total budget={heavy_ratio+recent_ratio:.0%}, decay_lambda={decay_lambda})")


def reset_h2o_caches(model, decay_lambda: float = None) -> None:
    for module in model.modules():
        if isinstance(module, H2OAttention):
            if decay_lambda is not None:
                module.decay_lambda = decay_lambda
            module._kv_k         = None
            module._kv_v         = None
            module._kv_scores    = None
            module._kv_pos       = None
            module._h2o_budget   = None
            module._h2o_heavy_k  = None
            module._h2o_recent_k = None
            module._h2o_step     = 0
            module._evict_scores = None
            module._b_idx        = None
            module._zero_score   = None
            module._step_pos_buf = None
            module._prefill_len  = 0
