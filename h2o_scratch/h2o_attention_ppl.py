import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.opt.modeling_opt import OPTAttention

def _gather_kv(tensor, indices):
    bsz, n_heads, _, head_dim = tensor.shape
    idx = indices.unsqueeze(1).unsqueeze(-1).expand(bsz, n_heads, -1, head_dim)
    return tensor.gather(2, idx)

class H2OAttentionPPL(OPTAttention):
    heavy_ratio:  float = 0.10
    recent_ratio: float = 0.10
    _acc_scores: Optional[torch.Tensor] = None
    _prefill_len: int = 0 

    def _budget(self, seq_len):
        ref = self._prefill_len if self._prefill_len > 0 else seq_len
        H = max(1, int(ref * self.heavy_ratio))  if self.heavy_ratio  > 0 else 0
        R = max(1, int(ref * self.recent_ratio)) if self.recent_ratio > 0 else 0
        return H, R

    def _prefill_eviction_mask(self, attn_weights):
        bsz, n_heads, seq_len, _ = attn_weights.shape
        device = attn_weights.device
        avg    = attn_weights.detach().mean(dim=1)  
        keep   = torch.zeros(bsz, seq_len, seq_len, dtype=torch.bool, device=device)
        scores = torch.zeros(bsz, seq_len, device=device)

        for q in range(seq_len):
            ctx = q + 1
            scores[:, :ctx] += avg[:, q, :ctx]
            H, R = self._budget(ctx)
            if ctx <= H + R:
                keep[:, q, :ctx] = True
            else:
                rs = ctx - R if R > 0 else ctx
                if R > 0:
                    keep[:, q, rs:ctx] = True
                if H > 0 and rs > 0:
                    _, idx = scores[:, :rs].topk(min(H, rs), dim=-1)
                    keep[:, q].scatter_(1, idx,
                        torch.ones(bsz, min(H, rs), dtype=torch.bool, device=device))

                keep[:, q, 0] = True
        return keep.unsqueeze(1)   

    @staticmethod
    def _gather_kv(t, idx):
        B, H, _, D = t.shape
        i = idx.unsqueeze(1).unsqueeze(-1).expand(B, H, idx.shape[1], D)
        return t.gather(2, i)

    def forward(self, hidden_states, key_value_states=None,
                past_key_value=None, attention_mask=None,
                layer_head_mask=None, output_attentions=False, **kwargs):

        bsz, tgt_len, _ = hidden_states.shape

        # PREFILL: apply causal eviction mask
        if tgt_len > 1:
            self._acc_scores = None
            self._prefill_len = tgt_len

            # compute raw attention to build eviction mask
            q = self.q_proj(hidden_states) * self.scaling
            k = self.k_proj(hidden_states)
            q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            raw = torch.matmul(q, k.transpose(-1, -2))
            raw = raw + torch.triu(
                torch.full((tgt_len, tgt_len), float('-inf'),
                           device=hidden_states.device, dtype=raw.dtype), diagonal=1)
            soft = F.softmax(raw, dim=-1, dtype=torch.float32).to(raw.dtype)
            keep_mask = self._prefill_eviction_mask(soft)  # (B,1,S,S)
            evict_bias = torch.zeros_like(soft[:, :1]).masked_fill(
                ~keep_mask, float('-inf'))
            attention_mask = (attention_mask + evict_bias
                              if attention_mask is not None else evict_bias)

            out = super().forward(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                **kwargs)
            return out

        with torch.no_grad():
            q_  = (self.q_proj(hidden_states) * self.scaling
                   ).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k_  = self.k_proj(hidden_states
                   ).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k_cached = past_key_value[0] if past_key_value is not None else None
            k_full   = torch.cat([k_cached, k_], dim=2) if k_cached is not None else k_
            w   = F.softmax(
                torch.matmul(q_, k_full.transpose(-1, -2)),
                dim=-1, dtype=torch.float32).to(q_.dtype)
            step_scores = w.squeeze(2).mean(dim=1).detach()  # (B, L+1)

        raw_out = super().forward(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            **kwargs)

        # Unpack to handle both 2-tuple and 3-tuple
        if len(raw_out) == 3:
            attn_out, ret_w, new_pkv = raw_out
        else:
            attn_out, new_pkv = raw_out
            ret_w = None

        if new_pkv is None:
            return raw_out

        cache_len = new_pkv[0].shape[2]

        # Accumulate
        if self._acc_scores is None:
            self._acc_scores = step_scores[:, :cache_len].clone()
        else:
            if self._acc_scores.shape[1] < cache_len:
                pad = torch.zeros(bsz, cache_len - self._acc_scores.shape[1],
                                  device=new_pkv[0].device, dtype=self._acc_scores.dtype)
                self._acc_scores = torch.cat([self._acc_scores, pad], dim=1)
            self._acc_scores += step_scores[:, :cache_len]

        H, R = self._budget(cache_len)
        budget = H + R

        if cache_len > budget:
            prot = cache_len - R
            if H > 0 and prot > 0:
                search_from = min(1, prot)  
                keep_k = min(H, prot - search_from)
                if keep_k > 0:
                    _, hi = self._acc_scores[:, search_from:prot].topk(
                        keep_k, dim=-1, largest=True)
                    hi = hi + search_from  
                else:
                    hi = torch.zeros(bsz, 0, dtype=torch.long, device=new_pkv[0].device)
                sink = torch.zeros(bsz, 1, dtype=torch.long, device=new_pkv[0].device)
                hi = torch.cat([sink, hi], dim=1)
            else:
                hi = torch.zeros(bsz, 1, dtype=torch.long, device=new_pkv[0].device)
            ri = torch.arange(cache_len - R, cache_len,
                              device=new_pkv[0].device).unsqueeze(0).expand(bsz, -1)
            keep, _ = torch.cat([hi, ri], dim=1).sort(dim=1)
            keep = torch.unique(keep, dim=1)
            new_pkv = (self._gather_kv(new_pkv[0], keep),
                       self._gather_kv(new_pkv[1], keep))
            self._acc_scores = self._acc_scores.gather(1, keep)

        if len(raw_out) == 3:
            return attn_out, ret_w, new_pkv
        return attn_out, new_pkv