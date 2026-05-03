import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.opt.modeling_opt import OPTAttention

def _gather_kv(tensor, indices):
    bsz, n_heads, _, head_dim = tensor.shape
    idx = indices.unsqueeze(1).unsqueeze(-1).expand(bsz, n_heads, -1, head_dim)
    return tensor.gather(2, idx)

class H2OAttentionDecay(OPTAttention):
    heavy_ratio:  float = 0.10
    recent_ratio: float = 0.10
    decay_lambda: float = 1.0  
    _h2o_score_cache: Optional[torch.Tensor] = None

    def _budget(self, seq_len):
        heavy_k  = max(1, int(seq_len * self.heavy_ratio))  if self.heavy_ratio  > 0 else 0
        recent_k = max(1, int(seq_len * self.recent_ratio)) if self.recent_ratio > 0 else 0
        return heavy_k, recent_k

    def _prefill_eviction_mask(self, attn_weights):
        bsz, n_heads, seq_len, _ = attn_weights.shape
        device = attn_weights.device
        avg_weights = attn_weights.detach().mean(dim=1)
        keep_mask = torch.zeros(bsz, seq_len, seq_len, dtype=torch.bool, device=device)

        running_scores = torch.zeros(bsz, seq_len, device=device)

        for q in range(seq_len):
            ctx_len = q + 1

            running_scores[:, :ctx_len] = (
                self.decay_lambda * running_scores[:, :ctx_len]
                + avg_weights[:, q, :ctx_len]
            )

            heavy_k, recent_k = self._budget(ctx_len)
            if ctx_len <= heavy_k + recent_k:
                keep_mask[:, q, :ctx_len] = True
            else:
                recent_start = ctx_len - recent_k if recent_k > 0 else ctx_len
                if recent_k > 0:
                    keep_mask[:, q, recent_start:ctx_len] = True
                if heavy_k > 0 and recent_start > 0:
                    _, heavy_idx = running_scores[:, :recent_start].topk(
                        min(heavy_k, recent_start), dim=-1)
                    keep_mask[:, q, :].scatter_(1, heavy_idx,
                        torch.ones(bsz, min(heavy_k, recent_start),
                                   dtype=torch.bool, device=device))
        return keep_mask.unsqueeze(1)

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None, layer_head_mask=None, output_attentions=False,
                **kwargs):
        bsz, tgt_len, _ = hidden_states.shape

        if past_key_value is None:
            past_key_value = kwargs.pop("past_key_values", None)

        if tgt_len > 1 and (self.heavy_ratio + self.recent_ratio) < 1.0:
            q = self.q_proj(hidden_states) * self.scaling
            k = self.k_proj(hidden_states)
            head_dim = self.head_dim
            q = q.view(bsz, tgt_len, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, tgt_len, self.num_heads, head_dim).transpose(1, 2)
            raw = torch.matmul(q, k.transpose(-1, -2))
            raw = raw + torch.triu(
                torch.full((tgt_len, tgt_len), float('-inf'),
                           device=hidden_states.device, dtype=raw.dtype), diagonal=1)
            soft = F.softmax(raw, dim=-1, dtype=torch.float32).to(raw.dtype)
            keep_mask = self._prefill_eviction_mask(soft)
            evict_bias = torch.zeros_like(soft[:, :1]).masked_fill(~keep_mask, float('-inf'))
            attention_mask = (attention_mask + evict_bias) if attention_mask is not None else evict_bias

        raw_out = super().forward(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            **kwargs)

        return raw_out


def patch_model_with_decay(model, heavy_ratio=0.10, recent_ratio=0.10, decay_lambda=1.0):
    patched = 0
    for m in model.modules():
        if type(m) is OPTAttention:
            m.__class__       = H2OAttentionDecay
            m.heavy_ratio     = heavy_ratio
            m.recent_ratio    = recent_ratio
            m.decay_lambda    = decay_lambda
            m._h2o_score_cache = None
            patched += 1
    print(f"[H2O+Decay] Patched {patched} layers  "
          f"(heavy={heavy_ratio:.0%}, recent={recent_ratio:.0%}, lambda={decay_lambda})")

def reset_h2o_caches(model):
    for m in model.modules():
        if isinstance(m, H2OAttentionDecay):
            m._h2o_score_cache = None