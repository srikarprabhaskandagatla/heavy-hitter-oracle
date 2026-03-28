import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers.models.opt.modeling_opt import OPTAttention


#  Helper: gather rows from a 4-D KV tensor
def _gather_kv(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    bsz, n_heads, _, head_dim = tensor.shape
    idx = indices.unsqueeze(1).unsqueeze(-1)     
    idx = idx.expand(bsz, n_heads, -1, head_dim)   
    return tensor.gather(2, idx)


#  H2OAttention  —  drop-in replacement for OPTAttention
class H2OAttention(OPTAttention):
    heavy_ratio: float = 0.10
    recent_ratio: float = 0.10
    _h2o_score_cache: Optional[torch.Tensor] = None   

    def _budget(self, seq_len: int) -> Tuple[int, int]:
        heavy_k  = max(1, int(seq_len * self.heavy_ratio))
        recent_k = max(1, int(seq_len * self.recent_ratio))
        return heavy_k, recent_k

    def _evict(
        self,
        key_states:   torch.Tensor, 
        value_states: torch.Tensor,  
        attn_weights: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_heads, seq_len, head_dim = key_states.shape

        # accumulate scores (average over heads)
        curr_scores = attn_weights.squeeze(2).mean(dim=1)   

        if self._h2o_score_cache is None or \
           self._h2o_score_cache.shape[1] != seq_len:
            self._h2o_score_cache = curr_scores.detach().clone()
        else:
            self._h2o_score_cache = self._h2o_score_cache + curr_scores.detach()

        heavy_k, recent_k = self._budget(seq_len)
        budget = heavy_k + recent_k

        if seq_len <= budget:
            return key_states, value_states

        # protect the most recent recent_k tokens
        recent_start = seq_len - recent_k
        recent_idx   = torch.arange(recent_start, seq_len,
                                    device=key_states.device)
        recent_idx   = recent_idx.unsqueeze(0).expand(bsz, -1)

        # pick heavy hitters from non-recent tokens
        non_recent_scores = self._h2o_score_cache[:, :recent_start]
        k_select          = min(heavy_k, recent_start)
        _, heavy_idx      = non_recent_scores.topk(k_select, dim=-1)
        heavy_idx, _      = heavy_idx.sort(dim=-1)

        keep_idx   = torch.cat([heavy_idx, recent_idx], dim=-1)
        new_keys   = _gather_kv(key_states,   keep_idx)
        new_values = _gather_kv(value_states, keep_idx)
        self._h2o_score_cache = self._h2o_score_cache.gather(1, keep_idx)

        return new_keys, new_values

    def _prefill_eviction_mask(
        self,
        attn_weights: torch.Tensor,  
    ) -> torch.Tensor:
        bsz, n_heads, seq_len, _ = attn_weights.shape
        device = attn_weights.device

        avg_weights = attn_weights.detach().mean(dim=1)  

        keep_mask = torch.zeros(bsz, seq_len, seq_len,
                                dtype=torch.bool, device=device)

        # Accumulate scores token by token to simulate decode
        running_scores = torch.zeros(bsz, seq_len, device=device)

        for q in range(seq_len):
            ctx_len = q + 1   

            # update running scores with current query's attention
            running_scores[:, :ctx_len] += avg_weights[:, q, :ctx_len]

            heavy_k, recent_k = self._budget(ctx_len)
            budget = heavy_k + recent_k

            if ctx_len <= budget:
                keep_mask[:, q, :ctx_len] = True
            else:
                recent_start = ctx_len - recent_k
                keep_mask[:, q, recent_start:ctx_len] = True
                scores_non_recent = running_scores[:, :recent_start]
                k_select = min(heavy_k, recent_start)
                _, heavy_idx = scores_non_recent.topk(k_select, dim=-1)
                keep_mask[:, q, :].scatter_(
                    1, heavy_idx,
                    torch.ones(bsz, k_select, dtype=torch.bool, device=device)
                )

        return keep_mask.unsqueeze(1)

    def forward(
        self,
        hidden_states:    torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value:   Optional[Tuple[torch.Tensor]] = None,
        attention_mask:   Optional[torch.Tensor] = None,
        layer_head_mask:  Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:

        bsz, tgt_len, _ = hidden_states.shape
        is_decode  = (tgt_len == 1)
        is_prefill = (tgt_len > 1)

        # PREFILL mode
        if is_prefill and (self.heavy_ratio + self.recent_ratio) < 1.0:
            query_states = self.q_proj(hidden_states) * self.scaling
            key_states   = self.k_proj(hidden_states)

            head_dim = self.head_dim
            q = query_states.view(bsz, tgt_len, self.num_heads, head_dim) \
                            .transpose(1, 2)                  # (B,H,S,D)
            k = key_states.view(bsz, tgt_len, self.num_heads, head_dim) \
                          .transpose(1, 2)

            raw_scores = torch.matmul(q, k.transpose(-1, -2))

            causal_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float('-inf'),
                           device=hidden_states.device,
                           dtype=raw_scores.dtype),
                diagonal=1
            )
            raw_scores = raw_scores + causal_mask

            soft_weights = F.softmax(raw_scores, dim=-1, dtype=torch.float32) \
                            .to(raw_scores.dtype)

            keep_mask = self._prefill_eviction_mask(soft_weights)

            evict_bias = torch.zeros_like(soft_weights[:, :1, :, :])
            evict_bias = evict_bias.masked_fill(~keep_mask, float('-inf'))

            if attention_mask is not None:
                attention_mask = attention_mask + evict_bias
            else:
                attention_mask = evict_bias

        attn_output, attn_weights_out, new_past_kv = super().forward(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=True,
        )

        if is_decode and new_past_kv is not None:
            k, v = new_past_kv
            k, v = self._evict(k, v, attn_weights_out)
            new_past_kv = (k, v)

        if not output_attentions:
            attn_weights_out = None

        return attn_output, attn_weights_out, new_past_kv


def patch_model_with_h2o(model, heavy_ratio: float = 0.10,
                          recent_ratio: float = 0.10) -> None:

    patched = 0
    for module in model.modules():
        if type(module) is OPTAttention:
            module.__class__        = H2OAttention
            module.heavy_ratio      = heavy_ratio
            module.recent_ratio     = recent_ratio
            module._h2o_score_cache = None
            patched += 1
    print(f"[H2O] Patched {patched} OPTAttention layers "
          f"(heavy={heavy_ratio:.0%}, recent={recent_ratio:.0%}, "
          f"total budget={heavy_ratio+recent_ratio:.0%})")


def reset_h2o_caches(model) -> None:
    for module in model.modules():
        if isinstance(module, H2OAttention):
            module._h2o_score_cache = None