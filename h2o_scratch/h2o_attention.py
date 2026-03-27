import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.models.opt.modeling_opt import OPTAttention

#  Helper: soft-select rows from a 4-D KV tensor
def _gather_kv(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    bsz, n_heads, _, head_dim = tensor.shape
    idx = indices.unsqueeze(1).unsqueeze(-1)       
    idx = idx.expand(bsz, n_heads, -1, head_dim)    
    return tensor.gather(2, idx)

class H2OAttention(OPTAttention):
    heavy_ratio: float = 0.10
    recent_ratio: float = 0.10
    _h2o_score_cache: Optional[torch.Tensor] = None   # (batch, seq_len)

    def _budget(self, seq_len: int) -> Tuple[int, int]:
        heavy_k  = max(1, int(seq_len * self.heavy_ratio))
        recent_k = max(1, int(seq_len * self.recent_ratio))
        return heavy_k, recent_k

    def _evict(
        self,
        key_states:   torch.Tensor,   # (B, H, S, D)
        value_states: torch.Tensor,   # (B, H, S, D)
        attn_weights: torch.Tensor,   # (B, H, 1, S)  — current token
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_heads, seq_len, head_dim = key_states.shape

        # step 1: accumulate scores
        # Average over heads → (B, S)
        curr_scores = attn_weights.squeeze(2).mean(dim=1)

        if self._h2o_score_cache is None or \
           self._h2o_score_cache.shape[1] != seq_len:
            self._h2o_score_cache = curr_scores.detach().clone()
        else:
            self._h2o_score_cache = self._h2o_score_cache + curr_scores.detach()

        heavy_k, recent_k = self._budget(seq_len)
        budget = heavy_k + recent_k

        # step 2: skip eviction if still within budget
        if seq_len <= budget:
            return key_states, value_states

        # step 3: always protect the most recent recent_k tokens 
        recent_start = seq_len - recent_k
        recent_idx   = torch.arange(recent_start, seq_len,
                                    device=key_states.device)       
        recent_idx   = recent_idx.unsqueeze(0).expand(bsz, -1)      

        # step 4: pick heavy hitters from non-recent tokens 
        non_recent_scores = self._h2o_score_cache[:, :recent_start]    
        k_select          = min(heavy_k, recent_start)
        _, heavy_idx      = non_recent_scores.topk(k_select, dim=-1)  
        heavy_idx, _      = heavy_idx.sort(dim=-1)

        # step 5: merge indices and gather 
        keep_idx = torch.cat([heavy_idx, recent_idx], dim=-1)  

        new_keys   = _gather_kv(key_states,   keep_idx)
        new_values = _gather_kv(value_states, keep_idx)

        # trim score cache to kept positions
        self._h2o_score_cache = self._h2o_score_cache.gather(1, keep_idx)

        return new_keys, new_values

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

        attn_output, attn_weights_out, new_past_kv = super().forward(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=True,  
        )

        # apply H2O eviction on the *returned* past KV cache 
        # Only evict during the decode phase (not prefill).
        # Prefill: hidden_states.shape[1] > 1
        # Decode : hidden_states.shape[1] == 1
        if new_past_kv is not None and hidden_states.shape[1] == 1:
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
            module.__class__      = H2OAttention
            module.heavy_ratio    = heavy_ratio
            module.recent_ratio   = recent_ratio
            module._h2o_score_cache = None
            patched += 1
    print(f"[H2O] Patched {patched} OPTAttention layers "
          f"(heavy={heavy_ratio:.0%}, recent={recent_ratio:.0%}, "
          f"total budget={heavy_ratio+recent_ratio:.0%})")

def reset_h2o_caches(model) -> None:
    for module in model.modules():
        if isinstance(module, H2OAttention):
            module._h2o_score_cache = None