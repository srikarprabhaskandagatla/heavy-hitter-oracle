# H2O: Heavy-Hitter Oracle (Re-Implementation from Scratch)
Standard LLM inference stores every past token's Key and Value vectors in a **KV cache** that grows linearly with sequence length. For OPT-6.7B at FP16, a batch of 16 at length 512 adds ~4 GB on top of the 13.3 GB model weights - forcing small batch sizes and slow throughput.

H2O's insight: **a small fraction of tokens consistently receives the majority of attention** across all layers. These "Heavy Hitters" can be identified cheaply at runtime by tracking cumulative attention scores. By keeping only the top-H heavy hitters plus the R most recent tokens and evicting everything else, the KV cache stays at a fixed bounded size with near-zero accuracy loss.

This repo re-implements the H2O eviction algorithm **from scratch** in PyTorch, validates it against the authors' released code, and runs the full suite of experiments from the paper.

## Files At Present
#### Let's first discuss the scratch H2O Implementation.
---
### `h2o_scratch/h2o_attention.py`
The core of the project. Implements `H2OAttention`, a subclass of HuggingFace's `OPTAttention` that overrides only `forward()` to insert the KV eviction step. All Q/K/V projection weights are inherited unchanged - only the cache management logic is new.

Key components inside:
| Function / Class | What it does |
|---|---|
| `H2OAttention._evict()` | Accumulates column-sum attention scores, selects top-H heavy hitters + R recent tokens, gathers their K/V rows, discards the rest |
| `H2OAttention._budget()` | Converts `heavy_ratio` and `recent_ratio` to integer token counts given current sequence length |
| `_gather_kv()` | Efficient `torch.gather` over dimension 2 of a (batch, heads, seq, dim) tensor - extracts only the kept rows in one CUDA call |
| `patch_model_with_h2o()` | Iterates all model modules, switches `__class__` from `OPTAttention` to `H2OAttention` in-place, sets ratios |
| `reset_h2o_caches()` | Clears `_h2o_score_cache` on all H2OAttention layers between independent samples |

**Prefill vs decode distinction:** eviction only activates when `hidden_states.shape[1] == 1` (one token at a time, i.e. decode phase). During prefill the full prompt is processed normally.

---
### `h2o_scratch/h2o_authors_wrapper.py`
A thin bridge to the authors' released code so both implementations can be called from the same scripts with the same interface.

Key function: `patch_model_with_authors_h2o(model, heavy_ratio, recent_ratio)`

Three bugs encountered and fixed here (documented in the source):
1. **Wrong path** - authors' code lives in `h2o_hf/`, not `h2o/`
2. **Config mismatch** - `convert_kvcache_opt_heavy_recent` needs the full `model.config` (with `num_attention_heads`, `hidden_size`, etc.), not a minimal dataclass
3. **Device + dtype** - authors' patch creates new linear layers on CPU in float32; must call `.half().cuda()` after patching to match the FP16 model on GPU

**Important limitation discovered:** the authors' implementation indexes `attn_weights[:, token_index, :]` assuming the full attention matrix (shape `(batch, seq_len, seq_len)`). During `model.generate()`, this dimension is always 1, causing an immediate `IndexError`. The authors' code therefore **only works with lm-eval loglikelihood scoring** (Experiments 1 and 3) and cannot be used for throughput benchmarking (Experiment 2). Our scratch implementation supports both.

---
### `h2o_scratch/__init__.py`
Package-level exports. Imports `H2OAttention`, `patch_model_with_h2o`, `reset_h2o_caches` from `h2o_attention.py` and `patch_model_with_authors_h2o` from `h2o_authors_wrapper.py` so all four are importable from `h2o_scratch` directly.


---
## Reference Base Paper
**H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, et al. NeurIPS 2023

[Paper](https://arxiv.org/abs/2306.14048)
[Authors' Official Code](https://github.com/FMInference/H2O). 