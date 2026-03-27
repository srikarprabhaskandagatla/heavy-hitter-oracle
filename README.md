## H2O: Heavy-Hitter Oracle (Re-Implementation from Scratch)
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

#### Let's discuss the experiment analysis files
---
### `scripts/00_sparsity_analysis.py` - Observation
**Purpose:** Empirically verify the paper's central motivation - that a small fraction of tokens accumulates most attention mass - before implementing anything.

**What it does:**
1. Loads OPT-6.7B and runs 20 WikiText-103 samples through the model with `output_attentions=True`
2. At each of the 32 layers, averages attention weights over heads to computes column-sum (how much attention each key position received)
3. For each sample and layer, measures what fraction of total attention mass is captured by the top {5, 10, 20, 50}% of tokens
4. Additionally collects the raw per-token score distributions for CDF plotting

**Outputs:**
| File | Description |
|---|---|
| `plots/sparsity_layerwise.png` | Per-layer line plot: top-X% tokens vs fraction of attention mass, with std shading across 20 samples |
| `plots/sparsity_heatmap.png` | Layer × budget heatmap - shows concentration at a glance |
| `plots/sparsity_cdf.png` | CDF of accumulated token scores for 4 representative layers - the steeper the curve, the more extreme the heavy-hitter concentration |
| `results/sparsity_stats.json` | Full numeric summary: mean and std per layer per ratio - copy these numbers into the report |

---
### `scripts/01_accuracy_eval.py` - Experiment 1
**Purpose:** Measure whether H2O eviction degrades downstream task accuracy vs full KV cache, and validate our scratch implementation against the authors'.

**What it does - four sequential parts:**
| Part | What runs | Model state |
|---|---|---|
| Part 1 | Full cache baseline | Fresh model, no patch |
| Part 2 | Scratch H2O at budgets {4, 10, 20, 50}% | Same model, patched once, ratios updated in-place between budgets |
| Part 3 | Authors' H2O at same budgets | Fresh model reload per budget (authors' patch is structural - cannot update ratios in-place) |
| Part 4 | Local-only ablation at same budgets | Single patched model, `heavy_ratio=0` throughout |

**KV budget split:** each budget is split 50/50 between heavy hitters and recent tokens (e.g., 20% budget to `heavy_ratio=0.10`, `recent_ratio=0.10`), matching the paper's default.

**Evaluation:** uses `lm-eval-harness` loglikelihood scoring - the model ranks answer choices by log-probability, accuracy = fraction correct.

**Outputs:**
| File | Description |
|---|---|
| `plots/accuracy_vs_budget.png` | One subplot per task; three curve families (Scratch H2O, Authors H2O, Local-only) with jitter and distinct markers so overlapping points stay visible; full-cache dashed reference line |
| `results/accuracy_results.json` | All raw numbers keyed by method label |
| `results/accuracy_table.txt` | Formatted table ready to paste into the report |


## Our Implementation vs Authors' Implementation
| Property | Ours (`h2o_attention.py`) | Authors' (`modify_opt.py`) |
|---|---|---|
| Works with `model.generate()` | Yes | No - crashes with IndexError |
| Works with lm-eval scoring | Yes | Yes |
| Update budget without model reload | Yes - update `.heavy_ratio` in-place | No - must reload and re-patch |
| Design | Subclass `OPTAttention`, override `forward()` | Replace class structurally, new linear layers |
| Score accumulation | Running sum during decode, trimmed with cache | Recomputed from full attention matrix each call |

---


> **COMPSCI 690AB - Milestone Report**
For this milestone report, we have completed the **Observation** (layer-wise attention sparsity analysis on WikiText-103) and **Experiment 1** (downstream task accuracy evaluation on COPA and Winogrande). Experiments 2 (throughput) and 3 (ablation) are reserved and still under testing for the final report.

---
## Reference Base Paper
**H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, et al. NeurIPS 2023

[Paper](https://arxiv.org/abs/2306.14048)

[Authors' Official Code](https://github.com/FMInference/H2O). 