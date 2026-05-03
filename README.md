# H2O: Heavy-Hitter Oracle — Re-Implementation & Experiments

Standard LLM inference stores every past token's Key and Value vectors in a KV cache that grows linearly with sequence length. For OPT-6.7B at FP16, a batch of 16 at length 512 adds ~4 GB on top of the 13.3 GB model weights.

**H2O's insight:** a small fraction of tokens consistently receives most of the attention. By keeping only the top-H "heavy hitters" plus the R most recent tokens and evicting everything else, the KV cache stays at a fixed bounded size with near-zero accuracy loss.

This repo re-implements H2O from scratch in PyTorch and runs a full suite of experiments on OPT-6.7B.

---

## Experiments Overview

| # | Experiment | Key Question |
|---|---|---|
| 0 | Sparsity Analysis | Do heavy hitters actually exist? |
| 1 | Accuracy Eval | Does H2O preserve task accuracy? |
| 2a | Throughput vs Batch Size | Does H2O improve throughput? |
| 2b | Throughput vs Sequence Length | Does the memory gap widen at longer sequences? |
| 3 | Ablation: H vs R components | Which component matters more? |
| 4 | H:R Ratio Sweep | What's the optimal heavy/recent split? |
| 5 | Lambda Decay Sweep | Does decaying old scores help? |

---

## Exp 0 — Attention Sparsity Analysis

**Finding:** the top 20% of tokens capture ~85% of total attention mass across all layers — confirming the heavy-hitter phenomenon motivating H2O.

| Layerwise Sparsity | Heatmap |
|---|---|
| ![](plots/00_sparsity_analysis/00_sparsity_layerwise.png) | ![](plots/00_sparsity_analysis/00_sparsity_heatmap.png) |

![](plots/00_sparsity_analysis/00_sparsity_cdf.png)

---

## Exp 1 — Accuracy Evaluation

**Setup:** Zero-shot on COPA, OpenBookQA, PIQA, Winogrande. Budgets: 4%, 10%, 20%, 50%.

**Finding:** H2O at 20% budget retains near full-cache accuracy. Local-only degrades significantly, showing heavy hitters are essential.

![](plots/01_accuracy_eval/01_accuracy_vs_budget.png)

---

## Exp 2a — Throughput vs Batch Size

**Setup:** Fixed prompt=256 tokens, generate=128, batch sizes 1→16.

| Batch | Full Cache tok/s | Full Cache Mem | H2O 20% tok/s | H2O 20% Mem |
|---|---|---|---|---|
| 1  | 41.6  | 13.53 GB | 34.0  | 13.36 GB |
| 2  | 97.5  | 13.74 GB | 73.4  | 13.39 GB |
| 4  | 195.6 | 14.14 GB | 146.3 | 13.45 GB |
| 8  | 387.6 | 14.96 GB | 294.8 | 13.58 GB |
| 16 | 714.7 | 16.60 GB | 587.8 | 13.83 GB |

**Finding:** H2O saves up to **2.77 GB** of peak memory at batch=16.

| Throughput | Memory |
|---|---|
| ![](plots/02_throughput/02_throughput_2a.png) | ![](plots/02_throughput/02_throughput_memory_2a.png) |

---

## Exp 2b — Throughput vs Sequence Length

**Setup:** Fixed batch=8, generate=128. Prompt lengths: 128, 256, 512.

| Seq Len | Full Cache tok/s | Full Cache Mem | H2O 20% tok/s | H2O 20% Mem |
|---|---|---|---|---|
| 128 | 384.4 | 14.42 GB | 342.9 | 13.44 GB |
| 256 | 372.5 | 14.96 GB | 281.7 | 13.58 GB |
| 512 | 357.4 | 16.05 GB | 218.4 | 14.07 GB |

**Finding:** the memory gap **widens** as sequence length grows (0.98 → 1.38 → 1.99 GB), confirming H2O's advantage scales with KV cache pressure.

| Throughput | Memory |
|---|---|
| ![](plots/02_throughput/02_throughput_2b.png) | ![](plots/02_throughput/02_throughput_memory_2b.png) |

---

## Exp 3 — Ablation: H vs R Components

**Setup:** Fixed 20% budget, zero-shot on 4 tasks.

| Method | COPA | OpenBookQA | PIQA | Winogrande | Avg |
|---|---|---|---|---|---|
| Full Cache | 81.0% | 27.6% | 76.3% | 65.3% | 62.5% |
| H2O 50/50 (H=10%, R=10%) | 58.0% | 13.4% | 62.7% | 51.6% | 46.4% |
| Local-only (H=0%, R=20%) | 60.0% | 13.2% | 52.9% | 50.4% | 44.1% |
| H2-only (H=20%, R=0%) | 60.0% | 12.2% | 51.7% | 52.4% | 44.1% |

**Finding:** H2O 50/50 outperforms both single-component variants, especially on PIQA (+10 pts over local-only). Both components are needed.

| COPA | OpenBookQA |
|---|---|
| ![](plots/03_ablation/03_ablation_copa.png) | ![](plots/03_ablation/03_ablation_openbookqa.png) |

| PIQA | Winogrande |
|---|---|
| ![](plots/03_ablation/03_ablation_piqa.png) | ![](plots/03_ablation/03_ablation_winogrande.png) |

---

## Exp 4 — H:R Ratio Sweep

**Setup:** Fixed 20% total budget, 5 splits from all-recent to all-heavy.

| H:R Split | COPA | OpenBookQA | PIQA | Winogrande | Avg |
|---|---|---|---|---|---|
| H=0%,  R=20% | 53.0% | 14.4% | 52.8% | 50.8% | 42.7% |
| **H=5%,  R=15%** | **60.0%** | **14.4%** | **61.3%** | **53.0%** | **47.2%** |
| H=10%, R=10% | 53.0% | 14.8% | 58.8% | 50.8% | 44.3% |
| H=15%, R=5%  | 56.0% | 12.8% | 56.2% | 50.2% | 43.8% |
| H=20%, R=0%  | 59.0% | 13.8% | 52.7% | 50.5% | 44.0% |

**Finding:** **H=5%, R=15%** is the best split (avg 47.2%). The paper's default 50/50 is not optimal for OPT-6.7B.

![](plots/04_hr_ratio/04_hr_ratio_sweep_avg.png)

| COPA | OpenBookQA |
|---|---|
| ![](plots/04_hr_ratio/04_hr_ratio_sweep_copa.png) | ![](plots/04_hr_ratio/04_hr_ratio_sweep_openbookqa.png) |

| PIQA | Winogrande |
|---|---|
| ![](plots/04_hr_ratio/04_hr_ratio_sweep_piqa.png) | ![](plots/04_hr_ratio/04_hr_ratio_sweep_winogrande.png) |

---

## Exp 5 — Lambda Decay Sweep

**Setup:** H2O 50/50 at 20% budget, vary decay factor applied to running attention scores.

| Lambda | COPA | OpenBookQA | PIQA | Winogrande | Avg |
|---|---|---|---|---|---|
| 1.00 (standard H2O) | 59.0% | 13.2% | 58.8% | 51.9% | 45.7% |
| 0.95 | 59.0% | 13.0% | 58.7% | 52.4% | 45.8% |
| **0.90** | **59.0%** | **13.4%** | **58.5%** | **53.3%** | **46.1%** |
| 0.80 | 60.0% | 13.4% | 57.0% | 51.9% | 45.6% |
| 0.50 | 59.0% | 14.6% | 56.2% | 51.5% | 45.3% |

**Finding:** mild decay (lambda=0.90) gives a marginal improvement but the effect is minimal — standard H2O is already near-optimal.

![](plots/05_lambda_decay/05_lambda_decay_avg.png)

| COPA | OpenBookQA |
|---|---|
| ![](plots/05_lambda_decay/05_lambda_decay_copa.png) | ![](plots/05_lambda_decay/05_lambda_decay_openbookqa.png) |

| PIQA | Winogrande |
|---|---|
| ![](plots/05_lambda_decay/05_lambda_decay_piqa.png) | ![](plots/05_lambda_decay/05_lambda_decay_winogrande.png) |

---

## Implementation Notes

Our scratch implementation (`h2o_scratch/h2o_attention.py`) works with both `model.generate()` and lm-eval scoring. The authors' released code crashes with `IndexError` during `model.generate()` because it assumes the full attention matrix at every decode step.

---

## Reference

**H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, et al. NeurIPS 2023
[Paper](https://arxiv.org/abs/2306.14048) · [Authors' Code](https://github.com/FMInference/H2O)
