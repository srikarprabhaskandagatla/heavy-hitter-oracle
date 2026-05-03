import os, sys, time, json, os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError as e:
    print(f"\n[DEBUG] ERROR: {e}\n")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent / "h2o_scratch"))
from h2o_attention import patch_model_with_h2o, reset_h2o_caches, H2OAttention

MODEL_NAME = "facebook/opt-6.7b"

RESULTS_DIR = os.environ.get("RESULTS_DIR")
RESULTS_DIR_PLOTS = os.environ.get("RESULTS_DIR_PLOTS")

RESULTS_FILE = f"{RESULTS_DIR}/04_hr_ratio.json"

BUDGET = 0.20
TASKS  = ["copa", "openbookqa", "piqa", "winogrande"]

RATIOS = [
    {"name": "H=0%,  R=20%", "heavy": 0.00, "recent": 0.20},  # Local-only — already known
    {"name": "H=5%,  R=15%", "heavy": 0.05, "recent": 0.15},  # NEW
    {"name": "H=10%, R=10%", "heavy": 0.10, "recent": 0.10},  # H2O default — already known
    {"name": "H=15%, R=5%",  "heavy": 0.15, "recent": 0.05},  # NEW
    {"name": "H=20%, R=0%",  "heavy": 0.20, "recent": 0.00},  # H2-only — already known
]

def run_eval(model, tokenizer, label):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = evaluator.simple_evaluate(
        model=lm, tasks=TASKS, num_fewshot=0, log_samples=False)
    out = {}
    for task in TASKS:
        r = res["results"][task]
        acc = r.get("acc,none") or r.get("acc_norm,none") or r.get("acc", 0.0)
        out[task] = round(float(acc) * 100, 2)
    avg = sum(out.values()) / len(out)
    print(f"  [{label}]  " +
          " | ".join(f"{t}: {out[t]:.1f}%" for t in TASKS) +
          f"  | Avg: {avg:.1f}%")
    return out


if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
    print(f"Resuming — already have: {list(all_results.keys())}")
else:
    all_results = {
        "H=0%,  R=20%": {"copa": 53.0,  "openbookqa": 14.4,  "piqa": 52.77, "winogrande": 50.75},
        "H=10%, R=10%": {"copa": 53.0,  "openbookqa": 14.8,  "piqa": 58.76, "winogrande": 50.75},
        "H=20%, R=0%":  {"copa": 59.0,  "openbookqa": 13.8,  "piqa": 52.72, "winogrande": 50.51},
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Known points from Exp 3 prepopulated. Running 2 new conditions...\n")

print(f"\nLoading {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

patch_model_with_h2o(model, heavy_ratio=0.05, recent_ratio=0.15)

NEW_RATIOS = [r for r in RATIOS if r["name"] not in all_results]

for cond in NEW_RATIOS:
    print("\n" + "="*60)
    print(f"Running: {cond['name']}")
    print("="*60)

    for m in model.modules():
        if isinstance(m, H2OAttention):
            m.heavy_ratio  = cond["heavy"]
            m.recent_ratio = cond["recent"]
    reset_h2o_caches(model)

    t0 = time.time()
    all_results[cond["name"]] = run_eval(model, tokenizer, cond["name"])
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved.")

# Final table 
print("\n" + "="*70)
print("EXPERIMENT 4 RESULTS — H:R Ratio Sweep (20% KV Budget, OPT-6.7B)")
print("="*70)
print(f"{'Ratio':<18}" + "".join(f"{t.upper():>12}" for t in TASKS) + f"{'Avg':>8}")
print("-"*70)
for name in [r["name"] for r in RATIOS]:
    accs = all_results[name]
    avg  = sum(accs.values()) / len(accs)
    print(f"{name:<18}" +
          "".join(f"{accs[t]:>11.1f}%" for t in TASKS) +
          f"{avg:>7.1f}%")
print("="*70)

# Plots
with open(RESULTS_FILE) as f:
    all_results = json.load(f)

# Load Full Cache baseline from 3a results
results_3a_file = f"{RESULTS_DIR}/03_ablation_3a.json"
FULL_CACHE = {}
if os.path.exists(results_3a_file):
    with open(results_3a_file) as f:
        FULL_CACHE = json.load(f).get("Full Cache (baseline)", {})

RATIO_ORDER = ["H=0%,  R=20%", "H=5%,  R=15%", "H=10%, R=10%",
               "H=15%, R=5%",  "H=20%, R=0%"]
x_labels    = ["0/20", "5/15", "10/10", "15/5", "20/0"]
H2O_COLOR   = "#1D9E75"
FC_COLOR    = "#2C7BB6"

os.makedirs(f"{RESULTS_DIR_PLOTS}/04_hr_ratio", exist_ok=True)

# One plot per task
for task in TASKS:
    fig, ax = plt.subplots(figsize=(9, 6))

    if FULL_CACHE:
        full_acc = FULL_CACHE.get(task, 0.0)
        ax.axhline(full_acc, color=FC_COLOR, linestyle="--", linewidth=2,
                   alpha=0.8, label=f"Full Cache ({full_acc:.1f}%)")

    y = [all_results[r][task] for r in RATIO_ORDER]
    x = list(range(len(RATIO_ORDER)))
    best_idx = y.index(max(y))

    ax.plot(x, y, marker="o", color=H2O_COLOR, linewidth=2, label="H2O (20% Budget)")
    ax.plot(x[best_idx], y[best_idx], marker="o", color="#F0A500",
            markersize=12, zorder=5, label=f"Best: {x_labels[best_idx]} ({y[best_idx]:.1f}%)")
    for xi, val in zip(x, y):
        ax.annotate(f"{val:.1f}%", (xi, val), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, color=H2O_COLOR)

    ax.set_xlabel("H:R Split (Heavy% / Recent%)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"H:R Ratio Sweep — {task.upper()} - opt-6.7b\n"
                 f"(20% KV Budget, Zero-shot)",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = f"{RESULTS_DIR_PLOTS}/04_hr_ratio/04_hr_ratio_sweep_{task}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

# Avg across all tasks
fig, ax = plt.subplots(figsize=(9, 6))

if FULL_CACHE:
    full_avg = sum(FULL_CACHE.values()) / len(FULL_CACHE)
    ax.axhline(full_avg, color=FC_COLOR, linestyle="--", linewidth=2,
               alpha=0.8, label=f"Full Cache Avg ({full_avg:.1f}%)")

y_avg = [sum(all_results[r][t] for t in TASKS) / len(TASKS) for r in RATIO_ORDER]
x = list(range(len(RATIO_ORDER)))
best_idx = y_avg.index(max(y_avg))

ax.plot(x, y_avg, marker="o", color=H2O_COLOR, linewidth=2, label="H2O (20% Budget)")
ax.plot(x[best_idx], y_avg[best_idx], marker="o", color="#F0A500",
        markersize=12, zorder=5, label=f"Best: {x_labels[best_idx]} ({y_avg[best_idx]:.1f}%)")
for xi, val in zip(x, y_avg):
    ax.annotate(f"{val:.1f}%", (xi, val), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=10, color=H2O_COLOR)

ax.set_xlabel("H:R Split (Heavy% / Recent%)", fontsize=12)
ax.set_ylabel("Average Accuracy (%)", fontsize=12)
ax.set_title("H:R Ratio Sweep — Average Across All Tasks - opt-6.7b\n"
             "(20% KV Budget, Zero-shot)",
             fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
fig.tight_layout()
path = f"{RESULTS_DIR_PLOTS}/04_hr_ratio/04_hr_ratio_sweep_avg.png"
fig.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.close()

print("\nAll Exp 4 plots saved.")