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
RESULTS_FILE      = f"{RESULTS_DIR}/03_ablation.json"

BUDGET = 0.20
TASKS  = ["copa", "openbookqa", "piqa", "winogrande"]

CONDITIONS = [
    {"name": "Local-only  (H=0,   R=20%)", "heavy": 0.00,        "recent": BUDGET     },
    {"name": "H2-only (H=20%, R=0)",        "heavy": BUDGET,      "recent": 0.00       },
    {"name": "H2O 50/50 (H=10%, R=10%)",   "heavy": BUDGET / 2,  "recent": BUDGET / 2 },
]

def run_eval(model, tokenizer, label):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = evaluator.simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, log_samples=False)
    out = {}
    for task in TASKS:
        r = res["results"][task]
        acc = r.get("acc,none") or r.get("acc_norm,none") or r.get("acc", 0.0)
        out[task] = round(float(acc) * 100, 2)
    avg = sum(out.values()) / len(out)
    print(f"  [{label}]  " + " | ".join(f"{t}: {out[t]:.1f}%" for t in TASKS) + f"  | Avg: {avg:.1f}%")
    return out

# Load existing results to resume from where we left off
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
    print(f"Resuming — already have: {list(all_results.keys())}")
else:
    all_results = {}

print(f"\nLoading {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Full cache baseline 
if "Full Cache (baseline)" not in all_results:
    print("="*60 + "\nFull Cache baseline\n" + "="*60)
    t0 = time.time()
    all_results["Full Cache (baseline)"] = run_eval(model, tokenizer, "Full Cache")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
    with open(RESULTS_FILE, "w") as f: json.dump(all_results, f, indent=2)
    print("Saved.")
else:
    print("Skipping Full Cache baseline (already in results).")

# H2O conditions
patch_model_with_h2o(model, heavy_ratio=0.0, recent_ratio=BUDGET)

for cond in CONDITIONS:
    if cond["name"] in all_results:
        print(f"Skipping {cond['name']} (already in results).")
        continue
    print("\n" + "="*60 + f"\n{cond['name']}\n" + "="*60)
    for m in model.modules():
        if isinstance(m, H2OAttention):
            m.heavy_ratio  = cond["heavy"]
            m.recent_ratio = cond["recent"]
    reset_h2o_caches(model)
    t0 = time.time()
    all_results[cond["name"]] = run_eval(model, tokenizer, cond["name"])
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
    with open(RESULTS_FILE, "w") as f: json.dump(all_results, f, indent=2)
    print("Saved.")

# Plots
with open(RESULTS_FILE) as f:
    results = json.load(f)

ALL_METHODS = [
    ("Local-only  (H=0,   R=20%)", "Local-only", "#D85A30"),
    ("H2-only (H=20%, R=0)",        "H2-only",    "#1D9E75"),
    ("H2O 50/50 (H=10%, R=10%)",   "H2O 50/50",  "#378ADD"),
]
FC_COLOR = "#2C7BB6"

os.makedirs(f"{RESULTS_DIR_PLOTS}/03_ablation", exist_ok=True)

full_cache = results.get("Full Cache (baseline)", {})
present = [(key, lbl, col) for key, lbl, col in ALL_METHODS if key in results]

for task in TASKS:
    fig, ax = plt.subplots(figsize=(9, 6))

    full_acc = full_cache.get(task, 0.0)
    ax.axhline(full_acc, color=FC_COLOR, linestyle="--", linewidth=2,
               alpha=0.8, label=f"Full Cache ({full_acc:.1f}%)")

    for i, (key, lbl, color) in enumerate(present):
        acc = results[key].get(task, 0.0)
        ax.bar(i, acc, color=color, alpha=0.85, width=0.5, label=lbl)
        ax.annotate(f"{acc:.1f}%", (i, acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, color=color)

    ax.set_xlabel("KV Budget Allocation Strategy", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Ablation — {task.upper()} - opt-6.7b\n"
                 f"(20% KV Budget, Zero-shot)",
                 fontsize=13)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([lbl for _, lbl, _ in present], fontsize=10)
    ax.set_ylim(0, full_acc + 18)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = f"{RESULTS_DIR_PLOTS}/03_ablation/exp3_ablation_{task}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

print("All Exp 3a plots saved.")
