import os, sys, time, json
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
from h2o_attention_decay import patch_model_with_decay, reset_h2o_caches, H2OAttentionDecay

MODEL_NAME = "facebook/opt-6.7b"

RESULTS_DIR = os.environ.get("RESULTS_DIR")
RESULTS_DIR_PLOTS = os.environ.get("RESULTS_DIR_PLOTS")
RESULTS_FILE = f"{RESULTS_DIR}/05_lambda_decay.json"

HEAVY_RATIO  = 0.10
RECENT_RATIO = 0.10
TASKS = ["copa", "openbookqa", "piqa", "winogrande"]

LAMBDAS = [
    {"name": "lambda=1.00 (standard H2O)", "lambda": 1.00}, 
    {"name": "lambda=0.95",                 "lambda": 0.95}, 
    {"name": "lambda=0.90",                 "lambda": 0.90},  
    {"name": "lambda=0.80",                 "lambda": 0.80},
    {"name": "lambda=0.50",                 "lambda": 0.50},  
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

# Resume / prepopulate lambda=1.0 from Exp 3 
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
    print(f"Resuming — already have: {list(all_results.keys())}")
else:
    all_results = {
        "lambda=1.00 (standard H2O)": {
            "copa": 53.0, "openbookqa": 14.8,
            "piqa": 58.76, "winogrande": 50.75
        },
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print("lambda=1.0 baseline prepopulated from Exp 3.")
    print("Running 4 new decay values...\n")

print(f"\nLoading {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Patch model once, update lambda per condition
patch_model_with_decay(model, heavy_ratio=HEAVY_RATIO, recent_ratio=RECENT_RATIO,
                       decay_lambda=1.0)

NEW_LAMBDAS = [l for l in LAMBDAS if l["name"] not in all_results]

for cond in NEW_LAMBDAS:
    print("\n" + "="*60)
    print(f"Running: {cond['name']}")
    print("="*60)

    for m in model.modules():
        if isinstance(m, H2OAttentionDecay):
            m.decay_lambda = cond["lambda"]
    reset_h2o_caches(model)

    t0 = time.time()
    all_results[cond["name"]] = run_eval(model, tokenizer, cond["name"])
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved.")

# Final table 
print("\n" + "="*75)
print("EXPERIMENT 5 RESULTS — Decay Lambda Sweep (H2O 20% Budget, OPT-6.7B)")
print("="*75)
print(f"{'Lambda':<25}" + "".join(f"{t.upper():>12}" for t in TASKS) + f"{'Avg':>8}")
print("-"*75)
for cond in LAMBDAS:
    name = cond["name"]
    accs = all_results.get(name)
    if not accs:
        continue
    avg = sum(accs.values()) / len(accs)
    print(f"{name:<25}" +
          "".join(f"{accs[t]:>11.1f}%" for t in TASKS) +
          f"{avg:>7.1f}%")
print("="*75)

# Plots 
with open(RESULTS_FILE) as f:
    all_results = json.load(f)

# Load Full Cache baseline from 3a results
results_3a_file = f"{RESULTS_DIR}/03_ablation_3a.json"
FULL_CACHE = {}
if os.path.exists(results_3a_file):
    with open(results_3a_file) as f:
        FULL_CACHE = json.load(f).get("Full Cache (baseline)", {})

LAMBDA_ORDER = ["lambda=1.00 (standard H2O)", "lambda=0.95", "lambda=0.90", "lambda=0.80", "lambda=0.50"]
x_labels     = ["1.00\n(H2O)", "0.95", "0.90", "0.80", "0.50"]
x            = list(range(len(LAMBDA_ORDER)))
H2O_COLOR    = "#1D9E75"
FC_COLOR     = "#2C7BB6"
BEST_COLOR   = "#F0A500"

os.makedirs(f"{RESULTS_DIR_PLOTS}/05_lambda_decay", exist_ok=True)

# One plot per task 
for task in TASKS:
    fig, ax = plt.subplots(figsize=(9, 6))

    if FULL_CACHE:
        full_acc = FULL_CACHE.get(task, 0.0)
        ax.axhline(full_acc, color=FC_COLOR, linestyle="--", linewidth=2,
                   alpha=0.8, label=f"Full Cache ({full_acc:.1f}%)")

    y = [all_results[n][task] for n in LAMBDA_ORDER if n in all_results]
    x_valid = x[:len(y)]
    best_idx = y.index(max(y))

    ax.plot(x_valid, y, marker="o", color=H2O_COLOR, linewidth=2, label="H2O + Decay")
    ax.plot(x_valid[best_idx], y[best_idx], marker="o", color=BEST_COLOR,
            markersize=12, zorder=5,
            label=f"Best: lambda={x_labels[best_idx].split(chr(10))[0]} ({y[best_idx]:.1f}%)")
    for xi, val in zip(x_valid, y):
        ax.annotate(f"{val:.1f}%", (xi, val), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, color=H2O_COLOR)

    ax.set_xlabel("Decay Factor Lambda", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Decay Lambda Sweep — {task.upper()}\n"
                 f"(H2O 20% Budget, Zero-shot (OPT 6.7b))",
                 fontsize=13)
    ax.set_xticks(x_valid)
    ax.set_xticklabels(x_labels[:len(y)], fontsize=10)
    all_vals = y + ([FULL_CACHE.get(task, max(y))] if FULL_CACHE else [])
    ax.set_ylim(max(0, min(all_vals) - 5), max(all_vals) + 10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = f"{RESULTS_DIR_PLOTS}/05_lambda_decay/05_lambda_decay_{task}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

# Avg across all tasks 
fig, ax = plt.subplots(figsize=(9, 6))

if FULL_CACHE:
    full_avg = sum(FULL_CACHE.values()) / len(FULL_CACHE)
    ax.axhline(full_avg, color=FC_COLOR, linestyle="--", linewidth=2,
               alpha=0.8, label=f"Full Cache Avg ({full_avg:.1f}%)")

valid_names = [n for n in LAMBDA_ORDER if n in all_results]
y_avg = [sum(all_results[n][t] for t in TASKS) / len(TASKS) for n in valid_names]
x_valid = x[:len(y_avg)]
best_idx = y_avg.index(max(y_avg))

ax.plot(x_valid, y_avg, marker="o", color=H2O_COLOR, linewidth=2, label="H2O + Decay")
ax.plot(x_valid[best_idx], y_avg[best_idx], marker="o", color=BEST_COLOR,
        markersize=12, zorder=5,
        label=f"Best: lambda={x_labels[best_idx].split(chr(10))[0]} ({y_avg[best_idx]:.1f}%)")
for xi, val in zip(x_valid, y_avg):
    ax.annotate(f"{val:.1f}%", (xi, val), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=10, color=H2O_COLOR)

ax.set_xlabel("Decay Factor Lambda", fontsize=12)
ax.set_ylabel("Average Accuracy (%)", fontsize=12)
ax.set_title("Decay Lambda Sweep — Average Across All Tasks\n"
             "(H2O 20% Budget, Zero-shot (OPT 6.7b))",
             fontsize=13)
ax.set_xticks(x_valid)
ax.set_xticklabels(x_labels[:len(y_avg)], fontsize=10)
all_vals_avg = y_avg + ([full_avg] if FULL_CACHE else [])
ax.set_ylim(max(0, min(all_vals_avg) - 5), max(all_vals_avg) + 10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
fig.tight_layout()
path = f"{RESULTS_DIR_PLOTS}/05_lambda_decay/05_lambda_decay_avg.png"
fig.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.close()

print("\nAll Exp 5 plots saved.")
