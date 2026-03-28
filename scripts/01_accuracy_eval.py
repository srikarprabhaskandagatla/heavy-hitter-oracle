import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

parser = argparse.ArgumentParser()
parser.add_argument("--model",    default="facebook/opt-6.7b")
parser.add_argument("--tasks",    nargs="+",
                    default=["copa", "openbookqa", "piqa", "winogrande"])
parser.add_argument("--budgets",  nargs="+", type=float,
                    default=[0.04, 0.10, 0.20, 0.50, 1.00],
                    help="Total KV budget as fraction of seq len. "
                         "1.0 = full cache. Split 50/50 between heavy/recent.")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_fewshot", type=int, default=0,
                    help="Number of few-shot examples (0 = zero-shot)")
parser.add_argument("--out_dir",  default=".")
parser.add_argument("--limit",    type=int, default=None,
                    help="Limit samples per task (for quick testing)")
args = parser.parse_args()

os.makedirs(f"{args.out_dir}/results", exist_ok=True)
os.makedirs(f"{args.out_dir}/plots",   exist_ok=True)

try:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError as e:
    print(f"\n[DEBUG] ERROR: {e}\n")
    sys.exit(1)

# H2O scratch patch
sys.path.insert(0, str(Path(__file__).parent.parent / "h2o_scratch"))
from h2o_attention import patch_model_with_h2o, reset_h2o_caches, H2OAttention
from h2o_authors_wrapper import patch_model_with_authors_h2o

# Load base model
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"\nLoading {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print(f"GPU memory after load: "
      f"{torch.cuda.memory_allocated()/1e9:.1f} GB")


# Helper: Run lm-eval on current model state
def run_eval(model, tokenizer, tasks, label):
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=False,
    )
    out = {}
    for task in tasks:
        r = results["results"][task]
        # lm-eval returns different metric keys per task
        acc = (r.get("acc,none")
               or r.get("acc_norm,none")
               or r.get("acc", 0.0))
        out[task] = round(float(acc) * 100, 2) 
    print(f"  [{label}] " + " | ".join(f"{t}: {out[t]:.1f}%" for t in tasks))
    return out

# Helper: Reload a clean base model
def _load_fresh_model():
    print("Loading fresh base model")
    m = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    m.eval()
    return m


if __name__ == "__main__":
    # PART 1 - Full KV cache baseline
    all_results = {}   # {label: {task: acc_pct}}

    label = "Full cache (baseline)"
    print(f"\n{'='*60}\nRunning: {label}\n{'='*60}")
    t0 = time.time()
    all_results[label] = run_eval(model, tokenizer, args.tasks, label)
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

    with open(f"{args.out_dir}/results/01_accuracy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


    # PART 2 - Scratch H2O at each budget
    scratch_budgets = [b for b in args.budgets if b < 1.0]

    patch_model_with_h2o(model, heavy_ratio=scratch_budgets[0] / 2,
                                recent_ratio=scratch_budgets[0] / 2)
    model._h2o_patched = True

    for budget in scratch_budgets:
        heavy_r = budget / 2.0
        recent_r = budget / 2.0
        label = f"Scratch H2O {int(budget*100)}% budget"

        for m in model.modules():
            if isinstance(m, H2OAttention):
                m.heavy_ratio  = heavy_r
                m.recent_ratio = recent_r
        reset_h2o_caches(model)

        print(f"\n{'='*60}\nRunning: {label}\n{'='*60}")
        t0 = time.time()
        all_results[label] = run_eval(model, tokenizer, args.tasks, label)
        print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

        with open(f"{args.out_dir}/results/01_accuracy_results.json", "w") as f:
            json.dump(all_results, f, indent=2)


    # PART 3 - Authors' H2O at each budget
    del model
    torch.cuda.empty_cache()

    for budget in scratch_budgets:
        heavy_r  = budget / 2.0
        recent_r = budget / 2.0
        label    = f"Authors H2O {int(budget*100)}% Budget"

        print(f"\n{'='*60}\nRunning: {label}\n{'='*60}")

        model_authors = _load_fresh_model()
        model_authors = patch_model_with_authors_h2o(model_authors, heavy_r, recent_r)

        t0 = time.time()
        all_results[label] = run_eval(model_authors, tokenizer, args.tasks, label)
        print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

        del model_authors
        torch.cuda.empty_cache()

        with open(f"{args.out_dir}/results/01_accuracy_results.json", "w") as f:
            json.dump(all_results, f, indent=2)


    # PART 4 - Local-only ablation swept across all budgets
    print(f"\n{'='*60}")
    print("Running: Local-only ablation sweep across all budgets")
    print(f"{'='*60}")

    model_local = _load_fresh_model()
    patch_model_with_h2o(model_local, heavy_ratio=0.0,
                                      recent_ratio=scratch_budgets[0])

    for budget in scratch_budgets:
        label = f"Local-only {int(budget*100)}%"
        for m in model_local.modules():
            if isinstance(m, H2OAttention):
                m.heavy_ratio  = 0.0
                m.recent_ratio = budget
        reset_h2o_caches(model_local)

        print(f"\n--- {label} ---")
        t0 = time.time()
        all_results[label] = run_eval(model_local, tokenizer, args.tasks, label)
        print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

        with open(f"{args.out_dir}/results/01_accuracy_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    del model_local
    torch.cuda.empty_cache()

    # Formatted table
    tasks_upper = [t.upper() for t in args.tasks]
    header_row  = f"{'Method':<35}" + "".join(f"{t:>14}" for t in tasks_upper)
    sep         = "-" * len(header_row)
    lines       = [sep, header_row, sep]
    for method, task_acc in all_results.items():
        row = f"{method:<35}" + "".join(
            f"{task_acc.get(t, 0.0):>13.1f}%" for t in args.tasks
        )
        lines.append(row)
    lines.append(sep)
    table_str = "\n".join(lines)

    with open(f"{args.out_dir}/results/01_accuracy_table.txt", "w") as f:
        f.write(table_str + "\n")
    print("\n=== ACCURACY TABLE ===")
    print(table_str)

    # Plot: accuracy vs budget
    budgets_pct_sorted = sorted([int(b * 100) for b in scratch_budgets])
    use_log = len(budgets_pct_sorted) > 1 

    FAMILIES = [
        ("Scratch H2O", "% budget", -0.5, "#378ADD", "o", "-" ),
        ("Authors H2O", "% Budget", +0.0, "#1D9E75", "s", "--"),
        ("Local-only",  "%",        +0.5, "#D85A30", "^", ":" ),
    ]

    fig, axes = plt.subplots(1, len(args.tasks),
                            figsize=(5 * len(args.tasks), 4),
                            sharey=False)
    if len(args.tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, args.tasks):
        # Full-cache dashed reference
        full_acc = all_results.get("Full cache (baseline)", {}).get(task, None)
        if full_acc is not None:
            ax.axhline(full_acc, color="gray", linestyle="--",
                    linewidth=1.2, alpha=0.6, label="Full Cache")

        for family, key_suffix, jitter, color, marker, lstyle in FAMILIES:
            x_vals, y_vals = [], []
            for b_pct in budgets_pct_sorted:
                key = f"{family} {b_pct}{key_suffix}"
                if key in all_results and task in all_results[key]:
                    x_vals.append(b_pct + jitter)
                    y_vals.append(all_results[key][task])
            if x_vals:
                ax.plot(x_vals, y_vals,
                        marker=marker, linestyle=lstyle,
                        label=family, linewidth=2, color=color,
                        markersize=7, markeredgewidth=1.2,
                        markeredgecolor="white", zorder=3)

        ax.set_xlabel("KV budget (% of Full Cache)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(task.upper(), fontsize=12)

        if use_log:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_xticks(budgets_pct_sorted)
        else:
            ax.set_xticks(budgets_pct_sorted)
            margin = max(5, budgets_pct_sorted[0] * 0.4)
            ax.set_xlim(budgets_pct_sorted[0] - margin,
                        budgets_pct_sorted[-1] + margin)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("H2O Accuracy vs KV Budget - OPT-6.7B", fontsize=13, y=1.01)
    fig.tight_layout()
    plot_path = f"{args.out_dir}/plots/01_accuracy_vs_budget.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved {plot_path}")