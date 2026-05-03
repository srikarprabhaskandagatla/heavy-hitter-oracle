import os, sys, gc, math, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTAttention

sys.path.insert(0, str(Path(__file__).parent.parent / "h2o_scratch"))
from h2o_attention_ppl import H2OAttentionPPL

sys.path.insert(0, str(Path(__file__).parent.parent / "h2o_authors" / "h2o_hf" / "utils_hh"))
from modify_opt import OPTAttention_Mask

MODEL_NAME = "facebook/opt-6.7b"
CONTEXT_LEN = 512
TARGET_LEN = 256
N_WINDOWS = 20
STRIDE = 256
DEVICE = "cuda"

RESULTS_DIR = os.environ.get("RESULTS_DIR")
RESULTS_DIR_PLOTS = os.environ.get("RESULTS_DIR_PLOTS")
RESULTS_FILE = f"{RESULTS_DIR}/06_perplexity.json"

DATASETS = ["wikitext-103-v1", "wikitext-2-v1"]

CONDITIONS = [
    {"name": "Full Cache",      "mode": "full",   "heavy": None,  "recent": None},
    {"name": "Authors H2O 20%", "mode": "authors","heavy": 0.10,  "recent": 0.10},
    {"name": "Ours H2O 20%",    "mode": "ours",   "heavy": 0.10,  "recent": 0.10},
    {"name": "Ours H2O 50%",    "mode": "ours",   "heavy": 0.25,  "recent": 0.25},
    {"name": "Local-only 20%",  "mode": "ours",   "heavy": 0.00,  "recent": 0.20},
]

CONDITION_ORDER = [c["name"] for c in CONDITIONS]

H2O_COLOR = "#1D9E75"
FC_COLOR = "#2C7BB6"
AUTHORS_COLOR = "#9B59B6"
LOCAL_COLOR = "#D85A30"
OURS50_COLOR = "#378ADD"

COLORS = {
    "Full Cache":      FC_COLOR,
    "Authors H2O 20%": AUTHORS_COLOR,
    "Ours H2O 20%":    H2O_COLOR,
    "Ours H2O 50%":    OURS50_COLOR,
    "Local-only 20%":  LOCAL_COLOR,
}


def patch_ours(model, heavy_frac, recent_frac):
    patched = 0
    for m in model.modules():
        if type(m) is OPTAttention:
            m.__class__    = H2OAttentionPPL
            m.heavy_ratio  = heavy_frac
            m.recent_ratio = recent_frac
            m._acc_scores  = None
            patched += 1
    budget_h = max(1, int(heavy_frac * CONTEXT_LEN)) if heavy_frac > 0 else 0
    budget_r = max(1, int(recent_frac * CONTEXT_LEN))
    print(f"  [Ours]    Patched {patched} layers  "
          f"heavy={budget_h} recent={budget_r} "
          f"({100*(budget_h+budget_r)/CONTEXT_LEN:.0f}% of {CONTEXT_LEN})")


def patch_authors(model, heavy_frac, recent_frac):
    patched = 0
    for m in model.modules():
        if type(m) is OPTAttention:
            m.__class__             = OPTAttention_Mask
            m.heavy_budget_ratio    = heavy_frac
            m.recent_budget_ratio   = recent_frac
            # State fields expected by OPTAttention_Mask.forward
            m.attention_masks_next  = None
            m.heavy_budget          = None
            m.recent_budget         = None
            m.cache_budget          = None
            m.previous_scores       = None
            m.input_length          = []
            m.cache_budget_records  = []
            patched += 1
    budget_h = int(heavy_frac  * CONTEXT_LEN)
    budget_r = int(recent_frac * CONTEXT_LEN)
    print(f"  [Authors] Patched {patched} layers  "
          f"heavy={budget_h} recent={budget_r} "
          f"({100*(budget_h+budget_r)/CONTEXT_LEN:.0f}% of {CONTEXT_LEN})")


def unpatch_model(model):
    for m in model.modules():
        if isinstance(m, (H2OAttentionPPL, OPTAttention_Mask)):
            m.__class__ = OPTAttention
            for attr in ("heavy_ratio", "recent_ratio", "_acc_scores", "_prefill_len",
                         "heavy_budget_ratio", "recent_budget_ratio",
                         "attention_masks_next", "heavy_budget", "recent_budget",
                         "cache_budget", "previous_scores",
                         "input_length", "cache_budget_records"):
                m.__dict__.pop(attr, None)


def reset_scores(model):
    for m in model.modules():
        if isinstance(m, H2OAttentionPPL):
            m._acc_scores = None
            m._prefill_len = 0
        elif isinstance(m, OPTAttention_Mask):
            m._reset_masks()


# Perplexity computation 
@torch.no_grad()
def compute_perplexity(model, all_ids, label, needs_reset=False):
    total_nll, total_tokens, windows_done = 0.0, 0, 0
    n_avail = all_ids.shape[1]

    for begin in range(0, n_avail - CONTEXT_LEN - TARGET_LEN, STRIDE):
        if windows_done >= N_WINDOWS:
            break

        context = all_ids[:, begin : begin + CONTEXT_LEN].to(DEVICE)
        targets = all_ids[:, begin + CONTEXT_LEN :
                              begin + CONTEXT_LEN + TARGET_LEN].to(DEVICE)

        if needs_reset:
            reset_scores(model)

        out = model(context, use_cache=True)
        past_kv = out.past_key_values

        total_nll += F.cross_entropy(
            out.logits[:, -1, :], targets[:, 0], reduction="sum").item()
        total_tokens += 1

        for t in range(TARGET_LEN - 1):
            out     = model(targets[:, t:t+1],
                            past_key_values=past_kv,
                            use_cache=True)
            past_kv = out.past_key_values
            total_nll    += F.cross_entropy(
                out.logits[:, -1, :], targets[:, t+1], reduction="sum").item()
            total_tokens += 1

        windows_done += 1
        print(f"  [{label}] win {windows_done}/{N_WINDOWS}  "
              f"ppl={math.exp(total_nll/total_tokens):.2f}")

        del past_kv, out
        gc.collect(); torch.cuda.empty_cache()

    ppl = math.exp(total_nll / total_tokens)
    print(f"  [{label}]  FINAL PPL = {ppl:.3f}")
    return ppl

def load_data(dataset_name):
    from datasets import load_dataset
    print(f"Loading WikiText ({dataset_name}, validation) ...")
    ds  = load_dataset("wikitext", dataset_name, split="validation")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    text = " ".join(t for t in ds["text"] if len(t.split()) >= 80)
    ids  = tok(text, return_tensors="pt").input_ids
    print(f"  Tokens: {ids.shape[1]:,}")
    return ids


def load_model():
    gc.collect(); torch.cuda.empty_cache()
    print(f"Loading {MODEL_NAME} ...")
    m = OPTForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={"": 0})
    m.eval()
    print(f"  VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return m


def plot_results(all_results, dataset_name):
    short = dataset_name.replace("-v1", "").replace("wikitext-", "wt")
    os.makedirs(f"{RESULTS_DIR_PLOTS}/06_perplexity", exist_ok=True)

    names = [n for n in CONDITION_ORDER if n in all_results.get(dataset_name, {})]
    ppls  = [all_results[dataset_name][n] for n in names]
    fc    = all_results[dataset_name].get("Full Cache", ppls[0])
    x     = list(range(len(names)))

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.axhline(fc, color=FC_COLOR, linestyle="--", linewidth=2,
               alpha=0.8, label=f"Full Cache ({fc:.2f})")

    best_idx = ppls.index(min(ppls))

    for xi, (name, val) in enumerate(zip(names, ppls)):
        color = COLORS.get(name, H2O_COLOR)
        ax.plot(xi, val, marker="o", color=color, markersize=8, zorder=4)

    ax.plot(x, ppls, linewidth=2, color=H2O_COLOR, label="PPL", zorder=3)
    ax.plot(x[best_idx], ppls[best_idx], marker="o", color="#F0A500",
            markersize=13, zorder=5,
            label=f"Best: {names[best_idx]} ({ppls[best_idx]:.2f})")

    for xi, (name, val) in enumerate(zip(names, ppls)):
        ax.annotate(f"{val:.2f}", (xi, val), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10, color=COLORS.get(name, H2O_COLOR))

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=10)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title(
        f"Perplexity vs KV-Cache Policy — {dataset_name}\n"
        f"OPT-6.7B · context={CONTEXT_LEN}, target={TARGET_LEN} · {N_WINDOWS} windows",
        fontsize=13)

    all_vals = ppls + [fc]
    ax.set_ylim(max(0, min(all_vals) - 5), max(all_vals) + 10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = f"{RESULTS_DIR_PLOTS}/06_perplexity/06_perplexity_{short}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_combined(all_results):
    os.makedirs(f"{RESULTS_DIR_PLOTS}/06_perplexity", exist_ok=True)
    datasets_done = [d for d in DATASETS if d in all_results]
    if len(datasets_done) < 2:
        return

    import numpy as np
    x_labels = [d.replace("-v1", "").replace("wikitext-", "WikiText-") for d in datasets_done]
    conds_present = [c for c in CONDITIONS
                     if all(all_results[d].get(c["name"]) is not None for d in datasets_done)]

    n_datasets = len(datasets_done)
    n_conds    = len(conds_present)
    width      = 0.8 / n_conds
    x          = np.arange(n_datasets)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cond in enumerate(conds_present):
        name   = cond["name"]
        vals   = [all_results[d][name] for d in datasets_done]
        offset = (i - n_conds / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width=width * 0.9,
                        color=COLORS.get(name, H2O_COLOR), label=name, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                    color=COLORS.get(name, H2O_COLOR))

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title("Perplexity Comparison Across Datasets\n"
                 f"OPT-6.7B · context={CONTEXT_LEN}, target={TARGET_LEN} · {N_WINDOWS} windows",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = f"{RESULTS_DIR_PLOTS}/06_perplexity/06_perplexity_combined.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        print(f"Resuming — already have: {list(all_results.keys())}")
    else:
        all_results = {}
        print("No existing results. Running all conditions...\n")

    model = load_model()

    for dataset_name in DATASETS:
        if dataset_name not in all_results:
            all_results[dataset_name] = {}

        all_ids = load_data(dataset_name)
        done    = all_results[dataset_name]

        for cond in CONDITIONS:
            name   = cond["name"]
            mode   = cond["mode"]
            heavy  = cond["heavy"]
            recent = cond["recent"]

            if name in done:
                print(f"  Skipping [{dataset_name}] {name} — already computed")
                continue

            print("\n" + "="*60)
            print(f"Dataset: {dataset_name}  |  Condition: {name}")
            print("="*60)

            if mode == "full":
                print("  Full Cache — no eviction")
                needs_reset = False
            elif mode == "authors":
                patch_authors(model, heavy, recent)
                needs_reset = True
            else:
                patch_ours(model, heavy, recent)
                needs_reset = True

            t0 = time.time()
            done[name] = compute_perplexity(model, all_ids, name, needs_reset=needs_reset)
            print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

            if mode != "full":
                unpatch_model(model)

            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            print("Saved.")

    del model; gc.collect(); torch.cuda.empty_cache()

    # Final tables
    for dataset_name in DATASETS:
        if dataset_name not in all_results:
            continue
        res = all_results[dataset_name]
        fc  = res.get("Full Cache", 0)
        print("\n" + "="*65)
        print(f"EXPERIMENT 6 RESULTS — {dataset_name} (OPT-6.7B)")
        print("="*65)
        print(f"{'Condition':<22}  {'PPL':>8}  {'Delta vs Full Cache':>20}")
        print("-"*65)
        for name in CONDITION_ORDER:
            if name not in res:
                continue
            ppl   = res[name]
            delta = f"+{ppl-fc:.3f}" if name != "Full Cache" else "—"
            print(f"{name:<22}  {ppl:>8.3f}  {delta:>20}")
        print("="*65)

    with open(RESULTS_FILE) as f:
        all_results = json.load(f)

    for dataset_name in DATASETS:
        if dataset_name in all_results:
            plot_results(all_results, dataset_name)
    plot_combined(all_results)
    print("\nAll Exp 6 plots saved.")


if __name__ == "__main__":
    main()
