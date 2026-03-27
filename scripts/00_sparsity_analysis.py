import argparse, json, os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="facebook/opt-6.7b")
parser.add_argument("--n_samples",  type=int, default=20)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--top_ratios", nargs="+", type=float,
                    default=[0.05, 0.10, 0.20, 0.50],
                    help="Fractions of tokens to treat as heavy hitters")
parser.add_argument("--out_dir",    default=".")
args = parser.parse_args()

os.makedirs(f"{args.out_dir}/plots",   exist_ok=True)
os.makedirs(f"{args.out_dir}/results", exist_ok=True)

if __name__ == "__main__":
    print(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"{n_layers} layers | device map: {model.hf_device_map}")

    # Dataset
    print(f"Loading WikiText-103")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
    texts = [t for t in dataset["text"] if len(t.split()) >= 80]
    texts = texts[:args.n_samples]
    print(f"Using {len(texts)} samples")

    layer_mass = {l: {r: [] for r in args.top_ratios} for l in range(n_layers)}

    layer_col_sums = {l: [] for l in range(n_layers)}

    for idx, text in enumerate(texts):
        print(f"sample {idx+1}/{len(texts)}.", end="\r")
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (1, heads, seq, seq) per layer
        for l, attn in enumerate(outputs.attentions):
            attn_mean = attn[0].float().mean(dim=0).cpu().numpy()
            col_sum = attn_mean.sum(axis=0)  
            col_sum = col_sum / (col_sum.sum() + 1e-9)

            for r in args.top_ratios:
                k = max(1, int(len(col_sum) * r))
                top_mass = float(np.sort(col_sum)[::-1][:k].sum())
                layer_mass[l][r].append(top_mass)

            # Store the per-token normalised scores for the CDF plot.
            layer_col_sums[l].append(col_sum.copy())

        # free attention tensors immediately
        del outputs
        torch.cuda.empty_cache()
    print("\nDone collecting attention stats.")

    # Aggregate
    stats = {}
    for l in range(n_layers):
        stats[l] = {}
        for r in args.top_ratios:
            vals = layer_mass[l][r]
            stats[l][r] = {"mean": float(np.mean(vals)),
                        "std":  float(np.std(vals))}

    # save numeric summary
    with open(f"{args.out_dir}/results/00_sparsity_stats.json", "w") as f:
        json.dump({str(k): {str(rr): vv for rr, vv in v.items()}
                for k, v in stats.items()}, f, indent=2)
    print("Saved results/sparsity_stats.json")

    # ========== PLOTTING ==========
    # Plot 1: per-layer line plot 
    layers = list(range(n_layers))
    colors = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for r, c in zip(args.top_ratios, colors):
        means = [stats[l][r]["mean"] for l in layers]
        stds  = [stats[l][r]["std"]  for l in layers]
        ax.plot(layers, means, label=f"Top {int(r*100)}% tokens", linewidth=2, color=c)
        ax.fill_between(layers,
                        [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)],
                        alpha=0.12, color=c)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Fraction of Total Attention Mass", fontsize=12)
    ax.set_title(f"Layer-wise Attention Sparsity - {args.model.split('/')[-1]} "
                f"on WikiText-103 ({len(texts)} samples)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path1 = f"{args.out_dir}/plots/00_sparsity_layerwise.png"
    fig.savefig(path1, dpi=150)
    print(f"Saved {path1}")


    # Plot 2: heatmap - layer × top-ratio 
    heat = np.array([[stats[l][r]["mean"] for r in args.top_ratios]
                    for l in layers])

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    im = ax2.imshow(heat.T, aspect="auto", vmin=0, vmax=1,
                    cmap="YlOrRd", origin="lower")
    ax2.set_xlabel("Layer index", fontsize=12)
    ax2.set_yticks(range(len(args.top_ratios)))
    ax2.set_yticklabels([f"Top {int(r*100)}%" for r in args.top_ratios])
    ax2.set_title("Attention Mass Concentration by Layer", fontsize=13)
    fig2.colorbar(im, ax=ax2, label="Fraction of Attention Mass")
    fig2.tight_layout()
    path2 = f"{args.out_dir}/plots/00_sparsity_heatmap.png"
    fig2.savefig(path2, dpi=150)
    print(f"Saved {path2}")


    # Plot 3: CDF of accumulated token attention scores
    rep_layers = [
        0,
        n_layers // 3,
        2 * n_layers // 3,
        n_layers - 1,
    ]
    rep_colors = colors[:len(rep_layers)]

    fig3, ax3 = plt.subplots(figsize=(9, 6))

    for l, c in zip(rep_layers, rep_colors):
        pooled = np.concatenate(layer_col_sums[l])  
        pooled = pooled / (pooled.sum() + 1e-9)
        sorted_scores = np.sort(pooled)[::-1]            
        cdf           = np.cumsum(sorted_scores)
        x_frac        = np.linspace(0, 1, len(cdf))
        ax3.plot(x_frac, cdf, label=f"Layer {l}", linewidth=2, color=c)

    ax3.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                label="50% Attention Mass")
    ax3.set_xlabel("Fraction of Tokens (sorted by attention received, descending)",
                fontsize=12)
    ax3.set_ylabel("Cumulative attention mass", fontsize=12)
    ax3.set_title(
        f"Heavy-Hitter Distribution: CDF of Accumulated Attention Scores\n"
        f"{args.model.split('/')[-1]} on WikiText-103 ({len(texts)} samples)",
        fontsize=12,
    )
    ax3.legend(fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.25)
    fig3.tight_layout()
    path3 = f"{args.out_dir}/plots/00_sparsity_cdf.png"
    fig3.savefig(path3, dpi=150)
    print(f"Saved {path3}")


    # Print summary for report
    print("\n======== REPORT ========")
    for r in args.top_ratios:
        avg = np.mean([stats[l][r]["mean"] for l in layers])
        print(f"  Top {int(r*100):2d}% of tokens carry on average "
            f"{avg*100:.1f}% of total attention mass (across all layers)")
    print("==========================\n")