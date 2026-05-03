import os
import sys, gc, time, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent / "h2o_scratch"))
from h2o_attention_throughput import patch_model_with_h2o, reset_h2o_caches

MODEL_NAME = "facebook/opt-6.7b"

RESULTS_DIR = os.environ.get("RESULTS_DIR")
RESULTS_DIR_PLOTS = os.environ.get("RESULTS_DIR_PLOTS")
RESULTS_FILE = f"{RESULTS_DIR}/02_throughput_2a.json"

PROMPT_LEN = 256
GENERATE_LEN = 128
BATCH_SIZES  = [1, 2, 4, 8, 16]
BUDGET = 0.20
WARMUP_RUNS = 2
TIMED_RUNS = 3

print(f"\nLoading tokenizer for {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

dummy_ids  = torch.randint(100, 50000, (1, PROMPT_LEN))
dummy_text = tokenizer.decode(dummy_ids[0])

model = None


def measure_throughput(model, batch_size, label):
    inputs = tokenizer(
        [dummy_text] * batch_size,
        return_tensors="pt",
        max_length=PROMPT_LEN,
        truncation=True,
        padding=True,
    ).to("cuda")

    total_new_tokens = batch_size * GENERATE_LEN

    for _ in range(WARMUP_RUNS):
        try:
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=GENERATE_LEN,
                               do_sample=False, use_cache=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{label}] OOM during warmup at batch_size={batch_size} — skipping")
            torch.cuda.empty_cache()
            return None, None

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    elapsed_times = []

    for _ in range(TIMED_RUNS):
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=GENERATE_LEN,
                               do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            elapsed_times.append(time.perf_counter() - t0)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{label}] OOM during timed run at batch_size={batch_size} — skipping")
            torch.cuda.empty_cache()
            return None, None

    avg_elapsed    = np.mean(elapsed_times)
    tokens_per_sec = total_new_tokens / avg_elapsed
    peak_mem_gb    = torch.cuda.max_memory_allocated() / 1e9

    print(f"  [{label}] batch={batch_size:>2}  "
          f"tok/s={tokens_per_sec:>7.1f}  "
          f"peak_mem={peak_mem_gb:.2f} GB  "
          f"avg_time={avg_elapsed:.2f}s")

    return round(tokens_per_sec, 2), round(peak_mem_gb, 3)


all_results = {"full_cache": {}, "h2o_20pct": {}}

# Part 1: Full cache 
print("\n" + "="*60)
print("EXP 2a PART 1: Full Cache Throughput")
print(f"Fixed prompt={PROMPT_LEN}, generate={GENERATE_LEN} tokens")
print("="*60)

del model
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

for bs in BATCH_SIZES:
    tps, mem = measure_throughput(model, bs, "Full Cache")
    if tps is not None:
        all_results["full_cache"][bs] = {"tokens_per_sec": tps, "peak_mem_gb": mem}
    torch.cuda.empty_cache()

with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print("Full cache results saved.")

# Part 2: H2O 20% ──
print("\n" + "="*60)
print("EXP 2a PART 2: H2O 20% Budget Throughput")
print("="*60)

del model
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()
patch_model_with_h2o(model, heavy_ratio=BUDGET/2, recent_ratio=BUDGET/2)

for bs in BATCH_SIZES:
    reset_h2o_caches(model)
    tps, mem = measure_throughput(model, bs, "H2O 20%")
    if tps is not None:
        all_results["h2o_20pct"][bs] = {"tokens_per_sec": tps, "peak_mem_gb": mem}
    torch.cuda.empty_cache()

with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print("H2O results saved.")

# Table
print("\n" + "="*70)
print("EXP 2a RESULTS — Throughput vs Batch Size (OPT-6.7B)")
print("="*70)
print(f"{'Batch':>6} | {'FC tok/s':>17} | {'FC Mem':>14} | "
      f"{'H2O tok/s':>10} | {'H2O Mem':>8} | {'Speedup':>7}")
print("-"*70)
for bs in BATCH_SIZES:
    fc = all_results["full_cache"].get(bs)
    h2 = all_results["h2o_20pct"].get(bs)
    fc_tps_s = f"{fc['tokens_per_sec']:>10.1f}" if fc else "     OOM"
    fc_mem_s = f"{fc['peak_mem_gb']:>8.2f} GB"  if fc else "     OOM"
    h2_tps_s = f"{h2['tokens_per_sec']:>10.1f}" if h2 else "     OOM"
    h2_mem_s = f"{h2['peak_mem_gb']:>8.2f} GB"  if h2 else "     OOM"
    speedup  = f"{h2['tokens_per_sec']/fc['tokens_per_sec']:.2f}x" if fc and h2 else "   N/A"
    print(f"{bs:>6} | {fc_tps_s:>17} | {fc_mem_s:>14} | "
          f"{h2_tps_s:>10} | {h2_mem_s:>8} | {speedup:>7}")
print("="*70)

# Plots
with open(RESULTS_FILE) as f:
    results = json.load(f)

fc = results["full_cache"]
h2 = results["h2o_20pct"]

def get(d, k):
    return d.get(str(k)) or d.get(k)

common_bs  = [bs for bs in BATCH_SIZES if get(fc, bs) and get(h2, bs)]
fc_tps_v   = [get(fc, bs)["tokens_per_sec"] for bs in common_bs]
h2_tps_v   = [get(h2, bs)["tokens_per_sec"] for bs in common_bs]
fc_mem_v   = [get(fc, bs)["peak_mem_gb"]    for bs in common_bs]
h2_mem_v   = [get(h2, bs)["peak_mem_gb"]    for bs in common_bs]

FC_COLOR  = "#2C7BB6"
H2O_COLOR = "#1D9E75"

# Throughput plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(common_bs, fc_tps_v, marker="o", color=FC_COLOR,
        linewidth=2, label="Full Cache")
ax.plot(common_bs, h2_tps_v, marker="o", color=H2O_COLOR,
        linewidth=2, label="H2O 20% Budget")
for x, y in zip(common_bs, fc_tps_v):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=10, color=FC_COLOR)
for x, y in zip(common_bs, h2_tps_v):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=10, color=H2O_COLOR)
ax.set_xlabel("Batch Size", fontsize=12)
ax.set_ylabel("Throughput (tokens / sec)", fontsize=12)
ax.set_title(f"Throughput vs Batch Size - opt-6.7b\n"
             f"(Prompt={PROMPT_LEN}, Generate={GENERATE_LEN} tokens)",
             fontsize=13)
ax.set_xticks(common_bs)
ax.set_xticklabels([str(bs) for bs in common_bs])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
fig.tight_layout()
path = f"{RESULTS_DIR_PLOTS}/02_throughput_2a.png"
fig.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.close()

# Memory plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(common_bs, fc_mem_v, marker="o", color=FC_COLOR,
        linewidth=2, label="Full Cache")
ax.plot(common_bs, h2_mem_v, marker="o", color=H2O_COLOR,
        linewidth=2, label="H2O 20% Budget")
ax.fill_between(common_bs, fc_mem_v, h2_mem_v,
                alpha=0.12, color=H2O_COLOR, label="Memory saved by H2O")
for x, fc_y, h2_y in zip(common_bs, fc_mem_v, h2_mem_v):
    saved = fc_y - h2_y
    ax.annotate(f"−{saved:.2f} GB", (x, (fc_y + h2_y) / 2),
                textcoords="offset points", xytext=(10, 0),
                ha="left", fontsize=10, color=H2O_COLOR, style="italic")
ax.set_xlabel("Batch Size", fontsize=12)
ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
ax.set_title(f"Peak Memory vs Batch Size - opt-6.7b\n"
             f"(Prompt={PROMPT_LEN}, Generate={GENERATE_LEN} tokens)",
             fontsize=13)
ax.set_xticks(common_bs)
ax.set_xticklabels([str(bs) for bs in common_bs])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
fig.tight_layout()
path = f"{RESULTS_DIR_PLOTS}/02_throughput_memory_2a.png"
fig.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.close()
print("Both Exp 2a plots saved.")
