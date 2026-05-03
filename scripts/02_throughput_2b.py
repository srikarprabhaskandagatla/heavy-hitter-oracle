import sys, gc, time, json, os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent / "h2o_scratch"))
from h2o_attention_throughput import patch_model_with_h2o, reset_h2o_caches

MODEL_NAME = "facebook/opt-6.7b"

print(f"\nLoading tokenizer for {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = None

RESULTS_DIR = os.environ.get("RESULTS_DIR")
RESULTS_DIR_PLOTS = os.environ.get("RESULTS_DIR_PLOTS")

SEQ_LENGTHS = [128, 256, 512]
GENERATE_LEN = 128
FIXED_BATCH = 8
WARMUP_RUNS = 2
TIMED_RUNS = 3
RESULTS_FILE_2B = f"{RESULTS_DIR}/02_throughput_2b.json"

def measure_seqlen_throughput(model, prompt_len, label):
    dummy_ids = torch.randint(100, 50000, (1, prompt_len))
    dummy_text = tokenizer.decode(dummy_ids[0])

    inputs = tokenizer(
        [dummy_text] * FIXED_BATCH,
        return_tensors="pt",
        max_length=prompt_len,
        truncation=True,
        padding=True,
    ).to("cuda")

    total_new_tokens = FIXED_BATCH * GENERATE_LEN

    # Warmup
    for _ in range(WARMUP_RUNS):
        try:
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=GENERATE_LEN,
                               do_sample=False, use_cache=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{label}] OOM at prompt_len={prompt_len} — skipping")
            torch.cuda.empty_cache()
            return None, None

    # Timed runs
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
            print(f"  [{label}] OOM during timed run — skipping")
            torch.cuda.empty_cache()
            return None, None

    avg_elapsed    = np.mean(elapsed_times)
    tokens_per_sec = total_new_tokens / avg_elapsed
    peak_mem_gb    = torch.cuda.max_memory_allocated() / 1e9

    print(f"  [{label}] prompt_len={prompt_len}  "
          f"tok/s={tokens_per_sec:>7.1f}  "
          f"peak_mem={peak_mem_gb:.2f} GB")

    return round(tokens_per_sec, 2), round(peak_mem_gb, 3)


all_results_2b = {"full_cache": {}, "h2o_20pct": {}}

# Part 1: Full cache across sequence lengths
print("\n" + "="*60)
print("EXP 2b PART 1: Full Cache — varying sequence length")
print(f"Fixed batch={FIXED_BATCH}, generate={GENERATE_LEN} tokens")
print("="*60)

# Reload clean model
del model
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16,
    device_map="auto")
model.eval()

for seq_len in SEQ_LENGTHS:
    tps, mem = measure_seqlen_throughput(model, seq_len, "Full Cache")
    if tps is not None:
        all_results_2b["full_cache"][seq_len] = {
            "tokens_per_sec": tps, "peak_mem_gb": mem}
    torch.cuda.empty_cache()

with open(RESULTS_FILE_2B, "w") as f:
    json.dump(all_results_2b, f, indent=2)
print("Full cache seq-length results saved.")

# Part 2: H2O across sequence lengths
print("\n" + "="*60)
print("EXP 2b PART 2: H2O 20% — varying sequence length")
print("="*60)

del model
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16,
    device_map="auto")
model.eval()
patch_model_with_h2o(model, heavy_ratio=0.10, recent_ratio=0.10)

for seq_len in SEQ_LENGTHS:
    reset_h2o_caches(model)
    tps, mem = measure_seqlen_throughput(model, seq_len, "H2O 20%")
    if tps is not None:
        all_results_2b["h2o_20pct"][seq_len] = {
            "tokens_per_sec": tps, "peak_mem_gb": mem}
    torch.cuda.empty_cache()

with open(RESULTS_FILE_2B, "w") as f:
    json.dump(all_results_2b, f, indent=2)
print("H2O seq-length results saved.")

# Table 
print("\n" + "="*70)
print("EXP 2b RESULTS — Throughput vs Sequence Length (batch=8)")
print("="*70)
print(f"{'SeqLen':>8} | {'FC tok/s':>10} | {'FC Mem':>8} | "
      f"{'H2O tok/s':>10} | {'H2O Mem':>8} | {'Mem Saved':>10}")
print("-"*70)
for seq_len in SEQ_LENGTHS:
    fc = all_results_2b["full_cache"].get(str(seq_len)) or \
         all_results_2b["full_cache"].get(seq_len)
    h2 = all_results_2b["h2o_20pct"].get(str(seq_len)) or \
         all_results_2b["h2o_20pct"].get(seq_len)
    if fc and h2:
        mem_saved = fc["peak_mem_gb"] - h2["peak_mem_gb"]
        print(f"{seq_len:>8} | {fc['tokens_per_sec']:>10.1f} | "
              f"{fc['peak_mem_gb']:>7.2f}G | "
              f"{h2['tokens_per_sec']:>10.1f} | "
              f"{h2['peak_mem_gb']:>7.2f}G | "
              f"{mem_saved:>+9.2f}G")
    else:
        print(f"{seq_len:>8} | {'OOM':>10} | {'':>8} | {'OOM':>10}")
print("="*70)
print("\nKey question: does memory gap widen at longer sequences?")


RESULTS_FILE_2B = f"{RESULTS_DIR}/02_throughput_2b.json"
with open(RESULTS_FILE_2B) as f:
    results = json.load(f)

fc = results["full_cache"]
h2 = results["h2o_20pct"]
seq_lens = SEQ_LENGTHS

def get(d, k):
    return d.get(str(k)) or d.get(k)

fc_tps = [get(fc, s)["tokens_per_sec"] for s in seq_lens if get(fc, s)]
h2_tps = [get(h2, s)["tokens_per_sec"] for s in seq_lens if get(h2, s)]
fc_mem = [get(fc, s)["peak_mem_gb"]    for s in seq_lens if get(fc, s)]
h2_mem = [get(h2, s)["peak_mem_gb"]    for s in seq_lens if get(h2, s)]
valid_lens = [s for s in seq_lens if get(fc, s) and get(h2, s)]

FC_COLOR  = "#2C7BB6"
H2O_COLOR = "#1D9E75"

# Throughput plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(valid_lens, fc_tps, marker="o", color=FC_COLOR,
        linewidth=2, label="Full Cache")
ax.plot(valid_lens, h2_tps, marker="o", color=H2O_COLOR,
        linewidth=2, label="H2O 20% Budget")
for x, y in zip(valid_lens, fc_tps):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=10, color=FC_COLOR)
for x, y in zip(valid_lens, h2_tps):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=10, color=H2O_COLOR)
ax.set_xlabel("Prompt Length (tokens)", fontsize=12)
ax.set_ylabel("Throughput (tokens / sec)", fontsize=12)
ax.set_title(f"Throughput vs Sequence Length - opt-6.7b\n"
             f"(Batch={FIXED_BATCH}, Generate={GENERATE_LEN} tokens)",
             fontsize=13)
ax.set_xticks(valid_lens)
ax.set_xticklabels([str(s) for s in valid_lens])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
fig.tight_layout()
path = f"{RESULTS_DIR_PLOTS}/02_throughput_2b.png"
fig.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.close()

# Memory plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(valid_lens, fc_mem, marker="o", color=FC_COLOR,
        linewidth=2, label="Full Cache")
ax.plot(valid_lens, h2_mem, marker="o", color=H2O_COLOR,
        linewidth=2, label="H2O 20% Budget")
ax.fill_between(valid_lens, fc_mem, h2_mem,
                alpha=0.12, color=H2O_COLOR, label="Memory saved by H2O")
for x, fc_y, h2_y in zip(valid_lens, fc_mem, h2_mem):
    saved = fc_y - h2_y
    ax.annotate(f"−{saved:.2f} GB", (x, (fc_y + h2_y) / 2),
                textcoords="offset points", xytext=(10, 0),
                ha="left", fontsize=10, color=H2O_COLOR, style="italic")
ax.set_xlabel("Prompt Length (tokens)", fontsize=12)
ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
ax.set_title(f"Peak Memory vs Sequence Length - opt-6.7b\n"
             f"(Batch={FIXED_BATCH}, Generate={GENERATE_LEN} tokens)",
             fontsize=13)
ax.set_xticks(valid_lens)
ax.set_xticklabels([str(s) for s in valid_lens])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
fig.tight_layout()
path = f"{RESULTS_DIR_PLOTS}/02_throughput_memory_2b.png"
fig.savefig(path, dpi=150)
print(f"Saved: {path}")
plt.close()
print("Both Exp 2b plots saved.")