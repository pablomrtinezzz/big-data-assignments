#!/usr/bin/env python3
"""
Empirical Benchmarking Harness for Part A: Dense Optimisations.
Measures Baseline (i-j-k), Loop Reordered (i-k-j), and Cache Blocked across matrix sizes n.
Follows course protocol:
  - 1 warm-up run (excluded)
  - 5 measured repetitions per configuration
  - Central tendency: Median; Dispersion: IQR
  - Records Peak RSS via getrusage
"""

import sys
import os
import subprocess
import json
import time
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DENSE_BIN = os.path.join(PROJECT_DIR, "src", "dense", "matmul_dense")

def build_dense():
    subprocess.run(["make", "-C", os.path.join(PROJECT_DIR, "src", "dense")], check=True)

def run_single_config(method: str, n: int, seed: int = 42, warmups: int = 1, repetitions: int = 5, block_size: int = 64) -> dict:
    cmd = [
        DENSE_BIN,
        "--method", method,
        "--n", str(n),
        "--seed", str(seed),
        "--warmups", str(warmups),
        "--repetitions", str(repetitions),
        "--block-size", str(block_size),
        "--json"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

def main():
    build_dense()
    methods = ["baseline", "reordered", "blocked"]
    sizes = [32, 64, 128, 256, 512, 1024, 1536, 2048]
    seed = 42
    warmups = 1
    repetitions = 5
    block_size = 64

    raw_records = []
    summary_records = []

    print("=" * 80)
    print("STARTING DENSE MATRIX MULTIPLICATION BENCHMARKS (Part A)")
    print(f"Sizes: {sizes} | Repetitions: {repetitions} | Warmups: {warmups}")
    print("=" * 80)

    for n in sizes:
        print(f"\n>>> Benchmarking N = {n} ...")
        baseline_median = None

        for m in methods:
            # Measure configuration
            print(f"  Method: {m:10s} ... ", end="", flush=True)
            t0 = time.time()
            data = run_single_config(m, n, seed, warmups, repetitions, block_size)
            elapsed = time.time() - t0

            raw_times = data["raw_times"]
            peak_rss = data["peak_rss_bytes"]
            checksum = data["checksum"]

            # Save individual raw repetition rows
            for rep_idx, t_val in enumerate(raw_times):
                raw_records.append({
                    "assignment": 2,
                    "category": "dense",
                    "method": m,
                    "n": n,
                    "block_size": block_size,
                    "seed": seed,
                    "repetition": rep_idx + 1,
                    "time_sec": t_val,
                    "peak_rss_bytes": peak_rss,
                    "checksum": checksum
                })

            med = float(np.median(raw_times))
            q75, q25 = np.percentile(raw_times, [75, 25])
            iqr = float(q75 - q25)
            mean = float(np.mean(raw_times))
            std = float(np.std(raw_times, ddof=1)) if len(raw_times) > 1 else 0.0

            if m == "baseline":
                baseline_median = med
                speedup = 1.0
            else:
                speedup = (baseline_median / med) if (baseline_median and med > 0) else 1.0

            summary_records.append({
                "assignment": 2,
                "category": "dense",
                "method": m,
                "n": n,
                "block_size": block_size,
                "seed": seed,
                "repetitions": repetitions,
                "time_median_sec": med,
                "time_iqr_sec": iqr,
                "time_mean_sec": mean,
                "time_std_sec": std,
                "time_min_sec": min(raw_times),
                "time_max_sec": max(raw_times),
                "speedup_vs_baseline": speedup,
                "peak_rss_mb": peak_rss / (1024.0 * 1024.0),
                "checksum": checksum
            })

            print(f"Median: {med:8.6f} s | IQR: {iqr:8.6f} s | Speedup: {speedup:6.2f}x | RSS: {peak_rss / (1024*1024):5.2f} MB")

    raw_df = pd.DataFrame(raw_records)
    raw_path = os.path.join(RESULTS_DIR, "raw_dense_benchmarks.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\n[SAVED] Raw dense measurements: {raw_path} ({len(raw_df)} rows)")

    sum_df = pd.DataFrame(summary_records)
    sum_path = os.path.join(RESULTS_DIR, "summary_dense_benchmarks.csv")
    sum_df.to_csv(sum_path, index=False)
    print(f"[SAVED] Aggregated dense summaries: {sum_path} ({len(sum_df)} configurations)")

if __name__ == "__main__":
    main()
