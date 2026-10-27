#!/usr/bin/env python3
"""
Empirical Benchmarking Harness for Part B: Sparse Matrices & Sparsity Scaling.
Measures Gustavson SpGEMM (CSR) across matrix sizes n and sparsity levels rho.
Compares directly against Dense Reordered (i-k-j) on matched logical inputs.
Tracks:
  - Kernel time (median & IQR)
  - Conversion cost (dense-to-CSR)
  - Input non-zero count (NNZ_in) & Output non-zero count (NNZ_out)
  - Non-zero fill-in ratio (NNZ_out / NNZ_in)
  - Peak RSS and theoretical CSR structural storage bytes
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

SPARSE_BIN = os.path.join(PROJECT_DIR, "src", "sparse", "matmul_sparse")
DENSE_BIN = os.path.join(PROJECT_DIR, "src", "dense", "matmul_dense")

def build_all():
    subprocess.run(["make", "-C", os.path.join(PROJECT_DIR, "src", "sparse")], check=True)
    subprocess.run(["make", "-C", os.path.join(PROJECT_DIR, "src", "dense")], check=True)

def run_sparse(n: int, sparsity: float, seed: int = 42, warmups: int = 1, repetitions: int = 5) -> dict:
    cmd = [
        SPARSE_BIN,
        "--n", str(n),
        "--sparsity", str(sparsity),
        "--seed", str(seed),
        "--warmups", str(warmups),
        "--repetitions", str(repetitions),
        "--json"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

def run_dense_ref(n: int, seed: int = 42, warmups: int = 1, repetitions: int = 3) -> dict:
    cmd = [
        DENSE_BIN,
        "--method", "reordered",
        "--n", str(n),
        "--seed", str(seed),
        "--warmups", str(warmups),
        "--repetitions", str(repetitions),
        "--json"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

def main():
    build_all()
    sizes = [256, 512, 1024, 2048]
    sparsities = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    seed = 42
    warmups = 1
    repetitions = 5

    raw_records = []
    summary_records = []

    print("=" * 85)
    print("STARTING SPARSE CSR & BREAK-EVEN BENCHMARKS (Part B)")
    print(f"Sizes: {sizes} | Sparsities: {[f'{s*100:.1f}%' for s in sparsities]}")
    print("=" * 85)

    # First, collect dense reordered reference times for each N
    dense_ref_times = {}
    print("\n>>> Collecting Dense Reordered (i-k-j) baseline reference times ...")
    for n in sizes:
        d_res = run_dense_ref(n, seed=seed, warmups=1, repetitions=3)
        dense_ref_times[n] = d_res["time_median_sec"]
        print(f"  Dense N={n:4d}: Median = {dense_ref_times[n]:8.6f} s")

    for n in sizes:
        t_dense_ref = dense_ref_times[n]
        dense_storage_bytes = 3 * n * n * 8 # 3 matrices of doubles

        print(f"\n>>> Benchmarking Sparse Scaling for N = {n} (Dense Ref: {t_dense_ref:.4f} s) ...")

        for s in sparsities:
            print(f"  Sparsity {s*100:5.1f}% ... ", end="", flush=True)
            res = run_sparse(n, s, seed=seed, warmups=warmups, repetitions=repetitions)

            raw_times = res["raw_times"]
            med_time = res["kernel_time_median_sec"]
            iqr_time = res["kernel_time_iqr_sec"]
            conv_time = res["conversion_time_sec"]
            nnz_a = res["nnz_a"]
            nnz_b = res["nnz_b"]
            nnz_c = res["nnz_c"]
            fill_in = res["fill_in_ratio"]
            out_sparsity = res["output_sparsity"]
            csr_bytes = res["csr_storage_bytes"]
            peak_rss = res["peak_rss_bytes"]
            checksum = res["checksum"]

            speedup_kernel = t_dense_ref / med_time if med_time > 0 else 1.0
            speedup_total = t_dense_ref / (med_time + conv_time) if (med_time + conv_time) > 0 else 1.0
            mem_reduction = 1.0 - (float(csr_bytes) / float(dense_storage_bytes))

            for rep_idx, t_val in enumerate(raw_times):
                raw_records.append({
                    "assignment": 2,
                    "category": "sparse",
                    "n": n,
                    "target_sparsity": s,
                    "output_sparsity": out_sparsity,
                    "seed": seed,
                    "repetition": rep_idx + 1,
                    "kernel_time_sec": t_val,
                    "conversion_time_sec": conv_time,
                    "nnz_in": nnz_a + nnz_b,
                    "nnz_out": nnz_c,
                    "fill_in_ratio": fill_in,
                    "csr_storage_bytes": csr_bytes,
                    "peak_rss_bytes": peak_rss,
                    "checksum": checksum
                })

            summary_records.append({
                "assignment": 2,
                "category": "sparse",
                "n": n,
                "target_sparsity": s,
                "target_density": 1.0 - s,
                "output_sparsity": out_sparsity,
                "nnz_a": nnz_a,
                "nnz_b": nnz_b,
                "nnz_c": nnz_c,
                "fill_in_ratio": fill_in,
                "seed": seed,
                "repetitions": repetitions,
                "kernel_time_median_sec": med_time,
                "kernel_time_iqr_sec": iqr_time,
                "conversion_time_sec": conv_time,
                "dense_reordered_time_sec": t_dense_ref,
                "speedup_kernel_vs_dense": speedup_kernel,
                "speedup_total_vs_dense": speedup_total,
                "csr_storage_bytes": csr_bytes,
                "dense_storage_bytes": dense_storage_bytes,
                "memory_reduction_ratio": mem_reduction,
                "peak_rss_mb": peak_rss / (1024.0 * 1024.0),
                "checksum": checksum
            })

            print(f"Time: {med_time:8.6f} s | Speedup: {speedup_kernel:7.2f}x | Fill-in: {fill_in:4.2f}x | OutSparsity: {out_sparsity*100:5.1f}%")

    raw_df = pd.DataFrame(raw_records)
    raw_path = os.path.join(RESULTS_DIR, "raw_sparse_benchmarks.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\n[SAVED] Raw sparse measurements: {raw_path} ({len(raw_df)} rows)")

    sum_df = pd.DataFrame(summary_records)
    sum_path = os.path.join(RESULTS_DIR, "summary_sparse_benchmarks.csv")
    sum_df.to_csv(sum_path, index=False)
    print(f"[SAVED] Aggregated sparse summaries: {sum_path} ({len(sum_df)} configurations)")

if __name__ == "__main__":
    main()
