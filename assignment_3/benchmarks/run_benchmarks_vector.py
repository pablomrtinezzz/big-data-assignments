#!/usr/bin/env python3
"""
Vectorisation Benchmarking Harness for Assignment 3:
Evaluates ARM NEON 128-bit SIMD Vectorisation against Scalar Kernels
across thread counts and matrix sizes.

Demonstrates:
  - Isolated SIMD speedup: S_simd = T_scalar / T_simd
  - Combined Speedup: S_total(p) = S_thread(p) * S_simd
  - Multiplicative interaction between SIMD and multi-threading
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

PARALLEL_BIN = os.path.join(PROJECT_DIR, "src", "matmul_parallel")

def run_single(method: str, n: int, threads: int = 1, seed: int = 42, warmups: int = 1, repetitions: int = 5) -> dict:
    cmd = [
        PARALLEL_BIN,
        "--method", method,
        "--n", str(n),
        "--threads", str(threads),
        "--seed", str(seed),
        "--warmups", str(warmups),
        "--repetitions", str(repetitions),
        "--json"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

def main():
    sizes = [64, 128, 256, 512, 1024, 1536, 2048]
    threads_list = [1, 2, 4, 8, 16]
    seed = 42
    warmups = 1
    repetitions = 5

    raw_records = []
    summary_records = []

    print("=" * 80)
    print("STARTING ARM NEON VECTORISATION BENCHMARKS (Assignment 3 Extension)")
    print(f"Sizes: {sizes}")
    print(f"Threads: {threads_list}")
    print(f"Repetitions: {repetitions} | Warmups: {warmups}")
    print("=" * 80)

    for n in sizes:
        print(f"\n>>> Benchmarking size n = {n} ...")

        # 1. Pure Sequential comparison: seq_ikj (scalar) vs seq_simd (NEON)
        res_seq_scalar = run_single("seq_ikj", n, threads=1, seed=seed, warmups=warmups, repetitions=repetitions)
        res_seq_simd = run_single("seq_simd", n, threads=1, seed=seed, warmups=warmups, repetitions=repetitions)

        t_seq_scalar = res_seq_scalar["time_median_sec"]
        t_seq_simd = res_seq_simd["time_median_sec"]
        simd_seq_speedup = t_seq_scalar / t_seq_simd if t_seq_simd > 0 else 0.0

        print(f"  [SEQUENTIAL] Scalar: {t_seq_scalar:.5f}s | NEON SIMD: {t_seq_simd:.5f}s | SIMD Speedup: {simd_seq_speedup:.2f}x")

        # Record raw sequential
        for rep_idx, t_val in enumerate(res_seq_scalar["raw_times"]):
            raw_records.append({
                "assignment": 3,
                "variant": "scalar",
                "method": "seq_ikj",
                "n": n,
                "threads": 1,
                "repetition": rep_idx + 1,
                "time_sec": t_val,
                "peak_rss_bytes": res_seq_scalar["peak_rss_bytes"]
            })
        for rep_idx, t_val in enumerate(res_seq_simd["raw_times"]):
            raw_records.append({
                "assignment": 3,
                "variant": "simd",
                "method": "seq_simd",
                "n": n,
                "threads": 1,
                "repetition": rep_idx + 1,
                "time_sec": t_val,
                "peak_rss_bytes": res_seq_simd["peak_rss_bytes"]
            })

        summary_records.append({
            "variant": "scalar",
            "method": "seq_ikj",
            "n": n,
            "threads": 1,
            "time_median_sec": t_seq_scalar,
            "time_iqr_sec": res_seq_scalar["time_iqr_sec"],
            "gflops": res_seq_scalar["gflops"],
            "simd_factor": 1.0,
            "total_speedup": 1.0
        })
        summary_records.append({
            "variant": "simd",
            "method": "seq_simd",
            "n": n,
            "threads": 1,
            "time_median_sec": t_seq_simd,
            "time_iqr_sec": res_seq_simd["time_iqr_sec"],
            "gflops": res_seq_simd["gflops"],
            "simd_factor": simd_seq_speedup,
            "total_speedup": simd_seq_speedup
        })

        # 2. Parallel comparison: omp_ikj (scalar) vs omp_simd (NEON) across threads
        for p in threads_list:
            res_omp_scalar = run_single("omp_ikj", n, threads=p, seed=seed, warmups=warmups, repetitions=repetitions)
            res_omp_simd = run_single("omp_simd", n, threads=p, seed=seed, warmups=warmups, repetitions=repetitions)

            t_omp_scalar = res_omp_scalar["time_median_sec"]
            t_omp_simd = res_omp_simd["time_median_sec"]
            simd_factor = t_omp_scalar / t_omp_simd if t_omp_simd > 0 else 0.0
            total_speedup = t_seq_scalar / t_omp_simd if t_omp_simd > 0 else 0.0

            print(f"  [PARALLEL p={p:2d}] Scalar: {t_omp_scalar:.5f}s ({res_omp_scalar['gflops']:.1f} GFLOPS) | "
                  f"SIMD: {t_omp_simd:.5f}s ({res_omp_simd['gflops']:.1f} GFLOPS) | "
                  f"SIMD Gain: {simd_factor:.2f}x | Total: {total_speedup:.2f}x")

            for rep_idx, t_val in enumerate(res_omp_scalar["raw_times"]):
                raw_records.append({
                    "assignment": 3,
                    "variant": "scalar",
                    "method": "omp_ikj",
                    "n": n,
                    "threads": p,
                    "repetition": rep_idx + 1,
                    "time_sec": t_val,
                    "peak_rss_bytes": res_omp_scalar["peak_rss_bytes"]
                })
            for rep_idx, t_val in enumerate(res_omp_simd["raw_times"]):
                raw_records.append({
                    "assignment": 3,
                    "variant": "simd",
                    "method": "omp_simd",
                    "n": n,
                    "threads": p,
                    "repetition": rep_idx + 1,
                    "time_sec": t_val,
                    "peak_rss_bytes": res_omp_simd["peak_rss_bytes"]
                })

            summary_records.append({
                "variant": "scalar",
                "method": "omp_ikj",
                "n": n,
                "threads": p,
                "time_median_sec": t_omp_scalar,
                "time_iqr_sec": res_omp_scalar["time_iqr_sec"],
                "gflops": res_omp_scalar["gflops"],
                "simd_factor": 1.0,
                "total_speedup": t_seq_scalar / t_omp_scalar if t_omp_scalar > 0 else 0.0
            })
            summary_records.append({
                "variant": "simd",
                "method": "omp_simd",
                "n": n,
                "threads": p,
                "time_median_sec": t_omp_simd,
                "time_iqr_sec": res_omp_simd["time_iqr_sec"],
                "gflops": res_omp_simd["gflops"],
                "simd_factor": simd_factor,
                "total_speedup": total_speedup
            })

    raw_df = pd.DataFrame(raw_records)
    summary_df = pd.DataFrame(summary_records)

    raw_path = os.path.join(RESULTS_DIR, "raw_vector_benchmarks.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary_vector_benchmarks.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("VECTORISATION BENCHMARKS COMPLETE")
    print(f"Raw data saved to:     {raw_path} ({len(raw_df)} measurements)")
    print(f"Summary data saved to: {summary_path} ({len(summary_df)} configurations)")
    print("=" * 80)

if __name__ == "__main__":
    main()
