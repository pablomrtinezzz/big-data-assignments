#!/usr/bin/env python3
"""
Empirical Benchmarking Harness for Assignment 3:
Shared-Memory Parallel Matrix Multiplication.

Evaluates:
  - Strong scaling across matrix sizes n in {64, 128, 256, 512, 1024, 1536, 2048}
  - Thread counts p in {1, 2, 4, 6, 8, 12, 16}
  - Scheduling comparison: static vs dynamic
  - Protocol: 1 warmup run (discarded), 5 measured repetitions
  - Metrics: Median time, IQR dispersion, Speedup S(p), Efficiency E(p), Peak RSS
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

def build_parallel():
    subprocess.run(["make", "-C", os.path.join(PROJECT_DIR, "src")], check=True)

def run_single(method: str, n: int, threads: int = 1, sched: str = "static", chunk_size: int = 0,
               seed: int = 42, warmups: int = 1, repetitions: int = 5) -> dict:
    cmd = [
        PARALLEL_BIN,
        "--method", method,
        "--n", str(n),
        "--threads", str(threads),
        "--schedule", sched,
        "--chunk-size", str(chunk_size),
        "--seed", str(seed),
        "--warmups", str(warmups),
        "--repetitions", str(repetitions),
        "--json"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

def main():
    build_parallel()

    sizes = [64, 128, 256, 512, 1024, 1536, 2048]
    threads_list = [1, 2, 4, 6, 8, 12, 16]
    seed = 42
    warmups = 1
    repetitions = 5

    raw_records = []
    summary_records = []

    print("=" * 80)
    print("STARTING PARALLEL SCALING BENCHMARKS (Assignment 3)")
    print(f"Sizes: {sizes}")
    print(f"Threads: {threads_list}")
    print(f"Repetitions: {repetitions} | Warmups: {warmups}")
    print("=" * 80)

    # Dictionary to hold sequential baseline medians: seq_baselines[n]
    seq_baselines = {}

    for n in sizes:
        print(f"\n>>> Benchmarking size n = {n} ...")

        # 1. Sequential baseline (seq_ikj)
        t0 = time.time()
        res_seq = run_single("seq_ikj", n, threads=1, seed=seed, warmups=warmups, repetitions=repetitions)
        elapsed = time.time() - t0
        med_seq = res_seq["time_median_sec"]
        seq_baselines[n] = med_seq

        print(f"  [SEQUENTIAL] seq_ikj  | Median: {med_seq:.5f}s | IQR: {res_seq['time_iqr_sec']:.5f}s | RSS: {res_seq['peak_rss_bytes']/(1024*1024):.1f}MB")

        for rep_idx, t_val in enumerate(res_seq["raw_times"]):
            raw_records.append({
                "assignment": 3,
                "category": "scaling",
                "method": "seq_ikj",
                "n": n,
                "threads": 1,
                "schedule": "none",
                "chunk_size": 0,
                "repetition": rep_idx + 1,
                "time_sec": t_val,
                "peak_rss_bytes": res_seq["peak_rss_bytes"],
                "checksum": res_seq["checksum"]
            })

        summary_records.append({
            "method": "seq_ikj",
            "n": n,
            "threads": 1,
            "schedule": "none",
            "chunk_size": 0,
            "time_median_sec": med_seq,
            "time_iqr_sec": res_seq["time_iqr_sec"],
            "gflops": res_seq["gflops"],
            "speedup": 1.0,
            "efficiency": 1.0,
            "peak_rss_bytes": res_seq["peak_rss_bytes"],
            "checksum": res_seq["checksum"]
        })

        # 2. Canonical sequential baseline (seq_ijk) for comparison
        if n <= 1024:
            res_seq_ijk = run_single("seq_ijk", n, threads=1, seed=seed, warmups=warmups, repetitions=repetitions)
            med_seq_ijk = res_seq_ijk["time_median_sec"]
            for rep_idx, t_val in enumerate(res_seq_ijk["raw_times"]):
                raw_records.append({
                    "assignment": 3,
                    "category": "canonical",
                    "method": "seq_ijk",
                    "n": n,
                    "threads": 1,
                    "schedule": "none",
                    "chunk_size": 0,
                    "repetition": rep_idx + 1,
                    "time_sec": t_val,
                    "peak_rss_bytes": res_seq_ijk["peak_rss_bytes"],
                    "checksum": res_seq_ijk["checksum"]
                })
            summary_records.append({
                "method": "seq_ijk",
                "n": n,
                "threads": 1,
                "schedule": "none",
                "chunk_size": 0,
                "time_median_sec": med_seq_ijk,
                "time_iqr_sec": res_seq_ijk["time_iqr_sec"],
                "gflops": res_seq_ijk["gflops"],
                "speedup": med_seq / med_seq_ijk if med_seq_ijk > 0 else 0.0,
                "efficiency": (med_seq / med_seq_ijk) if med_seq_ijk > 0 else 0.0,
                "peak_rss_bytes": res_seq_ijk["peak_rss_bytes"],
                "checksum": res_seq_ijk["checksum"]
            })

        # 3. OpenMP parallel scaling (omp_ikj) across all thread counts
        for p in threads_list:
            res_omp = run_single("omp_ikj", n, threads=p, sched="static", seed=seed, warmups=warmups, repetitions=repetitions)
            med_omp = res_omp["time_median_sec"]
            speedup = med_seq / med_omp if med_omp > 0 else 0.0
            efficiency = speedup / p

            print(f"  [PARALLEL]   omp_ikj  | p={p:2d} | Median: {med_omp:.5f}s | Speedup: {speedup:.2f}x | Eff: {efficiency*100:.1f}%")

            for rep_idx, t_val in enumerate(res_omp["raw_times"]):
                raw_records.append({
                    "assignment": 3,
                    "category": "scaling",
                    "method": "omp_ikj",
                    "n": n,
                    "threads": p,
                    "schedule": "static",
                    "chunk_size": 0,
                    "repetition": rep_idx + 1,
                    "time_sec": t_val,
                    "peak_rss_bytes": res_omp["peak_rss_bytes"],
                    "checksum": res_omp["checksum"]
                })

            summary_records.append({
                "method": "omp_ikj",
                "n": n,
                "threads": p,
                "schedule": "static",
                "chunk_size": 0,
                "time_median_sec": med_omp,
                "time_iqr_sec": res_omp["time_iqr_sec"],
                "gflops": res_omp["gflops"],
                "speedup": speedup,
                "efficiency": efficiency,
                "peak_rss_bytes": res_omp["peak_rss_bytes"],
                "checksum": res_omp["checksum"]
            })

        # 4. Scheduling comparison: Dynamic scheduling on heterogeneous cores (p in 4, 8, 16)
        if n >= 512:
            for p in [4, 8, 16]:
                for chunk in [8, 32]:
                    res_dyn = run_single("omp_ikj", n, threads=p, sched="dynamic", chunk_size=chunk,
                                         seed=seed, warmups=warmups, repetitions=repetitions)
                    med_dyn = res_dyn["time_median_sec"]
                    sp_dyn = med_seq / med_dyn if med_dyn > 0 else 0.0
                    eff_dyn = sp_dyn / p

                    for rep_idx, t_val in enumerate(res_dyn["raw_times"]):
                        raw_records.append({
                            "assignment": 3,
                            "category": "scheduling",
                            "method": "omp_ikj",
                            "n": n,
                            "threads": p,
                            "schedule": f"dynamic_{chunk}",
                            "chunk_size": chunk,
                            "repetition": rep_idx + 1,
                            "time_sec": t_val,
                            "peak_rss_bytes": res_dyn["peak_rss_bytes"],
                            "checksum": res_dyn["checksum"]
                        })

                    summary_records.append({
                        "method": "omp_ikj",
                        "n": n,
                        "threads": p,
                        "schedule": f"dynamic_{chunk}",
                        "chunk_size": chunk,
                        "time_median_sec": med_dyn,
                        "time_iqr_sec": res_dyn["time_iqr_sec"],
                        "gflops": res_dyn["gflops"],
                        "speedup": sp_dyn,
                        "efficiency": eff_dyn,
                        "peak_rss_bytes": res_dyn["peak_rss_bytes"],
                        "checksum": res_dyn["checksum"]
                    })

    raw_df = pd.DataFrame(raw_records)
    summary_df = pd.DataFrame(summary_records)

    raw_path = os.path.join(RESULTS_DIR, "raw_parallel_benchmarks.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary_parallel_benchmarks.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("PARALLEL BENCHMARKS COMPLETE")
    print(f"Raw data saved to:     {raw_path} ({len(raw_df)} measurements)")
    print(f"Summary data saved to: {summary_path} ({len(summary_df)} configurations)")
    print("=" * 80)

if __name__ == "__main__":
    main()
