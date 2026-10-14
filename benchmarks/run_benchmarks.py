#!/usr/bin/env python3
import sys
import os
import subprocess
import json
import time
import csv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
SIZES = [32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
REPETITIONS = 5
SEED = 42
TIME_BUDGET_SEC = 60.0  # Declared budget per repetition

def get_warmup_count(n):
    if n <= 256:
        return 3
    elif n <= 512:
        return 2
    else:
        return 1

IMPLEMENTATIONS = [
    {
        "id": "c_opt",
        "name": "C (Clang -O2)",
        "cmd": lambda n, w, r: ["./src/c/matmul_opt", "--n", str(n), "--seed", str(SEED), "--warmups", str(w), "--repetitions", str(r), "--json"]
    },
    {
        "id": "c_o0",
        "name": "C (Clang -O0)",
        "cmd": lambda n, w, r: ["./src/c/matmul_o0", "--n", str(n), "--seed", str(SEED), "--warmups", str(w), "--repetitions", str(r), "--json"]
    },
    {
        "id": "java",
        "name": "Java (OpenJDK 22)",
        "cmd": lambda n, w, r: ["java", "-cp", "src/java/build", "src.java.MatrixMultiplication", "--n", str(n), "--seed", str(SEED), "--warmups", str(w), "--repetitions", str(r), "--json"]
    },
    {
        "id": "python",
        "name": "Python (CPython 3.14)",
        "cmd": lambda n, w, r: [sys.executable, "src/python/matmul.py", "--n", str(n), "--seed", str(SEED), "--warmups", str(w), "--repetitions", str(r), "--json"]
    }
]

def main():
    print("================================================================================")
    print("STARTING COMPREHENSIVE MATRIX MULTIPLICATION BENCHMARK SUITE")
    print(f"System: macOS Darwin arm64 | Time Budget: {TIME_BUDGET_SEC}s per rep | Repetitions: {REPETITIONS}")
    print(f"Tested sizes: {SIZES}")
    print("================================================================================")

    # Compile binaries
    print("\n[1/4] Ensuring build artifacts are up to date...")
    subprocess.run(["make", "-C", "src/c"], check=True)
    subprocess.run(["make", "-C", "src/java"], check=True)

    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = results_dir / "raw_benchmark_results.csv"
    summary_csv_path = results_dir / "summary_benchmark_results.csv"

    raw_records = []
    summary_records = []

    # Map to track if an implementation has timed out
    timed_out = {impl["id"]: False for impl in IMPLEMENTATIONS}

    for n in SIZES:
        w = get_warmup_count(n)
        print(f"\n--------------------------------------------------------------------------------")
        print(f"BENCHMARKING MATRIX SIZE: N = {n}x{n} (Warmups: {w}, Repetitions: {REPETITIONS})")
        print(f"--------------------------------------------------------------------------------")

        for impl in IMPLEMENTATIONS:
            impl_id = impl["id"]
            impl_name = impl["name"]

            if timed_out[impl_id]:
                print(f"  [{impl_name:<20}] SKIPPED (Exceeded declared budget of {TIME_BUDGET_SEC}s in earlier size)")
                summary_records.append({
                    "n": n,
                    "impl_id": impl_id,
                    "impl_name": impl_name,
                    "status": "TIMEOUT",
                    "repetitions": 0,
                    "warmups": w,
                    "median_time_sec": None,
                    "iqr_time_sec": None,
                    "min_time_sec": None,
                    "max_time_sec": None,
                    "mean_time_sec": None,
                    "std_time_sec": None,
                    "payload_bytes": 3 * n * n * 8,
                    "heap_bytes": None,
                    "peak_rss_bytes": None,
                    "checksum": None
                })
                continue

            cmd = impl["cmd"](n, w, REPETITIONS)
            # Timeout allowance: warmups * budget + reps * budget + 30s buffer
            process_timeout = (w + REPETITIONS) * TIME_BUDGET_SEC + 30.0

            print(f"  [{impl_name:<20}] Running...", end="", flush=True)
            t_start = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=process_timeout,
                    check=True
                )
                t_elapsed = time.time() - t_start
                data = json.loads(proc.stdout)
                times = data["times_sec"]
                
                # Check if any single run exceeded the per-rep budget
                if any(t > TIME_BUDGET_SEC for t in times):
                    print(f" TIMEOUT: repetition exceeded {TIME_BUDGET_SEC}s limit!")
                    timed_out[impl_id] = True
                    status = "TIMEOUT"
                else:
                    status = "SUCCESS"
                    print(f" Done in {t_elapsed:.2f}s | Median: {data['median_time_sec']:.6f}s | IQR: {data['iqr_sec']:.6f}s | Peak RSS: {data.get('peak_rss_bytes', 0)/(1024*1024):.1f}MB")

                # Store raw repetitions
                for rep_idx, t_val in enumerate(times, 1):
                    raw_records.append({
                        "n": n,
                        "impl_id": impl_id,
                        "impl_name": impl_name,
                        "repetition": rep_idx,
                        "time_sec": t_val,
                        "status": status,
                        "payload_bytes": data["payload_bytes"],
                        "heap_bytes": data.get("heap_bytes"),
                        "peak_rss_bytes": data.get("peak_rss_bytes"),
                        "checksum": data["checksum"]
                    })

                # Calculate statistics
                arr = np.array(times)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

                summary_records.append({
                    "n": n,
                    "impl_id": impl_id,
                    "impl_name": impl_name,
                    "status": status,
                    "repetitions": REPETITIONS,
                    "warmups": w,
                    "median_time_sec": data["median_time_sec"],
                    "iqr_time_sec": data["iqr_sec"],
                    "min_time_sec": data["min_time_sec"],
                    "max_time_sec": data["max_time_sec"],
                    "mean_time_sec": mean_val,
                    "std_time_sec": std_val,
                    "payload_bytes": data["payload_bytes"],
                    "heap_bytes": data.get("heap_bytes"),
                    "peak_rss_bytes": data.get("peak_rss_bytes"),
                    "checksum": data["checksum"]
                })

            except subprocess.TimeoutExpired:
                print(f" TIMEOUT: Exceeded {TIME_BUDGET_SEC}s budget!")
                timed_out[impl_id] = True
                summary_records.append({
                    "n": n,
                    "impl_id": impl_id,
                    "impl_name": impl_name,
                    "status": "TIMEOUT",
                    "repetitions": 0,
                    "warmups": w,
                    "median_time_sec": None,
                    "iqr_time_sec": None,
                    "min_time_sec": None,
                    "max_time_sec": None,
                    "mean_time_sec": None,
                    "std_time_sec": None,
                    "payload_bytes": 3 * n * n * 8,
                    "heap_bytes": None,
                    "peak_rss_bytes": None,
                    "checksum": None
                })
            except Exception as e:
                print(f" FAILED: {e}")
                summary_records.append({
                    "n": n,
                    "impl_id": impl_id,
                    "impl_name": impl_name,
                    "status": f"ERROR: {e}",
                    "repetitions": 0,
                    "warmups": w,
                    "median_time_sec": None,
                    "iqr_time_sec": None,
                    "min_time_sec": None,
                    "max_time_sec": None,
                    "mean_time_sec": None,
                    "std_time_sec": None,
                    "payload_bytes": 3 * n * n * 8,
                    "heap_bytes": None,
                    "peak_rss_bytes": None,
                    "checksum": None
                })

    # Write raw CSV
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "impl_id", "impl_name", "repetition", "time_sec", "status",
            "payload_bytes", "heap_bytes", "peak_rss_bytes", "checksum"
        ])
        writer.writeheader()
        writer.writerows(raw_records)
    print(f"\n[INFO] Raw measurements saved to: {raw_csv_path.relative_to(PROJECT_ROOT)}")

    # Write summary CSV
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "impl_id", "impl_name", "status", "repetitions", "warmups",
            "median_time_sec", "iqr_time_sec", "min_time_sec", "max_time_sec",
            "mean_time_sec", "std_time_sec", "payload_bytes", "heap_bytes",
            "peak_rss_bytes", "checksum"
        ])
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"[INFO] Summary statistics saved to: {summary_csv_path.relative_to(PROJECT_ROOT)}")
    print("\nBENCHMARK RUN COMPLETE.")

if __name__ == "__main__":
    main()
