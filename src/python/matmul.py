#!/usr/bin/env python3
import sys
import time
import argparse
import json
import resource
import tracemalloc
from pathlib import Path

# Add project root to sys.path so common.prng is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.prng import PRNG

def matmul_ijk(A, B, C, n: int):
    """
    Standard canonical triple-loop dense matrix multiplication (i-j-k).
    Storage layout: list[list[float]] (nested lists of IEEE 754 floats).
    Arithmetic work: exactly 2 * n^3 operations (n^3 multiplications, n^3 additions).
    """
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s

def compute_checksum(C, n: int) -> float:
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += C[i][j]
    return s

def calculate_iqr(sorted_vals):
    m = len(sorted_vals)
    if m == 1:
        return 0.0
    mid = m // 2
    if mid % 2 == 1:
        q1 = sorted_vals[mid // 2]
        q3 = sorted_vals[m - 1 - (mid // 2)]
    else:
        q1 = 0.5 * (sorted_vals[mid // 2 - 1] + sorted_vals[mid // 2])
        q3 = 0.5 * (sorted_vals[m - mid + mid // 2 - 1] + sorted_vals[m - mid + mid // 2])
    return q3 - q1

def main():
    parser = argparse.ArgumentParser(description="Dense Matrix Multiplication (Python baseline)")
    parser.add_argument("--n", type=int, default=64, help="Matrix dimension N")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--warmups", type=int, default=0, help="Number of warm-up runs")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of measured runs")
    parser.add_argument("--json", action="store_true", help="Output results in JSON")
    parser.add_argument("--print", action="store_true", help="Print matrix C")
    parser.add_argument("--test-identity", action="store_true", help="Set B as identity matrix")
    parser.add_argument("--test-zero", action="store_true", help="Set B as zero matrix")
    args = parser.parse_args()

    n = args.n
    if n <= 0 or args.repetitions <= 0:
        sys.stderr.write(f"Invalid parameters: n={n}, repetitions={args.repetitions}\n")
        sys.exit(1)

    tracemalloc.start()

    prng = PRNG(args.seed)
    A = prng.fill_matrix(n)

    if args.test_identity:
        B = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    elif args.test_zero:
        B = [[0.0 for _ in range(n)] for _ in range(n)]
    else:
        B = prng.fill_matrix(n)

    C = [[0.0 for _ in range(n)] for _ in range(n)]

    heap_bytes = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    payload_bytes = 3 * n * n * 8

    # Warm-up runs
    for _ in range(args.warmups):
        matmul_ijk(A, B, C, n)

    # Measured repetitions
    times_sec = []
    last_checksum = 0.0

    for _ in range(args.repetitions):
        # Reset output matrix
        for i in range(n):
            for j in range(n):
                C[i][j] = 0.0

        t0 = time.perf_counter()
        matmul_ijk(A, B, C, n)
        t1 = time.perf_counter()

        times_sec.append(t1 - t0)
        last_checksum = compute_checksum(C, n)

    sorted_times = sorted(times_sec)
    m = len(sorted_times)
    if m % 2 == 1:
        median_time = sorted_times[m // 2]
    else:
        median_time = 0.5 * (sorted_times[m // 2 - 1] + sorted_times[m // 2])

    iqr_sec = calculate_iqr(sorted_times)
    min_time = sorted_times[0]
    max_time = sorted_times[-1]

    # Peak RSS (bytes on macOS)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if getattr(args, 'print') and n <= 10:
        print(f"Result matrix C ({n}x{n}):")
        for row in C:
            print(" ".join(f"{val:10.6f}" for val in row))

    if args.json:
        result = {
            "language": "python",
            "n": n,
            "seed": args.seed,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "times_sec": times_sec,
            "median_time_sec": median_time,
            "iqr_sec": iqr_sec,
            "min_time_sec": min_time,
            "max_time_sec": max_time,
            "payload_bytes": payload_bytes,
            "heap_bytes": heap_bytes,
            "peak_rss_bytes": peak_rss,
            "checksum": last_checksum
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Python (n={n}): Repetitions={args.repetitions}, Median={median_time:.6f} s, "
              f"IQR={iqr_sec:.6f} s, Min={min_time:.6f} s, Max={max_time:.6f} s, "
              f"Heap={heap_bytes / (1024*1024):.2f} MB, Checksum={last_checksum:.6f}")

if __name__ == "__main__":
    main()
