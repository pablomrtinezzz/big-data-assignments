#!/usr/bin/env python3
"""
Automated Correctness Verification Suite for Assignment 3:
Parallel Matrix Multiplication and Optional Vectorisation.

Validates:
  1. Analytical edge cases (Zero matrix, Identity matrix).
  2. Numerical equivalence against NumPy BLAS reference across sizes, seeds, and thread counts.
  3. Safe concurrent access and race-free consistency across OpenMP threads.
"""

import sys
import os
import subprocess
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_DIR)

from common.prng import PRNG

PARALLEL_BIN = os.path.join(PROJECT_DIR, "src", "matmul_parallel")

def build_parallel():
    make_cmd = ["make", "-C", os.path.join(PROJECT_DIR, "src")]
    res = subprocess.run(make_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Build failed:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

def run_cmd(method: str, n: int, threads: int = 1, seed: int = 42, test_flag: str = None, sched: str = "static") -> dict:
    cmd = [
        PARALLEL_BIN,
        "--method", method,
        "--n", str(n),
        "--threads", str(threads),
        "--schedule", sched,
        "--seed", str(seed),
        "--repetitions", "1",
        "--warmups", "0",
        "--json"
    ]
    if test_flag:
        cmd.append(test_flag)
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

def test_analytical_cases():
    print("=" * 70)
    print("TEST SUITE 1: Analytical Edge Cases (Identity & Zero)")
    print("=" * 70)
    methods = ["seq_ikj", "seq_simd", "omp_ikj", "omp_simd", "seq_ijk", "omp_ijk"]
    thread_counts = [1, 2, 4, 8]
    n = 64
    seed = 42

    prng = PRNG(seed)
    A = np.array(prng.fill_matrix(n), dtype=np.float64)
    expected_identity_sum = float(np.sum(A))

    results = []

    for m in methods:
        t_list = [1] if m.startswith("seq") else thread_counts
        for p in t_list:
            # 1. Zero Matrix Test (A * 0 = 0)
            res_zero = run_cmd(m, n, threads=p, seed=seed, test_flag="--test-zero")
            assert abs(res_zero["checksum"]) < 1e-15, f"{m} (p={p}) failed Zero Matrix test!"
            print(f"  [PASS] {m:10s} (p={p:2d}) | A * 0 = 0 (checksum={res_zero['checksum']:.2e})")

            # 2. Identity Matrix Test (A * I = A)
            res_ident = run_cmd(m, n, threads=p, seed=seed, test_flag="--test-identity")
            rel_err = abs(res_ident["checksum"] - expected_identity_sum) / abs(expected_identity_sum)
            assert rel_err < 1e-12, f"{m} (p={p}) failed Identity test (rel_err={rel_err})"
            print(f"  [PASS] {m:10s} (p={p:2d}) | A * I = A (rel_err={rel_err:.2e} < 1e-12)")

            results.append({
                "category": "analytical",
                "method": m,
                "threads": p,
                "n": n,
                "status": "PASSED",
                "zero_checksum": res_zero["checksum"],
                "identity_rel_err": rel_err
            })

    return results

def test_against_numpy_blas():
    print("\n" + "=" * 70)
    print("TEST SUITE 2: Numerical Verification against NumPy BLAS")
    print("=" * 70)
    methods = ["seq_ikj", "seq_simd", "omp_ikj", "omp_simd", "seq_ijk", "omp_ijk"]
    sizes = [16, 32, 64, 128, 256]
    seeds = [42, 101, 777]
    tolerance = 1e-12

    results = []

    for n in sizes:
        for seed in seeds:
            prng_a = PRNG(seed)
            A = np.array(prng_a.fill_matrix(n), dtype=np.float64)
            prng_b = PRNG(seed + 1)
            B = np.array(prng_b.fill_matrix(n), dtype=np.float64)

            # Ground truth via NumPy BLAS
            C_ref = np.matmul(A, B)
            ref_checksum = float(np.sum(C_ref))

            for m in methods:
                threads = 4 if m.startswith("omp") else 1
                res = run_cmd(m, n, threads=threads, seed=seed)
                cand_checksum = res["checksum"]

                rel_err = abs(cand_checksum - ref_checksum) / abs(ref_checksum)
                assert rel_err < tolerance, (
                    f"Numerical mismatch: {m} (p={threads}, n={n}, seed={seed}) "
                    f"rel_err={rel_err:.3e} exceeds tolerance {tolerance}"
                )

                results.append({
                    "category": "numpy_blas_comparison",
                    "method": m,
                    "threads": threads,
                    "n": n,
                    "seed": seed,
                    "rel_err": rel_err,
                    "status": "PASSED"
                })

            print(f"  [PASS] n={n:3d} | seed={seed:3d} | All {len(methods)} kernels match NumPy BLAS (rel_err < 1e-12)")

    return results

def test_concurrency_race_free():
    print("\n" + "=" * 70)
    print("TEST SUITE 3: Multi-Threaded Concurrency & Race Condition Verification")
    print("=" * 70)
    # Check that for the same seed, running with 1, 2, 4, 8, 16 threads produces identical checksums
    n = 128
    seed = 42
    threads_to_test = [1, 2, 4, 6, 8, 12, 16]

    res_seq = run_cmd("seq_ikj", n, threads=1, seed=seed)
    seq_sum = res_seq["checksum"]

    results = []
    for m in ["omp_ikj", "omp_simd"]:
        for p in threads_to_test:
            res_omp = run_cmd(m, n, threads=p, seed=seed)
            omp_sum = res_omp["checksum"]
            rel_diff = abs(omp_sum - seq_sum) / abs(seq_sum)
            assert rel_diff < 1e-12, f"Race condition detected in {m} with {p} threads! rel_diff={rel_diff}"
            print(f"  [PASS] {m:10s} | threads={p:2d} | Checksum match across threads: diff={rel_diff:.2e}")
            results.append({
                "category": "concurrency_consistency",
                "method": m,
                "threads": p,
                "rel_diff_vs_seq": rel_diff,
                "status": "PASSED"
            })

    return results

if __name__ == "__main__":
    build_parallel()
    r1 = test_analytical_cases()
    r2 = test_against_numpy_blas()
    r3 = test_concurrency_race_free()
    total_tests = len(r1) + len(r2) + len(r3)
    print("\n" + "=" * 70)
    print(f"ALL {total_tests} CORRECTNESS AND CONCURRENCY TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
