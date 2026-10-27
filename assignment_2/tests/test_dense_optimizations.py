#!/usr/bin/env python3
"""
Automated Correctness Verification Suite for Part A: Dense Optimisations.
Validates:
  1. Analytical edge cases (Zero matrix, Identity matrix).
  2. Numerical equivalence against NumPy BLAS across matrix sizes and seeds.
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

DENSE_BIN = os.path.join(PROJECT_DIR, "src", "dense", "matmul_dense")

def build_dense():
    make_cmd = ["make", "-C", os.path.join(PROJECT_DIR, "src", "dense")]
    res = subprocess.run(make_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Build failed:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

def run_dense_cmd(method: str, n: int, seed: int = 42, test_flag: str = None, block_size: int = 64) -> dict:
    cmd = [DENSE_BIN, "--method", method, "--n", str(n), "--seed", str(seed), "--json"]
    if test_flag:
        cmd.append(test_flag)
    if method == "blocked":
        cmd.extend(["--block-size", str(block_size)])
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

def test_special_cases():
    print("=" * 70)
    print("TEST SUITE 1: Dense Analytical Special Cases (Identity & Zero)")
    print("=" * 70)
    methods = ["baseline", "reordered", "blocked"]
    n = 64
    seed = 42

    prng = PRNG(seed)
    A = np.array(prng.fill_matrix(n), dtype=np.float64)
    expected_identity_sum = float(np.sum(A))

    for m in methods:
        # Test Zero
        res_zero = run_dense_cmd(m, n, seed, "--test-zero")
        assert abs(res_zero["checksum"]) < 1e-15, f"{m} failed Zero Matrix test!"
        print(f"  [PASS] {m:10s} | A * 0 = 0 (exact 0.0)")

        # Test Identity
        res_ident = run_dense_cmd(m, n, seed, "--test-identity")
        rel_err = abs(res_ident["checksum"] - expected_identity_sum) / abs(expected_identity_sum)
        assert rel_err < 1e-12, f"{m} failed Identity test (rel_err={rel_err})"
        print(f"  [PASS] {m:10s} | A * I = A (rel_err = {rel_err:.2e})")

def test_against_numpy():
    print("\n" + "=" * 70)
    print("TEST SUITE 2: Dense Numerical Verification Against NumPy Reference Standard")
    print("=" * 70)
    print(f"{'N':>5} | {'Seed':>6} | {'NumPy Checksum':>16} | {'Baseline Err':>13} | {'Reordered Err':>14} | {'Blocked Err':>12} | Status")
    print("-" * 88)

    test_configs = [
        (4, 42),
        (8, 42),
        (16, 42),
        (32, 42),
        (64, 42),
        (128, 42),
        (256, 42),
        (16, 999),
        (64, 2026),
    ]

    records = []
    tol = 1e-11

    for n, seed in test_configs:
        prng_a = PRNG(seed)
        A = np.array(prng_a.fill_matrix(n), dtype=np.float64)
        prng_b = PRNG(seed + 1)
        B = np.array(prng_b.fill_matrix(n), dtype=np.float64)

        C_ref = A @ B
        ref_sum = float(np.sum(C_ref))

        res_base = run_dense_cmd("baseline", n, seed)
        res_reord = run_dense_cmd("reordered", n, seed)
        res_block = run_dense_cmd("blocked", n, seed, block_size=32 if n < 64 else 64)

        err_base = abs(res_base["checksum"] - ref_sum) / abs(ref_sum)
        err_reord = abs(res_reord["checksum"] - ref_sum) / abs(ref_sum)
        err_block = abs(res_block["checksum"] - ref_sum) / abs(ref_sum)

        assert err_base < tol, f"Baseline error {err_base} exceeds {tol} for N={n}"
        assert err_reord < tol, f"Reordered error {err_reord} exceeds {tol} for N={n}"
        assert err_block < tol, f"Blocked error {err_block} exceeds {tol} for N={n}"

        print(f"{n:5d} | {seed:6d} | {ref_sum:16.6f} | {err_base:13.2e} | {err_reord:14.2e} | {err_block:12.2e} | PASS")

        records.append({
            "n": n,
            "seed": seed,
            "ref_checksum": ref_sum,
            "err_baseline": err_base,
            "err_reordered": err_reord,
            "err_blocked": err_block,
            "status": "PASS"
        })

    return records

if __name__ == "__main__":
    build_dense()
    test_special_cases()
    results = test_against_numpy()
    print("\n[SUCCESS] All dense implementations verified against analytical and NumPy references.")
