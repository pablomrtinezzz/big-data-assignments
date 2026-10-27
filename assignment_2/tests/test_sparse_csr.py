#!/usr/bin/env python3
"""
Automated Correctness Verification Suite for Part B: Sparse Matrices (CSR & SpGEMM).
Validates:
  1. Analytical edge cases (Zero matrix, Identity matrix).
  2. Numerical equivalence & non-zero count (NNZ) against SciPy CSR standard.
"""

import sys
import os
import subprocess
import json
import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_DIR)

from common.prng import PRNG

SPARSE_BIN = os.path.join(PROJECT_DIR, "src", "sparse", "matmul_sparse")

def build_sparse():
    make_cmd = ["make", "-C", os.path.join(PROJECT_DIR, "src", "sparse")]
    res = subprocess.run(make_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Build failed:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

def run_sparse_cmd(n: int, sparsity: float, seed: int = 42, test_flag: str = None) -> dict:
    cmd = [SPARSE_BIN, "--n", str(n), "--sparsity", str(sparsity), "--seed", str(seed), "--json"]
    if test_flag:
        cmd.append(test_flag)
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

def test_sparse_special_cases():
    print("=" * 70)
    print("TEST SUITE 1: Sparse Analytical Special Cases (Identity & Zero)")
    print("=" * 70)
    n = 64
    sparsity = 0.90
    seed = 42

    prng = PRNG(seed)
    A_dense = np.array(prng.fill_sparse_matrix(n, sparsity), dtype=np.float64)
    expected_sum = float(np.sum(A_dense))
    expected_nnz = int(np.count_nonzero(A_dense))

    # Test Zero
    res_zero = run_sparse_cmd(n, sparsity, seed, "--test-zero")
    assert abs(res_zero["checksum"]) < 1e-15, "Sparse failed Zero Matrix test!"
    assert res_zero["nnz_c"] == 0, f"Expected 0 NNZ for Zero test, got {res_zero['nnz_c']}"
    print("  [PASS] Sparse CSR | A * 0 = 0 (exact 0.0, NNZ=0)")

    # Test Identity
    res_ident = run_sparse_cmd(n, sparsity, seed, "--test-identity")
    rel_err = abs(res_ident["checksum"] - expected_sum) / abs(expected_sum)
    assert rel_err < 1e-12, f"Sparse failed Identity test (rel_err={rel_err})"
    assert res_ident["nnz_c"] == expected_nnz, f"NNZ mismatch: expected {expected_nnz}, got {res_ident['nnz_c']}"
    print(f"  [PASS] Sparse CSR | A * I = A (rel_err = {rel_err:.2e}, NNZ matched = {expected_nnz})")

def test_against_scipy():
    print("\n" + "=" * 70)
    print("TEST SUITE 2: Sparse Numerical Verification Against SciPy CSR Reference")
    print("=" * 70)
    print(f"{'N':>5} | {'Sparsity':>8} | {'Seed':>6} | {'SciPy NNZ':>9} | {'C NNZ':>6} | {'Rel Error':>11} | Status")
    print("-" * 65)

    test_configs = [
        (8, 0.50, 42),
        (16, 0.80, 42),
        (32, 0.90, 42),
        (64, 0.95, 42),
        (128, 0.98, 42),
        (256, 0.99, 42),
        (64, 0.90, 999),
        (128, 0.95, 2026),
    ]

    records = []
    tol = 1e-11

    for n, sparsity, seed in test_configs:
        prng_a = PRNG(seed)
        A_dense = np.array(prng_a.fill_sparse_matrix(n, sparsity), dtype=np.float64)
        prng_b = PRNG(seed + 1)
        B_dense = np.array(prng_b.fill_sparse_matrix(n, sparsity), dtype=np.float64)

        A_csr = sp.csr_matrix(A_dense)
        B_csr = sp.csr_matrix(B_dense)

        C_ref = A_csr.dot(B_csr)
        ref_sum = float(C_ref.sum())
        ref_nnz = int(C_ref.nnz)

        res_c = run_sparse_cmd(n, sparsity, seed)

        err_c = abs(res_c["checksum"] - ref_sum) / abs(ref_sum)
        assert err_c < tol, f"Checksum error {err_c} exceeds {tol} for N={n}, rho={sparsity}"
        assert res_c["nnz_c"] == ref_nnz, f"NNZ mismatch: C has {res_c['nnz_c']}, SciPy has {ref_nnz}"

        print(f"{n:5d} | {sparsity*100:7.1f}% | {seed:6d} | {ref_nnz:9d} | {res_c['nnz_c']:6d} | {err_c:11.2e} | PASS")

        records.append({
            "n": n,
            "sparsity": sparsity,
            "seed": seed,
            "ref_nnz": ref_nnz,
            "c_nnz": res_c["nnz_c"],
            "fill_in_ratio": res_c["fill_in_ratio"],
            "rel_err": err_c,
            "status": "PASS"
        })

    return records

if __name__ == "__main__":
    build_sparse()
    test_sparse_special_cases()
    results = test_against_scipy()
    print("\n[SUCCESS] Custom CSR SpGEMM verified against analytical and SciPy references.")
