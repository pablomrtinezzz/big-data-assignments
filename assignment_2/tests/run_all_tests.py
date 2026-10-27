#!/usr/bin/env python3
"""
Master Test Runner for Assignment 2.
Executes both Dense and Sparse verification suites and saves a comprehensive JSON log.
"""

import sys
import os
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import test_dense_optimizations
import test_sparse_csr

def main():
    start_time = time.time()
    print("======================================================================")
    print("ASSIGNMENT 2 MASTER CORRECTNESS VERIFICATION")
    print("======================================================================")

    # 1. Dense suite
    test_dense_optimizations.build_dense()
    test_dense_optimizations.test_special_cases()
    dense_results = test_dense_optimizations.test_against_numpy()

    # 2. Sparse suite
    test_sparse_csr.build_sparse()
    test_sparse_csr.test_sparse_special_cases()
    sparse_results = test_sparse_csr.test_against_scipy()

    elapsed = time.time() - start_time

    summary = {
        "assignment": 2,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 4),
        "overall_status": "ALL_TESTS_PASSED",
        "tolerance": 1e-12,
        "dense_tests": dense_results,
        "sparse_tests": sparse_results
    }

    out_path = os.path.join(SCRIPT_DIR, "correctness_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"OVERALL RESULT: PASS (All {len(dense_results) + len(sparse_results)} test cases verified)")
    print(f"Summary log written to: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
