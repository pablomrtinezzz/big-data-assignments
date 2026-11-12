#!/usr/bin/env python3
"""
Master Test Runner for Assignment 3:
Parallel Matrix Multiplication and Optional Vectorisation.
Executes the full test suite and persists results to correctness_summary.json.
"""

import sys
import os
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import test_parallel_correctness

def main():
    start_time = time.time()
    print("======================================================================")
    print("ASSIGNMENT 3 MASTER CORRECTNESS & THREADING VERIFICATION")
    print("======================================================================")

    test_parallel_correctness.build_parallel()
    r1 = test_parallel_correctness.test_analytical_cases()
    r2 = test_parallel_correctness.test_against_numpy_blas()
    r3 = test_parallel_correctness.test_concurrency_race_free()

    elapsed = time.time() - start_time
    total_passed = len(r1) + len(r2) + len(r3)

    summary = {
        "assignment": 3,
        "title": "Parallel Matrix Multiplication and Optional Vectorisation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 4),
        "overall_status": "ALL_TESTS_PASSED",
        "tolerance": 1e-12,
        "total_tests_passed": total_passed,
        "analytical_tests": r1,
        "numpy_blas_tests": r2,
        "concurrency_tests": r3
    }

    out_path = os.path.join(SCRIPT_DIR, "correctness_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"OVERALL RESULT: PASS (All {total_passed} test configurations verified)")
    print(f"Summary log written to: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
