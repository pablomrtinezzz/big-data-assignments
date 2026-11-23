"""
Correctness suite for the distributed implementations (run before benchmarking).

Checks, on a real multi-executor Spark cluster (local-cluster[2,1,1024]):
  1. SplitMix64 NumPy generator == Assignment 3 C generator (checksum match).
  2. Local C block kernel == NumPy reference on rectangular blocks.
  3. Known answers: identity, zero matrix, hand-computed 2x2 / 3x3.
  4. Both Spark strategies vs. NumPy reference for several n, including n not
     divisible by the number of panels / grid side (ragged blocks), n < panels.
  5. bcast is bit-identical to the sequential kernel (same summation order).

Tolerance for float64 (block2d changes the k-summation order):
  |C - C_ref| <= ATOL + RTOL * |C_ref|, RTOL = 1e-12, ATOL = 1e-12 * n.
Inputs are in [0,1), so entries are O(n/4) and the standard bound
|err| <= n * u * sum|a||b| (u = 1.1e-16) gives ~n^2 u/4, well below these.
Writes tests/correctness_summary.json.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from prng_np import generate_inputs  # noqa: E402
import spark_matmul as sm  # noqa: E402

A3_BIN = os.path.join(ROOT, "..", "assignment_3", "src", "matmul_parallel")
RESULTS = []


def check(name, ok, **info):
    RESULTS.append({"test": name, "passed": bool(ok), **info})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info if not ok else ''}")


def close(C, R, n):
    return bool(np.all(np.abs(C - R) <= 1e-12 * n + 1e-12 * np.abs(R)))


def main():
    # 1. generator equivalence with the Assignment 3 C harness
    if os.path.exists(A3_BIN):
        for n in (16, 128):
            out = subprocess.run([A3_BIN, "--method", "seq_ikj", "--n", str(n), "--repetitions", "1",
                                  "--warmups", "0", "--json"], capture_output=True, text=True, check=True)
            ref = json.loads(out.stdout)["checksum"]
            A, B = generate_inputs(n)
            got = sm.block_matmul(A, B).sum()
            check(f"prng+kernel checksum vs A3 C n={n}", abs(got - ref) <= 1e-9 * abs(ref),
                  a3=ref, a4=float(got))
    else:
        check("A3 binary present", False, path=A3_BIN)

    # 2. local kernel on rectangular blocks
    rng = np.random.default_rng(0)
    for m, k, n in [(1, 1, 1), (3, 5, 7), (64, 17, 33), (128, 128, 128)]:
        a, b = rng.random((m, k)), rng.random((k, n))
        check(f"kernel rect {m}x{k}x{n}", close(sm.block_matmul(a, b), a @ b, k))

    spark, sc, startup = sm.start_session(sm.ClusterConfig(workers=2, cores=1))
    check("cluster has 2 executors", len(sm.executors(sc)) == 2, startup_s=startup)
    try:
        # 3. known answers
        A2 = np.array([[1., 2.], [3., 4.]]); B2 = np.array([[5., 6.], [7., 8.]])
        R2 = np.array([[19., 22.], [43., 50.]])
        A3 = np.arange(1., 10.).reshape(3, 3); R3 = A3 @ A3
        for strat, p in (("bcast", 16), ("block2d", 4)):
            C, _ = sm.run_multiply(sc, strat, A2, B2, p, f"t2-{strat}", measure=False)
            check(f"{strat} 2x2 hand-computed", np.array_equal(C, R2))
            C, _ = sm.run_multiply(sc, strat, A3, A3, p, f"t3-{strat}", measure=False)
            check(f"{strat} 3x3 hand-computed", np.array_equal(C, R3))
            for n in (8, 61):
                A, _ = generate_inputs(n)
                C, _ = sm.run_multiply(sc, strat, A, np.eye(n), p, f"id-{strat}-{n}", measure=False)
                check(f"{strat} A*I=A n={n}", np.array_equal(C, A))
                C, _ = sm.run_multiply(sc, strat, A, np.zeros((n, n)), p, f"z-{strat}-{n}", measure=False)
                check(f"{strat} A*0=0 n={n}", not C.any())

        # 4./5. random inputs, ragged partitions, reference comparison
        for n in (1, 5, 37, 100, 256, 513):
            A, B = generate_inputs(n)
            R = A @ B
            S = sm.block_matmul(A, B)
            for strat, p in (("bcast", 16), ("bcast", 3), ("block2d", 4), ("block2d", 3)):
                C, t = sm.run_multiply(sc, strat, A, B, p, f"r-{strat}-{p}-{n}", measure=False)
                err = float(np.max(np.abs(C - R))) if n else 0.0
                check(f"{strat}(p={p}) vs NumPy n={n}", close(C, R, n), max_abs_err=err)
                if strat == "bcast":
                    check(f"bcast(p={p}) bit-identical to sequential kernel n={n}", np.array_equal(C, S))
    finally:
        spark.stop()

    passed = sum(r["passed"] for r in RESULTS)
    summary = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "passed": passed,
               "total": len(RESULTS), "tolerance": "|C-R| <= 1e-12*n + 1e-12*|R|", "results": RESULTS}
    with open(os.path.join(HERE, "correctness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{passed}/{len(RESULTS)} passed")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
