"""
Sequential and shared-memory parallel baselines, re-measured on the same machine
and in the same session window as the distributed runs.

Reuses the validated Assignment 3 harness unchanged
(assignment_3/src/matmul_parallel, git commit fb1d3e8):
  seq_ikj  -- scalar i-k-j sequential kernel (same inner loop as kernels/blockmm.c)
  omp_ikj  -- OpenMP parallel-for over rows, schedule(static), p threads
Timer: CLOCK_MONOTONIC around the kernel only (inputs generated outside).
Memory: getrusage ru_maxrss (peak RSS of the process).
"""
import argparse
import csv
import json
import os
import platform
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
A3_BIN = os.path.abspath(os.path.join(ROOT, "..", "assignment_3", "src", "matmul_parallel"))
OUT = os.path.join(HERE, "results", "raw_baselines.csv")


def run(method, n, threads, reps, warmups):
    cmd = [A3_BIN, "--method", method, "--n", str(n), "--threads", str(threads),
           "--schedule", "static", "--repetitions", str(reps), "--warmups", str(warmups), "--json"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="false")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return json.loads(out.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,1024,2048,4096")
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for n in map(int, a.sizes.split(",")):
        configs = [("seq_ikj", 1)] + [("omp_ikj", int(p)) for p in a.threads.split(",")]
        for method, p in configs:
            r = run(method, n, p, a.reps, a.warmups)
            print(f"{method:8s} n={n:5d} p={p} median={r['time_median_sec']:.4f}s "
                  f"rss={r['peak_rss_bytes']/2**20:.1f}MiB")
            # raw_times are sorted by the C harness; repetition index is order of the sorted sample
            for i, t in enumerate(r["raw_times"]):
                rows.append({"method": method, "n": n, "threads": p, "rep": i, "time_s": t,
                             "peak_rss_bytes": r["peak_rss_bytes"], "checksum": r["checksum"],
                             "seed": r["seed"], "warmups": r["warmups"],
                             "dtype": "float64", "host": platform.node()})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
