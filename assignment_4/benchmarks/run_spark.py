"""
Distributed benchmark: Spark local-cluster, strong scaling, weak scaling and
executor topology (same total cores, different process layout).

For each cluster configuration a fresh SparkSession is started (start-up time
recorded once per session). Per (strategy, n): 1 warm-up (discarded) + R
measured repetitions. Every measured result is validated against a NumPy
reference (tolerance as in tests/test_correctness.py).

Budgets (declared before running): max 120 s per single multiplication,
total resident memory <= 6 GB of the 8 GB host.
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from prng_np import generate_inputs  # noqa: E402
import spark_matmul as sm  # noqa: E402

RES = os.path.join(HERE, "results")
PANELS, GRID = 16, 4          # decomposition fixed across all worker counts
STRONG_N = [512, 1024, 2048, 3072]
WEAK_N = {1: 1536, 2: 1920, 4: 2432, 8: 3072}   # n^3 ~ W * 1536^3 (rounded to x64)
# n = 4096 was attempted first and caused host paging for W >= 2 on the 8 GB
# machine (see results/raw_spark_n4096_attempt.csv); it is outside the budget.
TIME_BUDGET_S = 120.0


def vm_counters():
    """Cumulative macOS VM pageout/swapout counters (pages) from vm_stat."""
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    vals = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            vals[k.strip()] = int(v.strip().rstrip(".") or 0) if v.strip().rstrip(".").isdigit() else 0
    return vals.get("Pageouts", 0), vals.get("Swapouts", 0)


def plan(mode):
    """List of (ClusterConfig, [(experiment, strategy, n)])."""
    jobs = []
    if mode in ("all", "scaling"):
        for w in (1, 2, 4, 8):
            exps = [("strong", s, n) for n in STRONG_N for s in ("bcast", "block2d")]
            if WEAK_N[w] not in STRONG_N:
                exps += [("weak", s, WEAK_N[w]) for s in ("bcast", "block2d")]
            jobs.append((sm.ClusterConfig(workers=w, cores=1), exps))
    if mode in ("all", "topology"):
        for w, c in ((1, 4), (2, 2)):          # 4x1 is measured in the scaling block
            exps = [("topology", s, 2048) for s in ("bcast", "block2d")]
            jobs.append((sm.ClusterConfig(workers=w, cores=c, worker_mem_mb=2048,
                                          executor_mem="1g"), exps))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "scaling", "topology"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(RES, "raw_spark.csv"))
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    inputs, refs = {}, {}

    def get(n):
        if n not in inputs:
            inputs[n] = generate_inputs(n)
            refs[n] = inputs[n][0] @ inputs[n][1]
        return inputs[n], refs[n]

    rows, sessions = [], []
    write_header = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    writer = None
    for cfg, exps in plan(a.mode):
        spark, sc, startup = sm.start_session(cfg)
        ex = sm.executors(sc)
        sessions.append({"config": cfg.label, "workers": cfg.workers, "cores": cfg.cores,
                         "executor_mem": cfg.executor_mem, "startup_s": startup,
                         "executors_registered": len(ex), "spark": spark.version,
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"== cluster {cfg.master}: startup {startup:.2f}s, executors={len(ex)}", flush=True)
        for exp, strat, n in exps:
            (A, B), R = get(n)
            param = PANELS if strat == "bcast" else GRID
            for w in range(a.warmups):
                sm.run_multiply(sc, strat, A, B, param, f"w-{strat}-{n}-{w}", measure=False)
            for rep in range(a.reps):
                tag = f"{cfg.label}-{strat}-{n}-{rep}"
                po0, so0 = vm_counters()
                C, t = sm.run_multiply(sc, strat, A, B, param, tag)
                po1, so1 = vm_counters()
                t["pageouts"], t["swapouts"] = po1 - po0, so1 - so0
                ok = bool(np.all(np.abs(C - R) <= 1e-12 * n + 1e-12 * np.abs(R)))
                row = {"experiment": exp, "strategy": strat, "n": n, "workers": cfg.workers,
                       "cores_per_worker": cfg.cores, "total_cores": cfg.workers * cfg.cores,
                       "decomposition": f"{PANELS} panels" if strat == "bcast" else f"{GRID}x{GRID} grid",
                       "rep": rep, "dtype": "float64", "seed": 42, "warmups": a.warmups,
                       "valid": ok, "checksum": float(C.sum()), "max_abs_err": float(np.max(np.abs(C - R))),
                       "machines": 1, "host": platform.node()}
                row.update({k: v for k, v in t.items()})
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    if write_header:
                        writer.writeheader()
                writer.writerow(row)
                fh.flush()
                rows.append(row)
                print(f"  {exp:8s} {strat:7s} n={n:5d} {cfg.label} rep={rep} e2e={t['e2e']:.3f}s "
                      f"job={t['job']:.3f}s ok={ok} execRSS={t['exec_peak_rss_max']/2**20:.0f}MiB", flush=True)
                if t["e2e"] > TIME_BUDGET_S:
                    print("  time budget exceeded; skipping remaining reps", flush=True)
                    break
        spark.stop()
        inputs.clear(); refs.clear()
        import gc; gc.collect()
        time.sleep(5)
    fh.close()
    sp = os.path.join(RES, "sessions.json")
    old = json.load(open(sp)) if os.path.exists(sp) else []
    json.dump(old + sessions, open(sp, "w"), indent=2)
    print(f"done: {len(rows)} measured rows -> {a.out}")


if __name__ == "__main__":
    main()
