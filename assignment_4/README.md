# Assignment 4: Distributed Matrix Multiplication (Apache Spark)

**Course:** Big Data (Academic Year 2026–2027) · GCID · ULPGC
**Author:** Pablo Martínez Suárez
**GitHub Repository:** <https://github.com/pablomrtinezzz/big-data-assignments>
**Report:** [`Assignment04_Martinez_Report.pdf`](./Assignment04_Martinez_Report.pdf)

## 1. Research question

How does a distributed dense matrix multiplication implemented with Apache Spark
scale with the number of worker processes, and how much of its end-to-end time is
computation versus distribution, communication and framework overhead, compared
with the validated sequential and OpenMP implementations from Assignment 3?

## 2. What is implemented

| Component | File | Notes |
|---|---|---|
| Local block kernel | `kernels/blockmm.c` | Scalar i-k-j kernel of A3 (`matmul_seq_ikj_scalar`), generalised to `C += A·B` on rectangular blocks; called from Spark tasks via `ctypes` |
| Spark strategy `bcast` | `src/spark_matmul.py` | A split into 16 row panels (RDD partitions), B broadcast once per executor, no shuffle |
| Spark strategy `block2d` | `src/spark_matmul.py` | 4×4 block grid; replicate–join–reduceByKey (MLlib `BlockMatrix` scheme): 2 shuffles |
| Input generator | `src/prng_np.py` | Vectorised SplitMix64, bit-identical to A1–A3 (`seed=42` for A, `43` for B) |
| Sequential + OpenMP baselines | `../assignment_3/src/matmul_parallel.c` | Reused unchanged (commit `fb1d3e8`) |
| Memory sampler | `src/spark_matmul.py::MemorySampler` | `psutil`, 20 ms, peak RSS per executor process tree (JVM + Python daemon + workers) |
| Spark metrics | `src/spark_matmul.py::stage_metrics` | Spark monitoring REST API (run time, shuffle bytes, result size, GC) |

## 3. Environment

Single Apple M2 (4 P + 4 E cores, 8 GB RAM), macOS 26, Python 3.14, PySpark 4.2.0,
OpenJDK 22, Apple clang 21. Exact versions: `benchmarks/results/environment.json`.

**Execution environment label:** *single-machine Spark standalone cluster*
(`local-cluster[W,C,M]`): one Master and W Workers inside the driver JVM, each
Worker launching **one separate executor JVM process** with C task slots and its own
Python worker processes. Executors communicate through the loopback network
(Netty block transfer). There is **1 physical machine**: results characterise
framework, serialisation and process overhead, not inter-machine network
bandwidth or aggregate multi-machine memory.

## 4. Reproduce

```bash
pip install -r requirements.txt      # Java 17+ required by Spark 4
make build                           # libblockmm + A3 binary
make test                            # 55 correctness checks -> tests/correctness_summary.json
make benchmark                       # baselines + Spark (≈45 min) -> benchmarks/results/raw_*.csv
make plots                           # summaries, figures, Typst tables
make report                          # Assignment04_Martinez_Report.pdf (Typst)
```

Individual pieces:

```bash
python3 benchmarks/run_baselines.py --sizes 512,1024,2048,3072,4096 --threads 1,2,4,8
python3 benchmarks/run_spark.py --mode scaling     # strong + weak scaling, W = 1,2,4,8
python3 benchmarks/run_spark.py --mode topology    # 4 cores as 1x4, 2x2 (4x1 in scaling)
```

## 5. Experimental protocol (summary)

* Sizes: n ∈ {512, 1024, 2048, 3072} (strong scaling); weak scaling with
  n³ ∝ W: n = 1536, 1920, 2432, 3072 for W = 1, 2, 4, 8.
* n = 4096 was attempted first and caused host paging for W ≥ 2 on the 8 GB
  machine; those runs are kept in `benchmarks/results/raw_spark_n4096_attempt.csv`
  (full log of that sweep: `aborted_first_pass_*`).
* Per-run system-wide swap-out (`vm_stat`) is recorded; configurations with
  > 64 MiB swap-out are flagged as paging-affected (†) and kept.
* Decomposition fixed across W: 16 row panels (`bcast`), 4×4 grid (`block2d`).
* 1 warm-up (discarded) + 5 measured repetitions per configuration; median and IQR.
* **End-to-end boundary:** A and B in driver memory → partition + serialise →
  distributed job → C assembled in driver memory. Spark session start-up is
  measured separately (`sessions.json`). Input generation is excluded.
* Every measured result is checked against NumPy (`|C−R| ≤ 1e-12·n + 1e-12·|R|`).
* Budgets: ≤ 120 s per multiplication, ≤ 6 GB resident memory in total.
* Thread control: `OMP_NUM_THREADS=1` etc. in executors; 1 task slot per executor.

## 6. Outputs

```
benchmarks/results/raw_baselines.csv   raw A3 baseline timings (sorted per configuration)
benchmarks/results/raw_spark.csv       one row per measured Spark repetition (time phases, Spark metrics, RSS)
benchmarks/results/sessions.json       cluster start-up per configuration
benchmarks/results/summary_*.csv       medians, IQR, speedup, efficiency
report/figures/fig*.png|pdf            figures used in the report
report/tables_autogen.typ              tables used in the report
```

## 7. Main findings

* No Spark configuration beats the sequential C kernel on one machine (best S = 0.86,
  `bcast`, W = 2, n = 3072); OpenMP reaches S = 1.62 (p = 8).
* W = 1 → 2 gives a framework-internal speedup of 1.12–1.54; W ≥ 4 is slower
  because each executor replicates the runtime and B, and the 8 GB host pages.
* `block2d` shuffles 12 matrix volumes per multiplication (403 MB at n = 2048);
  `bcast` moves W·8n² bytes and is faster for W ≤ 8.
* Session start-up: 2.1–6.9 s per cluster configuration.

## 8. AI disclosure

Claude Code (Anthropic) was used to assist with scaffolding the scripts and
drafting the report text. All measurements come from the scripts in this folder;
results and references were checked by the author.
