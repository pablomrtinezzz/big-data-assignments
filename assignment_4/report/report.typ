#set page(
  paper: "a4",
  margin: (x: 1.6cm, top: 2.0cm, bottom: 2.0cm),
  header: align(right)[
    #text(size: 8pt, fill: luma(100))[
      Big Data · Individual Assignment 4 · Academic Year 2026–2027
    ]
  ],
  footer: context align(center)[
    #text(size: 8pt, fill: luma(100))[
      #counter(page).display("1 of 1", both: true)
    ]
  ]
)
#set text(font: "Times New Roman", size: 9.5pt, lang: "en")
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.1")
#show figure.caption: set text(size: 8pt)
#show table: set text(size: 7.6pt)

#import "tables_autogen.typ": table_strong, table_breakdown, table_startup, table_attempt

#align(center)[
  #v(0.5em)
  #text(size: 15.5pt, weight: "bold")[
    Distributed Matrix Multiplication with Apache Spark: Strong and Weak Scaling, Communication and Framework Overhead on a Single-Machine Standalone Cluster
  ]
  #v(0.8em)
  #text(size: 11pt, weight: "medium")[*Pablo Martínez Suárez*]
  #v(0.2em)
  #text(size: 9pt, style: "italic")[
    Universidad de Las Palmas de Gran Canaria (ULPGC) \
    Grado en Ciencia e Ingeniería de Datos · Big Data (2026–2027) · Individual Assignment 4
  ]
  #v(0.3em)
  #text(size: 9pt)[
    *GitHub Repository:* #link("https://github.com/pablomrtinezzz/big-data-assignments/tree/main/assignment_4")
  ]
  #v(1.0em)
]

#rect(width: 100%, stroke: 0.5pt + luma(180), inset: 10pt, radius: 2pt, fill: luma(250))[
  #text(weight: "bold")[Abstract] —
  We implement dense float64 matrix multiplication with Apache Spark (PySpark 4.2 RDDs) using two partitionings, row panels with a broadcast of $B$ (`bcast`) and a $4 times 4$ block grid joined and reduced through two shuffles (`block2d`), both running the scalar C kernel validated in Assignment 3, and we compare them with the sequential and OpenMP versions of that kernel. Experiments run on a *single-machine Spark standalone cluster* (Apple M2, 8 GB) with $W in {1,2,4,8}$ separate executor processes, $n in {512, ..., 3072}$, 5 measured repetitions after a warm-up, an explicit end-to-end boundary (inputs in driver memory to assembled result in driver memory), Spark task metrics and per-executor RSS sampling. All 210 distributed results agree with NumPy (max. abs. error $6.5 times 10^(-12)$). Spark never beats the sequential kernel on this host: the best configuration (`bcast`, $W=2$, $n=3072$) reaches $S = 0.86$, while OpenMP reaches $S = 1.62$ with 8 threads. At $W=1$ the framework adds 0.38 s (`bcast`) and 1.95 s (`block2d`, 403 MiB of shuffle, 12 times the size of one matrix) to a 1.32 s kernel at $n=2048$, plus 2–7 s of session start-up per cluster. Going from one to two executors gives a framework-internal speedup of 1.12–1.54, but every additional executor holds a private copy of the runtime and of $B$: total resident memory at $n=3072$ grows from 1.9 GB ($W=1$) to 2.2–3.0 GB at $W=8$ (a lower bound, because part of it is already swapped out), the host starts paging at $W >= 4$, and end-to-end time *increases* up to 4.6 times. The aggregate kernel time of `bcast` grows 3.6 times from $W=1$ to $W=4$ because private replicas of $B$ compete for one memory system, whereas the cache-sized blocks of `block2d` grow only 1.7 times. Weak-scaling efficiency ($n^3 prop W$) falls to 0.44–0.71 at $W=2$ and below 0.1 at $W >= 4$. Within one machine, distribution therefore costs time and memory without adding resources; the measurements quantify the overheads a multi-machine deployment would have to amortise, but they do not measure network performance.

  #v(0.4em)
  #text(weight: "bold")[Keywords] — Distributed computing, Apache Spark, RDD, broadcast join, block matrix multiplication, strong scaling, weak scaling, shuffle, serialisation overhead, memory replication.
]

#v(1em)

#columns(2, gutter: 1.5em)[

= Introduction and Objective

Assignments 1–3 measured dense $n times n$ matrix multiplication in a single process: language runtimes (A1), loop reordering, blocking and sparse storage (A2), and OpenMP threads with optional NEON vectorisation (A3). Shared-memory parallelism is bounded by the cores, memory bandwidth and RAM of one machine. Distributed frameworks remove the single-process restriction by partitioning data across *executor* processes that may live on different machines, at the price of serialisation, scheduling and data movement @zaharia2012rdd @dean2008mapreduce.

*Research question.* _How does the end-to-end time of a Spark matrix multiplication scale with the number of executor processes $W$ (strong scaling at fixed $n$, weak scaling at fixed work per worker), which fraction of that time is local computation versus distribution, communication and framework overhead, and how does it compare with the validated sequential and OpenMP implementations of Assignment 3 on the same hardware?_

*Factors:* number of executors $W in {1,2,4,8}$, matrix size $n$, partitioning strategy (row panels + broadcast vs. 2-D blocks + shuffle), and process layout for a fixed core count. *Outcomes:* end-to-end time, phase times, Spark task metrics, data moved, and peak resident memory per executor.

*Hypotheses (stated before measuring).*
H1: for small $n$ the fixed per-job cost of Spark (scheduling, Python worker start, serialisation) dominates and distribution is slower than the sequential baseline.
H2: the broadcast strategy moves less data than the 2-D block strategy for $W <= 8$ and is therefore faster on one machine.
H3: because all executors share one memory system, speedup will be bounded by the same bandwidth ceiling observed for OpenMP in A3, and lower than OpenMP because every executor holds a private copy of $B$.

= Background

*Spark RDDs.* An RDD is an immutable, partitioned collection; transformations are lazy and are executed as *stages* separated by *shuffles*; each partition is processed by one *task* in an executor JVM @zaharia2012rdd @zaharia2016spark. In PySpark, every task's data is pickled in the driver or in the JVM and processed by a separate Python worker process attached to the executor. *Broadcast variables* ship a read-only value once per executor; *shuffles* redistribute key–value records between stages through files and the block-transfer service.

*Distributed matrix multiplication.* For $C = A B$ the classical decompositions are 1-D (row panels of $A$ with replicated $B$) and 2-D (block grids), as in Cannon's algorithm @cannon1969 and SUMMA @vandegeijn1997summa. On a $g times g$ grid, $C_(i j) = sum_k A_(i k) B_(k j)$; MapReduce-style systems implement it by replicating each $A_(i k)$ to the $g$ products $(i,j,k)$ and each $B_(k j)$ to the $g$ products $(i,j,k)$, joining on $(i,j,k)$, and reducing over $k$. This is the scheme of Spark MLlib's `BlockMatrix.multiply` @zadeh2016matrix. Communication volume is the critical cost: 2-D schemes move $Theta(n^2 sqrt(P))$ words versus $Theta(n^2 P)$ for naive replication, and lower bounds exist for memory-constrained algorithms @irony2004communication.

*Scaling metrics.* With $T_"seq"(n)$ the median sequential time and $T_W(n)$ the median distributed time, speedup $S = T_"seq"/T_W$ and efficiency $E = S/W$ @amdahl1967validity. For weak scaling the *work* must be held constant per worker. Dense multiplication costs $2n^3$ flops, so keeping $n$ per worker constant would multiply the work per worker by $W^2$; we instead choose $n_W = n_1 W^(1/3)$ so that $n_W^3 / W$ is constant @gustafson1988reevaluating, and define weak efficiency $E_"weak"(W) = (n_W^3 \/ (W T_W)) \/ (n_1^3 \/ T_1)$.

= Methodology

== Implementations

All variants share the scalar loop-reordered $i$-$k$-$j$ kernel validated in A2–A3 (`matmul_seq_ikj_scalar`, auto-vectorisation disabled), so the arithmetic is identical and only the distribution changes.

/ Sequential C (A3): `seq_ikj`, reused unchanged (commit `fb1d3e8`).
/ OpenMP (A3): `omp_ikj`, `parallel for` over rows, `schedule(static)`, $p in {1,2,4,8}$.
/ Spark `bcast`: $A$ is cut into 16 contiguous row panels, one RDD partition each; $B$ is a broadcast variable; each task computes its $C$ panel with the C kernel (via `ctypes`, shared library `kernels/blockmm.c` shipped with `addFile`) and returns it to the driver with `collect()`. One stage, no shuffle. The summation order equals the sequential kernel, so the result is bit-identical to it.
/ Spark `block2d`: $A$ and $B$ are cut into a $4 times 4$ grid (16 blocks each, 64 block products). `flatMap` replicates blocks to keys $(i,j,k)$; `join` (shuffle 1) pairs them; `map` multiplies with the C kernel; `reduceByKey` (shuffle 2) sums over $k$ in place; `collect()` returns the 16 blocks of $C$. Three stages.

The decomposition (16 panels / $4 times 4$ grid) is *fixed for every $W$*, so the algorithm, the number of tasks and the data volume per task do not change with resources; only placement does. Results are assembled into a contiguous $n times n$ array in the driver.

== Environment and cluster configuration

#figure(
  table(
    columns: (auto, 1fr), stroke: none, inset: (x: 3pt, y: 1.5pt), align: (left, left),
    table.hline(stroke: 0.8pt),
    [CPU], [Apple M2: 4 performance + 4 efficiency cores, 8 logical CPUs (no SMT)],
    [RAM / OS], [8 GB unified memory; macOS 26.6.2 (Darwin 25.6)],
    [Machines], [*1 physical machine* (no network hardware involved)],
    [Framework], [Apache Spark / PySpark 4.2.0, OpenJDK 22.0.2, Python 3.14.7, NumPy 2.5.3],
    [Compiler], [Apple clang 21, `-O3 -std=c99`; A3 binary also `-fopenmp` (libomp)],
    [Master], [`local-cluster[W,C,M]` (Spark standalone master + $W$ workers)],
    [Executors], [$W$ separate executor JVMs, $C$ task slots each, 640 MiB heap (1 GiB for $C>1$)],
    [Python], [1 Python worker process per task slot, reused across tasks],
    [Driver], [2 GiB heap, `maxResultSize` 4 GiB],
    [Threads], [`OMP/OPENBLAS/VECLIB_NUM_THREADS=1` on executors],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Hardware, software and Spark configuration. Exact versions in `results/environment.json`.],
) <tab-env>

We distinguish three levels: *physical machines* (always 1), *workers/executors* (separate OS processes, each a JVM plus Python workers, the unit we scale as $W$) and *task slots/cores* (concurrent tasks per executor). The standalone `local-cluster` master launches each executor as an independent process with its own heap, block manager and shuffle files, communicating with the driver and other executors through Netty over the loopback interface @spark2026standalone. *Environment label: single-machine pseudo-distributed cluster.* It exposes the framework's partitioning, serialisation, scheduling, shuffle and per-process memory behaviour, but it does *not* measure inter-machine network bandwidth or latency, nor aggregate memory capacity of several machines (see @sec-limits). The course specification permits local multi-worker execution only with the instructor's approval; this study is labelled accordingly.

== Inputs, correctness and tolerances

Inputs are float64 matrices filled row-major by SplitMix64 @steele2014splitmix (seed 42 for $A$, 43 for $B$), values in $[0,1)$. The NumPy generator (`src/prng_np.py`) reproduces the A3 C generator bit for bit; this is checked through the A3 checksum. The correctness suite (`tests/test_correctness.py`, 55 checks, all passing) runs on a 2-executor cluster and covers: hand-computed $2 times 2$ and $3 times 3$ products, $A I = A$, $A dot 0 = 0$, ragged partitions ($n in {1,5,37,61,100,513}$ with 3 or 16 panels and $3 times 3$ or $4 times 4$ grids, including empty panels when $n < 16$), comparison with NumPy, and bit-identity of `bcast` with the sequential kernel. Because `block2d` changes the order of the $k$-summation, results may differ in the last bits; we accept $|C - R| <= 10^(-12) n + 10^(-12)|R|$, far above the forward-error bound $n u sum|a||b| approx n^2 u \/ 4$ with $u = 2^(-53)$ @higham2002accuracy and far below any real error. *Every measured repetition* of the benchmark is checked against a NumPy reference with this tolerance.

== Timing boundaries and protocol

- *End-to-end* ($T_"e2e"$, primary metric, monotonic `perf_counter` in the driver): starts with $A$ and $B$ already in driver memory and ends when the full $C$ is assembled in driver memory. It is the sum of three *sequential, non-overlapping* driver phases: *distribute* (slicing, pickling and handing partitions/broadcast to the JVM), *job* (the action: task launch, data fetch, kernels, shuffles, result transfer) and *assemble* (copying returned blocks into $C$).
- *Kernel time*: measured inside each task around the C call and summed over tasks ($Sigma_"kernel"$). Tasks run concurrently, so $Sigma_"kernel"$ is *CPU work*, not wall time; $Sigma_"kernel"\/W$ is the ideal critical-path compute. Inside *job*, compute, (de)serialisation, shuffle I/O and result transfer overlap across tasks, so they are *not* additive; we report the remainder $"job" - Sigma_"kernel"\/W$ as aggregated framework + transfer + imbalance cost.
- *Spark task metrics* (monitoring REST API @spark2026monitoring): shuffle write/read bytes, result size, executor run/deserialise time, JVM GC time, summed over the stages of the job.
- *Session start-up* (creating the SparkSession until all $W$ executors register) is excluded from $T_"e2e"$ and reported separately, once per cluster configuration. Input generation is excluded.
- *Baselines*: the A3 harness times only the kernel (`CLOCK_MONOTONIC`); allocation and input generation are outside. The boundaries therefore differ: baselines exclude data movement because there is none in shared memory. We keep them in separate columns and only compare them explicitly as "kernel-only" vs. "end-to-end".
- *Repetitions*: 1 warm-up (discarded) + 5 measured repetitions per configuration, all raw values kept; we report the median and IQR @hoefler2015scientific. The warm-up absorbs Python worker start-up and the first load of the shared library.

== Resource measurements

*Memory* is sampled every 20 ms with `psutil` during each multiplication: peak RSS of each executor *process tree* (executor JVM + Python daemon + Python workers) and of the driver (Python + driver JVM). RSS of forked Python workers counts shared copy-on-write pages more than once, so the sum over a tree is an upper bound. The baselines report peak RSS (`getrusage`). *Data transfer* is measured with Spark's shuffle and result-size metrics; the broadcast volume is computed analytically ($8n^2$ bytes per executor) because PySpark broadcast traffic is not exposed as a task metric. *Paging* is tracked with the macOS `vm_stat` swap-out counter before and after every repetition.

== Sizes and budgets

Declared before running: at most 120 s per multiplication and at most 6 GB resident in total on the 8 GB host (shared with the desktop session). Strong scaling uses $n in {512, 1024, 2048, 3072}$; weak scaling uses $n_W = 1536 dot W^(1/3)$ rounded to multiples of 64: $n = 1536, 1920, 2432, 3072$ for $W = 1, 2, 4, 8$ (work ratios 1, 1.95, 3.98, 8). A process-layout experiment keeps 4 cores and $n=2048$ fixed and varies executors × cores: $1 times 4$, $2 times 2$, $4 times 1$. A first sweep also tried $n = 4096$; it is reported as the *largest attempted* size in @sec-attempt.

= Results

All 210 measured distributed multiplications passed the tolerance check (largest absolute deviation from NumPy $6.5 times 10^(-12)$; `bcast` results were bit-identical to the sequential kernel). Configurations whose median system-wide swap-out during the run exceeded 64 MiB are marked "†" in tables and with hollow markers in figures; they are reported, not removed.

== Strong scaling and comparison with A3 baselines

#figure(
  image("figures/fig1_strong_scaling.png", width: 100%),
  caption: [Strong scaling. Left/centre: median end-to-end time (error bars: IQR, 5 repetitions) of both Spark strategies vs. $W$, OpenMP kernel time vs. threads $p$, and the sequential kernel (dashed). Right: speedup over the sequential C kernel at $n = 2048$. Hollow markers: host paging during the run.],
  placement: top, scope: "parent",
) <fig-strong>

@fig-strong and @tab-strong answer the primary question. *No Spark configuration is faster than the sequential C kernel.* The best distributed results are obtained with $W = 2$: at $n = 2048$ `bcast` needs 1.585 s ($S = 0.83$) and `block2d` 2.194 s ($S = 0.60$); at $n = 3072$ they reach 5.40 s ($S = 0.86$) and 5.79 s ($S = 0.80$), against 4.65 s sequential and 2.88 s for OpenMP with 8 threads ($S = 1.62$). OpenMP itself reproduces the A3 findings: its $p = 1$ version is about $2 times$ slower than the sequential kernel and it saturates near $S approx 1.6$–$1.8$ because the $i$-$k$-$j$ kernel is memory-bandwidth bound.

Relative to the *same distributed code* on one executor, adding a second executor helps: the framework-internal speedup $T_1\/T_2$ is 1.12–1.28 for `bcast` and 1.40–1.54 for `block2d` for $n >= 512$. Beyond two executors the time *increases* for every size: at $n = 2048$, $W = 8$ is 3.6 (`bcast`) and 3.4 (`block2d`) times slower than $W = 1$, and at $n = 3072$ the slowdown reaches 4.6. These $W >= 4$ points coincide with heavy paging (@sec-memory), so they measure the host's memory limit rather than the algorithm.

#figure(
  table_strong,
  caption: [Strong scaling (5 repetitions, median and IQR in seconds). Baselines: kernel-only time; Spark: end-to-end time. $S = T_"seq"\/T$, $E = S\/p$ or $S\/W$. RSS: process peak (baselines) or largest per-executor peak (Spark, executor JVM + Python workers; driver excluded). † host paging.],
  placement: top, scope: "parent",
) <tab-strong>

== Where the time goes

#figure(
  image("figures/fig2_time_breakdown.png", width: 100%),
  caption: [Decomposition of the median end-to-end time at $n = 2048$. Driver phases are sequential and additive; inside the job, "kernel compute" is the ideal critical path $Sigma_"kernel"\/W$ and the grey remainder aggregates overlapping scheduling, (de)serialisation, shuffle, result transfer and load imbalance. \* host paging.],
  placement: top, scope: "parent",
) <fig-breakdown>

@fig-breakdown and @tab-breakdown separate computation from overhead. At $W = 1$, $n = 2048$, the `bcast` job takes 1.73 s, of which 1.35 s is kernel time; the remaining 0.38 s (20 % of $T_"e2e"$) is task scheduling, pickling 16 panels, shipping $B$ to the Python worker and returning 33.7 MiB of results. The driver phases are small (0.11 s distribution, 0.01 s assembly). `block2d` performs the same arithmetic (1.32 s) but its job takes 3.27 s: replicating every block $g = 4$ times and summing 64 partial products produces 403 MB of shuffle writes per multiplication (@fig-comm), 12 times the 33.5 MB of one input matrix, and 1.95 s (58 %) of overhead. Exactly as predicted by the replication scheme, the shuffle volume is $(2g + g) dot 8n^2 = 12 dot 8 n^2$ bytes, independent of $W$.

The kernel time itself is *not* constant. The sum of in-task kernel times for `bcast` grows from 1.35 s ($W=1$) to 2.24 s ($W=2$) and 4.88 s ($W=4$): each executor process holds its own copy of $B$ (32 MiB at $n=2048$), so $W$ processes stream $W$ different copies of $B$ through the shared L2 caches and the single memory controller. For `block2d` the operands are $512 times 512$ blocks (2 MiB) that fit in cache, and the aggregate kernel time grows only from 1.32 to 1.40 s ($W=2$) and 2.23 s ($W=4$). This is the same bandwidth ceiling that limits OpenMP (aggregate CPU time $p dot T_p$ grows from 2.64 s at $p=1$ to 3.35 s at $p=4$ and 5.72 s at $p=8$), worsened by replication.

#figure(
  table_breakdown,
  caption: [Phase times (medians, s) and Spark metrics. Distr./Assem.: driver phases; Job: action wall time; $Sigma"kernel"\/W$: ideal compute critical path. Shuffle and result sizes from the Spark REST API. GC: JVM garbage collection summed over tasks. Swap-out: system-wide during the run. † host paging.],
  placement: top, scope: "parent",
) <tab-breakdown>

Fixed costs dominate small problems (@fig-weak, left, and @tab-strong): at $n = 512$, `bcast` needs 0.15–0.20 s and `block2d` 0.61–0.88 s for a 0.021 s computation, i.e., 7–42 times the sequential time. Creating a session and registering the executors cost a further 2.1–6.9 s per configuration (@tab-startup), more than a whole $n = 2048$ multiplication; any job that is not reused amortises it poorly.

#figure(
  table_startup,
  caption: [Spark session start-up (one sample per cluster configuration), excluded from $T_"e2e"$.],
) <tab-startup>

== Data movement and memory <sec-memory>

#figure(
  image("figures/fig3_communication_memory.png", width: 100%),
  caption: [Left: data moved per multiplication at $W = 4$: analytical broadcast volume of `bcast` ($W dot 8n^2$) and measured shuffle writes of `block2d`, vs. the size of three dense matrices. Right: peak RSS per executor process tree and its sum over executors at $n = 3072$, vs. the OpenMP process.],
  placement: top, scope: "parent",
) <fig-comm>

`bcast` moves $B$ once per executor ($W dot 8 n^2$ bytes, 288 MiB at $W=4$, $n=3072$) and returns $C$ once; `block2d` moves 865 MiB of shuffle data per multiplication at $n=3072$ regardless of $W$. Hence H2 is confirmed for $W <= 8$: the broadcast strategy moves less data and is faster at $W = 1, 2$. Its volume grows linearly with $W$, whereas the 2-D volume grows with the grid side $g$, so the ranking could reverse for large clusters; that regime was not tested.

Memory is the binding constraint on this host. The OpenMP process needs 218 MiB at $n = 3072$. A single Spark executor peaks at 0.70–0.77 GB and the driver at 1.16–1.38 GB, about 9 times more in total, because the data exist simultaneously as NumPy arrays, pickled bytes in the driver JVM, broadcast/shuffle blocks in each executor JVM, and unpickled copies in each Python worker. The sum over executors grows with $W$ (from 0.7 GB to 1.4 GB for `bcast` and 2.2 GB for `block2d` at $W = 8$), while the per-executor RSS *drops* at $W >= 4$ because the operating system is already evicting pages: the system-wide swap-out reached 3.2 GiB (`bcast`) and 7.8 GiB (`block2d`) per multiplication at $W = 4$, $n = 3072$. With the desktop session open, `memory_pressure` reported only 29–34 % of the 8 GB as free (about 2.3–2.7 GB), so the declared 6 GB budget could not be fully used by the experiment.

== Weak scaling and process layout

#figure(
  image("figures/fig4_size_weak_topology.png", width: 100%),
  caption: [Left: median time vs. $n$ for $W = 8$, OpenMP $p = 8$ and sequential C. Centre: weak-scaling efficiency with $n_W = 1536 dot W^(1/3)$ (constant work per worker). Right: 4 cores at $n = 2048$ arranged as executors × cores (bars: median, whiskers: IQR/2). Hollow markers: host paging.],
  placement: top, scope: "parent",
) <fig-weak>

With work per worker held constant ($n = 1536, 1920, 2432, 3072$), the time per unit of work per worker should stay constant. It does not: weak efficiency falls to 0.44 (`bcast`) and 0.71 (`block2d`) at $W=2$ and to 0.03–0.08 at $W = 4, 8$, where paging dominates (@fig-weak, centre). Even before paging, `bcast` loses efficiency faster because its per-executor $B$ replica grows as $n^2$ while the work per worker is fixed.

The layout experiment fixes 4 cores and $n = 2048$ (@fig-weak, right). Two executors with two slots each are fastest for `bcast` (1.86 s, versus 2.22 s for $1 times 4$ and 2.44 s for $4 times 1$); for `block2d` $1 times 4$ and $2 times 2$ are equivalent (2.84 and 2.80 s) and $4 times 1$ is slowest (3.09 s). Fewer, larger executors share one JVM, block manager and broadcast copy, and avoid the per-process memory and start-up costs that made $W = 4$ page. In PySpark each task slot still has its own Python worker with its own unpickled copy of $B$, which explains why $1 times 4$ is not better than $2 times 2$ for `bcast`.

== Largest attempted size: $n = 4096$ <sec-attempt>

The first sweep included $n = 4096$ (three matrices: 403 MB). It met the 120 s limit, but not the memory budget: with $W >= 2$ the host paged heavily (executor RSS fell below 0.3 GB while swap filled) and the IQR grew to 2.5–21 s, so the sweep was stopped after $W = 4$ (`bcast`) and $W = 2$ (`block2d`) and the size was removed from the scaling study. The measured values are kept (`results/raw_spark_n4096_attempt.csv`) and shown in @tab-attempt. *All configurations with $n <= 3072$ met the 120 s limit; the memory budget was the binding one. Paging appeared at $W >= 4$ for almost every size and at $n = 3072$ already for $W = 1$; the region free of paging for all strategies is $W <= 2$, $n <= 2048$. Largest tested size within the time budget: $n = 3072$; largest attempted: $n = 4096$.*

#figure(
  table_attempt,
  caption: [$n = 4096$. Baselines from the final sweep; Spark rows from the aborted first sweep (different session, same machine), not directly comparable with the final medians. $Sigma"kernel"$: aggregate in-task kernel time.],
) <tab-attempt>

= Discussion

*Answer to the research question.* On one machine, Spark's distributed multiplication scales positively only from one to two executors (framework-internal speedup 1.12–1.54) and is always slower than the sequential and OpenMP kernels that perform the same arithmetic (H1 confirmed for all sizes, not only small ones). The gap has three measured causes: (i) fixed framework costs (0.15–0.9 s per job plus seconds of session start-up), (ii) data movement and serialisation that grow with $n^2$ and, for `block2d`, with the replication factor (12 matrix copies of shuffle), and (iii) per-process replication of the runtime and of the operands, which turns the single memory system into the bottleneck: it inflates the kernel time itself and, from $W = 4$, exhausts physical memory. H3 is confirmed: Spark hits the same bandwidth ceiling as OpenMP, earlier and harder because it holds private copies of $B$.

*Why a distributed framework at all?* The study isolates the costs that a real cluster must amortise. On separate machines, executors would add memory bandwidth, caches and RAM instead of competing for them, so the replication that hurts here would buy capacity; but the transfers that use loopback here would cross a network, typically far slower than the 100 GB/s unified memory of the M2. Our measured volumes (e.g., 865 MiB of shuffle or $W dot 72$ MiB of broadcast per multiplication at $n = 3072$) give the input for such an estimate, but no network time was measured and none is claimed. The results also explain the design of practical libraries: MLlib's `BlockMatrix` uses the same replication scheme and relies on large blocks and BLAS kernels, and communication-avoiding algorithms such as SUMMA or 2.5-D reduce exactly the volume that dominates `block2d` @vandegeijn1997summa @irony2004communication.

*Alternative explanations considered.* The slowdown at $W >= 4$ could have been attributed to the efficiency cores of the M2 (2.42 GHz vs. 3.5 GHz for the performance cores, A3). This contributes, but cannot explain a 3–5 times slowdown, and the swap-out counters (up to 8.7 GiB during a 28 s run) show paging. The faster kernel of `block2d` at $W >= 2$ is a cache effect of its smaller operand blocks, an algorithmic difference documented here, not a property of the framework.

== Limitations <sec-limits>

- *One physical machine.* The cluster is pseudo-distributed (separate processes, loopback transport). Inter-machine bandwidth, latency, stragglers and aggregate memory are not measured; local multi-worker execution requires the instructor's approval.
- *Shared, non-dedicated host.* The desktop session (browser, database server, IDE) used part of the 8 GB, so free memory varied; paging was measured with system-wide counters, which also include other processes.
- *Memory metric.* RSS sums over forked Python workers count shared pages repeatedly, and RSS under paging underestimates the footprint. Driver memory is reported separately.
- *Overlapping phases.* Within the job, compute, serialisation and transfers overlap; only the driver phases are additive. The remainder $"job" - Sigma_"kernel"\/W$ mixes framework cost and load imbalance.
- *Timing boundaries differ.* Baselines time only the kernel, Spark times end-to-end; the comparison is therefore conservative in favour of the baselines, but data movement is intrinsic to the distributed version.
- *One session per configuration*, so start-up has a single sample per configuration; 5 repetitions per point.

= Conclusions

Within the tested scope (one 8-core, 8 GB machine; $n <= 3072$; $W <= 8$ executor processes; PySpark RDDs with a scalar C kernel) we establish that:
+ Spark distribution adds 0.15–0.9 s per job, 2–7 s of session start-up and, for the 2-D scheme, 12 matrix volumes of shuffle; as a result it never outperforms the sequential kernel on one machine ($S <= 0.86$), whereas OpenMP reaches $S = 1.62$.
+ The row-panel + broadcast strategy is faster than the 2-D block strategy for $W <= 8$ because it moves less data, but its per-executor copy of $B$ makes its kernel time grow fastest with $W$.
+ Memory, not CPU, limits scaling: each executor process replicates the runtime and operands, total footprint is about 9 times the OpenMP process, and from $W = 4$ the host pages and time increases with resources.
+ For a fixed core count, fewer executors with several cores each are faster than many single-core executors.

What remains unresolved is the multi-machine regime, where replication adds capacity but transfers cross a real network; testing it requires a cluster, BLAS-level kernels and larger $n$ to amortise the fixed costs.

= Reproducibility, Materials and AI Disclosure

Code, raw CSVs (every repetition), session metadata, environment file, analysis script and this report's sources are in the repository (`assignment_4/`): #link("https://github.com/pablomrtinezzz/big-data-assignments/tree/main/assignment_4"). The A3 baselines are reused from commit `fb1d3e8`. `make test`, `make benchmark`, `make plots` and `make report` regenerate every number, figure and table. *Generative AI:* Claude Code (Anthropic) was used to help scaffold scripts and draft text; all measurements were produced by the scripts above and the references and conclusions were checked by the author.

#bibliography("references.bib", style: "ieee")


]
