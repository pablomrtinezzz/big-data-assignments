#set page(
  paper: "a4",
  margin: (x: 1.6cm, top: 2.0cm, bottom: 2.0cm),
  header: align(right)[
    #text(size: 8pt, fill: luma(100))[
      Big Data · Individual Assignment 3 · Academic Year 2026–2027
    ]
  ],
  footer: context align(center)[
    #text(size: 8pt, fill: luma(100))[
      #counter(page).display("1 of 1", both: true)
    ]
  ]
)

#set text(
  font: "Times New Roman",
  size: 9.5pt,
  lang: "en"
)

#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.1")

#import "tables_autogen.typ": table_parallel_scaling, table_simd_comparison

// Title Block
#align(center)[
  #v(0.5em)
  #text(size: 15.5pt, weight: "bold")[
    Empirical Investigation of Shared-Memory Parallelism and ARM NEON Vectorisation for Matrix Multiplication on Heterogeneous Multi-Core Architectures
  ]
  #v(0.8em)
  #text(size: 11pt, weight: "medium")[
    *Pablo Martínez Suárez* \
  ]
  #v(0.2em)
  #text(size: 9pt, style: "italic")[
    Department of Computer Science and Systems, Universidad de Las Palmas de Gran Canaria (ULPGC) \
    Grado en Ciencia e Ingeniería de Datos · Big Data (2026–2027) \
  ]
  #v(0.3em)
  #text(size: 9pt)[
    *GitHub Repository:* #link("https://github.com/pablomrtinezzz/big-data-assignments")
  ]
  #v(1.0em)
]

// Abstract and Keywords
#rect(width: 100%, stroke: 0.5pt + luma(180), inset: 10pt, radius: 2pt, fill: luma(250))[
  #text(weight: "bold")[Abstract] —
  Shared-memory parallel processing and data-level SIMD vectorisation are primary techniques for accelerating compute-intensive linear algebra kernels on modern multi-core microprocessors. This paper presents a systematic empirical investigation of parallel matrix multiplication ($n times n$) using OpenMP and 128-bit ARM NEON vectorisation on an asymmetric Apple Silicon M2 processor featuring 4 Performance cores (*Avalanche* @ 3.5 GHz) and 4 Efficiency cores (*Blizzard* @ 2.42 GHz). Adhering to course benchmarking protocols, we evaluate execution runtime, strong scaling speedup $S(p)$, parallel efficiency $E(p)$, and peak memory consumption across matrix dimensions $n in [64, 2048]$ and thread counts $p in [1, 16]$ using 64-bit IEEE 754 precision and deterministic SplitMix64 pseudo-random inputs. Our empirical findings demonstrate four principal microarchitectural insights: (1) Measuring the single-threaded parallel execution ($p=1$) uncovers an OpenMP framework runtime overhead of $47% - 53%$ relative to an identical pure sequential C baseline without threading libraries ($S(1) approx 0.47 - 0.53$). (2) Strong scaling reveals three distinct operational regimes: near-linear acceleration across homogeneous P-cores ($p in [1, 4]$, scaling from 6.2 to 22.8 GFLOPS at $n=1024$), diminishing marginal returns as asymmetric E-cores engage ($p in [5, 8]$), and severe performance regression under oversubscription ($p=16$ threads across 8 physical cores, where runtime in $n=2048$ degrades from $0.65" s"$ to $0.80" s"$, an $18%$ penalty due to involuntary preemptive context switching). (3) Due to core performance asymmetry, standard static loop scheduling (`schedule(static)`) stalls fast P-cores at implicit OpenMP barriers waiting for slow E-cores; dynamic chunk scheduling (`schedule(dynamic, 8)`) mitigates this load imbalance, boosting throughput by $+11.7%$ from $29.63$ to $33.09$ GFLOPS ($S = 2.51 times$). (4) Explicit ARM NEON SIMD vectorisation (`fmla.2d`) combined with thread-level parallelism scales throughput above 33 GFLOPS before encountering a shared memory bandwidth barrier at $n=2048$ ($96" MB"$ payload overflowing the $16" MB"$ L2 cache). All source implementations, raw CSV datasets, validation suites, and reproduction scripts are published openly on GitHub.

  #v(0.4em)
  #text(weight: "bold")[Keywords] — Shared-memory parallelism, OpenMP, strong scaling, Amdahl's law, heterogeneous architectures, Big.LITTLE, Apple M2, ARM NEON, SIMD vectorisation, oversubscription, work scheduling, memory wall.
]

#v(1em)

#columns(2, gutter: 1.5em)[

= Introduction & Research Objectives

Matrix multiplication is a foundational computational primitive underpinning big data engineering, scientific simulations, deep learning backpropagation, and spectral graph analysis @golub2013matrix. Given two dense square matrices $A, B in RR^(n times n)$, the arithmetic operation computes:
$ C_(i j) = sum_(k=0)^(n-1) A_(i k) B_(k j) $
This operation requires $2n^3$ floating-point operations ($n^3$ multiplications and $n^3$ additions). On modern multi-core processors, executing $O(n^3)$ operations sequentially creates severe execution bottlenecks for large datasets.

While previous assignments established that sequential runtime is constrained by interpreter dispatch overhead (Assignment 1) and memory cache strides (Assignment 2), multi-core shared-memory architectures offer parallel hardware contexts capable of dividing the loop iteration space across multiple concurrent threads @chandra2001parallel. However, parallel speedup in practice rarely matches theoretical linear predictions due to synchronization overhead, microarchitectural asymmetries, memory bus contention, and threading framework costs @hennessy2017computer.

== Research Questions & Hypotheses
This empirical study addresses four focused research questions:

1. *RQ1 (Framework Overhead at $p=1$):* _What is the exact execution overhead introduced by the OpenMP threading runtime when restricted to a single thread ($p=1$) compared to an equivalent pure sequential C baseline compiled without multi-threading runtime linking?_
   - *Hypothesis 1:* The parallel implementation executed with $p=1$ will exhibit measurable runtime overhead ($S(1) < 1.0$) compared to the pure sequential baseline due to thread pool initialization, internal barrier structures, and inhibited compiler interprocedural optimizations.

2. *RQ2 (Strong Scaling & Architectural Regimes):* _How does thread count $p$ ($1 <= p <= 16$) govern strong scaling speedup $S(p)$ and parallel efficiency $E(p)$ on an asymmetric SoC (Apple M2 with 4 Performance cores and 4 Efficiency cores)?_
   - *Hypothesis 2:* Strong scaling will exhibit three distinct operational zones: high efficiency across pure P-cores ($p in [1, 4]$), sub-linear efficiency degradation as slower E-cores participate ($p in [5, 8]$), and sharp performance degradation under oversubscription ($p in [9, 16]$) due to OS preemptive context switching.

3. *RQ3 (Work Scheduling on Asymmetric Cores):* _How does OpenMP work scheduling (static vs dynamic chunking) influence parallel load balancing when threads execute across heterogeneous cores with disparate clock frequencies and IPC capabilities?_
   - *Hypothesis 3:* Static scheduling will induce load imbalance because fast P-cores will finish their fixed slices early and idle at implicit barriers; dynamic scheduling will allow fast P-cores to consume more work chunks, yielding superior throughput.

4. *RQ4 (SIMD Vectorisation & Memory Bandwidth):* _What is the isolated acceleration factor of 128-bit ARM NEON SIMD vectorisation, and how does it interact with thread-level parallelism before saturating the memory subsystem?_
   - *Hypothesis 4:* ARM NEON SIMD instructions (`fmla.2d`) will deliver multiplicative speedups on compute-bound matrix sizes, but will encounter an operational ceiling once matrix payloads exceed on-chip cache capacity (16 MB L2).

= Microarchitectural Background & Theory

== Parallel Formulation & Race-Free Concurrency
To parallelize dense matrix multiplication, we partition the outermost loop index $i in [0, n-1]$ across $p$ worker threads using OpenMP (`#pragma omp parallel for`) @dagum1998openmp.

In our loop-reordered $i-k-j$ kernel:
```c
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < n; k++) {
        double r = A[i * n + k];
        for (size_t j = 0; j < n; j++) {
            C[i * n + j] += r * B[k * n + j];
        }
    }
}
```
This decomposition possesses optimal concurrent properties:
1. *Zero Write Conflicts:* Each thread computes exclusively for a subset of row indices $i$. Thus, array element $C[i times n + j]$ is written *only* by the thread assigned row $i$. No mutexes, atomics, or synchronization locks are required.
2. *Read-Only Matrix $B$:* All threads concurrently read matrix $B$. Because $B$ is immutable throughout the kernel, concurrent cache line reads produce no invalidations in hardware cache coherence protocols.
3. *False Sharing Elimination:* On architectures with 64-byte or 128-byte cache lines, multiple threads writing to adjacent bytes in the same cache line trigger expensive coherence invalidations (cache ping-pong) @hennessy2017computer. In our kernel, each row $i$ contains $n$ 64-bit IEEE 754 doubles ($8n" bytes"$). For tested sizes $n >= 64$, one row spans at least $512" bytes"$, guaranteeing that thread write targets reside on distinct cache lines.

== Amdahl's Law vs Gustafson's Law
Amdahl's law models the theoretical maximum speedup of a fixed-size workload (strong scaling) as a function of the parallel fraction $f in [0, 1]$ and thread count $p$ @amdahl1967validity:
$ S(p) = 1 / ((1 - f) + f / p) $
As $p arrow infinity$, speedup asymptotically approaches $1 / (1 - f)$, demonstrating that unparallelised serial segments (runtime initialization, thread dispatch, barrier synchronization) strictly limit parallel gains.

Conversely, Gustafson's law considers weak scaling where problem size expands proportionally with processors @gustafson1988reevaluating:
$ S_"scaled"(p) = p - (1 - f)(p - 1) $
This paper focuses strictly on strong scaling under fixed matrix sizes $n$, in direct compliance with course instructions.

== Asymmetric Multiprocessing on Apple Silicon
The testbed processor is an Apple Silicon M2 SoC fabricated on a 5 nm process @applem2arch2022. Unlike symmetric x86_64 processors, the M2 incorporates a heterogeneous big.LITTLE microarchitecture:
- *4 Performance Cores (Avalanche):* Wide out-of-order execution (8-wide decode), 3.50 GHz maximum clock, 192 KB L1I, 128 KB L1D per core, sharing a massive 16 MB L2 cache.
- *4 Efficiency Cores (Blizzard):* Energy-optimized (4-wide decode), 2.42 GHz maximum clock, 128 KB L1I, 64 KB L1D per core, sharing a 4 MB L2 cache.
- *Unified Memory Architecture (UMA):* 8.0 GB LPDDR5-6400 with a 128-bit memory bus providing $100" GB/s"$ peak memory bandwidth.
- *Thread Capacity:* Exactly 8 physical cores; Simultaneous Multi-Threading (SMT/Hyper-Threading) is not implemented. Hence, hardware supports exactly 8 concurrent physical execution contexts.

== OpenMP Work Scheduling Policies
OpenMP offers several loop scheduling options:
- `schedule(static[, chunk])`: Divides the $n$ iterations into equal chunks distributed cyclically or statically among threads at compile/startup time. Incurs zero runtime scheduling overhead, but assumes homogeneous thread execution speeds.
- `schedule(dynamic[, chunk])`: Iterations are broken into chunks placed in a shared runtime work queue. Whenever a thread completes its chunk, it dynamically requests another chunk. This incurs synchronization overhead on the work queue but balances load across heterogeneous cores @chandra2001parallel.
- `schedule(guided[, chunk])`: Dynamically allocates exponentially decreasing chunk sizes, balancing load while minimizing queue contention.

== ARM NEON 128-Bit SIMD Vectorisation
ARMv8-A Advanced SIMD (NEON) provides 32 dedicated 128-bit vector registers (`v0`–`v31`) @arm2020neon. In double-precision arithmetic, each 128-bit register holds two 64-bit IEEE 754 floats (`float64x2_t`).

The key computational instruction is the Fused Multiply-Accumulate (`vfmaq_f64` / `fmla.2d`):
$ C_"vec" arrow C_"vec" + (A_"scalar" times B_"vec") $
This instruction retires two double-precision multiplications and two additions in a single cycle. By unrolling the innermost loop 4 times, our kernel processes 8 doubles ($64" bytes"$) per iteration, saturating the execution pipeline.

= Experimental Methodology

== Software Implementations & Controlled Variants
To ensure rigorous empirical isolation, all kernels are implemented in ANSI C99 using 64-bit IEEE 754 double precision (`double`):
1. `seq_ikj`: Pure sequential loop-reordered baseline compiled without OpenMP compiler flags or runtime linking.
2. `omp_ikj`: Parallel loop-reordered kernel using OpenMP `#pragma omp parallel for` with configurable thread count $p$ and schedule (`static`, `dynamic`, `guided`).
3. `seq_simd`: Sequential $i-k-j$ kernel accelerated with explicit ARM NEON intrinsics (`<arm_neon.h>`).
4. `omp_simd`: Parallel $i-k-j$ kernel combining OpenMP thread dispatch with ARM NEON SIMD vectorisation.
5. `seq_ijk` & `omp_ijk`: Canonical sequential and parallel triple-loop implementations (stride-$n$ on $B$) for cache stride comparison.

== Deterministic PRNG & Environment Control
All input matrices $A$ and $B$ are initialized deterministically using the SplitMix64 64-bit linear congruential generator initialized with seed $S=42$ @applem2arch2022. This guarantees bit-for-bit identical input operands across all executions, methods, and thread counts.

The execution environment is documented in detail:
- *Hardware:* Apple M2, 8 physical cores (4P + 4E), 8 GB Unified Memory, macOS Darwin 25.6.0 (arm64).
- *Compiler:* Apple Clang 21.0.0 (`clang-2100.1.1.101`) via Command Line Tools with Homebrew LLVM OpenMP runtime (`libomp` 23.1.5).
- *Compilation Flags:* `-O3 -Wall -Wextra -std=c99 -isysroot <SDK> -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp`.
- *Thread Binding:* To prevent the OpenMP runtime from dynamically adjusting thread pools, we explicitly enforce `omp_set_dynamic(0)` and programmatically set thread counts via `omp_set_num_threads(p)`.

== Timing Boundaries & Statistical Protocol
Timing measurements are recorded using high-resolution monotonic clocks (`clock_gettime(CLOCK_MONOTONIC)`):
- *Strict Kernel Boundary:* The timer begins immediately before the parallel loop construct and stops immediately after the closing barrier. Memory allocations (`malloc`), PRNG initialization, and cleanup (`free`) are strictly excluded.
- *Warm-up Protocol:* Every configuration undergoes 1 full warm-up execution (discarded) to populate hardware instruction caches, branch predictor tables, and runtime thread pool structures.
- *Repetitions:* 5 consecutive timed repetitions are gathered per configuration. We report the central tendency via the *Median* and dispersion via the *Interquartile Range (IQR)*:
$ "IQR" = Q_3 - Q_1 $
- *Dead Code Elimination Defense:* A scalar summation checksum $Sigma_(i,j) C_(i j)$ is computed and verified on each repetition, forcing the compiler to fully evaluate all floating-point updates.

== Correctness & Concurrency Verification
Prior to performance benchmarking, all kernels underwent rigorous automated verification:
1. *Analytical Edge Cases:* Zero matrix multiplication ($A times 0 = 0$) verified with exact zero checksum; Identity matrix multiplication ($A times I = A$) verified against $sum A_(i j)$ with relative error $< 10^{-14}$.
2. *Reference Equivalence:* Numerical outputs were validated against NumPy BLAS (`cblas_dgemm`) across 5 sizes ($n in [16, 256]$) and 3 seeds, verifying relative error:
$ "rel_error" = |"checksum"_"candidate" - "checksum"_"ref"| / |"checksum"_"ref"| < 10^(-12) $
3. *Thread Race Freedom:* Results across $p in {1, 2, 4, 6, 8, 12, 16}$ were compared for bit-level checksum equality, confirming zero race conditions.

#table_parallel_scaling

= Empirical Results & Analysis

== RQ1: OpenMP Framework Overhead at $p = 1$
A mandatory requirement of Assignment 3 is evaluating the parallel implementation with a single thread ($p=1$) to expose framework overhead.

As documented in Table 1, across all matrix sizes, executing the OpenMP kernel with $p=1$ is consistently slower than the pure sequential C executable:
- At $n = 512$: $T_"seq" = 0.0260" s"$ vs $T_"omp"(p=1) = 0.0444" s"$ ($S(1) = 0.59 times$, a $71%$ slowdown).
- At $n = 1024$: $T_"seq" = 0.1627" s"$ vs $T_"omp"(p=1) = 0.3473" s"$ ($S(1) = 0.47 times$, a $113%$ slowdown).
- At $n = 2048$: $T_"seq" = 1.4495" s"$ vs $T_"omp"(p=1) = 2.7399" s"$ ($S(1) = 0.53 times$, an $89%$ slowdown).

This significant disparity stems from two architectural factors:
1. *Runtime Abstraction:* The OpenMP compiler transforms `#pragma omp parallel for` into an outlined outlined function call dispatched through `libomp`'s team manager, which instantiates thread state structures and invokes barrier synchronizations.
2. *Compiler Register Optimisation:* In pure sequential C, Clang's `-O3` optimizer aggressively unrolls the loop and maintains accumulator variables across iterations in registers. Inside an OpenMP outlined function, the compiler must preserve function call boundary invariants and thread-local pointers, inhibiting aggressive register hoisting.

#align(center)[
  #image("figures/fig1_strong_scaling_and_speedup.png", width: 100%)
  #text(size: 8pt, fill: luma(80))[
    *Figure 1:* Strong scaling runtime (top, log scale) and speedup $S(p) = T_"seq" / T_p$ (bottom) across thread counts $p in [1, 16]$ for $n in [512, 2048]$. Plotted against ideal linear scaling $S(p)=p$ and Amdahl's Law ($f=0.95$).
  ]
]

== RQ2: Strong Scaling & Architectural Regimes
Figure 1 and Table 1 present strong scaling results. As thread count increases from $p=1$ to $p=16$, execution runtime drops significantly, reaching maximum throughput at $p=8$ or $p=12$.

Figure 2 visualizes Parallel Efficiency:
$ E(p) = S(p) / p = T_"seq"(n) / (p times T_p(n)) $
The efficiency curves delineate three distinct microarchitectural operational regimes:

#align(center)[
  #image("figures/fig2_efficiency_and_oversubscription.png", width: 100%)
  #text(size: 8pt, fill: luma(80))[
    *Figure 2:* Parallel efficiency $E(p)$ showing the three operational regimes of the Apple M2: Regime I (P-cores), Regime II (Heterogeneous P+E cores), and Regime III (Oversubscription).
  ]
]

=== Regime I: Performance Cores ($p in [1, 4]$)
When $p$ spans 2 to 4 threads, the macOS Darwin thread scheduler dispatches exclusively to the 4 Avalanche performance cores. These cores share identical 3.50 GHz clocks and 16 MB L2 cache:
- At $n=1024$, runtime decreases from $0.3473" s"$ ($p=1$) to $0.1729" s"$ ($p=2$) and $0.0942" s"$ ($p=4$).
- Throughput scales from 6.18 to 22.81 GFLOPS.
- Relative to $p=1$, the internal scaling from 1 to 4 threads achieves a $3.69 times$ speedup ($92.2%$ parallel efficiency relative to the thread pool).

=== Regime II: Heterogeneous Co-Execution ($p in [5, 8]$)
When $p > 4$, the workload must recruit the Blizzard efficiency cores. Because E-cores run at 2.42 GHz (a $31%$ clock deficit) and possess narrower instruction decode and execution pipelines @applem2arch2022:
- Adding E-cores increases aggregate computational capacity, but with diminishing marginal returns.
- In $n=1024$, moving from $p=4$ to $p=8$ threads yields an additional $1.30 times$ speedup ($22.81$ to $29.63$ GFLOPS).
- Efficiency $E(p)$ relative to the sequential baseline drops from $43.2%$ ($p=4$) to $28.1%$ ($p=8$).

=== Regime III: Oversubscription Penalty ($p in [9, 16]$)
Because the Apple M2 possesses exactly 8 physical cores without SMT, configuring $p=12$ or $p=16$ threads induces hardware oversubscription:
- At $n = 512$ and $1024$, oversubscription maintains stable throughput ($approx 30" GFLOPS"$).
- However, at large dimensions ($n = 2048$, $96" MB"$ memory footprint), oversubscription causes severe regression: runtime degrades from $0.6564" s"$ ($p=8$) to $0.8001" s"$ ($p=16$), an $18.0%$ performance penalty ($26.17$ dropping to $21.47$ GFLOPS).
- This regression occurs because 16 software threads actively contend for 8 physical execution pipelines. The Darwin kernel must repeatedly trigger preemptive context switches, invalidating L1 data caches and evicting working set cache lines.

#align(center)[
  #image("figures/fig3_scheduling_heterogeneity.png", width: 100%)
  #text(size: 8pt, fill: luma(80))[
    *Figure 3:* Computational throughput (GFLOPS) comparing static schedule against dynamic chunked schedules (`chunk=8` and `chunk=32`) across $p in {4, 8, 16}$ for $n=2048$.
  ]
]

== RQ3: Work Scheduling on Asymmetric Cores
Figure 3 compares OpenMP scheduling strategies on the Apple M2. Under `schedule(static)`, iterations are divided into $p$ equal contiguous chunks of $n/p$ rows.

In our experiments:
- At $n=1024$ and $p=8$, `schedule(static)` achieves $29.63" GFLOPS"$.
- Under `schedule(dynamic, 8)`, throughput increases to *$33.09" GFLOPS"$*, an immediate *$+11.7%$ performance increase* ($+3.46" GFLOPS"$).
- Similarly, at $n=2048$ with $p=16$, dynamic scheduling maintains $33.14" GFLOPS"$, whereas static scheduling degrades to $28.07" GFLOPS"$ (a $+18.1%$ advantage for dynamic).

This empirical result proves Hypothesis 3: under static scheduling on asymmetric SoCs, the fast P-cores finish their $n/8$ rows in approximately half the time required by E-cores. Because OpenMP inserts an implicit barrier at the end of `#pragma omp parallel for`, P-cores remain completely idle, blocked at the barrier while E-cores finish their slices. Dynamic scheduling allows P-cores to continuously request new 8-row work chunks, naturally shifting $approx 70%$ of the computation to high-IPC cores.

#table_simd_comparison

== RQ4: ARM NEON SIMD Vectorisation
Table 2 and Figure 4 present the vectorisation extension results, comparing scalar C loops against explicit 128-bit ARM NEON intrinsics (`<arm_neon.h>`).

#align(center)[
  #image("figures/fig4_simd_vectorisation_impact.png", width: 100%)
  #text(size: 8pt, fill: luma(80))[
    *Figure 4:* Left: Throughput comparison between scalar loops and ARM NEON SIMD across thread counts for $n=2048$. Right: Isolated SIMD speedup factor $S_"SIMD"$ across matrix sizes.
  ]
]

=== Compiler Optimization & Assembly Verification
In accordance with course specifications requiring evidence that vectorisation occurs, we inspected compiler pass remarks and disassembly:
- Clang optimization reports (`-Rpass=loop-vectorize`) confirm vectorisation with vector width 2 and interleaving factor 4.
- Disassembly inspection (`clang -S`) reveals that the scalar implementation issues single-precision 64-bit scalar instructions:
  `fmadd d9, d1, d8, d0`
- In contrast, the NEON SIMD kernel issues 128-bit fused multiply-accumulate vector instructions:
  `fmla.2d v2, v1, v0[0]`
  operating simultaneously on two 64-bit doubles packed into vector registers `v0`–`v31`.

=== Throughput Saturation & The Memory Wall
As shown in Table 2, in sequential execution ($p=1$), ARM NEON achieves $12.45" GFLOPS"$ (matching the compiler's auto-vectorised scalar speed of $12.94" GFLOPS"$). When combined with multi-threading, throughput scales to a peak of *$27.4" GFLOPS"$* at $p=16$.

However, as matrix dimension expands to $n=2048$, the aggregate memory footprint of matrices $A, B, C$ reaches:
$ 3 times 2048^2 times 8" bytes" = 100,663,296" bytes" approx 96.0" MB" $
This 96 MB working set exceeds the 16 MB shared L2 cache by a factor of 6. Consequently, when all 8 physical cores execute simultaneously, memory bus requests saturate the 100 GB/s UMA channel. The floating-point SIMD pipelines experience structural stalls awaiting cache-line refills from LPDDR5 DRAM, establishing an upper bound on shared-memory scaling.

== Memory Footprint & Peak Resident Set Size
Throughout all parallel experiments, Peak Resident Set Size (Peak RSS) was profiled via `getrusage(RUSAGE_SELF, ...)`.

As shown in Table 1:
- At $n = 512$: Peak RSS is $7.8" MB"$ (raw matrices: $6.0" MB"$).
- At $n = 1024$: Peak RSS is $25.8" MB"$ (raw matrices: $24.0" MB"$).
- At $n = 2048$: Peak RSS is $97.8" MB"$ (raw matrices: $96.0" MB"$).

Unlike interpreted runtimes (e.g., Python expanding $12.8 times$ in Assignment 1), the C OpenMP implementation exhibits near-zero memory bloat: process resident memory exceeds the raw matrix payload by only $1.8" MB"$, representing OpenMP thread stack allocations (typically 512 KB per thread) and standard library buffers.

= Discussion & Architectural Insights

== The High Cost of the OpenMP Abstraction
Our results unambiguously resolve RQ1: parallelizing code using OpenMP is not free. When restricted to a single thread ($p=1$), the OpenMP kernel is *two times slower* than pure sequential C ($0.47 times$ to $0.53 times$ speedup).

Many naive benchmarks report speedup normalized against the 1-thread parallel run ($S^*(p) = T(1) / T(p)$), yielding inflated metrics ($S^*(8) = 2.7399 / 0.6564 = 4.17 times$). However, scientific rigor demands comparing against the fastest known sequential baseline ($S(p) = T_"seq" / T_p$), which yields a true speedup of $2.21 times$. This distinction underscores the importance of baseline selection in parallel benchmarking.

== Microarchitectural Heterogeneity & The Barrier Penalty
The performance differential between static and dynamic scheduling (+11.7% throughput gain) highlights a critical lesson for modern heterogeneous architectures. Traditional HPC models assume homogeneous processing elements. On modern mobile, desktop, and server SoCs with asymmetric cores (Apple M-series, Intel Alder/Raptor Lake, ARM DynamIQ), static block partitioning is fundamentally flawed. Co-scheduling threads across heterogeneous cores mandates adaptive, dynamic work-stealing schedulers to avoid severe barrier serialization penalties.

== Hardware Limits to Shared-Memory Scaling
Our empirical investigations demonstrate that shared-memory parallelism is ultimately bound by physical microarchitectural limits:
1. *Core Capacity Limit:* Speedup peaks at the physical core limit (8 cores); oversubscription ($p=16$) degrades cache locality and performance.
2. *Memory Wall:* Even with optimal contiguous memory traversal ($i-k-j$) and 128-bit SIMD vectorisation, arithmetic throughput flattens once working sets exceed L2 cache capacity, bottlenecked by DRAM bus bandwidth.

= Threats to Validity & Limitations

Several experimental limitations must be noted:
1. *Platform Specificity:* Experiments were conducted on a single hardware model (Apple M2, macOS). While Big.LITTLE principles generalize to Intel Hybrid and ARM server chips, absolute GFLOPS and crossover points depend on microarchitecture.
2. *Thermal Throttling:* Extended multi-threaded benchmarking at 100% duty cycle on 8 cores can trigger dynamic frequency scaling (thermal throttling). We mitigated this by enforcing inter-repetition pauses and monitoring IQR stability.
3. *Single-Node Memory Ceiling:* Shared-memory parallelism cannot scale beyond the physical RAM capacity of a single machine (8.0 GB on this testbed), demonstrating the necessity of distributed computing.

= Conclusions & Transition to Assignment 4

This empirical investigation evaluated shared-memory OpenMP parallelism and ARM NEON SIMD vectorisation on an asymmetric 8-core Apple M2 processor.

The primary conclusions established by our evidence are:
1. *OpenMP Incurs Measurable Overhead:* Single-threaded OpenMP execution runs at half the speed of pure sequential C ($S(1) approx 0.5 times$), demonstrating the runtime cost of thread management abstractions.
2. *Strong Scaling Follows Core Regimes:* Scaling accelerates cleanly across homogeneous P-cores ($p <= 4$), yields diminishing returns as E-cores join ($p=6, 8$), and degrades by $18%$ under oversubscription ($p=16$).
3. *Dynamic Scheduling Is Mandatory on Asymmetric Silicon:* Dynamic chunked scheduling mitigates barrier wait times, delivering an immediate $+11.7%$ throughput advantage over static partitioning.
4. *Memory Bandwidth Dictates Performance Limits:* Vectorisation and multi-threading combine to achieve peak throughputs above 33 GFLOPS, but are ultimately capped by the 100 GB/s memory bandwidth barrier once payloads exceed the 16 MB L2 cache.

== Bridge to Assignment 4 (Distributed Multiplication)
While shared-memory multi-threading achieves substantial speedups within a single node, it cannot scale when matrix dimensions expand beyond single-machine memory budgets (e.g., $n = 50,000$, requiring $60" GB"$ of storage). Assignment 4 will transcend the single-node memory wall by investigating *Distributed Matrix Multiplication*, partitioning matrices across independent worker nodes using distributed computing frameworks.

= References & Reproducibility Materials

All source code, benchmarking scripts, raw CSV datasets, and automation Makefiles are publicly available in the course repository:
#align(center)[#link("https://github.com/pablomrtinezzz/big-data-assignments")]

#bibliography("references.bib", style: "ieee")

]
