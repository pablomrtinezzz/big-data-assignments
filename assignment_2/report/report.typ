#set page(
  paper: "a4",
  margin: (x: 1.6cm, top: 2.0cm, bottom: 2.0cm),
  header: align(right)[
    #text(size: 8pt, fill: luma(100))[
      Big Data · Individual Assignment 2 · Academic Year 2026–2027
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

#import "tables_autogen.typ": table_dense_scaling, table_sparse_scaling

// Title Block
#align(center)[
  #v(0.5em)
  #text(size: 15.5pt, weight: "bold")[
    Empirical Evaluation of Cache-Aware Dense Algorithmic Optimisations and Compressed Sparse Row Representation for Matrix Multiplication
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
  Matrix multiplication is a cornerstone kernel in numerical linear algebra, scientific computing, and data analytics. On modern multi-tier memory hierarchies, execution performance is predominantly constrained by memory access patterns, cache utilization, and matrix density rather than raw arithmetic compute capability. This paper presents a rigorous empirical study addressing two interconnected research objectives: (1) evaluating how algorithmic microarchitectural transformations—specifically loop reordering ($i-k-j$) and cache blocking/tiling—accelerate sequential dense matrix multiplication relative to the canonical triple-loop baseline ($i-j-k$) on an Apple Silicon M2 processor, and (2) determining the empirical break-even / crossover conditions under which a Compressed Sparse Row (CSR) representation and Gustavson's SpGEMM kernel outperform optimized dense multiplication. Across square matrix dimensions $n in [32, 2048]$ with 64-bit IEEE 754 precision and controlled SplitMix64 deterministic inputs, loop reordering achieves up to a $22.40 times$ speedup over the baseline ($1.38" s"$ vs $30.90" s"$ at $n=2048$), with the speedup surging dramatically once working sets exceed the 16 MB shared L2 cache boundary. In the sparse domain, CSR multiplication is bottlenecked at moderate sparsities ($rho <= 70\%$) by pointer chasing and severe non-zero fill-in ($C = A B$ expanding up to $19.87 times$ denser than input operands). The crossover threshold where sparse CSR surpasses dense reordered computation occurs at $rho^* approx 75\% - 80\%$, reaching speedups of $51.96 times$ at $rho=99\%$ and $15,925 times$ at $rho=99.9\%$, while cutting structural storage by $99.6\%$. All implementations, raw CSV logs, and automated validation suites are openly accessible on GitHub.

  #v(0.4em)
  #text(weight: "bold")[Keywords] — Dense matrix multiplication, loop reordering, cache blocking, tiling, memory hierarchy, Apple M2, CSR, SpGEMM, Gustavson algorithm, sparsity crossover, fill-in effect.
]

#v(1em)

#columns(2, gutter: 1.5em)[

= Introduction & Research Objectives

Matrix multiplication is among the most computationally intensive kernels in data engineering, graph processing, and neural network computing @golub2013matrix. For dense square matrices $A, B in RR^(n times n)$, the canonical definition requires:
$ C_(i j) = sum_(k=0)^(n-1) A_(i k) B_(k j) $
This formulation entails $O(n^3)$ arithmetic operations ($n^3$ multiplications and $n^3$ additions). On modern superscalar microprocessors, floating-point units can retire multiple instructions per clock cycle. However, the throughput of matrix multiplication is fundamentally throttled by the *memory wall*—the latency and bandwidth discrepancy between the CPU execution pipeline and external DRAM @hennessy2017computer.

Furthermore, real-world data engineering workloads (e.g., social graphs, recommendation systems, NLP term-document matrices) frequently exhibit structural sparsity, where the overwhelming majority of matrix elements are zero @davis2006direct. Representing sparse matrices densely incurs prohibitive $O(n^2)$ memory waste and performs vast numbers of redundant trivial multiplications ($0 times x = 0$). Conversely, sparse storage formats introduce non-trivial metadata overhead, irregular memory access, and branch mispredictions.

== Research Questions & Hypotheses
This empirical study investigates two research questions:
1. *RQ1 (Dense Optimisation):* _How do loop reordering ($i-k-j$) and cache blocking (tiling) mitigate cache miss latency and accelerate sequential dense matrix multiplication compared to the canonical $i-j-k$ baseline across varying matrix sizes $n$?_
   - *Hypothesis 1:* Loop reordering ($i-k-j$) will transform memory accesses to matrix $B$ and $C$ into contiguous stride-1 traversals, enabling compiler SIMD vectorization and hardware streaming prefetchers. The relative speedup will jump substantially once working sets exceed the Apple M2 shared L2 cache capacity (16 MB).
2. *RQ2 (Sparse Representation & Break-Even):* _Under what sparsity thresholds $rho$ does a Compressed Sparse Row (CSR) representation and Gustavson's SpGEMM kernel achieve superior execution time and memory efficiency compared to optimized dense multiplication, accounting for non-zero fill-in?_
   - *Hypothesis 2:* At low to moderate sparsity ($rho <= 70\%$), dense multiplication will remain faster than CSR due to the structural fill-in phenomenon ($C = A B$ becoming nearly $100\%$ dense) and sparse indexing overhead. CSR will only demonstrate an operational advantage beyond a crossover threshold $rho^* approx 75\% - 80\%$, where arithmetic bypass compensates for pointer indirection.

= Background & Algorithmic Design

== Cache Locality & Stride Penalties in Dense Multiplication
In row-major contiguous storage (standard in C), elements within the same row are adjacent in memory: $"address"(M_(i j)) = "base" + (i times n + j) times 8" bytes"$. A 64-byte or 128-byte cache line loads 8 to 16 contiguous double-precision values simultaneously @applem2arch2022.

In the canonical $i-j-k$ loop order, the inner loop index is $k$:
- $A_(i k)$ is traversed with stride-1 along row $i$ (favorable spatial locality).
- $B_(k j)$ is traversed with stride-$n$ down column $j$. For $n = 1024$, successive accesses jump by $8,192" bytes"$, guaranteeing that every inner iteration accesses a distinct cache line, inducing catastrophic L1/L2 cache misses.

Reordering the nested loops to $i-k-j$ fundamentally alters the memory access pattern:
- $A_(i k)$ is loop-invariant with respect to $j$ and is hoisted into a hardware register.
- $B_(k j)$ is accessed sequentially along row $k$ with stride-1.
- $C_(i j)$ is updated sequentially along row $i$ with stride-1.
Both inner arrays achieve optimal spatial locality. Moreover, the compiler can automatically vectorize the inner stride-1 accumulation using ARM64 NEON SIMD instructions (`fmla.2d`) @goto2008anatomy.

== Cache Blocking (Tiling)
Cache blocking divides the global $n times n$ matrix into sub-blocks of dimension $B_"tile" times B_"tile"$ @lam1991cache. The working set of three sub-matrices must satisfy:
$ 3 times (B_"tile")^2 times 8" bytes" <= C_"L1D" $
On the Apple M2 with a 64 KB L1D data cache per performance core, choosing $B_"tile" = 64$ yields $3 times 64^2 times 8 = 98,304" bytes"$ (fitting across L1D and L2). Within each block, operations follow an $i-k-j$ traversal.

== Compressed Sparse Row (CSR) & Gustavson SpGEMM
The CSR format represents an $n times n$ sparse matrix with $"NNZ"$ non-zero elements using three arrays @davis2006direct:
1. `values`: array of non-zero entries (length $"NNZ"$, $8" bytes"$).
2. `col_indices`: column index for each entry (length $"NNZ"$, $8" bytes"$).
3. `row_ptr`: start pointer for each row (length $n+1$, $8" bytes"$).

Total structural memory required is:
$ S_"CSR" = 16 "NNZ" + 8(n + 1)" bytes" $
Comparing this with dense storage ($S_"dense" = 8 n^2" bytes"$), CSR achieves storage parity ($S_"CSR" < S_"dense"$) when density $d = "NNZ" / n^2 < 0.50$ (i.e., sparsity $rho > 50\%$).

To compute $C = A B$ directly in CSR format without dense expansion, we implement Gustavson's row-oriented algorithm @gustavson1978two. For each row $i$ of $A$:
1. A dense accumulator array `accum` and marker array `marker` of length $n$ are maintained.
2. For each non-zero $A_(i k)$, we iterate across non-zeros $B_(k j)$ and accumulate $A_(i k) B_(k j)$ into `accum[j]`.
3. Column indices are tracked in $O(1)$ time via `marker`.
4. Only non-zero entries are written back into the CSR structures of $C$.

The theoretical arithmetic complexity of Gustavson's SpGEMM is proportional to the number of non-zero floating-point operations:
$ "FLOPs"_"CSR" = 2 sum_(i=0)^(n-1) sum_(k in "nnz"(A_i)) "nnz"(B_k) $

== The Non-Zero Fill-In Phenomenon
An unavoidable characteristic of sparse matrix multiplication is non-zero fill-in: even when input operands $A$ and $B$ are highly sparse, the output matrix $C = A B$ can become substantially denser @davis2006direct. If the probability of a non-zero at $(i, k)$ in $A$ is $d_A$ and at $(k, j)$ in $B$ is $d_B$, the probability that $C_(i j) != 0$ is:
$ P(C_(i j) != 0) = 1 - (1 - d_A d_B)^n $
For $n = 1024$ and input sparsity $rho = 90\%$ ($d = 0.10$), $d^2 = 0.01$. The probability of an output entry being non-zero is $1 - (0.99)^(1024) approx 0.99997$ (virtually $100\%$ dense). This theoretical reality causes massive structural expansion and indexing work.

= Experimental Methodology

== Hardware & Host Environment
All benchmarks were executed on an isolated Apple M2 workstation under controlled single-user conditions:
- *Microprocessor:* Apple M2 (ARMv8.5-A architecture, 8 physical cores: 4 Avalanche P-cores @ 3.50 GHz, 4 Blizzard E-cores @ 2.42 GHz).
- *Cache Subsystem:* P-core L1D: 64 KB, L1I: 128 KB; shared L2 cache: 16 MB.
- *Main Memory:* 8.0 GB Unified Memory (LPDDR5-6400, 100 GB/s bandwidth).
- *Operating System:* macOS Darwin 25.6.0 (Kernel 26.6.2, Build 25G83, arm64).
- *Toolchain:* Apple Clang 21.0.0 (`-O2 -Wall -Wextra -std=c99`), CPython 3.14.7, NumPy 2.5.3, SciPy 1.18.1.

== Input Generation & Deterministic Control
To ensure strict comparability across methods, matrices were generated using the 64-bit SplitMix64 pseudo-random generator with recorded seeds (seed 42 for $A$, seed 43 for $B$). For sparse matrices, elements were populated with non-zero values according to the target sparsity $rho in [0.50, 0.999]$.

== Timing Boundary & Measurement Protocol
- *Clock:* Monotonic timer `clock_gettime(CLOCK_MONOTONIC)`.
- *Boundary:* The timer captures strictly the multiplication kernel execution. Memory allocation and dense-to-CSR conversion are measured separately.
- *Repetitions:* 1 warm-up run (discarded) followed by 5 measured repetitions per configuration. Central tendency is reported via the *Median*, and dispersion via the *Interquartile Range (IQR)*.
- *Memory Metric:* Peak Resident Set Size (RSS) via `getrusage(RUSAGE_SELF)`.

== Automated Correctness Verification
All implementations underwent automated verification before benchmarking:
1. *Analytical cases:* Identity matrix ($A I = A$) and Zero matrix ($A times 0 = 0$).
2. *Reference standards:* Outputs were verified against NumPy BLAS (`np.dot`) and SciPy (`scipy.sparse.csr_matrix.dot`), confirming relative error $epsilon < 10^(-11)$ and $100\%$ exact match on output non-zero counts (NNZ).

= Empirical Results

#figure(
  image("figures/fig1_dense_scaling_and_speedup.pdf", width: 100%),
  caption: [Execution runtime scaling (top) and speedup factor over baseline (bottom) for dense algorithms across matrix dimension $n$. Cache boundaries for Apple M2 (L1D: 64 KB, L2: 16 MB) are indicated.]
) <fig_dense>

== Part A: Dense Optimisation Scaling
@fig_dense and @table_dense present the execution times and speedups for the three dense implementations: Canonical Baseline ($i-j-k$), Loop Reordered ($i-k-j$), and Cache Blocked ($B=64, i-k-j$).

#figure(
  table_dense_scaling,
  caption: [Dense matrix multiplication empirical results across dimensions $n$ (Medians over 5 repetitions, Apple M2, `-O2`).]
) <table_dense>

Several critical patterns emerge from the empirical data:
1. *Sub-Cache Scaling ($n <= 512$):* For small matrices fitting inside the 16 MB L2 cache ($n <= 512$), loop reordering and cache blocking achieve immediate $4.0 times - 7.3 times$ speedups over the baseline by eliminating column-stride cache misses on $B$.
2. *The DRAM Latency Cliff ($n = 2048$):* At $n = 2048$, three dense $2048 times 2048$ matrices of doubles require $3 times 2048^2 times 8 = 100.66" MB"$, vastly exceeding the 16 MB L2 cache. The canonical baseline execution time degrades catastrophically to $30.900" s"$, as each inner iteration suffers a main memory DRAM access.
3. *Peak Dense Speedup:* In contrast, Loop Reordered ($i-k-j$) completes in just $1.3797" s"$, yielding an extraordinary *$22.40 times$ speedup* over baseline ($18.03 times$ for Blocked). Stride-1 access allows hardware streaming prefetchers and NEON SIMD lanes to remain saturated.

#figure(
  image("figures/fig2_sparsity_crossover.pdf", width: 100%),
  caption: [Runtime comparison between Gustavson CSR SpGEMM and Dense Reordered across sparsity levels $rho in [50\%, 99.9\%]$, illustrating the empirical crossover break-even point.]
) <fig_sparse_crossover>

== Part B: Sparse Matrices & Break-Even Analysis
@fig_sparse_crossover and @table_sparse display the comparative scaling of Gustavson's CSR kernel against the Dense Reordered reference on identical logical matrices.

#figure(
  table_sparse_scaling,
  caption: [Sparse CSR empirical performance, non-zero fill-in, speedup vs dense reordered, and memory reduction for $n = 1024$ (Apple M2, `-O2`).]
) <table_sparse>

#figure(
  image("figures/fig3_fillin_and_memory.pdf", width: 100%),
  caption: [Non-zero fill-in expansion ratio (left) and structural memory savings vs dense storage (right) as a function of input sparsity $rho$ for $n = 1024$.]
) <fig_fillin_memory>

Key findings from the sparse matrix evaluation:
1. *Crossover / Break-Even Threshold ($rho^*$):* At $50\%$ sparsity, dense reordered multiplication is $1.27 times$ faster than CSR SpGEMM ($0.159" s"$ vs $0.201" s"$). The crossover point occurs at $rho^* approx 75\% - 80\%$. Above $80\%$ sparsity, CSR consistently outperforms dense computation.
2. *Exponential Scaling at High Sparsity:* At $rho = 98\%$, CSR achieves an $11.61 times$ speedup. At $rho = 99.9\%$, the CSR kernel executes in $0.000010" s"$, delivering an astonishing *$15,925 times$ speedup* over dense multiplication.
3. *Fill-In Quantification:* As shown in @fig_fillin_memory (left), fill-in peaks at $rho = 95\%$, where output non-zeros expand by $18.46 times$ ($"NNZ"_"out" = 967,935$ from $"NNZ"_"in" = 104,853$). At $rho = 50\%$ and $70\%$, the output matrix is $100\%$ dense ($1,048,576$ entries), proving that input sparsity does not guarantee output sparsity.
4. *Memory Savings:* CSR structural memory savings (@fig_fillin_memory, right) cross parity at $rho = 33.3\%$ and achieve $88.0\%$ memory reduction at $rho = 99\%$, and $99.6\%$ at $rho = 99.9\%$.

= Discussion & Architectural Analysis

== Why Loop Reordering Dominates Canonical Baseline
The experimental evidence confirms Hypothesis 1. In canonical $i-j-k$, the inner loop executes:
```c
C[i*n + j] += A[i*n + k] * B[k*n + j];
```
Because $B$ is stored in row-major layout, advancing $k$ advances by $n$ elements in memory ($n times 8" bytes"$). For $n >= 512$, this step exceeds the cache line size, resulting in a cache miss on virtually every multiply-accumulate operation. The CPU execution units stall waiting for data transfers.

In $i-k-j$ reordering:
```c
double r = A[i*n + k];
for (size_t j = 0; j < n; j++)
    C[i*n + j] += r * B[k*n + j];
```
Here, `r` is held in an ARM64 floating-point register. Both `B[k*n + j]` and `C[i*n + j]` are read and written sequentially. An L1D cache line fetch satisfies 8 consecutive iterations. Furthermore, Apple Clang vectorizes the inner loop into NEON instructions (`fmla.2d`), doubling arithmetic throughput.

== Reordered vs Blocked on Apple Silicon
Interestingly, plain Loop Reordered ($i-k-j$) slightly outperforms Cache Blocked at large $n$ ($1.38" s"$ vs $1.71" s"$ at $n=2048$). On the Apple M2, the hardware stream prefetcher is extraordinarily aggressive when detecting contiguous stride-1 access, masking DRAM latency. Cache blocking introduces loop overhead (6 nested loops instead of 3) and boundary condition branches that marginally diminish prefetching throughput.

== Anatomy of the Sparse Break-Even Threshold
The data validates Hypothesis 2: *a percentage of zeros alone does not guarantee superior performance*. At $rho = 50\%$, CSR SpGEMM is slower than dense multiplication for three concrete architectural reasons:
1. *Loss of Vectorization:* Gustavson's SpGEMM relies on indirect indexing (`B->col_indices[q]`) and dynamic hash/accumulator updates, precluding compiler auto-vectorization.
2. *Branch Misprediction:* Testing non-zero row boundaries and markers induces frequent pipeline flushes.
3. *Output Fill-In Overhead:* At $rho = 50\%$, the product matrix $C$ is fully dense. Gustavson's algorithm must dynamically allocate, sort, and compress $1,048,576$ entries, incurring heavy dynamic memory management overhead.

Only when sparsity exceeds $80\%$ do the skipped trivial multiplications ($0 times x$) outbalance the indexing penalty, unlocking orders-of-magnitude acceleration.

== Threats to Validity & Limitations
- *Single-Threaded Scope:* All evaluations were sequential; multi-threaded shared-memory scaling will be investigated in Assignment 3.
- *Synthetic Uniform Sparsity:* Matrices were generated with uniform random sparsity. Real-world power-law graphs (e.g., scale-free networks) exhibit clustered non-zeros with different fill-in dynamics.

= Conclusions

This empirical study demonstrates the decisive impact of memory hierarchy optimization and sparse representations in matrix multiplication:
1. *Microarchitectural Loop Transformation:* Reordering loops from $i-j-k$ to $i-k-j$ eliminates column-stride cache thrashing, delivering up to a *$22.40 times$ speedup* on Apple M2, with the performance gap expanding drastically when working sets spill beyond the 16 MB L2 cache into DRAM.
2. *Sparse Representation Crossover:* Compressed Sparse Row (CSR) storage provides massive computational and memory benefits only beyond an empirical crossover threshold of *$rho^* approx 75\% - 80\%$*. Below this threshold, non-zero fill-in and pointer indirection make dense multiplication superior.
3. *Fill-In Impact:* Due to non-zero accumulation, output matrices are dramatically denser than input matrices (expanding up to $19.87 times$), necessitating careful pre-allocation in high-performance pipelines.

= References & Reproducibility Materials

All source code, automated test suites, benchmarking harnesses, raw CSV measurement tables, and Typst typesetting scripts are publicly available in the dedicated GitHub repository:
#align(center)[
  #link("https://github.com/pablomrtinezzz/big-data-assignments")
]

#v(1em)
#bibliography("references.bib", style: "ieee")

]
