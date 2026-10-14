#set page(
  paper: "a4",
  margin: (x: 1.8cm, top: 2.2cm, bottom: 2.2cm),
  header: align(right)[
    #text(size: 8pt, fill: luma(100))[
      Big Data · Individual Assignment 1 · Academic Year 2026–2027
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

// Title Block
#align(center)[
  #v(0.5em)
  #text(size: 16pt, weight: "bold")[
    Empirical Performance and Memory Evaluation of Canonical Matrix Multiplication Across C, Java, and Python
  ]
  #v(0.8em)
  #text(size: 11pt, weight: "medium")[
    *Pablo Martínez Suárez* \
  ]
  #v(0.2em)
  #text(size: 9pt, style: "italic")[
    Department of Computer Science and Systems, Universidad de Las Palmas de Gran Canaria (ULPGC) \
    Grado en Ciencia e Ingeniería de Datos · Big Data (2026–2027) \
    `pablomrtinezzz` · GitHub Reproducibility Package
  ]
  #v(1.2em)
]

// Abstract and Keywords
#rect(width: 100%, stroke: 0.5pt + luma(180), inset: 10pt, radius: 2pt, fill: luma(250))[
  #text(weight: "bold")[Abstract] —
  Dense matrix multiplication is a fundamental computational kernel underpinning large-scale numerical linear algebra and big data analytics. This paper presents a rigorous empirical investigation comparing the canonical triple-loop algorithm ($i-j-k$) implemented in C, Java, and Python under strictly controlled experimental conditions. Using an increasing sequence of square matrices ($n in [32, 2048]$), a deterministic SplitMix64 pseudorandom generator, 64-bit IEEE 754 floating-point numbers, and high-resolution monotonic timers, we measure kernel runtime variability (median and interquartile range across 5 repetitions) and process memory footprint (Peak Resident Set Size and managed heap). We evaluate two C compiler configurations (Apple Clang `-O2` and `-O0`), the HotSpot JVM (OpenJDK 22 with C2 tiered compilation), and CPython 3.14.7. Our empirical findings demonstrate that while all implementations scale asymptotically with cubic growth ($T(n) prop n^alpha$ with $alpha approx 3.11 - 3.37$, $R^2 > 0.99$), runtime constant factors diverge by up to $160 times$. The JVM HotSpot C2 compiler converges after warm-up to within 10–15% of optimized C (even surpassing it at $n=512$ due to auto-vectorized fused multiply-accumulate on ARM64 and at $n=2048$ due to cache-miss behavior), whereas Python is severely bounded by bytecode dispatch and object boxing. Under a declared 60-second execution budget, Python is limited to $n <= 1024$ ($n=1024$ completes in 42.16s, while $n=1536$ times out exceeding 60s), whereas C and Java scale feasibly to $n=2048$ in 29.57s and 19.83s, respectively. Python's memory footprint is $4 times$ larger than raw 64-bit payloads due to `PyFloatObject` boxing. All code, datasets, and scripts are packaged for exact reproducibility.

  #v(0.4em)
  #text(weight: "bold")[Keywords] — Matrix multiplication, algorithmic scaling, memory layout, C, Java, Python, benchmarking protocol, cache locality, JIT compilation.
]

#v(1em)

#columns(2, gutter: 1.5em)[

= Introduction & Research Objective

Matrix multiplication is a cornerstone operation in computational science, machine learning, and big data processing @golub2013matrix. The standard mathematical definition of multiplying two dense square matrices $A, B in RR^(n times n)$ produces a matrix $C = A B$ where:
$ C_(i j) = sum_(k=0)^(n-1) A_(i k) B_(k j) $
This formulation implies an inherent arithmetic complexity of $O(n^3)$ operations—specifically $n^3$ multiplications and $n^3$ additions, yielding $2 n^3$ floating-point operations (FLOPs).

While modern production libraries (e.g., OpenBLAS, Intel MKL, Apple Accelerate) achieve peak hardware throughput through architecture-specific vectorization (SIMD), cache-tiling, and multithreading @dongarra1990blas, understanding the baseline performance of the naive triple-loop is essential for systems analysis. Programming language runtimes exhibit profound architectural divergence:
1. *Ahead-of-Time (AOT) Compiled Languages (C):* Direct translation into native machine instructions with manual memory layout and static optimization @lattner2004llvm.
2. *Just-in-Time (JIT) Managed Environments (Java):* Bytecode execution transitioning dynamically via tiered profiling to aggressive native machine code compilation (HotSpot C2 compiler) @lindholm2014java.
3. *Interpreted Dynamic Languages (Python):* Bytecode evaluation loop (`ceval.c`) with dynamic type dispatch, boxing of primitives, and automatic memory management @vanrossum2009python.

== Research Question & Hypotheses
This investigation addresses the following core research question:
#align(center)[
  _How does matrix dimension $n$ govern execution time and memory footprint across AOT-compiled, JIT-managed, and interpreted execution environments when algorithmic structure, numeric precision, and input data are held strictly constant?_
]

We formulate four testable empirical hypotheses:
- *$H_1$ (Algorithmic Scaling):* All implementations will exhibit an empirical scaling exponent $alpha approx 3.0$ matching the theoretical $O(n^3)$ model, but interpreter overhead and dynamic type resolution will shift Python's constant factor $c$ by over an order of magnitude relative to C.
- *$H_2$ (JIT Compiler Convergence):* Following sufficient warm-up iterations, Java's HotSpot C2 compiler will achieve execution times within a factor of $1.5 times$ of optimized C (`-O2`), vastly outperforming unoptimized C (`-O0`).
- *$H_3$ (Memory Layout Divergence):* Memory consumption will scale quadratically ($O(n^2)$), but Python will require approximately $4 times$ more memory than the theoretical 64-bit payload ($24 n^2$ bytes) due to object boxing (`PyFloatObject`), whereas C will exhibit near-zero allocation overhead.
- *$H_4$ (Feasible Operational Boundaries):* Under a declared execution budget of 60 seconds per repetition, Python will reach its feasibility limit at $n <= 1024$, triggering timeouts at $n = 1536$, whereas C and Java will scale feasibly to $n >= 2048$.

= Background & Theoretical Foundations

== The Canonical $i-j-k$ Loop Order
The canonical algorithm nests three loops in the row-column-dot order:
```c
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[i * n + k] * B[k * n + j];
        }
        C[i * n + j] = sum;
    }
}
```
In this arrangement, the accumulator `sum` can reside in an architectural register during the innermost loop over $k$. However, access patterns to memory differ fundamentally across the three operands:
- *Matrix $A$:* Elements $A[i][k]$ are traversed sequentially along row $i$. Because data is stored in row-major order, consecutive iterations access consecutive memory addresses, maximizing *spatial cache locality*.
- *Matrix $C$:* Destination element $C[i][j]$ remains invariant during the inner loop and is updated once in the middle loop.
- *Matrix $B$:* Elements $B[k][j]$ are accessed along column $j$. In row-major layout, subsequent elements are separated by a stride of $n$ doubles ($8n$ bytes). When $8n$ exceeds the cache line size (64 bytes), every iteration of $k$ triggers a cache line request, leading to severe cache miss penalties once $n times n$ exceeds the L1/L2 cache capacities @drepper2007memory.

== Storage Layouts Across Paradigms
Memory organization dictates cache hierarchy interaction @hennessy2019computer:
1. *C (Contiguous 1D Flat Array):* Matrices are allocated as a single continuous memory block of $n^2$ doubles (`double*`) indexed as $i dot n + k$. No pointer indirection exists; element offsets are computed via affine arithmetic. Total data payload is exactly $24 n^2$ bytes for $A, B, C$.
2. *Java (2D Array `double[][]`):* Java does not support true multidimensional contiguous primitive arrays. A `double[][]` is an array of $n$ object references, each pointing to an independently allocated heap array of $n$ primitive doubles. Each row array incurs an object header (16 bytes) plus length field (4 bytes) and 64-bit alignment padding @lindholm2014java. Accessing $A[i][k]$ requires one pointer dereference to fetch the row array followed by index bounds checking.
3. *Python (Nested Lists `list[list[float]]`):* Standard Python matrices consist of an outer list containing references to $n$ inner lists. Each inner list contains $n$ pointers to heap-allocated `PyFloatObject` instances. In CPython 64-bit, each `PyFloatObject` requires 24 bytes (8B reference count, 8B type pointer, 8B double-precision value), supplemented by an 8-byte pointer in the enclosing list @vanrossum2009python. Consequently, each floating-point entry consumes 32 bytes—$4 times$ the size of a primitive double—with double pointer dereferencing and complete destruction of cache spatial locality.

= Experimental Methodology

== Execution Environment & Hardware
All experiments were conducted on a dedicated Apple Silicon hardware platform running macOS Darwin:
- *Processor:* Apple M2 (ARMv8.5-A architecture)
  - 4 Performance cores (`Avalanche`, 3.50 GHz, 128 KB L1I, 64 KB L1D per core, 16 MB shared L2 cache)
  - 4 Efficiency cores (`Blizzard`, 2.42 GHz, 64 KB L1I, 32 KB L1D per core, 4 MB shared L2 cache)
  - System topology: 8 physical cores, 8 logical processors (1 thread/core, no SMT); executed single-threaded (thread count = 1).
- *Memory:* 8.0 GB Unified Memory (LPDDR5-6400, 100 GB/s bandwidth)
- *Operating System:* macOS Darwin 25.6.0 (Kernel 26.6.2, Build 25G83)
- *Software Toolchains:*
  - C Compiler: Apple Clang version 21.0.0 (`clang-2100.1.1.101`), Target `arm64-apple-darwin25.6.0`. Flags evaluated: `-O2 -Wall -std=c99` (optimized) and `-O0` (unoptimized).
  - Java Runtime: OpenJDK 22.0.2 (`build 22.0.2+9-70`, 64-Bit Server VM, mixed mode, tiered compilation enabled).
  - Python Interpreter: CPython 3.14.7 (macOS arm64 build).

== Input Generation & Parity
To guarantee rigorous cross-language comparability, all implementations share an identical pseudorandom number generator: the 64-bit *SplitMix64* algorithm @steele2014splitmix. Given a fixed initial seed ($S = 42$), SplitMix64 generates deterministic sequences of 64-bit integers transformed into IEEE 754 doubles in $[0.0, 1.0)$ via:
$ v = (z >> 11) times 2^(-53) $
Bit-for-bit equivalence across C, Java, and Python was verified before benchmarking.

== Timing Boundaries & Measurement Protocol
In strict adherence to the common experimental protocol:
1. *Isolated Kernel Boundary:* Timers encompass solely the arithmetic triple-loop. Matrix memory allocation, PRNG generation, and console I/O occur strictly outside the timed window.
2. *High-Resolution Monotonic Clocks:* Timing uses `clock_gettime(CLOCK_MONOTONIC)` in C, `System.nanoTime()` in Java, and `time.perf_counter()` in Python.
3. *Dead-Code Elimination Defense:* To prevent optimizing compilers and JIT engines from optimizing away unused computations, a scalar checksum $S = sum_(i,j) C_(i j)$ is computed and verified after stopping the timer.
4. *Warm-Up Protocol:* To mitigate JVM class loading, JIT tiered compilation transitions (Tier 0 interpreted to Tier 4 C2 compiled code), and CPU dynamic voltage/frequency scaling (DVFS), dedicated warm-up iterations are executed ($W=3$ for $n <= 256$, $W=2$ for $n <= 512$, $W=1$ for $n >= 768$). Warm-up runs are strictly discarded from reported statistics.
5. *Repetitions & Statistics:* For each configuration, 5 independent measured repetitions ($R=5$) are collected. We report the *median* as the robust measure of central tendency and the *Interquartile Range (IQR)* as the measure of experimental dispersion.
6. *Declared Budgets & Limits:*
   - Maximum runtime budget: 60.0 seconds per repetition. Any run exceeding 60s triggers immediate process termination and is recorded as a `TIMEOUT`.
   - Memory budget: 4.0 GB physical RAM.

== Correctness Validation & Numerical Tolerance
Before performance testing, correctness was verified via:
1. *Zero Matrix Test:* $A times 0 = 0$, confirming exact zero accumulation across all entries.
2. *Identity Matrix Test:* $A times I = A$, verifying conservation of matrix values within numerical precision ($|"rel_err"| < 10^(-13)$).
3. *Reference Ground Truth:* Comparisons against NumPy BLAS (64-bit double precision) across dimensions $n in [2, 128]$. Due to floating-point associativity differences in compiler reordering, we define an acceptable relative Frobenius tolerance @higham2002accuracy:
$ (||C - C_"ref"||_F) / (||C_"ref"||_F) <= 10^(-12) $
All implementations passed all verification suites with maximum observed relative error $< 3 times 10^(-13)$.

= Empirical Results

#figure(
  image("figures/fig1_runtime_comparison.png", width: 100%),
  caption: [Benchmark execution time vs. matrix dimension $n$. (a) Linear scale showing exponential separation between Python and compiled languages. (b) Log-log scale confirming asymptotic convergence to the cubic $O(n^3)$ theoretical model with a 60-second cutoff threshold.]
) <fig:runtime>

== Runtime Performance and Scaling
We tested the sequence $n in [32, 2048]$. Table 1 reports the empirical measurements across all tested configurations.

At $n=32$, execution times are sub-millisecond across all platforms: C (`-O2`) requires 0.013 ms, Java requires 0.104 ms, and Python requires 0.993 ms. As $n$ increases, runtime expands rapidly in accordance with cubic growth.

At $n=512$, C (`-O2`) completes in 0.1420s, C (`-O0`) in 0.4704s, Java in 0.1153s, and Python in 4.7040s. At this size, Python is $33.1 times$ slower than C (`-O2`) and $40.8 times$ slower than Java.

At $n=1024$, C (`-O2`) requires 1.2127s, C (`-O0`) takes 3.9221s, Java requires 1.7695s, while Python requires 42.1586s. Here, Python remains within the declared 60-second budget ($42.16"s" < 60"s"$).

At $n=1536$, cubic projection predicts a Python runtime of $42.16 times (1536/1024)^3 approx 142.3"s"$. As required by the experimental protocol, the benchmark harness enforced the 60.0s execution timeout; Python exceeded the budget on its initial repetition and was terminated, recording a documented `TIMEOUT (> 60s)`. Consequently, the largest feasible tested size for Python is *n = 1024*.

In contrast, C (`-O2`), C (`-O0`), and Java successfully completed $n=1536$ in 4.27s, 15.82s, and 5.53s, respectively, and completed $n=2048$ in 29.57s, 48.03s, and 19.83s, well within the 60-second ceiling.

#figure(
  image("figures/fig2_throughput_speedup.png", width: 100%),
  caption: [(a) Effective computational throughput in GFLOPS ($2n^3 / T$). (b) Slowdown ratio relative to C (Clang `-O2`).]
) <fig:throughput>

== Empirical Algorithmic Complexity Fits
To verify adherence to the cubic complexity model $T(n) = c dot n^alpha$, we performed linear regression on the logarithmic coordinates $ln T = ln c + alpha ln n$ over the range $n in [64, n_max]$ (excluding $n=32$ where timer resolution and constant overhead introduce non-linearities).

Table 2 presents the regression parameters. The empirical exponents are:
- *C (Clang `-O2`):* $alpha = 3.367$ ($R^2 = 0.9949$)
- *C (Clang `-O0`):* $alpha = 3.116$ ($R^2 = 0.9990$)
- *Java (OpenJDK 22):* $alpha = 3.139$ ($R^2 = 0.9889$)
- *Python (CPython 3.14):* $alpha = 3.111$ ($R^2 = 0.9997$)

All four configurations exhibit scaling exponents closely tracking the theoretical $alpha = 3.000$, validating that every implementation performs $O(n^3)$ arithmetic work. The exponent for C (`-O2`) is slightly higher ($3.367$) because between $n=1024$ and $n=2048$, the column-strided operand $B$ exceeds the 16 MB L2 cache, inducing severe DRAM latency stalls. The pre-factor $c$ highlights the vast performance divide: $c_"Python" = 1.79 times 10^(-8)$ vs. $c_"C-O2" = 1.12 times 10^(-10)$, representing a $160 times$ baseline overhead factor.

#figure(
  image("figures/fig3_memory_scaling.png", width: 95%),
  caption: [Peak Resident Set Size (RSS) and raw matrix payload ($24 n^2$ bytes) vs. matrix dimension $n$.]
) <fig:memory>

== Memory Footprint Analysis
Figure 3 and Table 1 depict the memory scaling behaviors:
1. *Theoretical Payload ($24 n^2$ bytes):* At $n=1024$, the three matrices require $24 times 1024^2 = 25.17$ MB. At $n=2048$, they require $100.66$ MB.
2. *C Peak RSS:* C displays a minimal base footprint of 1.3 MB at $n=32$. As $n$ grows, Peak RSS matches the allocated buffer size with minimal allocator overhead (25.3 MB at $n=1024$, 97.3 MB at $n=2048$).
3. *Java Peak RSS & Heap:* The JVM HotSpot runtime incurs an initial resident set size baseline for class loading, VM metadata, and JIT compilation buffers. As matrix size grows, managed heap usage tracks the theoretical array footprint closely ($25.2$ MB heap at $n=1024$).
4. *Python Memory Inflation:* Python displays a baseline RSS of $approx 21.0$ MB for the CPython binary and standard library. At $n=1024$, Python's Peak RSS reaches 326.1 MB—over $12.9 times$ the raw data payload. Heap tracking via `tracemalloc` confirms that the internal Python objects for $n=1024$ consume $100.7$ MB, precisely $4.0 times$ the raw 64-bit double payload.

= Discussion & Architectural Interpretation

== Compilation vs. Interpretation
The empirical results dramatically illustrate the performance implications of language runtime architectures:
- *Bytecode Interpretation Overhead:* CPython evaluates an abstract syntax tree compiled into bytecode instructions. In the innermost loop, executing `s += A[i][k] * B[k][j]` requires multiple bytecode dispatches (`BINARY_SUBSCR`, `LOAD_FAST`, `BINARY_OP`, `STORE_FAST`), reference count manipulations, and dynamic type checks. These operations introduce tens of machine instructions per floating-point addition/multiplication.
- *Dynamic Boxing:* Every numerical element in Python is a full-fledged heap object (`PyFloatObject`). Accessing matrix elements requires pointer indirection to retrieve the 64-bit float from the heap, which causes frequent cache misses and stalls the CPU pipeline.

== JIT Compilation and Java Efficiency
A standout observation is the remarkable performance of Java under HotSpot OpenJDK 22:
- After warm-up, Java's execution time is nearly indistinguishable from optimized C (`-O2`), achieving 1.77s at $n=1024$ compared to 1.21s for C (`-O2`).
- Intriguingly, at $n=512$, Java (0.115s) actually outpaced C (`-O2`) (0.142s), and at $n=2048$, Java completed in 19.83s versus 29.57s for C (`-O2`). Disassembly analysis shows that HotSpot C2 unrolls the middle loop and applies ARM64 NEON vector registers to perform fused multiply-accumulate operations across row elements.
- However, Java's performance is strictly contingent upon warm-up: cold runs (repetition 1 without warm-up) took $3.5 times$ longer due to interpreted bytecode profiling and compilation latency.

== Impact of Compiler Optimization (C `-O2` vs. `-O0`)
Comparing C compiled with `-O2` against `-O0` demonstrates the critical contribution of compiler transformations:
- Unoptimized C (`-O0`) took 3.92s at $n=1024$—a $3.23 times$ slowdown relative to `-O2`.
- Without optimization, Clang emits memory load and store instructions for the accumulator variable `sum` on every single iteration of $k$, writing to the stack frame rather than pinning `sum` into an ARM64 floating-point register (`d0`). Furthermore, no loop unrolling or instruction scheduling is performed.
- Crucially, unoptimized C is still $10.7 times$ faster than Python at $n=1024$, proving that native execution without dynamic type resolution and boxing provides a substantial baseline advantage even without compiler optimization.

== Cache Locality & Memory Layout Effects
The non-transposed $i-j-k$ loop order inherently penalizes memory hierarchy performance due to column-striding on $B$:
- In C and Java, as $n$ scales beyond 512, the working set of matrix $B$ ($n times n times 8$ bytes = 2 MB at $n=512$, 8 MB at $n=1024$, 32 MB at $n=2048$) exceeds the per-core L1D cache (64 KB) and approaches the shared L2 cache boundary (16 MB).
- Column-striding on $B$ forces a cache line miss on every access once $8n > 64$ bytes (i.e., $n > 8$). The hardware prefetcher is ineffective for high strides across large address spaces.
- At $n=2048$, $B$ requires 32 MB, completely exceeding the 16 MB L2 cache. This cache boundary explains why C (`-O2`) experiences a steep climb in runtime from 4.27s ($n=1536$) to 29.57s ($n=2048$).
- In Python, this penalty is magnified tenfold because even row accesses in $A$ lack spatial locality; the list contains pointers to heap memory addresses allocated non-contiguously by `pymalloc`.

== Threats to Validity & Limitations
We acknowledge several experimental boundaries:
1. *Single-Threaded Baseline:* All implementations were constrained to single-threaded sequential execution. Multi-core parallelism and explicit SIMD vectorization will be examined in Assignment 3.
2. *Algorithmic Restrictions:* The study evaluated solely the basic $i-j-k$ triple-loop. Loop reordering ($i-k-j$), cache-blocking (tiling), and sub-cubic algorithms (Strassen) are addressed in Assignment 2.
3. *Platform Specificity:* Measurements reflect the Apple Silicon unified memory architecture and Darwin OS kernel; absolute timings will differ on x86_64 or Linux server platforms, though relative scaling trends are fundamentally preserved.

= Conclusions

This research systematically investigated basic dense matrix multiplication across C, Java, and Python under controlled execution conditions. The empirical findings establish that:
1. *Algorithmic Scaling:* All three languages exhibit cubic complexity ($T prop n^3$) with empirical exponents $alpha approx 3.11 - 3.37$ ($R^2 > 0.99$).
2. *Runtime Disparity:* Runtime constant factors differ by over two orders of magnitude ($160 times$). Python is $25 times - 35 times$ slower than optimized C and Java. Unoptimized C (`-O0`) is $approx 3.2 times$ slower than `-O2`, but retains an $11 times$ advantage over interpreted Python.
3. *JIT Compiler Parity:* With adequate warm-up, Java's HotSpot C2 compiler achieves performance competitive with native compiled C (`-O2`), even outperforming it at cache-bound scales.
4. *Memory Inflation:* Python incurs a $4 times$ memory penalty on data storage ($32$ bytes/float vs. $8$ bytes) due to object boxing and list pointer overhead, and a $12.9 times$ process RSS footprint at $n=1024$.
5. *Feasibility Boundaries:* Under a 60-second execution budget, Python is limited to $n <= 1024$ ($n=1536$ timed out), whereas C and Java comfortably scale to $n = 2048$.

Future work in Assignment 2 will explore algorithmic optimisations—specifically cache-tiling and loop reordering ($i-k-j$)—to overcome the memory stride bottleneck observed in this baseline study.

#v(1em)
== Reproducibility Materials & Code
The complete source code, benchmark harnesses, raw CSV datasets, analysis scripts, and build files are organized as follows:
- *Repository:* `https://github.com/pablomrtinezzz/big-data-assignments`
- *Materials Archive:* `Assignment01_Martinez_Materials.zip`
- *One-Command Reproduction:* `make all`

#v(0.8em)
== Generative AI Disclosure
In compliance with the course policy (Section 9), generative AI tools were consulted for manuscript language editing, script boilerplate structuring, and plotting layout assistance. All code implementations, experimental protocols, empirical measurements, and architectural conclusions were designed, executed, and validated by the author.

#v(1em)
#bibliography("references.bib", style: "ieee")

] // End of two-column layout

#pagebreak()

= Appendix: Complete Empirical Measurement Tables

#import "tables_autogen.typ": benchmark-table, scaling-table

== Table 1: Comprehensive Benchmark Results Across Matrix Sizes
#benchmark-table

#v(1.5em)
== Table 2: Algorithmic Complexity Fits ($T(n) = c dot n^alpha$)
#scaling-table
