// ==============================================================================
// PRESENTATION: Assignment 1 - Matrix Multiplication Performance & Memory
// Author: Pablo Martínez Suárez (ULPGC - Big Data 2026–2027)
// Format: 16:9 widescreen presentation (4 slides maximum)
// ==============================================================================

#set page(
  paper: "presentation-16-9",
  margin: (x: 1.2cm, top: 0.8cm, bottom: 0.7cm),
  fill: rgb("#0B0F19")
)

#set text(
  font: ("Helvetica Neue", "Arial"),
  fill: rgb("#F8FAFC"),
  size: 9.5pt,
  weight: "regular"
)

// Brand and Theme Colors
#let c-bg-card = rgb("#151F32")
#let c-border = rgb("#22314E")
#let c-blue = rgb("#38BDF8")
#let c-green = rgb("#34D399")
#let c-amber = rgb("#FBBF24")
#let c-rose = rgb("#F87171")
#let c-purple = rgb("#A78BFA")
#let c-text-muted = rgb("#94A3B8")
#let c-text-dim = rgb("#64748B")

// Reusable card container
#let card(title: none, border-color: c-border, fill: c-bg-card, inset: 7pt, radius: 5pt, body) = {
  rect(
    width: 100%,
    stroke: 1pt + border-color,
    fill: fill,
    radius: radius,
    inset: inset
  )[
    #if title != none [
      #block(below: 5pt)[
        #text(weight: "bold", size: 9.5pt, fill: c-blue)[#title]
      ]
    ]
    #body
  ]
}

// Reusable stat metric box
#let stat-box(value, label, sublabel: none, accent: c-blue) = {
  rect(
    width: 100%,
    stroke: 1pt + c-border,
    fill: rgb("#111827"),
    radius: 4pt,
    inset: (x: 6pt, y: 5pt)
  )[
    #text(weight: "bold", size: 14pt, fill: accent)[#value] \
    #v(-3pt)
    #text(weight: "bold", size: 8pt, fill: rgb("#F1F5F9"))[#label]
    #if sublabel != none [
      \ #text(size: 6.8pt, fill: c-text-dim)[#sublabel]
    ]
  ]
}

// Slide Header component
#let slide-header(number, tag, title, subtitle) = [
  #grid(
    columns: (1fr, auto),
    gutter: 10pt,
    align: (left, right),
    [
      #box(fill: rgb("#1E293B"), radius: 3pt, inset: (x: 5pt, y: 2pt))[
        #text(size: 7pt, weight: "bold", fill: c-blue, tracking: 1.5pt)[#upper(tag)]
      ]
      #v(2pt)
      #text(size: 16pt, weight: "bold", fill: rgb("#FFFFFF"))[#title] \
      #v(-3pt)
      #text(size: 8.5pt, fill: c-text-muted)[#subtitle]
    ],
    [
      #v(4pt)
      #text(size: 9.5pt, weight: "bold", fill: c-text-dim)[SLIDE ]
      #text(size: 15pt, weight: "bold", fill: c-blue)[#number]
      #text(size: 9.5pt, fill: c-text-dim)[\/ 04]
    ]
  )
  #v(3pt)
  #line(length: 100%, stroke: 0.75pt + c-border)
  #v(5pt)
]

// ==============================================================================
// SLIDE 1: RESEARCH OBJECTIVE & EXPERIMENTAL RIGOR
// ==============================================================================
#slide-header(
  "01",
  "Experimental Rigor & Protocol",
  "Benchmarking Canonical Matrix Multiplication Across Paradigms",
  "Evaluating AOT (C), JIT (Java HotSpot), and Interpreted (CPython) Runtimes Under Identical Constraints"
)

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 10pt,
  [
    #card(title: "Research Objective & Theoretical Model", border-color: c-border)[
      #text(size: 9pt, fill: rgb("#E2E8F0"))[
        *Core Question:* How does matrix dimension $n$ govern runtime and memory when algorithmic structure ($i-j-k$), precision (IEEE 754 float64), and input data are held strictly identical?
      ]
      #v(4pt)
      #rect(width: 100%, stroke: none, fill: rgb("#0D1524"), radius: 4pt, inset: 6pt)[
        #grid(
          columns: (auto, 1fr),
          gutter: 6pt,
          [#text(weight: "bold", size: 8pt, fill: c-blue)[Kernel:]],
          [#text(size: 8pt)[Canonical triple-loop: $C_(i j) = sum_(k=0)^(n-1) A_(i k) B_(k j) arrow 2n^3$ FLOPs]],
          [#text(weight: "bold", size: 8pt, fill: c-blue)[Inputs:]],
          [#text(size: 8pt)[Bit-for-bit identical 64-bit SplitMix64 PRNG ($S=42$) across all languages]],
          [#text(weight: "bold", size: 8pt, fill: c-blue)[Timing:]],
          [#text(size: 8pt)[Isolated monotonic clock around *only* the inner arithmetic loop]]
        )
      ]
    ]

    #v(4pt)

    #card(title: "Hardware Specifications & Controlled Single-Threaded Testbed", border-color: c-border)[
      #grid(
        columns: (1fr, 1fr),
        gutter: 6pt,
        [
          #text(weight: "bold", size: 7.5pt, fill: c-blue)[HOST PROCESSOR] \
          #text(size: 8.5pt, weight: "bold")[Apple M2 (ARMv8.5-A)] \
          #text(size: 7.5pt, fill: c-text-muted)[4P (3.5GHz) + 4E (2.4GHz) \ 16 MB Shared L2 | Single-threaded]
        ],
        [
          #text(weight: "bold", size: 7.5pt, fill: c-blue)[MEMORY & SOFTWARE] \
          #text(size: 8.5pt, weight: "bold")[8.0 GB Unified Memory] \
          #text(size: 7.5pt, fill: c-text-muted)[100 GB/s LPDDR5-6400 \ Clang 21.0 | JDK 22.0.2 | Python 3.14]
        ]
      )
    ]
  ],
  [
    #card(title: "Experimental Benchmarking Protocol", border-color: c-blue)[
      #grid(
        columns: (1fr, 1fr),
        gutter: 6pt,
        stat-box("n = 32 - 2048", "Matrix Dimensions", sublabel: "10 discrete matrix sizes", accent: c-blue),
        stat-box("R = 5 runs", "Median + IQR", sublabel: "Robust to system noise", accent: c-green),
        stat-box("W = 1 - 3", "Dedicated Warm-ups", sublabel: "Reaching JIT steady-state", accent: c-purple),
        stat-box("60.0 s", "Timeout Budget", sublabel: "Per repetition cutoff", accent: c-rose)
      )
      
      #v(5pt)
      #rect(width: 100%, stroke: 0.5pt + rgb("#334155"), fill: rgb("#0F172A"), radius: 4pt, inset: 5pt)[
        #text(size: 7.5pt, fill: rgb("#CBD5E1"))[
          *Integrity & Defensive Design:*
          - *Dead-Code Elimination Defense:* Matrix checksum $S = sum C_(i j)$ verified after timing.
          - *Zero Allocation in Loop:* Flat buffers allocated prior to timing capture.
          - *Ground Truth:* Relative error $< 10^(-12)$ validated against NumPy BLAS ($A times 0$, $A times I$).
        ]
      ]
    ]
  ]
)

#pagebreak()

// ==============================================================================
// SLIDE 2: RUNTIME PERFORMANCE & THE 60-SECOND FEASIBILITY WALL
// ==============================================================================
#slide-header(
  "02",
  "Empirical Scaling & Limits",
  "Execution Time Divergence & Feasibility Wall",
  "Cubic Complexity Holds ($O(n^3)$), but Constant Factors Diverge by Over Two Orders of Magnitude"
)

#grid(
  columns: (1.25fr, 0.75fr),
  gutter: 10pt,
  [
    #card(title: "Empirical Runtime: Linear Scale (a) & Log-Log Scaling (b)", inset: 5pt)[
      #align(center)[
        #image("figures/fig1_runtime_comparison.png", height: 165pt)
      ]
    ]
    #v(4pt)
    #rect(width: 100%, stroke: 1pt + c-rose, fill: rgb("#1F131D"), radius: 4pt, inset: 6pt)[
      #text(weight: "bold", size: 8pt, fill: c-rose)[CRITICAL DISCOVERY: THE 60-SECOND TIMEOUT WALL] \
      #text(size: 7.8pt, fill: rgb("#F1F5F9"))[
        At $n = 1024$, Python takes *41.15s*. At $n = 1536$, cubic projection predicts *~139s*, exceeding the budget and triggering process abortion. *Python's operational ceiling is strictly $n = 1024$*, whereas C and Java scale seamlessly to $n = 2048$.
      ]
    ]
  ],
  [
    #card(title: "Execution Timings at $n = 1024$", border-color: c-border, inset: 5pt)[
      #grid(
        columns: (1fr, 1fr),
        gutter: 4pt,
        stat-box("1.21 s", "C (Clang -O2)", sublabel: "Baseline (1.78 GFLOPS)", accent: c-blue),
        stat-box("1.76 s", "Java (JDK 22)", sublabel: "1.45x slowdown vs C", accent: c-green),
        stat-box("3.80 s", "C (Clang -O0)", sublabel: "3.15x slowdown vs C -O2", accent: c-amber),
        stat-box("41.15 s", "Python 3.14", sublabel: "34.0x slower than C", accent: c-rose)
      )
    ]

    #v(4pt)
    #card(title: "Empirical Scaling Exponents ($T(n) = c dot n^alpha$)", inset: 5pt)[
      #text(size: 7.6pt)[
        - *C (`-O2`):* $alpha = 3.38$ ($R^2 = 0.995$) -- DRAM stride penalty
        - *C (`-O0`):* $alpha = 3.10$ ($R^2 = 0.999$)
        - *Java:* $alpha = 3.13$ ($R^2 = 0.990$)
        - *Python:* $alpha = 3.10$ ($R^2 = 1.000$)
        #v(2pt)
        #text(fill: c-blue, weight: "bold")[Constant Factor Overhead ($c_"Python" / c_"C-O2"$):] \
        $1.91 times 10^(-8) / 1.01 times 10^(-10) = bold(189 times" baseline gap")$
      ]
    ]
  ]
)

#pagebreak()

// ==============================================================================
// SLIDE 3: ARCHITECTURAL DEEP DIVE - THROUGHPUT & JIT CONVERGENCE
// ==============================================================================
#slide-header(
  "03",
  "Architectural Dissection",
  "Throughput Analysis & The Power of JIT Compilation",
  "Understanding Why Python Stalls and How Java HotSpot Matches Native Compiled C"
)

#grid(
  columns: (1.25fr, 0.75fr),
  gutter: 10pt,
  [
    #card(title: "Computational Throughput (GFLOPS) & Relative Slowdown", inset: 5pt)[
      #align(center)[
        #image("figures/fig2_throughput_speedup.png", height: 165pt)
      ]
    ]
    #v(4pt)
    #grid(
      columns: (1fr, 1fr),
      gutter: 6pt,
      card(title: "Python 0.06 GFLOPS Bottleneck", border-color: c-rose, inset: 5pt)[
        #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
          - *Interpreter Dispatch:* Inner loop opcode decoding in `ceval.c`.
          - *Dynamic Object Boxing:* Every number is a heap `PyFloatObject` (24 bytes).
          - *Pointer Indirection:* Stalls hardware CPU execution pipeline.
        ]
      ],
      card(title: "Java HotSpot C2 JIT Parity", border-color: c-green, inset: 5pt)[
        #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
          - *Tier 4 Compilation:* Unrolls inner loops and allocates registers.
          - *ARM64 NEON FMA:* Generates fused multiply-accumulate vector instructions.
          - *Outperforms C at $n=2048$:* 19.1s (Java) vs 29.0s (C) due to loop structure.
        ]
      ]
    )
  ],
  [
    #card(title: "Optimization Impact: C -O2 vs -O0", border-color: c-amber, inset: 5pt)[
      #stat-box("3.15x", "Compiler Speedup", sublabel: "-O2 vs -O0 at n = 1024", accent: c-amber)
      #v(3pt)
      #text(size: 7.5pt, fill: c-text-muted)[
        *Why is -O0 so slow?* \
        - Accumulator `sum` is spilled to RAM stack frame on *every iteration* of $k$.
        - No instruction scheduling or loop unrolling.
        #v(2pt)
        *Crucial Takeaway:* \
        Even unoptimized C (`-O0`) is *10.8x faster than Python*, proving that native compilation without dynamic boxing provides an immense baseline advantage.
      ]
    ]

    #v(4pt)
    #card(title: "Warm-Up Impact on Java", border-color: c-purple, inset: 5pt)[
      #stat-box("3.5x", "Warm-Up Speedup", sublabel: "Cold run vs Steady-state C2", accent: c-purple)
      #v(2pt)
      #text(size: 7.3pt, fill: c-text-muted)[
        Without warm-up, Java's first run is slowed down by Tier 0 bytecode interpretation and C1 profiling.
      ]
    ]
  ]
)

#pagebreak()

// ==============================================================================
// SLIDE 4: MEMORY FOOTPRINT & PRACTICAL BIG DATA TAKEAWAYS
// ==============================================================================
#slide-header(
  "04",
  "Memory & Big Data Implications",
  "Memory Overhead & Practical Takeaways",
  "Data Representation Dictates Memory Expansion and Hardware Cache Stride Bottlenecks"
)

#grid(
  columns: (1.15fr, 0.85fr),
  gutter: 10pt,
  [
    #card(title: "Memory Scaling: Process Peak RSS vs Raw Payload ($24n^2$ bytes)", inset: 5pt)[
      #align(center)[
        #image("figures/fig3_memory_scaling.png", height: 165pt)
      ]
    ]
    #v(4pt)
    #rect(width: 100%, stroke: 1pt + c-blue, fill: rgb("#0D1A2D"), radius: 4pt, inset: 6pt)[
      #text(weight: "bold", size: 8pt, fill: c-blue)[THE CACHE STRIDE WALL (THE $i-j-k$ BOTTLENECK)] \
      #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
        In row-major layout, accessing $B[k][j]$ along columns incurs a stride of $8n$ bytes. Once $8n > 64$B (cache line), *every iteration triggers a cache miss*. At $n = 2048$, Matrix $B$ consumes 32 MB, overflowing the 16 MB L2 cache and triggering steep DRAM latency stalls.
      ]
    ]
  ],
  [
    #card(title: "Memory Footprint at $n = 1024$", border-color: c-border, inset: 5pt)[
      #grid(
        columns: (1fr, 1fr),
        gutter: 4pt,
        stat-box("24.0 MB", "Raw Payload", sublabel: "3 matrices x 8 MB (24n²)", accent: rgb("#E2E8F0")),
        stat-box("25.3 MB", "C Peak RSS", sublabel: "1.05x raw payload", accent: c-blue),
        stat-box("23.8 MB", "Java Heap", sublabel: "Managed delta (†)", accent: c-green),
        stat-box("306.3 MB", "Python RSS", sublabel: "12.8x raw payload expansion", accent: c-rose)
      )
      #v(2pt)
      #text(size: 6.8pt, fill: c-text-dim)[
        † Java RSS includes JVM metaspace and runtime (~80MB base).
      ]
    ]

    #v(4pt)
    #card(title: "Core Takeaways for Data Engineering", border-color: c-green, inset: 5pt)[
      #text(size: 7.5pt, fill: rgb("#F1F5F9"))[
        1. *Never use pure Python loops* for numerical linear algebra; always rely on compiled C/C++ kernels (NumPy BLAS) or JIT runtimes.
        2. *Java HotSpot is production-grade* for high-throughput numeric engines when warm-up and flat arrays are leveraged.
        3. *Hardware awareness is mandatory:* Algorithmic $O(n^3)$ math cannot mask CPU cache hierarchy physics.
        4. *Bridge to Assignment 2:* Cache-tiling and loop reordering ($i-k-j$) will break through this cache stride wall.
      ]
    ]
  ]
)
