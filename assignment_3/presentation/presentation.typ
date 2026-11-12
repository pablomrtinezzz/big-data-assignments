// ==============================================================================
// PRESENTATION: Assignment 3 - Shared-Memory Parallelism & Vectorisation
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
  "Shared-Memory Parallelism & Vectorisation on Asymmetric Silicon",
  "Evaluating Strong Scaling, OpenMP Runtime Overhead, and Scheduling on Apple M2 (4P + 4E Cores)"
)

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 10pt,
  [
    #card(title: "Research Objective & Hypotheses", border-color: c-border)[
      #text(size: 9pt, fill: rgb("#E2E8F0"))[
        *Core Question:* How does thread count $p in [1, 16]$ govern execution time, speedup $S(p)$, and efficiency $E(p)$ of matrix multiplication on asymmetric multi-core hardware?
      ]
      #v(4pt)
      #rect(width: 100%, stroke: none, fill: rgb("#0D1524"), radius: 4pt, inset: 6pt)[
        #grid(
          columns: (auto, 1fr),
          gutter: 6pt,
          [#text(weight: "bold", size: 8pt, fill: c-blue)[Kernel:]],
          [#text(size: 8pt)[Loop-reordered $i-k-j$ (stride-1 on $B, C$). Parallelized over outer loop $i$.]],
          [#text(weight: "bold", size: 8pt, fill: c-blue)[Baseline:]],
          [#text(size: 8pt)[Identical algorithm in pure sequential C without OpenMP runtime linking.]],
          [#text(weight: "bold", size: 8pt, fill: c-blue)[Safety:]],
          [#text(size: 8pt)[Zero data races: each row $i$ of $C$ is private to its owning worker thread.]]
        )
      ]
    ]

    #v(4pt)

    #card(title: "Hardware Specifications & Execution Environment", border-color: c-border)[
      #grid(
        columns: (1fr, 1fr),
        gutter: 6pt,
        [
          #text(weight: "bold", size: 7.5pt, fill: c-blue)[ASYMMETRIC CPU] \
          #text(size: 8.5pt, weight: "bold")[Apple M2 (ARMv8.5-A)] \
          #text(size: 7.5pt, fill: c-text-muted)[4P (Avalanche @ 3.5 GHz) \ 4E (Blizzard @ 2.4 GHz) \ 8 physical execution contexts]
        ],
        [
          #text(weight: "bold", size: 7.5pt, fill: c-blue)[MEMORY & TOOLCHAIN] \
          #text(size: 8.5pt, weight: "bold")[8.0 GB Unified LPDDR5] \
          #text(size: 7.5pt, fill: c-text-muted)[100 GB/s bandwidth \ Clang 21.0 + OpenMP 23.1 \ Monotonic CLOCK_MONOTONIC]
        ]
      )
    ]
  ],
  [
    #card(title: "Controlled Common Protocol", border-color: c-border)[
      #grid(
        columns: (1fr, 1fr),
        gutter: 5pt,
        stat-box("8 Cores", "Physical Capacity", sublabel: "4P (High IPC) + 4E (Low Power)", accent: c-blue),
        stat-box("< 10⁻¹²", "Numerical Error", sublabel: "Verified vs NumPy BLAS", accent: c-green),
        stat-box("1 to 16", "Thread Range", sublabel: "Testing oversubscription p > 8", accent: c-amber),
        stat-box("5 Reps", "Statistical Rigor", sublabel: "Medians + IQR, 1 warmup disc.", accent: c-purple)
      )
      #v(4pt)
      #text(size: 7.5pt, fill: c-text-muted)[
        *Protocol Compliance (Course Spec p. 2, 6):*
        - Shared deterministic 64-bit SplitMix64 PRNG ($S=42$) across all tests.
        - Strict timing boundary: memory allocation and initialization excluded.
        - Checksum validation prevents Dead Code Elimination (DCE).
      ]
    ]
  ]
)

#pagebreak()

// ==============================================================================
// SLIDE 2: STRONG SCALING & ARCHITECTURAL REGIMES
// ==============================================================================
#slide-header(
  "02",
  "Empirical Scaling & Speedup",
  "Strong Scaling Behavior Across Three Microarchitectural Regimes",
  "Speedup Progression, OpenMP Framework Overhead at p=1, and Oversubscription Penalty"
)

#grid(
  columns: (1.2fr, 0.8fr),
  gutter: 10pt,
  [
    #card(title: "Strong Scaling Speedup Curves vs Ideal Linear & Amdahl's Law", inset: 5pt)[
      #align(center)[
        #image("figures/fig1_strong_scaling_and_speedup.png", height: 165pt)
      ]
    ]
  ],
  [
    #card(title: "The Three Scaling Regimes on Apple M2", border-color: c-border, inset: 6pt)[
      #text(weight: "bold", size: 8pt, fill: c-blue)[REGIME I: HOMOGENEOUS P-CORES ($p in [1, 4]$)] \
      #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
        - High IPC and linear scaling up to $22.8$ GFLOPS ($n=1024$).
        - At $p=1$, runtime shows $2.1 times$ slowdown vs pure sequential C due to thread-pool barrier initialization.
      ]
      #v(3pt)
      #text(weight: "bold", size: 8pt, fill: c-amber)[REGIME II: HETEROGENEOUS CO-EXECUTION ($p in [5, 8]$)] \
      #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
        - 4 Efficiency cores join: lower clock (2.42 GHz) and narrower decode width yield diminishing marginal returns.
        - Peak throughput reaches $29.6$ GFLOPS (Static) and $33.1$ GFLOPS (Dynamic).
      ]
      #v(3pt)
      #text(weight: "bold", size: 8pt, fill: c-rose)[REGIME III: OVERSUBSCRIPTION ($p in [9, 16]$)] \
      #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
        - More threads than 8 physical contexts causes involuntary OS context switching and cache thrashing.
        - At $n=2048$, runtime regresses from $0.65$s ($p=8$) to $0.80$s ($p=16$).
      ]
    ]

    #v(3pt)
    #grid(
      columns: (1fr, 1fr),
      gutter: 4pt,
      stat-box("0.47x", "p=1 Overhead", sublabel: "Framework cost vs pure seq", accent: c-rose),
      stat-box("2.51x", "Peak Speedup", sublabel: "At n=1024 (vs pure sequential)", accent: c-green)
    )
  ]
)

#pagebreak()

// ==============================================================================
// SLIDE 3: SCHEDULING STRATEGIES & HETEROGENEITY
// ==============================================================================
#slide-header(
  "03",
  "Load Balancing & Microarchitecture",
  "Mitigating Core Asymmetry via Dynamic Scheduling",
  "Static Work Distribution Stalls at Barrier; Dynamic Scheduling Delivers +11.7% Throughput Boost"
)

#grid(
  columns: (1.2fr, 0.8fr),
  gutter: 10pt,
  [
    #card(title: "Throughput (GFLOPS) Comparison: Static vs Dynamic Chunking", inset: 5pt)[
      #align(center)[
        #image("figures/fig3_scheduling_heterogeneity.png", height: 165pt)
      ]
    ]
  ],
  [
    #card(title: "The Heterogeneity Bottleneck on Asymmetric SoCs", border-color: c-border, inset: 6pt)[
      #text(weight: "bold", size: 8pt, fill: c-blue)[WHY STATIC SCHEDULING FAILS ON BIG.LITTLE] \
      #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
        Under `schedule(static)`, the loop space $n$ is partitioned into equal slices ($n / p$). \
        - Fast Performance cores complete their partition in half the time.
        - P-cores stall idle at the OpenMP implicit barrier waiting for slow Blizzard E-cores to complete their share.
      ]
      #v(4pt)
      #text(weight: "bold", size: 8pt, fill: c-green)[THE DYNAMIC LOAD-BALANCING ADVANTAGE] \
      #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
        With `schedule(dynamic, 8)`:
        - Threads pull small 8-row work chunks from a central work queue.
        - Faster P-cores process $approx 70%$ of the total rows; slower E-cores handle remaining chunks without causing barrier serialization.
      ]
    ]

    #v(4pt)
    #card(title: "Empirical Gain at p = 8 (n = 1024)", border-color: c-green, inset: 5pt)[
      #grid(
        columns: (1fr, 1fr),
        gutter: 4pt,
        stat-box("29.6 GFLOPS", "Static Schedule", sublabel: "Barrier wait overhead", accent: c-amber),
        stat-box("33.1 GFLOPS", "Dynamic (Chunk=8)", sublabel: "+11.7% throughput gain", accent: c-green)
      )
    ]
  ]
)

#pagebreak()

// ==============================================================================
// SLIDE 4: VECTORISATION, MEMORY LIMITS & TAKEAWAYS
// ==============================================================================
#slide-header(
  "04",
  "Vectorisation & Big Data Insights",
  "ARM NEON SIMD Vectorisation & High-Throughput Takeaways",
  "Isolating SIMD Gains, Hardware Memory Saturation, and the Transition to Distributed Computing"
)

#grid(
  columns: (1.15fr, 0.85fr),
  gutter: 10pt,
  [
    #card(title: "Throughput (GFLOPS) & SIMD Speedup Factor (ARM NEON)", inset: 5pt)[
      #align(center)[
        #image("figures/fig4_simd_vectorisation_impact.png", height: 165pt)
      ]
    ]
    #v(4pt)
    #rect(width: 100%, stroke: 1pt + c-blue, fill: rgb("#0D1A2D"), radius: 4pt, inset: 6pt)[
      #text(weight: "bold", size: 8pt, fill: c-blue)[MEMORY WALL & CACHE BANDWIDTH SATURATION] \
      #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
        At $n = 2048$, working set reaches $96$ MB, far exceeding the $16$ MB shared L2 cache. All 8 cores contend for the 100 GB/s unified memory bus. As bandwidth saturates, SIMD arithmetic execution units become memory-starved, capping parallel efficiency.
      ]
    ]
  ],
  [
    #card(title: "Assembly Vectorisation Evidence", border-color: c-border, inset: 5pt)[
      #text(size: 7.5pt, fill: rgb("#E2E8F0"))[
        - *Vector Instructions:* Clang generates `fmla.2d v, v, v[0]` (128-bit NEON FMA operating on two `float64` elements).
        - *Unrolling:* 4x unrolled inner loop issues 8 arithmetic updates per cycle, maximizing instruction-level parallelism (ILP).
      ]
    ]

    #v(4pt)
    #card(title: "Core Takeaways for Big Data Engineers", border-color: c-green, inset: 5pt)[
      #text(size: 7.5pt, fill: rgb("#F1F5F9"))[
        1. *Always measure p = 1:* OpenMP thread-pool setup incurs measurable overhead ($0.47 times$ relative to pure C).
        2. *Heterogeneous architectures require dynamic scheduling:* Equal static work chunks leave fast cores stalled at barriers.
        3. *Respect hardware core limits:* Oversubscribing ($p > 8$) degrades cache residency and triggers context switches.
        4. *Bridge to Assignment 4:* Shared-memory parallelism hits a physical memory bandwidth ceiling; scaling beyond $n > 10,000$ mandates distributed node partitioning (MPI/Spark).
      ]
    ]
  ]
)
