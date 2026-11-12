#!/usr/bin/env python3
"""
Publication-Quality Visualization and Typst Table Generator for Assignment 3:
Shared-Memory Parallelism and ARM NEON Vectorisation.

Generates:
  - fig1_strong_scaling_and_speedup: Strong scaling runtime and Speedup S(p) vs threads p
  - fig2_efficiency_and_oversubscription: Parallel efficiency E(p) highlighting 3 architectural zones
  - fig3_scheduling_heterogeneity: Static vs Dynamic scheduling on heterogeneous Apple M2 cores
  - fig4_simd_vectorisation_impact: Throughput (GFLOPS) for Scalar vs ARM NEON SIMD
  - tables_autogen.typ: Typst table code for the research paper
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BENCH_DIR = os.path.join(PROJECT_DIR, "benchmarks", "results")
REPORT_FIG_DIR = os.path.join(PROJECT_DIR, "report", "figures")
PRES_FIG_DIR = os.path.join(PROJECT_DIR, "presentation", "figures")
REPORT_DIR = os.path.join(PROJECT_DIR, "report")

os.makedirs(REPORT_FIG_DIR, exist_ok=True)
os.makedirs(PRES_FIG_DIR, exist_ok=True)

# Publication style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

def plot_strong_scaling_and_speedup(parallel_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 6.4), sharex=True, gridspec_kw={'height_ratios': [1.1, 1.0]})

    target_sizes = [512, 1024, 1536, 2048]
    colors = {512: "#2b83ba", 1024: "#abdda4", 1536: "#fdae61", 2048: "#d7191c"}
    markers = {512: "o", 1024: "s", 1536: "^", 2048: "D"}

    static_df = parallel_df[(parallel_df["method"] == "omp_ikj") & (parallel_df["schedule"] == "static")]

    for n in target_sizes:
        sub = static_df[static_df["n"] == n].sort_values("threads")
        if sub.empty:
            continue
        ax1.plot(sub["threads"], sub["time_median_sec"], label=f"$n = {n}$",
                 color=colors[n], marker=markers[n], linewidth=2.0)
        ax2.plot(sub["threads"], sub["speedup"], label=f"$n = {n}$",
                 color=colors[n], marker=markers[n], linewidth=2.0)

    # Subplot 1: Runtime
    ax1.set_ylabel("Median Kernel Runtime (s)")
    ax1.set_yscale("log")
    ax1.set_title("Strong Scaling Runtime across Thread Counts (Apple M2)", fontsize=11, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Subplot 2: Speedup
    threads = np.array([1, 2, 4, 6, 8, 12, 16])
    ax2.plot(threads, threads, "k--", label="Ideal Linear $S(p)=p$", alpha=0.6, linewidth=1.5)

    # Amdahl's Law theoretical curve with f = 0.95
    f_est = 0.95
    amdahl_p = np.linspace(1, 16, 100)
    amdahl_s = 1.0 / ((1.0 - f_est) + (f_est / amdahl_p))
    ax2.plot(amdahl_p, amdahl_s, ":", color="#7570b3", label=f"Amdahl's Law ($f={f_est}$)", linewidth=1.6)

    ax2.set_xlabel("Thread Count ($p$)")
    ax2.set_ylabel("Speedup $S(p) = T_{seq} / T_p$")
    ax2.set_xticks(threads)
    ax2.set_title("Observed Speedup vs. Theoretical Limits", fontsize=11, fontweight="bold")
    ax2.grid(True)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    for dest in [REPORT_FIG_DIR, PRES_FIG_DIR]:
        fig.savefig(os.path.join(dest, "fig1_strong_scaling_and_speedup.pdf"))
        fig.savefig(os.path.join(dest, "fig1_strong_scaling_and_speedup.png"))
    plt.close()
    print("  [GENERATED] fig1_strong_scaling_and_speedup (.pdf / .png)")

def plot_efficiency_and_regimes(parallel_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    target_sizes = [512, 1024, 1536, 2048]
    colors = {512: "#2b83ba", 1024: "#2ca25f", 1536: "#fa9fb5", 2048: "#e31a1c"}
    markers = {512: "o", 1024: "s", 1536: "^", 2048: "D"}

    static_df = parallel_df[(parallel_df["method"] == "omp_ikj") & (parallel_df["schedule"] == "static")]

    for n in target_sizes:
        sub = static_df[static_df["n"] == n].sort_values("threads")
        if sub.empty:
            continue
        ax.plot(sub["threads"], sub["efficiency"] * 100.0, label=f"$n = {n}$",
                color=colors[n], marker=markers[n], linewidth=2.0)

    # Shaded Architectural Regimes of Apple M2
    ax.axvspan(1, 4, color="#e0f3f8", alpha=0.5, label="Regime I: Pure P-Cores (3.5 GHz)")
    ax.axvspan(4, 8, color="#fee0d2", alpha=0.5, label="Regime II: Heterogeneous Cores (P + E)")
    ax.axvspan(8, 16, color="#e5f5e0", alpha=0.5, label="Regime III: Oversubscription ($p > 8$)")

    ax.axhline(100.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Thread Count ($p$)")
    ax.set_ylabel("Parallel Efficiency $E(p)$ (%)")
    ax.set_title("Parallel Efficiency $E(p) = S(p)/p$ across Architectural Regimes", fontsize=11, fontweight="bold")
    ax.set_xticks([1, 2, 4, 6, 8, 12, 16])
    ax.set_ylim(0, 115)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    for dest in [REPORT_FIG_DIR, PRES_FIG_DIR]:
        fig.savefig(os.path.join(dest, "fig2_efficiency_and_oversubscription.pdf"))
        fig.savefig(os.path.join(dest, "fig2_efficiency_and_oversubscription.png"))
    plt.close()
    print("  [GENERATED] fig2_efficiency_and_oversubscription (.pdf / .png)")

def plot_scheduling_heterogeneity(parallel_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    sub = parallel_df[(parallel_df["n"] == 2048) & (parallel_df["threads"].isin([4, 8, 16]))]
    if sub.empty:
        sub = parallel_df[(parallel_df["n"] == 1024) & (parallel_df["threads"].isin([4, 8, 16]))]

    sched_map = {
        "static": ("Static Schedule", "#3182bd", "o"),
        "dynamic_8": ("Dynamic (Chunk=8)", "#de2d26", "s"),
        "dynamic_32": ("Dynamic (Chunk=32)", "#31a354", "^")
    }

    x_threads = [4, 8, 16]
    width = 0.25
    x_indices = np.arange(len(x_threads))

    offset = -width
    for sched_key, (sched_name, color, _) in sched_map.items():
        data = sub[sub["schedule"] == sched_key].sort_values("threads")
        if len(data) == len(x_threads):
            gflops = data["gflops"].values
            ax.bar(x_indices + offset, gflops, width=width, label=sched_name, color=color, alpha=0.85, edgecolor="black")
            for i, val in enumerate(gflops):
                ax.text(x_indices[i] + offset, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)
            offset += width

    ax.set_xlabel("Thread Count ($p$)")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_title("Impact of Dynamic Scheduling on Asymmetric Cores ($n = 2048$)", fontsize=11, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{p} threads" for p in x_threads])
    ax.grid(axis="y")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    for dest in [REPORT_FIG_DIR, PRES_FIG_DIR]:
        fig.savefig(os.path.join(dest, "fig3_scheduling_heterogeneity.pdf"))
        fig.savefig(os.path.join(dest, "fig3_scheduling_heterogeneity.png"))
    plt.close()
    print("  [GENERATED] fig3_scheduling_heterogeneity (.pdf / .png)")

def plot_simd_vectorisation_impact(vector_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))

    # Left: Throughput comparison for n = 2048 across threads
    sub_2048 = vector_df[vector_df["n"] == 2048]
    threads = [1, 2, 4, 8, 16]

    scalar_gflops = []
    simd_gflops = []
    simd_speedup = []

    for p in threads:
        row_scal = sub_2048[(sub_2048["variant"] == "scalar") & (sub_2048["threads"] == p)]
        row_simd = sub_2048[(sub_2048["variant"] == "simd") & (sub_2048["threads"] == p)]
        if not row_scal.empty and not row_simd.empty:
            g_scal = row_scal["gflops"].values[0]
            g_simd = row_simd["gflops"].values[0]
            scalar_gflops.append(g_scal)
            simd_gflops.append(g_simd)
            simd_speedup.append(g_simd / g_scal)
        else:
            scalar_gflops.append(0)
            simd_gflops.append(0)
            simd_speedup.append(1)

    x = np.arange(len(threads))
    width = 0.35

    ax1.bar(x - width/2, scalar_gflops, width, label="Scalar (Non-Vectorised)", color="#9ecae1", edgecolor="black")
    ax1.bar(x + width/2, simd_gflops, width, label="ARM NEON SIMD (128-bit)", color="#3182bd", edgecolor="black")

    for i in range(len(threads)):
        ax1.text(x[i] - width/2, scalar_gflops[i] + 0.5, f"{scalar_gflops[i]:.1f}", ha="center", va="bottom", fontsize=7)
        ax1.text(x[i] + width/2, simd_gflops[i] + 0.5, f"{simd_gflops[i]:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax1.set_xlabel("Thread Count ($p$)")
    ax1.set_ylabel("Throughput (GFLOPS)")
    ax1.set_title("Throughput Scaling: Scalar vs SIMD", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(p) for p in threads])
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", fontsize=8)

    # Right: Pure SIMD Speedup Factor S_SIMD = T_scalar / T_simd across n
    sizes = [64, 128, 256, 512, 1024, 1536, 2048]
    simd_factors_seq = []
    simd_factors_p8 = []

    for n in sizes:
        row_seq_scal = vector_df[(vector_df["n"] == n) & (vector_df["threads"] == 1) & (vector_df["variant"] == "scalar")]
        row_seq_simd = vector_df[(vector_df["n"] == n) & (vector_df["threads"] == 1) & (vector_df["variant"] == "simd")]
        if not row_seq_scal.empty and not row_seq_simd.empty:
            simd_factors_seq.append(row_seq_simd["simd_factor"].values[0])
        else:
            simd_factors_seq.append(1.0)

        row_p8_scal = vector_df[(vector_df["n"] == n) & (vector_df["threads"] == 8) & (vector_df["variant"] == "scalar")]
        row_p8_simd = vector_df[(vector_df["n"] == n) & (vector_df["threads"] == 8) & (vector_df["variant"] == "simd")]
        if not row_p8_scal.empty and not row_p8_simd.empty:
            simd_factors_p8.append(row_p8_simd["simd_factor"].values[0])
        else:
            simd_factors_p8.append(1.0)

    ax2.plot(sizes, simd_factors_seq, "o-", color="#e6550d", label="Sequential ($p = 1$)", linewidth=1.8)
    ax2.plot(sizes, simd_factors_p8, "s-", color="#756bb1", label="Parallel ($p = 8$)", linewidth=1.8)
    ax2.axhline(2.0, color="gray", linestyle=":", label="Theoretical $2\\times$ Peak")

    ax2.set_xlabel("Matrix Dimension ($n$)")
    ax2.set_ylabel("SIMD Speedup Factor ($T_{scal} / T_{simd}$)")
    ax2.set_title("Isolated Vectorisation Gain ($S_{SIMD}$)", fontsize=10, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    for dest in [REPORT_FIG_DIR, PRES_FIG_DIR]:
        fig.savefig(os.path.join(dest, "fig4_simd_vectorisation_impact.pdf"))
        fig.savefig(os.path.join(dest, "fig4_simd_vectorisation_impact.png"))
    plt.close()
    print("  [GENERATED] fig4_simd_vectorisation_impact (.pdf / .png)")

def generate_typst_tables(parallel_df: pd.DataFrame, vector_df: pd.DataFrame):
    out_path = os.path.join(REPORT_DIR, "tables_autogen.typ")

    # Table 1: Strong Scaling Table for n in [512, 1024, 2048]
    static_df = parallel_df[(parallel_df["method"] == "omp_ikj") & (parallel_df["schedule"] == "static")]

    typ_content = """// Automatically generated by generate_plots_and_tables.py
// Do not edit manually.

#let table_parallel_scaling = [
  #figure(
    caption: [Empirical strong scaling measurements for shared-memory matrix multiplication ($i-k-j$) on Apple M2 (4P + 4E cores). Values represent medians of 5 measured repetitions preceded by 1 warmup run.],
    table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto),
      align: (center, center, right, right, right, right, right, right),
      stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else { (bottom: 0.3pt + luma(200)) },
      table.header(
        [*$n$*], [*Threads ($p$)*], [*Median Time (s)*], [*IQR (ms)*], [*GFLOPS*], [*Speedup $S(p)$*], [*Efficiency $E(p)$*], [*Peak RSS (MB)*]
      ),
"""
    for n in [512, 1024, 2048]:
        sub_n = static_df[static_df["n"] == n].sort_values("threads")
        for _, row in sub_n.iterrows():
            t_med = row["time_median_sec"]
            iqr_ms = row["time_iqr_sec"] * 1000.0
            gflops = row["gflops"]
            sp = row["speedup"]
            eff = row["efficiency"] * 100.0
            rss_mb = row["peak_rss_bytes"] / (1024.0 * 1024.0)
            p = int(row["threads"])
            typ_content += f'      [{n}], [{p}], [{t_med:.4f}], [{iqr_ms:.2f}], [{gflops:.2f}], [{sp:.2f}x], [{eff:.1f}%], [{rss_mb:.1f}],\n'

    typ_content += """    )
  )
]

#let table_simd_comparison = [
  #figure(
    caption: [ARM NEON SIMD vectorisation speedup and combined scaling performance for $n = 2048$ ($24n^2 = 96$ MB payload).],
    table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (center, right, right, right, right, right),
      stroke: (x, y) => if y == 0 { (bottom: 1.2pt + black) } else if y == 1 { (bottom: 0.8pt + black) } else { (bottom: 0.3pt + luma(200)) },
      table.header(
        [*Threads ($p$)*], [*Scalar Time (s)*], [*Scalar GFLOPS*], [*NEON SIMD Time (s)*], [*NEON GFLOPS*], [*SIMD Gain $S_"SIMD"$*]
      ),
"""
    sub_vec = vector_df[vector_df["n"] == 2048]
    for p in [1, 2, 4, 8, 16]:
        r_scal = sub_vec[(sub_vec["variant"] == "scalar") & (sub_vec["threads"] == p)]
        r_simd = sub_vec[(sub_vec["variant"] == "simd") & (sub_vec["threads"] == p)]
        if not r_scal.empty and not r_simd.empty:
            t_s = r_scal["time_median_sec"].values[0]
            g_s = r_scal["gflops"].values[0]
            t_v = r_simd["time_median_sec"].values[0]
            g_v = r_simd["gflops"].values[0]
            gain = g_v / g_s if g_s > 0 else 1.0
            typ_content += f'      [{p}], [{t_s:.4f}], [{g_s:.2f}], [{t_v:.4f}], [{g_v:.2f}], [{gain:.2f}x],\n'

    typ_content += """    )
  )
]
"""
    with open(out_path, "w") as f:
        f.write(typ_content)
    print(f"  [GENERATED] {out_path}")

def main():
    parallel_csv = os.path.join(BENCH_DIR, "summary_parallel_benchmarks.csv")
    vector_csv = os.path.join(BENCH_DIR, "summary_vector_benchmarks.csv")

    if not os.path.exists(parallel_csv):
        print(f"Error: {parallel_csv} not found.", file=sys.stderr)
        return

    parallel_df = pd.read_csv(parallel_csv)
    vector_df = pd.read_csv(vector_csv) if os.path.exists(vector_csv) else None

    print("Generating figures and tables...")
    plot_strong_scaling_and_speedup(parallel_df)
    plot_efficiency_and_regimes(parallel_df)
    plot_scheduling_heterogeneity(parallel_df)

    if vector_df is not None:
        plot_simd_vectorisation_impact(vector_df)
        generate_typst_tables(parallel_df, vector_df)

    print("All plots and tables successfully generated!")

if __name__ == "__main__":
    main()
