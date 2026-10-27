#!/usr/bin/env python3
"""
Publication-Quality Visualization and Typst Table Generation for Assignment 2.
Generates:
  - fig1_dense_scaling_and_speedup: Log-log scaling and speedup factor over baseline.
  - fig2_sparsity_crossover: Break-even analysis between CSR SpGEMM and Dense Reordered.
  - fig3_fillin_and_memory: Non-zero fill-in growth and memory footprint reduction.
  - tables_autogen.typ: Typst table source for the research paper.
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
REPORT_DIR = os.path.join(PROJECT_DIR, "report")

os.makedirs(REPORT_FIG_DIR, exist_ok=True)

# Styling configuration for IEEE publications
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
    "lines.linewidth": 1.75,
    "lines.markersize": 6,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

def plot_dense_scaling(dense_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 6.2), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.0]})

    color_map = {
        "baseline": "#d95f02",    # Vermilion / Orange
        "reordered": "#1b9e77",   # Teal / Green
        "blocked": "#7570b3"     # Purple
    }
    label_map = {
        "baseline": "Canonical Baseline (i-j-k)",
        "reordered": "Loop Reordered (i-k-j)",
        "blocked": "Cache Blocked (B=64, i-k-j)"
    }
    marker_map = {
        "baseline": "o",
        "reordered": "s",
        "blocked": "^"
    }

    # Top: Runtime scaling
    for m in ["baseline", "reordered", "blocked"]:
        sub = dense_df[dense_df["method"] == m].sort_values("n")
        ax1.plot(sub["n"], sub["time_median_sec"], label=label_map[m],
                 color=color_map[m], marker=marker_map[m])
        # Error bars using IQR
        ax1.fill_between(sub["n"],
                         sub["time_median_sec"] - 0.5 * sub["time_iqr_sec"],
                         sub["time_median_sec"] + 0.5 * sub["time_iqr_sec"],
                         color=color_map[m], alpha=0.15)

    # Reference O(n^3) cubic slope
    n_vals = dense_df["n"].unique()
    n_vals = np.sort(n_vals)
    c_cubic = dense_df[dense_df["method"] == "baseline"]["time_median_sec"].iloc[-1] / (n_vals[-1]**3)
    ax1.plot(n_vals, c_cubic * (n_vals**3), "k:", label="Theoretical O(n³)", alpha=0.7)

    # Annotations for cache limits on Apple M2 (64 KB L1D / core; 16 MB L2)
    # 3 matrices of doubles: 3 * n^2 * 8 bytes
    # n_L1D = sqrt(64 * 1024 / 24) ≈ 52
    # n_L2 = sqrt(16 * 1024 * 1024 / 24) ≈ 836
    ax1.axvline(x=52, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.text(54, 1e-4, "L1D Boundary (64 KB)", rotation=90, color="gray", fontsize=8, va="bottom")
    ax1.axvline(x=836, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.text(860, 1e-4, "L2 Boundary (16 MB)", rotation=90, color="gray", fontsize=8, va="bottom")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("Dense Matrix Multiplication Execution Time & Speedup (Apple M2)")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Bottom: Speedup vs baseline
    for m in ["reordered", "blocked"]:
        sub = dense_df[dense_df["method"] == m].sort_values("n")
        ax2.plot(sub["n"], sub["speedup_vs_baseline"], label=label_map[m],
                 color=color_map[m], marker=marker_map[m])

    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, label="Baseline (1.0x)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Matrix Dimension n")
    ax2.set_ylabel("Speedup Factor vs Baseline")
    ax2.set_xticks(n_vals)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.grid(True)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    pdf_path = os.path.join(REPORT_FIG_DIR, "fig1_dense_scaling_and_speedup.pdf")
    png_path = os.path.join(REPORT_FIG_DIR, "fig1_dense_scaling_and_speedup.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"[GENERATED] Figure 1: {pdf_path}")

def plot_sparsity_crossover(sparse_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sizes = sorted(sparse_df["n"].unique())
    colors = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c"]

    for idx, n in enumerate(sizes):
        sub = sparse_df[sparse_df["n"] == n].sort_values("target_sparsity")
        t_dense = sub["dense_reordered_time_sec"].iloc[0]

        ax.plot(sub["target_sparsity"] * 100.0, sub["kernel_time_median_sec"],
                marker="o", color=colors[idx % len(colors)], label=f"CSR SpGEMM (n={n})")
        ax.axhline(y=t_dense, color=colors[idx % len(colors)], linestyle=":", alpha=0.8,
                   label=f"Dense (i-k-j) Ref (n={n})")

    ax.set_yscale("log")
    ax.set_xlabel("Input Matrix Sparsity (% zeros)")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Sparse CSR vs Dense Reordered: Break-Even / Crossover Analysis")
    ax.grid(True)
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    pdf_path = os.path.join(REPORT_FIG_DIR, "fig2_sparsity_crossover.pdf")
    png_path = os.path.join(REPORT_FIG_DIR, "fig2_sparsity_crossover.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"[GENERATED] Figure 2: {pdf_path}")

def plot_fillin_and_memory(sparse_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))

    sub_1024 = sparse_df[sparse_df["n"] == 1024].sort_values("target_sparsity")
    if sub_1024.empty:
        sub_1024 = sparse_df[sparse_df["n"] == sparse_df["n"].max()].sort_values("target_sparsity")

    # Left: Fill-in ratio
    ax1.plot(sub_1024["target_sparsity"] * 100.0, sub_1024["fill_in_ratio"], "ro-", label="Fill-in Factor")
    ax1.set_xlabel("Input Sparsity (% zeros)")
    ax1.set_ylabel("Fill-in Ratio (NNZ_out / NNZ_in)")
    ax1.set_title("Non-Zero Fill-in Expansion")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # Right: Storage Reduction Ratio
    ax2.plot(sub_1024["target_sparsity"] * 100.0, sub_1024["memory_reduction_ratio"] * 100.0, "b^-", label="CSR Savings (%)")
    ax2.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, label="Storage Parity (0%)")
    ax2.axvline(x=33.3, color="gray", linestyle=":", alpha=0.8, label="Theory Parity (33.3%)")
    ax2.set_xlabel("Input Sparsity (% zeros)")
    ax2.set_ylabel("Memory Reduction vs Dense (%)")
    ax2.set_title("CSR Structural Memory Savings")
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    pdf_path = os.path.join(REPORT_FIG_DIR, "fig3_fillin_and_memory.pdf")
    png_path = os.path.join(REPORT_FIG_DIR, "fig3_fillin_and_memory.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"[GENERATED] Figure 3: {pdf_path}")

def generate_typst_tables(dense_df: pd.DataFrame, sparse_df: pd.DataFrame):
    out_path = os.path.join(REPORT_DIR, "tables_autogen.typ")
    with open(out_path, "w") as f:
        f.write("// Autogenerated Typst measurement tables for Assignment 2\n\n")

        # Table 1: Dense Performance
        f.write("#let table_dense_scaling = table(\n")
        f.write("  columns: (auto, auto, auto, auto, auto, auto, auto),\n")
        f.write("  stroke: (x, y) => if y == 0 { (bottom: 1.2pt) } else if y == 1 { (bottom: 0.8pt) } else { none },\n")
        f.write("  align: (center, center, center, center, center, center, center),\n")
        f.write("  table.header([*Dimension $n$*], [*Baseline Time (s)*], [*Reordered Time (s)*], [*Blocked Time (s)*], [*Speedup (Reord)*], [*Speedup (Block)*], [*Peak RSS (MB)*]),\n")

        for n in sorted(dense_df["n"].unique()):
            sub = dense_df[dense_df["n"] == n]
            t_base = sub[sub["method"] == "baseline"]["time_median_sec"].values[0]
            t_reord = sub[sub["method"] == "reordered"]["time_median_sec"].values[0]
            t_block = sub[sub["method"] == "blocked"]["time_median_sec"].values[0]
            sp_reord = sub[sub["method"] == "reordered"]["speedup_vs_baseline"].values[0]
            sp_block = sub[sub["method"] == "blocked"]["speedup_vs_baseline"].values[0]
            rss = sub["peak_rss_mb"].max()

            f.write(f"  [{n}], [{t_base:.4f}], [{t_reord:.4f}], [{t_block:.4f}], [{sp_reord:.2f}x], [{sp_block:.2f}x], [{rss:.2f}],\n")
        f.write(")\n\n")

        # Table 2: Sparse Crossover & Fill-in Table (for n=1024 or representative)
        rep_n = 1024 if 1024 in sparse_df["n"].unique() else sparse_df["n"].max()
        sub_sparse = sparse_df[sparse_df["n"] == rep_n].sort_values("target_sparsity")

        f.write("#let table_sparse_scaling = table(\n")
        f.write("  columns: (auto, auto, auto, auto, auto, auto, auto),\n")
        f.write("  stroke: (x, y) => if y == 0 { (bottom: 1.2pt) } else if y == 1 { (bottom: 0.8pt) } else { none },\n")
        f.write("  align: (center, center, center, center, center, center, center),\n")
        f.write(f"  table.header([*Sparsity $rho$*], [*Input NNZ*], [*Output NNZ*], [*Fill-in*], [*CSR Kernel Time*], [*Speedup vs Dense*], [*Memory Sav.*]),\n")

        for _, row in sub_sparse.iterrows():
            rho_pct = f"{row['target_sparsity']*100:.1f}%"
            nnz_in = int(row['nnz_a'] + row['nnz_b'])
            nnz_out = int(row['nnz_c'])
            fill = f"{row['fill_in_ratio']:.2f}x"
            t_csr = f"{row['kernel_time_median_sec']:.4f} s"
            sp = f"{row['speedup_kernel_vs_dense']:.2f}x"
            mem_sav = f"{row['memory_reduction_ratio']*100:.1f}%"
            f.write(f"  [{rho_pct}], [{nnz_in}], [{nnz_out}], [{fill}], [{t_csr}], [{sp}], [{mem_sav}],\n")
        f.write(")\n")

    print(f"[GENERATED] Typst measurement tables: {out_path}")

def main():
    dense_path = os.path.join(BENCH_DIR, "summary_dense_benchmarks.csv")
    sparse_path = os.path.join(BENCH_DIR, "summary_sparse_benchmarks.csv")

    if not os.path.exists(dense_path) or not os.path.exists(sparse_path):
        print(f"Error: Missing benchmark files ({dense_path} or {sparse_path})", file=sys.stderr)
        sys.exit(1)

    dense_df = pd.read_csv(dense_path)
    sparse_df = pd.read_csv(sparse_path)

    plot_dense_scaling(dense_df)
    plot_sparsity_crossover(sparse_df)
    plot_fillin_and_memory(sparse_df)
    generate_typst_tables(dense_df, sparse_df)
    print("\n[SUCCESS] Analysis and visualization artifacts generated successfully.")

if __name__ == "__main__":
    main()
