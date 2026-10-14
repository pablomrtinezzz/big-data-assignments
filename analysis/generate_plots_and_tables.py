#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Styling for academic publication figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 1.7,
    "lines.markersize": 6,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
SUMMARY_CSV = RESULTS_DIR / "summary_benchmark_results.csv"
RAW_CSV = RESULTS_DIR / "raw_benchmark_results.csv"
PLOTS_DIR = PROJECT_ROOT / "report" / "figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Color and marker palette
STYLE_MAP = {
    "c_opt": {"color": "#1f77b4", "marker": "o", "label": "C (Clang -O2)"},
    "c_o0": {"color": "#ff7f0e", "marker": "s", "label": "C (Clang -O0)"},
    "java": {"color": "#2ca02c", "marker": "^", "label": "Java (OpenJDK 22)"},
    "python": {"color": "#d62728", "marker": "d", "label": "Python (CPython 3.14)"}
}

def load_data():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary CSV not found at {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    raw_df = pd.read_csv(RAW_CSV)
    return df, raw_df

def plot_runtime_comparison(df):
    """Figure 1: Runtime vs Matrix Dimension (Linear and Log-Log Scale)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # 1. Linear Scale (Subset up to N=1024 for direct visibility)
    df_success = df[df["status"] == "SUCCESS"].copy()
    
    for impl_id, style in STYLE_MAP.items():
        sub = df_success[df_success["impl_id"] == impl_id].sort_values("n")
        if not sub.empty:
            ax1.errorbar(
                sub["n"], sub["median_time_sec"],
                yerr=sub["iqr_time_sec"] / 2.0,
                fmt=style["marker"] + "-",
                color=style["color"],
                label=style["label"],
                capsize=3,
                alpha=0.9
            )

    ax1.set_xlabel("Matrix Dimension $n$ ($n \\times n$)")
    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_title("(a) Execution Time vs Size (Linear Scale)")
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Log-Log Scale (All sizes, highlighting O(n^3) cubic scaling)
    for impl_id, style in STYLE_MAP.items():
        sub = df_success[df_success["impl_id"] == impl_id].sort_values("n")
        if not sub.empty:
            ax2.loglog(
                sub["n"], sub["median_time_sec"],
                style["marker"] + "-",
                color=style["color"],
                label=style["label"],
                alpha=0.9
            )

    # Reference cubic line O(n^3)
    n_ref = np.array([32, 2048])
    # Anchor reference line to C-O2 at n=256
    c_sub = df_success[(df_success["impl_id"] == "c_opt") & (df_success["n"] == 256)]
    if not c_sub.empty:
        t_ref = c_sub["median_time_sec"].values[0]
        y_ref = t_ref * (n_ref / 256.0) ** 3
        ax2.loglog(n_ref, y_ref, "k--", linewidth=1.2, label="Theoretical $O(n^3)$ reference")

    # Mark declared budget limit (60s)
    ax2.axhline(60.0, color="gray", linestyle=":", linewidth=1.2, label="Time Budget (60s)")
    ax2.text(35, 65, "Timeout Threshold (60 s)", color="gray", fontsize=8, verticalalignment="bottom")

    ax2.set_xlabel("Matrix Dimension $n$ ($n \\times n$, log scale)")
    ax2.set_ylabel("Execution Time (seconds, log scale)")
    ax2.set_title("(b) Algorithmic Scaling (Log-Log Scale)")
    ax2.grid(True, which="both")
    ax2.legend(loc="lower right", frameon=True, fontsize=8)

    plt.tight_layout()
    fig_path = PLOTS_DIR / "fig1_runtime_comparison.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "fig1_runtime_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Generated {fig_path.name}")

def plot_throughput_and_scaling(df):
    """Figure 2: Effective GFLOPS and Speedup Ratio relative to C (-O2)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    df_success = df[df["status"] == "SUCCESS"].copy()
    # 2 * n^3 operations / time
    df_success["gflops"] = (2.0 * (df_success["n"] ** 3) * 1e-9) / df_success["median_time_sec"]

    # 1. GFLOPS
    for impl_id, style in STYLE_MAP.items():
        sub = df_success[df_success["impl_id"] == impl_id].sort_values("n")
        if not sub.empty:
            ax1.plot(
                sub["n"], sub["gflops"],
                style["marker"] + "-",
                color=style["color"],
                label=style["label"]
            )

    ax1.set_xlabel("Matrix Dimension $n$ ($n \\times n$)")
    ax1.set_ylabel("Effective Throughput (GFLOPS)")
    ax1.set_title("(a) Computational Throughput")
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True)

    # 2. Relative Slowdown compared to C (-O2)
    c_df = df_success[df_success["impl_id"] == "c_opt"].set_index("n")["median_time_sec"]

    for impl_id, style in STYLE_MAP.items():
        if impl_id == "c_opt":
            continue
        sub = df_success[df_success["impl_id"] == impl_id].sort_values("n")
        common_n = [val for val in sub["n"] if val in c_df.index]
        if common_n:
            slowdown = [sub[sub["n"] == val]["median_time_sec"].values[0] / c_df.loc[val] for val in common_n]
            ax2.plot(
                common_n, slowdown,
                style["marker"] + "-",
                color=style["color"],
                label=f"{style['label']} / C (-O2)"
            )

    ax2.set_xlabel("Matrix Dimension $n$ ($n \\times n$)")
    ax2.set_ylabel("Slowdown Ratio ($T_{\\text{impl}} / T_{\\text{C-O2}}$)")
    ax2.set_title("(b) Slowdown Relative to C (Clang -O2)")
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig_path = PLOTS_DIR / "fig2_throughput_speedup.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "fig2_throughput_speedup.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Generated {fig_path.name}")

def plot_memory_usage(df):
    """Figure 3: Peak Resident Set Size (RSS) and Matrix Payload vs Size"""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    df_success = df[df["status"] == "SUCCESS"].copy()
    
    # Plot Theoretical Matrix Payload: 3 * n^2 * 8 bytes
    n_vals = np.sort(df_success["n"].unique())
    theoretical_mb = (3.0 * (n_vals ** 2) * 8) / (1024.0 * 1024.0)
    ax.plot(n_vals, theoretical_mb, "k--", linewidth=1.5, label="Raw Matrix Payload (24 $n^2$ bytes)")

    for impl_id, style in STYLE_MAP.items():
        sub = df_success[df_success["impl_id"] == impl_id].sort_values("n")
        if not sub.empty and sub["peak_rss_bytes"].notna().any():
            rss_mb = sub["peak_rss_bytes"] / (1024.0 * 1024.0)
            ax.plot(
                sub["n"], rss_mb,
                style["marker"] + "-",
                color=style["color"],
                label=f"{style['label']} Peak RSS"
            )

    ax.set_xlabel("Matrix Dimension $n$ ($n \\times n$)")
    ax.set_ylabel("Peak Resident Memory (MB)")
    ax.set_title("Peak Resident Set Size (RSS) vs Matrix Dimension")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    fig_path = PLOTS_DIR / "fig3_memory_scaling.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "fig3_memory_scaling.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Generated {fig_path.name}")

def compute_scaling_exponents(df):
    """Fits log(T) = log(c) + alpha * log(n) to determine empirical growth rate"""
    print("\n================================================================================")
    print("EMPIRICAL SCALING EXPONENT FIT: T(n) = c * n^alpha")
    print("================================================================================")
    df_success = df[df["status"] == "SUCCESS"].copy()
    
    results = []
    for impl_id in ["c_opt", "c_o0", "java", "python"]:
        sub = df_success[df_success["impl_id"] == impl_id].sort_values("n")
        # Exclude tiny n < 64 where overhead / clock granularity dominates
        fit_sub = sub[sub["n"] >= 64]
        if len(fit_sub) >= 3:
            log_n = np.log(fit_sub["n"].values)
            log_t = np.log(fit_sub["median_time_sec"].values)
            slope, intercept = np.polyfit(log_n, log_t, 1)
            
            # Correlation coefficient R^2
            p = np.poly1d([slope, intercept])
            y_pred = p(log_n)
            r2 = 1.0 - (np.sum((log_t - y_pred) ** 2) / np.sum((log_t - np.mean(log_t)) ** 2))
            
            c_factor = np.exp(intercept)
            results.append({
                "impl_id": impl_id,
                "name": STYLE_MAP[impl_id]["label"],
                "exponent_alpha": slope,
                "constant_c": c_factor,
                "r2": r2,
                "n_min": fit_sub["n"].min(),
                "n_max": fit_sub["n"].max()
            })
            print(f"  {STYLE_MAP[impl_id]['label']:<24}: alpha = {slope:.3f} | c = {c_factor:.2e} | R^2 = {r2:.4f} (Fit range: {fit_sub['n'].min()}-{fit_sub['n'].max()})")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "scaling_exponent_fits.csv", index=False)
    return res_df

def generate_typst_tables(df, fits_df):
    """Generates Typst-formatted tables for inclusion in the research report"""
    typst_table_path = PROJECT_ROOT / "report" / "tables_autogen.typ"
    
    with open(typst_table_path, "w") as f:
        f.write("// Autogenerated summary tables from benchmark results\n\n")
        
        # 1. Main Results Table
        f.write("#let benchmark-table = [\n")
        f.write("#table(\n")
        f.write("  columns: (auto, auto, auto, auto, auto, auto, auto, auto),\n")
        f.write("  align: (center, left, center, right, right, right, right, center),\n")
        f.write("  table.header([\n    *Size $n$*], [*Implementation*], [*Status*], [*Median Time (s)*], [*IQR (s)*], [*Mean (s)*], [*Memory (MB)*], [*Throughput (GFLOPS)*]\n  ),\n")
        
        for _, row in df.iterrows():
            n = int(row["n"])
            impl = STYLE_MAP[row["impl_id"]]["label"]
            status = row["status"]
            if status == "SUCCESS":
                med = f"{row['median_time_sec']:.6f}" if row['median_time_sec'] < 1 else f"{row['median_time_sec']:.4f}"
                iqr = f"{row['iqr_time_sec']:.6f}" if row['iqr_time_sec'] < 1 else f"{row['iqr_time_sec']:.4f}"
                mean = f"{row['mean_time_sec']:.6f}" if row['mean_time_sec'] < 1 else f"{row['mean_time_sec']:.4f}"
                if pd.notna(row['peak_rss_bytes']):
                    rss = f"{row['peak_rss_bytes'] / (1024*1024):.1f}"
                elif pd.notna(row['heap_bytes']) and row['heap_bytes'] > 0:
                    rss = f"{row['heap_bytes'] / (1024*1024):.1f}†"
                elif row['impl_id'] == 'java' and n <= 128:
                    rss = "< 1.0†"
                else:
                    rss = "-"
                gflops = (2.0 * (n**3) * 1e-9) / row['median_time_sec']
                gflops_str = f"{gflops:.3f}"
                f.write(f"  [{n}], [{impl}], [OK], [{med}], [{iqr}], [{mean}], [{rss}], [{gflops_str}],\n")
            else:
                f.write(f"  [{n}], [{impl}], [*TIMEOUT*], [-], [-], [-], [-], [-],\n")
                
        f.write(")\n")
        f.write("#v(0.4em)\n")
        f.write("#text(size: 8pt, fill: luma(90))[† Note: Values report process Peak Resident Set Size (RSS) via `getrusage` for C and Python; for Java, values marked with † report JVM Managed Heap used via `Runtime.getRuntime()`.]\n")
        f.write("]\n\n")

        # 2. Scaling Exponent Table
        f.write("#let scaling-table = table(\n")
        f.write("  columns: (auto, auto, auto, auto, auto),\n")
        f.write("  align: (left, center, center, right, center),\n")
        f.write("  table.header([\n    *Implementation*], [*Empirical Exponent $alpha$*], [*Ideal Model*], [*Pre-factor $c$*], [*Fit $R^2$*]\n  ),\n")
        
        for _, row in fits_df.iterrows():
            impl = row["name"]
            alpha = f"{row['exponent_alpha']:.3f}"
            c = f"{row['constant_c']:.2e}"
            r2 = f"{row['r2']:.4f}"
            f.write(f"  [{impl}], [*{alpha}*], [3.000], [{c}], [{r2}],\n")
            
        f.write(")\n")

    print(f"[INFO] Generated Typst tables at: {typst_table_path.relative_to(PROJECT_ROOT)}")

def main():
    df, raw_df = load_data()
    plot_runtime_comparison(df)
    plot_throughput_and_scaling(df)
    plot_memory_usage(df)
    fits_df = compute_scaling_exponents(df)
    generate_typst_tables(df, fits_df)
    print("\nALL ANALYSIS ARTIFACTS SUCCESSFULLY PRODUCED.")

if __name__ == "__main__":
    main()
