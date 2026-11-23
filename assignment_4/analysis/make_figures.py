"""
Summaries, figures and Typst tables from the raw CSVs.

Inputs : benchmarks/results/raw_baselines.csv, raw_spark.csv, sessions.json
Outputs: benchmarks/results/summary_*.csv, report/figures/*.png|pdf,
         report/tables_autogen.typ
Statistic: median over measured repetitions; variability: IQR (Q3-Q1).
Speedup S = T_seq_median / T_median, efficiency E = S / cores.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "benchmarks", "results")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

# validated categorical order (blue, orange, aqua, yellow) + neutral ink
C_BCAST, C_2D, C_OMP, C_SEQ = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
INK, GRID = "#0b0b0b", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": INK, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 300,
})
NMAX = 3072
NCLEAN = 2048
SIZES = [512, 1024, 2048, 3072]
STYLE = {"bcast": dict(color=C_BCAST, marker="o", label="Spark row-panel + broadcast"),
         "block2d": dict(color=C_2D, marker="s", label="Spark 2-D block (shuffle)"),
         "omp": dict(color=C_OMP, marker="^", label="OpenMP (A3), p threads"),
         "seq": dict(color=C_SEQ, ls="--", label="Sequential C (A3)")}


def q(x, p):
    return float(np.percentile(x, p))


def summarise(df, keys, col):
    g = df.groupby(keys)[col]
    out = g.agg(median="median", q1=lambda x: q(x, 25), q3=lambda x: q(x, 75), reps="count").reset_index()
    out["iqr"] = out.q3 - out.q1
    return out


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def main():
    base = pd.read_csv(os.path.join(RES, "raw_baselines.csv"))
    sp = pd.read_csv(os.path.join(RES, "raw_spark.csv"))
    sessions = json.load(open(os.path.join(RES, "sessions.json")))
    assert sp["valid"].all(), "invalid distributed result present"

    # ---------------- summaries ----------------
    bs = summarise(base, ["method", "n", "threads"], "time_s")
    rss_b = base.groupby(["method", "n", "threads"]).peak_rss_bytes.max().reset_index()
    bs = bs.merge(rss_b, on=["method", "n", "threads"])
    tseq = bs[bs.method == "seq_ikj"].set_index("n")["median"]
    bs["speedup"] = bs.apply(lambda r: tseq.get(r.n, np.nan) / r["median"], axis=1)
    bs["efficiency"] = bs.speedup / bs.threads
    bs.to_csv(os.path.join(RES, "summary_baselines.csv"), index=False)

    keys = ["experiment", "strategy", "n", "workers", "cores_per_worker", "total_cores"]
    ss = summarise(sp, keys, "e2e")
    extra = sp.groupby(keys).agg(
        job=("job", "median"), distribute=("distribute", "median"), assemble=("assemble", "median"),
        kernel_sum=("kernel_sum", "median"), shuffle_bytes=("shuffleWriteBytes", "median"),
        result_bytes=("resultSize", "median"), exec_rss_max=("exec_peak_rss_max", "max"),
        exec_rss_mean=("exec_peak_rss_mean", "median"), driver_rss=("driver_peak_rss", "max"),
        run_time_ms=("executorRunTime", "median"), deser_ms=("executorDeserializeTime", "median"),
        gc_ms=("jvmGcTime", "median"), tasks=("tasks", "median"),
        pageouts=("pageouts", "median"), swapouts=("swapouts", "median"),
        max_err=("max_abs_err", "max")).reset_index()
    ss = ss.merge(extra, on=keys)
    ss["speedup"] = ss.apply(lambda r: tseq.get(r.n, np.nan) / r["median"], axis=1)
    ss["efficiency"] = ss.speedup / ss.total_cores
    ss["gflops"] = 2 * ss.n.astype(float) ** 3 / ss["median"] / 1e9
    # speedup relative to the same distributed code on 1 worker (framework-internal scaling)
    one = ss[(ss.workers == 1) & (ss.cores_per_worker == 1)].set_index(["strategy", "n"])["median"]
    ss["self_speedup"] = ss.apply(lambda r: one.get((r.strategy, r.n), np.nan) / r["median"], axis=1)
    # paging flag: median system-wide swap-out during the run > 64 MiB (16 KiB pages)
    ss["paging"] = ss.swapouts * 16384 > 64 * 2**20
    ss.to_csv(os.path.join(RES, "summary_spark.csv"), index=False)

    def mark_paging(ax, x, y, flag, color):
        x, y, flag = np.asarray(x), np.asarray(y), np.asarray(flag, dtype=bool)
        if flag.any():
            ax.scatter(x[flag], y[flag], s=70, facecolors="white", edgecolors=color,
                       linewidths=1.6, zorder=4)

    strong = ss[ss.experiment == "strong"]
    W = [1, 2, 4, 8]

    # ---------------- Fig 1: strong scaling time + speedup ----------------
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, n in zip(axs[:2], (2048, NMAX)):
        for s in ("bcast", "block2d"):
            d = strong[(strong.strategy == s) & (strong.n == n)].sort_values("workers")
            ax.errorbar(d.workers, d["median"], yerr=[d["median"] - d.q1, d.q3 - d["median"]],
                        capsize=2, **STYLE[s])
            mark_paging(ax, d.workers, d["median"], d.paging, STYLE[s]["color"])
        o = bs[(bs.method == "omp_ikj") & (bs.n == n)].sort_values("threads")
        ax.plot(o.threads, o["median"], **STYLE["omp"])
        ax.axhline(tseq[n], **STYLE["seq"])
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(W); ax.set_xticklabels(W)
        ax.set_title(f"End-to-end time, n = {n}", fontsize=8.5)
        ax.set_xlabel("W (Spark) / p (OpenMP)")
        ax.set_ylabel("median time (s)")
    ax = axs[2]
    n = NCLEAN
    for s in ("bcast", "block2d"):
        d = strong[(strong.strategy == s) & (strong.n == n)].sort_values("workers")
        ax.plot(d.workers, d.speedup, **STYLE[s])
        mark_paging(ax, d.workers, d.speedup, d.paging, STYLE[s]["color"])
    o = bs[(bs.method == "omp_ikj") & (bs.n == n)].sort_values("threads")
    ax.plot(o.threads, o.speedup, **STYLE["omp"])
    ax.plot(W, W, color="#b5b4ae", lw=1, ls=":", label="ideal S = W")
    ax.set_ylim(0, 2.6)
    ax.set_xticks(W); ax.set_title(f"Speedup vs. seq. C, n = {NCLEAN}", fontsize=8.5)
    ax.set_xlabel("workers / threads"); ax.set_ylabel("S = T_seq / T")
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    hp = Line2D([], [], marker="o", ls="none", mfc="white", mec="#52514e", mew=1.6, ms=7)
    fig.legend(h + [h2[-1], hp], l + [l2[-1], "host paging (swap-out > 64 MiB)"], loc="lower center", ncol=3, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(FIG, "fig1_strong_scaling.png")); fig.savefig(os.path.join(FIG, "fig1_strong_scaling.pdf"))
    plt.close(fig)

    # ---------------- Fig 2: where the time goes (n = NCLEAN) ----------------
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, s in zip(axs, ("bcast", "block2d")):
        d = strong[(strong.strategy == s) & (strong.n == NCLEAN)].sort_values("workers")
        comp = d.kernel_sum / d.workers                 # ideal critical-path compute
        other = (d.job - comp).clip(lower=0)            # scheduling, (de)serialisation, transfer, imbalance
        x = np.arange(len(d))
        parts = [("driver partition + serialise", d.distribute, "#86b6ef"),
                 ("kernel compute (sum/W)", comp, C_BCAST if s == "bcast" else C_2D),
                 ("framework + transfer + imbalance", other, "#b5b4ae"),
                 ("driver assembly", d.assemble, "#52514e")]
        bottom = np.zeros(len(d))
        for lab, v, col in parts:
            ax.bar(x, v.values, bottom=bottom, color=col, width=0.6, label=lab,
                   edgecolor="white", linewidth=1)
            bottom += v.values
        ax.set_xticks(x); ax.set_xticklabels([f"W={w}" + (" *" if pg else "") for w, pg in zip(d.workers, d.paging)])
        ax.set_title(STYLE[s]["label"] + f", n = {NCLEAN}", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("median time (s)")
    h, l = axs[0].get_legend_handles_labels()
    h2, _ = axs[1].get_legend_handles_labels()
    fig.legend([h[0], h[1], h2[1], h[2], h[3]],
               [l[0], "kernel compute Σ/W (bcast)", "kernel compute Σ/W (block2d)", l[2], l[3]],
               loc="lower center", ncol=3, fontsize=7.2)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(FIG, "fig2_time_breakdown.png")); fig.savefig(os.path.join(FIG, "fig2_time_breakdown.pdf"))
    plt.close(fig)

    # ---------------- Fig 3: communication + memory ----------------
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axs[0]
    for s in ("bcast", "block2d"):
        d = strong[(strong.strategy == s) & (strong.workers == 4)].sort_values("n")
        if s == "bcast":   # B is broadcast once per executor: W * 8 n^2 bytes (analytical)
            vol = d.workers * 8 * d.n.astype(float) ** 2
            lab = "broadcast B (W·8n², analytical)"
        else:
            vol = d.shuffle_bytes.astype(float)
            lab = "shuffle write (Spark metrics)"
        ax.plot(d.n, vol / 2**20, **{**STYLE[s], "label": f"{s}: {lab}"})
    d = strong[(strong.strategy == "bcast") & (strong.workers == 4)].sort_values("n")
    ax.plot(d.n, 3 * 8 * d.n.astype(float) ** 2 / 2**20, color=C_SEQ, ls=":", lw=1.2,
            label="3 dense matrices, 24n² (reference)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(SIZES); ax.set_xticklabels(SIZES, fontsize=6.8, rotation=30)
    ax.set_xlabel("n"); ax.set_ylabel("MiB moved per multiplication")
    ax.set_title("Data movement, W = 4", fontsize=8.5); ax.legend(fontsize=6.5)
    ax = axs[1]
    for s in ("bcast", "block2d"):
        d = strong[(strong.strategy == s) & (strong.n == NMAX)].sort_values("workers")
        ax.plot(d.workers, d.exec_rss_max / 2**20, **{**STYLE[s], "label": f"{s}: peak RSS per executor"})
        ax.plot(d.workers, d.workers * d.exec_rss_mean / 2**20, **{**STYLE[s], "ls": ":",
                "label": f"{s}: sum over executors"})
    o = bs[(bs.method == "omp_ikj") & (bs.n == NMAX)].sort_values("threads")
    ax.plot(o.threads, o.peak_rss_bytes / 2**20, **{**STYLE["omp"], "label": "OpenMP process peak RSS"})
    ax.set_xticks(W); ax.set_xlabel("workers / threads"); ax.set_ylabel("MiB")
    ax.set_title(f"Peak resident memory, n = {NMAX}", fontsize=8.5); ax.legend(fontsize=6.2)
    save(fig, "fig3_communication_memory")

    # ---------------- Fig 4: size crossover + weak scaling + topology ----------------
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axs[0]
    for s in ("bcast", "block2d"):
        d = strong[(strong.strategy == s) & (strong.workers == 8)].sort_values("n")
        ax.plot(d.n, d["median"], **{**STYLE[s], "label": f"{s}, W=8"})
    o = bs[(bs.method == "omp_ikj") & (bs.threads == 8) & bs.n.isin(SIZES)].sort_values("n")
    ax.plot(o.n, o["median"], **{**STYLE["omp"], "label": "OpenMP p=8"})
    sq = bs[(bs.method == "seq_ikj") & bs.n.isin(SIZES)].sort_values("n")
    ax.plot(sq.n, sq["median"], color=C_SEQ, ls="--", marker="D", label="sequential C")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(SIZES); ax.set_xticklabels(SIZES, fontsize=6.8, rotation=30)
    ax.set_xlabel("n"); ax.set_ylabel("median time (s)")
    ax.set_title("Time vs. size", fontsize=8.5); ax.legend(fontsize=6.3)

    ax = axs[1]
    for s in ("bcast", "block2d"):
        d = ss[(ss.strategy == s) & (ss.cores_per_worker == 1) &
               (((ss.workers == 1) & (ss.n == 1536)) | ((ss.workers == 2) & (ss.n == 1920)) |
                ((ss.workers == 4) & (ss.n == 2432)) | ((ss.workers == 8) & (ss.n == 3072)))]
        d = d.drop_duplicates(["workers"]).sort_values("workers")
        work = d.n.astype(float) ** 3
        thr = work / d["median"] / d.workers
        eff = thr / thr.iloc[0]
        ax.plot(d.workers, eff, **STYLE[s])
        mark_paging(ax, d.workers, eff, d.paging, STYLE[s]["color"])
        if s == "bcast":
            ss_weak = d.assign(weak_eff=eff.values)
    ax.axhline(1, color="#b5b4ae", lw=1, ls=":")
    ax.set_xticks(W); ax.set_ylim(0, 1.15)
    ax.set_xlabel("workers W (n³ ∝ W)"); ax.set_ylabel("weak-scaling efficiency")
    ax.set_title("Weak scaling", fontsize=8.5)

    ax = axs[2]
    topo = ss[(ss.n == 2048) & (ss.total_cores == 4)].copy()
    topo["lab"] = topo.workers.astype(str) + "×" + topo.cores_per_worker.astype(str)
    order = ["1×4", "2×2", "4×1"]
    x = np.arange(len(order))
    for k, s in enumerate(("bcast", "block2d")):
        d = topo[topo.strategy == s].set_index("lab").reindex(order)
        ax.bar(x + (k - 0.5) * 0.36, d["median"], width=0.34, color=STYLE[s]["color"],
               yerr=d.iqr / 2, capsize=2, label=s, edgecolor="white", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7.5)
    ax.set_xlabel("executors × cores per executor")
    ax.set_ylim(0, 4.2)
    ax.set_ylabel("median time (s)"); ax.grid(axis="x", visible=False)
    ax.set_title("4 cores, n = 2048: layout", fontsize=8.5); ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    save(fig, "fig4_size_weak_topology")

    # ---------------- Typst tables ----------------
    def fmt(v, d=3):
        return f"{v:.{d}f}"

    lines = ["// Auto-generated by analysis/make_figures.py -- do not edit\n"]
    # Table: strong scaling n = 2048 and 4096
    rows = []
    for n in (1024, 2048, NMAX):
        r = bs[(bs.method == "seq_ikj") & (bs.n == n)].iloc[0]
        rows.append((str(n), "Sequential C", "1", fmt(r["median"]), fmt(r.iqr), "1.00", "--",
                     f"{r.peak_rss_bytes/2**20:.0f}"))
        for p in W:
            r = bs[(bs.method == "omp_ikj") & (bs.n == n) & (bs.threads == p)].iloc[0]
            rows.append(("", "OpenMP", str(p), fmt(r["median"]), fmt(r.iqr), fmt(r.speedup, 2),
                         fmt(r.efficiency, 2), f"{r.peak_rss_bytes/2**20:.0f}"))
        for s, name in (("bcast", "Spark bcast"), ("block2d", "Spark 2-D")):
            for w in W:
                r = strong[(strong.strategy == s) & (strong.n == n) & (strong.workers == w)].iloc[0]
                rows.append(("", name, str(w) + (" †" if r.paging else ""), fmt(r["median"]), fmt(r.iqr), fmt(r.speedup, 2),
                             fmt(r.efficiency, 2), f"{r.exec_rss_max/2**20:.0f}"))
    cells = ",\n    ".join(", ".join(f"[{c}]" for c in row) for row in rows)
    lines.append(f"""#let table_strong = table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto), column-gutter: 10pt,
  align: (center, left, center, right, right, right, right, right),
  stroke: none, inset: (x: 3pt, y: 1.6pt),
  table.hline(stroke: 0.8pt),
  table.header([*n*], [*Method*], [*p/W*], [*Median (s)*], [*IQR (s)*], [*S*], [*E*], [*RSS (MiB)*]),
  table.hline(stroke: 0.5pt),
    {cells},
  table.hline(stroke: 0.8pt),
)
""")
    # Table: overhead decomposition at 4096 and session start-up
    rows = []
    for s in ("bcast", "block2d"):
      for nn in (NCLEAN, NMAX):
        for w in W:
            sel = strong[(strong.strategy == s) & (strong.n == nn) & (strong.workers == w)]
            if sel.empty:
                continue
            r = sel.iloc[0]
            rows.append((s if (w == 1 and nn == NCLEAN) else "", str(nn) if w == 1 else "",
                         str(w) + (" †" if r.paging else ""), fmt(r.distribute), fmt(r.job), fmt(r.kernel_sum / w),
                         fmt(r.assemble), f"{r.shuffle_bytes/2**20:.0f}", f"{r.result_bytes/2**20:.0f}",
                         f"{r.gc_ms/1000:.2f}", f"{r.swapouts*16/1024:.0f}"))
    cells = ",\n    ".join(", ".join(f"[{c}]" for c in row) for row in rows)
    lines.append(f"""#let table_breakdown = table(
  columns: (auto, auto, auto, 1fr, 1fr, 1fr, 1fr, auto, auto, auto, auto),
  align: (left, center, center, right, right, right, right, right, right, right, right),
  stroke: none, inset: (x: 3pt, y: 1.6pt),
  table.hline(stroke: 0.8pt),
  table.header([*Strategy*], [*n*], [*W*], [*Distr. (s)*], [*Job (s)*], [*Σkernel/W (s)*], [*Assem. (s)*],
               [*Shuffle (MiB)*], [*Result (MiB)*], [*GC (s)*], [*Swap-out (MiB)*]),
  table.hline(stroke: 0.5pt),
    {cells},
  table.hline(stroke: 0.8pt),
)
""")
    st = pd.DataFrame(sessions)
    rows = [(r.config, str(r.executors_registered), fmt(r.startup_s, 2)) for r in st.itertuples()]
    cells = ",\n    ".join(", ".join(f"[{c}]" for c in row) for row in rows)
    lines.append(f"""#let table_startup = table(
  columns: (auto, auto, auto),
  align: (center, center, right),
  stroke: none, inset: (x: 4pt, y: 1.6pt),
  table.hline(stroke: 0.8pt),
  table.header([*Executors×cores*], [*Registered*], [*Start-up (s)*]),
  table.hline(stroke: 0.5pt),
    {cells},
  table.hline(stroke: 0.8pt),
)
""")
    att = pd.read_csv(os.path.join(RES, "raw_spark_n4096_attempt.csv"))
    a4 = summarise(att, ["strategy", "workers"], "e2e").merge(
        att.groupby(["strategy", "workers"]).agg(k=("kernel_sum", "median"),
                                                  rss=("exec_peak_rss_max", "max")).reset_index())
    t4 = bs[(bs.n == 4096)]
    rows = [("Sequential C", "1", fmt(t4[t4.method == "seq_ikj"]["median"].iloc[0], 2), "--", "--")]
    rows += [(f"OpenMP p={int(r.threads)}", "1 process", fmt(r.median, 2), fmt(r.iqr, 2), "--")
             for r in t4[t4.method == "omp_ikj"].itertuples()]
    rows += [(f"Spark {r.strategy}", f"W={r.workers}", fmt(r.median, 2), fmt(r.iqr, 2), fmt(r.k, 1))
             for r in a4.sort_values(["strategy", "workers"]).itertuples()]
    cells = ",\n    ".join(", ".join(f"[{c}]" for c in row) for row in rows)
    lines.append(f"""#let table_attempt = table(
  columns: (auto, auto, 1fr, 1fr, 1fr),
  align: (left, center, right, right, right),
  stroke: none, inset: (x: 3pt, y: 1.6pt),
  table.hline(stroke: 0.8pt),
  table.header([*Method*], [*Config*], [*Median (s)*], [*IQR (s)*], [*Σkernel (s)*]),
  table.hline(stroke: 0.5pt),
    {cells},
  table.hline(stroke: 0.8pt),
)
""")
    with open(os.path.join(ROOT, "report", "tables_autogen.typ"), "w") as f:
        f.write("\n".join(lines))

    # key numbers for the text
    key = {"weak_eff_bcast": ss_weak[["workers", "n", "median", "weak_eff"]].to_dict("records"),
           "max_abs_err": float(sp.max_abs_err.max()), "spark_rows": int(len(sp)),
           "baseline_rows": int(len(base))}
    json.dump(key, open(os.path.join(RES, "key_numbers.json"), "w"), indent=2, default=float)
    print(json.dumps(key, indent=2, default=float))


if __name__ == "__main__":
    main()
