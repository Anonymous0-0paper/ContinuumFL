#!/usr/bin/env python3
"""
compare_noniid_results.py
=========================
Collects every non-IID ContinuumFL experiment saved under ./results/ and
./results/noniid_sweep/, compares them on multiple axes (accuracy, F1,
communication cost, convergence speed) and proposes the best setup.

Outputs
-------
- Terminal table (rich / plain fallback)
- results/noniid_comparison/comparison_table.csv
- results/noniid_comparison/plots/*.png  (5 charts)
- results/noniid_comparison/recommendation.txt
"""

import os
import re
import sys
import json
import csv
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
OUT_DIR     = RESULTS_DIR / "noniid_comparison"
PLOT_DIR    = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# IEEE publication style
# ─────────────────────────────────────────────────────────────
from ieee_plot_style import (IEEE_COLORS as PALETTE, IEEE_HATCHES,
                              IEEE_LINESTYLES, IEEE_MARKERS, setup_ieee_style)
setup_ieee_style()

# ─────────────────────────────────────────────────────────────
# 1.  Load structured results  (femnist__intra*__inter*__comp*)
# ─────────────────────────────────────────────────────────────
FOLDER_RE = re.compile(
    r"(?P<dataset>\w+)__intra(?P<intra>[\d.]+)__inter(?P<inter>[\d.]+)__comp(?P<comp>\d+)pct"
)

records = []
metrics_by_exp: dict[str, pd.DataFrame] = {}

for folder in sorted(RESULTS_DIR.iterdir()):
    m = FOLDER_RE.match(folder.name)
    if not m:
        continue
    summary_path = folder / "summary.csv"
    metrics_path = folder / "metrics.csv"
    if not summary_path.exists():
        continue

    df_s = pd.read_csv(summary_path)
    if df_s.empty:
        continue
    row = df_s.iloc[0]

    intra     = float(m.group("intra"))
    inter     = float(m.group("inter"))
    comp_pct  = int(m.group("comp"))

    rec = {
        "source":              "structured",
        "exp_id":              folder.name,
        "dataset":             str(row.get("dataset", m.group("dataset"))),
        "intra_zone_alpha":    intra,
        "inter_zone_alpha":    inter,
        "compression_rate":    comp_pct / 100.0,
        "num_rounds":          int(row.get("num_rounds", 200)),
        "num_devices":         int(row.get("num_devices", 50)),
        "num_zones":           int(row.get("num_zones", 5)),
        "final_accuracy":      float(row.get("final_accuracy", 0)),
        "best_accuracy":       float(row.get("best_accuracy", 0)),
        "best_accuracy_round": int(row.get("best_accuracy_round", 0)),
        "final_loss":          float(row.get("final_loss", 9999)),
        "final_f1":            float(row.get("final_f1", 0)),
        "final_precision":     float(row.get("final_precision", 0)),
        "final_recall":        float(row.get("final_recall", 0)),
        "total_training_time": float(row.get("total_training_time", 0)),
        "average_round_time":  float(row.get("average_round_time", 0)),
        "total_comm_mb":       float(row.get("total_communication_cost", 0)),
        "convergence_rounds":  int(row.get("convergence_rounds", 0)),
        "total_rounds_run":    int(row.get("num_rounds", 200)),   # placeholder
    }

    # Pull actual rounds run from metrics.csv
    if metrics_path.exists():
        df_m = pd.read_csv(metrics_path).dropna(subset=["global_accuracy"])
        rec["total_rounds_run"] = len(df_m)
        metrics_by_exp[folder.name] = df_m

    records.append(rec)

# ─────────────────────────────────────────────────────────────
# 2.  Load noniid_sweep logs  (plain .log files)
# ─────────────────────────────────────────────────────────────
LOG_DIR = RESULTS_DIR / "noniid_sweep"
ROUND_RE = re.compile(
    r"Round\s+(\d+)\s*\|.*?Acc=([\d.]+)%.*?Loss=([\d.]+).*?P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)"
    r".*?Comm=([\d.]+)MB.*?Time=([\d.]+)s"
)
BEST_RE  = re.compile(r"best=([\d.]+)%@R(\d+)")
ES_RE    = re.compile(r"Early stopping.*no improvement for (\d+)/(\d+)")

if LOG_DIR.exists():
    for log_file in sorted(LOG_DIR.glob("noniid_alpha*.log")):
        alpha_m = re.search(r"noniid_alpha([\d.]+)\.log", log_file.name)
        if not alpha_m:
            continue
        alpha = float(alpha_m.group(1))

        text = log_file.read_text(errors="replace")

        rounds_data = []
        for rm in ROUND_RE.finditer(text):
            r_idx = int(rm.group(1))
            acc   = float(rm.group(2)) / 100.0   # convert % → fraction
            loss  = float(rm.group(3))
            prec  = float(rm.group(4))
            rec_  = float(rm.group(5))
            f1    = float(rm.group(6))
            comm  = float(rm.group(7))
            rtime = float(rm.group(8))
            rounds_data.append({
                "round": r_idx, "global_accuracy": acc, "global_loss": loss,
                "precision": prec, "recall": rec_, "f1": f1,
                "communication_cost_mb": comm, "round_time_s": rtime,
            })

        if not rounds_data:
            continue

        df_log = pd.DataFrame(rounds_data).drop_duplicates(subset="round").sort_values("round")
        exp_id = f"noniid_alpha{alpha}_log"
        metrics_by_exp[exp_id] = df_log

        # best accuracy  
        best_acc, best_rnd = 0.0, 0
        bm = list(BEST_RE.finditer(text))
        if bm:
            last_bm = bm[-1]
            best_acc = float(last_bm.group(1)) / 100.0
            best_rnd = int(last_bm.group(2))
        else:
            best_acc = df_log["global_accuracy"].max()
            best_rnd = int(df_log.loc[df_log["global_accuracy"].idxmax(), "round"])

        # convergence = first round ≥ 95 % of best_acc
        conv_rounds = 0
        threshold = 0.95 * best_acc
        above = df_log[df_log["global_accuracy"] >= threshold]
        if not above.empty:
            conv_rounds = int(above["round"].iloc[0])

        total_rounds = int(df_log["round"].max())
        total_comm   = float(df_log["communication_cost_mb"].sum())
        avg_rtime    = float(df_log["round_time_s"].mean())
        total_time   = float(df_log["round_time_s"].sum())
        final_row    = df_log.iloc[-1]

        rec = {
            "source":              "log",
            "exp_id":              exp_id,
            "dataset":             "femnist",
            "intra_zone_alpha":    100.0,      # fixed in noniid_sweep script
            "inter_zone_alpha":    alpha,
            "compression_rate":    0.1,         # default
            "num_rounds":          200,
            "num_devices":         50,
            "num_zones":           5,
            "final_accuracy":      float(final_row["global_accuracy"]),
            "best_accuracy":       best_acc,
            "best_accuracy_round": best_rnd,
            "final_loss":          float(final_row["global_loss"]),
            "final_f1":            float(final_row["f1"]),
            "final_precision":     float(final_row["precision"]),
            "final_recall":        float(final_row["recall"]),
            "total_training_time": total_time,
            "average_round_time":  avg_rtime,
            "total_comm_mb":       total_comm,
            "convergence_rounds":  conv_rounds,
            "total_rounds_run":    total_rounds,
        }

        # Avoid duplicating if we already have it from structured results
        existing_ids = {r["inter_zone_alpha"] for r in records if r["source"] == "structured"}
        if alpha not in existing_ids:
            records.append(rec)
        else:
            # Merge: prefer structured results but fill missing metrics from log
            for r in records:
                if r["source"] == "structured" and r["inter_zone_alpha"] == alpha:
                    if r["best_accuracy"] == 0:
                        r["best_accuracy"] = best_acc
                    if r["convergence_rounds"] == 0:
                        r["convergence_rounds"] = conv_rounds
                    # Keep the log metrics_by_exp for the richer per-round data
                    metrics_by_exp[r["exp_id"]] = df_log
                    break

# ─────────────────────────────────────────────────────────────
# 3.  Build comparison DataFrame
# ─────────────────────────────────────────────────────────────
if not records:
    print("❌  No experiment results found. Run some experiments first.")
    sys.exit(1)

df = pd.DataFrame(records).sort_values("inter_zone_alpha")

# Normalise metrics to [0,1] for composite scoring
def norm(series, higher_is_better=True):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    n = (series - mn) / (mx - mn)
    return n if higher_is_better else 1 - n

# Weights: best_accuracy 40%, final_f1 20%, convergence_rounds 20%, comm_cost 20%
df["_n_best_acc"]    = norm(df["best_accuracy"],       higher_is_better=True)
df["_n_f1"]          = norm(df["final_f1"],            higher_is_better=True)
df["_n_conv"]        = norm(df["convergence_rounds"],  higher_is_better=False)
df["_n_comm"]        = norm(df["total_comm_mb"],       higher_is_better=False)

df["composite_score"] = (
    0.40 * df["_n_best_acc"] +
    0.20 * df["_n_f1"]       +
    0.20 * df["_n_conv"]     +
    0.20 * df["_n_comm"]
)

df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# ─────────────────────────────────────────────────────────────
# 4.  Save CSV
# ─────────────────────────────────────────────────────────────
EXPORT_COLS = [
    "rank", "exp_id", "inter_zone_alpha", "intra_zone_alpha", "compression_rate",
    "best_accuracy", "final_accuracy", "final_f1", "final_loss",
    "convergence_rounds", "total_rounds_run", "total_comm_mb",
    "average_round_time", "composite_score",
]
df[EXPORT_COLS].to_csv(OUT_DIR / "comparison_table.csv", index=False)
print(f"✅  Table saved → {OUT_DIR / 'comparison_table.csv'}")

# ─────────────────────────────────────────────────────────────
# 5.  Terminal table
# ─────────────────────────────────────────────────────────────
DIVIDER = "─" * 120

def fmt_pct(v): return f"{v*100:.2f}%"
def fmt_f(v, d=3): return f"{v:.{d}f}"

print(f"\n{'':=<120}")
print(f"  ContinuumFL — Non-IID Experiment Comparison")
print(f"{'':=<120}")
header = (f"{'Rank':>4}  {'inter_α':>8}  {'intra_α':>7}  {'Comp%':>5}  "
          f"{'BestAcc':>8}  {'FinalAcc':>8}  {'F1':>6}  "
          f"{'ConvRnd':>7}  {'CommMB':>9}  {'Score':>6}  exp_id")
print(header)
print(DIVIDER)
for _, row in df.iterrows():
    print(
        f"{int(row['rank']):>4}  {row['inter_zone_alpha']:>8.1f}  "
        f"{row['intra_zone_alpha']:>7.0f}  "
        f"{row['compression_rate']*100:>5.0f}  "
        f"{fmt_pct(row['best_accuracy']):>8}  "
        f"{fmt_pct(row['final_accuracy']):>8}  "
        f"{fmt_f(row['final_f1']):>6}  "
        f"{int(row['convergence_rounds']):>7}  "
        f"{row['total_comm_mb']:>9.1f}  "
        f"{row['composite_score']:>6.3f}  "
        f"{row['exp_id']}"
    )
print(DIVIDER)

# ─────────────────────────────────────────────────────────────
# 6.  Plot 1 – Best Accuracy vs inter_zone_alpha
# ─────────────────────────────────────────────────────────────
def alpha_label(a):
    return f"α={a:.4g}"

plot_df = df.sort_values("inter_zone_alpha")
x_labels = [alpha_label(a) for a in plot_df["inter_zone_alpha"]]
colors   = [PALETTE[i % len(PALETTE)] for i in range(len(plot_df))]
hatches  = [IEEE_HATCHES[i % len(IEEE_HATCHES)] for i in range(len(plot_df))]

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(x_labels, plot_df["best_accuracy"] * 100,
              color=colors, hatch=hatches[0],  # per-bar hatch applied below
              edgecolor="black", linewidth=0.8, zorder=3)
for bar, h, c in zip(bars, hatches, colors):
    bar.set_hatch(h); bar.set_facecolor(c)
ax.set_xlabel("Inter-Zone Heterogeneity ($\\alpha$)  [lower = more non-IID]")
ax.set_ylabel("Best Global Accuracy (%)")
ax.set_title("Peak Accuracy vs. Non-IID Level")
ax.yaxis.grid(True, zorder=0)
lo = max(0, plot_df["best_accuracy"].min()*100 - 5)
ax.set_ylim(lo, min(100, plot_df["best_accuracy"].max()*100 + 7))
for bar, val in zip(bars, plot_df["best_accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8, color="black")
fig.tight_layout()
fig.savefig(PLOT_DIR / "01_best_accuracy_vs_alpha.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✅  Plot 1 saved → 01_best_accuracy_vs_alpha.png")

# ─────────────────────────────────────────────────────────────
# 7.  Plot 2 – Multi-metric radar / bar group
# ─────────────────────────────────────────────────────────────
metrics_show = ["_n_best_acc", "_n_f1", "_n_conv", "_n_comm"]
metric_labels = ["Best Acc\n(norm)", "F1\n(norm)", "Conv Speed\n(norm)", "Comm Eff\n(norm)"]

n_exp  = len(plot_df)
x      = np.arange(len(metrics_show))
width  = 0.8 / max(n_exp, 1)

fig, ax = plt.subplots(figsize=(7, 3.5))
for i, (_, row) in enumerate(plot_df.iterrows()):
    vals   = [row[m] for m in metrics_show]
    offset = (i - n_exp / 2) * width + width / 2
    ax.bar(x + offset, vals, width * 0.92,
           color=colors[i], hatch=IEEE_HATCHES[i % len(IEEE_HATCHES)],
           label=alpha_label(row["inter_zone_alpha"]),
           edgecolor="black", linewidth=0.5, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylabel("Normalised Score (higher = better)")
ax.set_title("Multi-Metric Comparison Across Non-IID Levels")
ax.yaxis.grid(True, zorder=0)
ax.set_ylim(0, 1.25)
ax.legend(loc="upper right", fontsize=7, ncol=2, title="$\\alpha$")
fig.tight_layout()
fig.savefig(PLOT_DIR / "02_multi_metric_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✅  Plot 2 saved → 02_multi_metric_bar.png")

# ─────────────────────────────────────────────────────────────
# 8.  Plot 3 – Accuracy learning curves (per-round)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
plotted = 0
for i, (_, row) in enumerate(plot_df.iterrows()):
    eid = row["exp_id"]
    if eid not in metrics_by_exp:
        continue
    dm = metrics_by_exp[eid].copy()
    col = "global_accuracy" if "global_accuracy" in dm.columns else None
    if col is None:
        continue
    rnd_col = "round" if "round" in dm.columns else dm.columns[0]
    dm  = dm.sort_values(rnd_col)
    acc = dm[col].values
    if acc.max() <= 1.01:
        acc = acc * 100
    n_pts   = len(acc)
    mevery  = max(1, n_pts // 8)
    ax.plot(dm[rnd_col].values, acc,
            color=colors[i],
            linestyle=IEEE_LINESTYLES[i % len(IEEE_LINESTYLES)],
            marker=IEEE_MARKERS[i % len(IEEE_MARKERS)],
            markevery=mevery, markersize=5,
            linewidth=1.5,
            label=alpha_label(row["inter_zone_alpha"]))
    plotted += 1

if plotted:
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.set_title("Accuracy Learning Curves — All Non-IID Settings")
    ax.yaxis.grid(True)
    ax.legend(loc="lower right", fontsize=7, ncol=2, title="$\\alpha$")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_learning_curves.png", dpi=300, bbox_inches="tight")
    print(f"✅  Plot 3 saved → 03_learning_curves.png")
else:
    print("⚠️   No per-round data found; skipping learning curve plot.")
plt.close(fig)

# ─────────────────────────────────────────────────────────────
# 9.  Plot 4 – Communication cost vs accuracy scatter
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for i, (_, row) in enumerate(plot_df.iterrows()):
    ax.scatter(
        row["total_comm_mb"] / 1000,
        row["best_accuracy"] * 100,
        color=colors[i],
        marker=IEEE_MARKERS[i % len(IEEE_MARKERS)],
        s=80, edgecolors="black", linewidths=0.7, zorder=5,
        label=alpha_label(row["inter_zone_alpha"])
    )
    ax.annotate(
        alpha_label(row["inter_zone_alpha"]),
        (row["total_comm_mb"] / 1000, row["best_accuracy"] * 100),
        textcoords="offset points", xytext=(7, 4),
        fontsize=8, color="black"
    )
ax.set_xlabel("Total Communication Cost (GB)")
ax.set_ylabel("Best Global Accuracy (%)")
ax.set_title("Accuracy vs. Communication Cost Trade-off")
ax.yaxis.grid(True, zorder=0)
ax.xaxis.grid(True, zorder=0)
ax.legend(fontsize=7, title="$\\alpha$", ncol=2)
fig.tight_layout()
fig.savefig(PLOT_DIR / "04_accuracy_vs_comm.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✅  Plot 4 saved → 04_accuracy_vs_comm.png")

# ─────────────────────────────────────────────────────────────
# 10. Plot 5 – Composite score ranking
# ─────────────────────────────────────────────────────────────
rank_df    = df.sort_values("composite_score", ascending=True)
labels     = [alpha_label(a) for a in rank_df["inter_zone_alpha"]]
scores     = rank_df["composite_score"].values
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(rank_df))]
bar_hatch  = [IEEE_HATCHES[i % len(IEEE_HATCHES)] for i in range(len(rank_df))]

fig, ax = plt.subplots(figsize=(7, max(3, len(rank_df) * 0.5)))
bars = ax.barh(labels, scores, color=bar_colors,
               edgecolor="black", linewidth=0.7, zorder=3)
for bar, h in zip(bars, bar_hatch):
    bar.set_hatch(h)
ax.set_xlabel("Composite Score (higher = better)")
ax.set_title("Overall Ranking — Composite Score\n"
             "(40% Acc + 20% F1 + 20% Conv + 20% Comm)")
ax.xaxis.grid(True, zorder=0)
ax.set_xlim(0, 1.15)
for bar, val in zip(bars, scores):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8, color="black")
fig.tight_layout()
fig.savefig(PLOT_DIR / "05_composite_ranking.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✅  Plot 5 saved → 05_composite_ranking.png")

# ─────────────────────────────────────────────────────────────
# 11. Best-setup recommendation
# ─────────────────────────────────────────────────────────────
best = df.iloc[0]   # rank 1

recommendation = f"""
╔══════════════════════════════════════════════════════════════╗
║          ContinuumFL — Non-IID Best Setup Recommendation    ║
╚══════════════════════════════════════════════════════════════╝

🏆  BEST CONFIGURATION:
    inter_zone_alpha  : {best['inter_zone_alpha']:.4g}  ({"extreme non-IID" if best['inter_zone_alpha'] <= 0.3 else "moderate non-IID" if best['inter_zone_alpha'] <= 5 else "near-IID"})
    intra_zone_alpha  : {best['intra_zone_alpha']:.4g}
    compression_rate  : {best['compression_rate']:.2f}  ({int(best['compression_rate']*100)}% top-k)
    dataset           : {best['dataset']}
    num_devices       : {int(best['num_devices'])}
    num_zones         : {int(best['num_zones'])}

📊  PERFORMANCE METRICS:
    Best accuracy     : {best['best_accuracy']*100:.2f}%  (round {int(best['best_accuracy_round'])})
    Final accuracy    : {best['final_accuracy']*100:.2f}%
    Final F1-score    : {best['final_f1']:.4f}
    Final loss        : {best['final_loss']:.4f}
    Convergence round : {int(best['convergence_rounds'])}
    Total comm. cost  : {best['total_comm_mb']:.1f} MB
    Avg round time    : {best['average_round_time']:.1f} s
    Composite score   : {best['composite_score']:.4f} / 1.000

📈  FULL RANKING SUMMARY:
"""
for _, row in df.iterrows():
    marker = "🥇" if row["rank"] == 1 else ("🥈" if row["rank"] == 2 else ("🥉" if row["rank"] == 3 else "  "))
    recommendation += (
        f"  {marker} #{int(row['rank']):<2}  α={row['inter_zone_alpha']:<6.4g}  "
        f"BestAcc={row['best_accuracy']*100:5.2f}%  F1={row['final_f1']:.3f}  "
        f"ConvRnd={int(row['convergence_rounds']):<4}  "
        f"CommMB={row['total_comm_mb']:>9.1f}  Score={row['composite_score']:.3f}\n"
    )

# Interpretation
recommendation += """
💡  INTERPRETATION:
    • inter_zone_alpha controls data heterogeneity between zones.
      Lower values  → zones see very different class distributions (hard non-IID).
      Higher values → zones see similar class distributions   (near-IID).

    • The composite score balances four criteria:
        40% — Peak accuracy achieved (higher is better)
        20% — F1-score  (higher is better)
        20% — Convergence speed (fewer rounds to reach 95% of peak is better)
        20% — Communication efficiency (less total bandwidth is better)

    • If your priority is maximum accuracy → choose the config with highest BestAcc.
    • If your priority is communication efficiency → choose highest Comm-Efficiency (lowest CommMB).
    • If your priority is fast convergence → choose lowest ConvRnd.

🚀  RECOMMENDED COMMAND:
    python main.py \\
        --dataset {b_dataset} \\
        --num_devices {b_num_devices} \\
        --num_zones {b_num_zones} \\
        --num_rounds 200 \\
        --inter_zone_alpha {b_inter_alpha} \\
        --intra_zone_alpha {b_intra_alpha} \\
        --compression_rate {b_comp_rate} \\
        --enable_early_stopping \\
        --early_stopping_patience 20

Plots saved to : {b_plot_dir}
Table saved to : {b_table_path}
""".format(
    b_dataset     = str(best["dataset"]),
    b_num_devices = int(best["num_devices"]),
    b_num_zones   = int(best["num_zones"]),
    b_inter_alpha = f"{best['inter_zone_alpha']:.4g}",
    b_intra_alpha = f"{best['intra_zone_alpha']:.4g}",
    b_comp_rate   = f"{best['compression_rate']:.2f}",
    b_plot_dir    = str(PLOT_DIR),
    b_table_path  = str(OUT_DIR / 'comparison_table.csv'),
)

print(recommendation)
(OUT_DIR / "recommendation.txt").write_text(recommendation)
print(f"✅  Recommendation saved → {OUT_DIR / 'recommendation.txt'}")
print(f"\n🎉  Analysis complete.  Results in: {OUT_DIR}\n")
