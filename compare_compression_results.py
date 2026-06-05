#!/usr/bin/env python3
"""
compare_compression_results.py
===============================
Collects every ContinuumFL compression-rate experiment from:
  1. results/femnist__intra*__inter*__comp*pct/   (structured summary.csv)
  2. results/compression_sweep/*.log               (plain log files)

Compares them across accuracy, F1, communication cost and convergence,
computes a composite score, then proposes the best compression rate.

Outputs
-------
- Terminal ranking table
- results/compression_comparison/comparison_table.csv
- results/compression_comparison/plots/*.png  (5 charts)
- results/compression_comparison/recommendation.txt
"""

import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
OUT_DIR     = RESULTS_DIR / "compression_comparison"
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
# 1. Load structured results  (femnist__intra*__inter*__compNpct)
# ─────────────────────────────────────────────────────────────
FOLDER_RE = re.compile(
    r"(?P<dataset>\w+)__intra(?P<intra>[\d.]+)__inter(?P<inter>[\d.]+)__comp(?P<comp>\d+)pct$"
)

records: list[dict] = []
metrics_by_exp: dict[str, pd.DataFrame] = {}

for folder in sorted(RESULTS_DIR.iterdir()):
    m = FOLDER_RE.match(folder.name)
    if not m:
        continue
    summary_path = folder / "summary.csv"
    metrics_path = folder / "metrics.csv"
    if not summary_path.exists():
        continue

    # summary.csv may have multiple rows (resumed runs) — keep last non-empty
    try:
        df_s = pd.read_csv(summary_path).dropna(subset=["best_accuracy"])
    except Exception:
        continue
    if df_s.empty:
        continue
    # prefer the row with the highest best_accuracy
    row = df_s.loc[df_s["best_accuracy"].idxmax()]

    comp_rate = int(m.group("comp")) / 100.0
    inter     = float(m.group("inter"))
    intra     = float(m.group("intra"))
    dataset   = str(m.group("dataset"))

    rec = {
        "source":              "structured",
        "exp_id":              folder.name,
        "dataset":             dataset,
        "intra_zone_alpha":    intra,
        "inter_zone_alpha":    inter,
        "compression_rate":    comp_rate,
        "comp_pct":            int(m.group("comp")),
        "num_rounds":          int(row.get("num_rounds",  200)),
        "num_devices":         int(row.get("num_devices",  50)),
        "num_zones":           int(row.get("num_zones",     5)),
        "final_accuracy":      float(row.get("final_accuracy",  0)),
        "best_accuracy":       float(row.get("best_accuracy",   0)),
        "best_accuracy_round": int(row.get("best_accuracy_round", 0)),
        "final_loss":          float(row.get("final_loss",    9999)),
        "final_f1":            float(row.get("final_f1",       0)),
        "final_precision":     float(row.get("final_precision", 0)),
        "final_recall":        float(row.get("final_recall",    0)),
        "total_training_time": float(row.get("total_training_time",     0)),
        "average_round_time":  float(row.get("average_round_time",      0)),
        "total_comm_mb":       float(row.get("total_communication_cost", 0)),
        "convergence_rounds":  int(row.get("convergence_rounds",         0)),
        "total_rounds_run":    int(row.get("num_rounds",                200)),
    }

    if metrics_path.exists():
        try:
            df_m = pd.read_csv(metrics_path).dropna(subset=["global_accuracy"])
            rec["total_rounds_run"] = len(df_m)
            metrics_by_exp[folder.name] = df_m
        except Exception:
            pass

    records.append(rec)

# ─────────────────────────────────────────────────────────────
# 2. Load compression_sweep log files
# ─────────────────────────────────────────────────────────────
LOG_DIR  = RESULTS_DIR / "compression_sweep"
ROUND_RE = re.compile(
    r"Round\s+(\d+)\s*\|.*?Acc=([\d.]+)%.*?Loss=([\d.]+).*?"
    r"P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+).*?Comm=([\d.]+)MB.*?Time=([\d.]+)s"
)
BEST_RE = re.compile(r"best=([\d.]+)%@R(\d+)")

if LOG_DIR.exists():
    for log_file in sorted(LOG_DIR.glob("compression_rate*.log")):
        rate_m = re.search(r"compression_rate([\d.]+)\.log", log_file.name)
        if not rate_m:
            continue
        comp_rate = float(rate_m.group(1))

        text = log_file.read_text(errors="replace")

        rows_data = []
        for rm in ROUND_RE.finditer(text):
            rows_data.append({
                "round":                int(rm.group(1)),
                "global_accuracy":      float(rm.group(2)) / 100.0,
                "global_loss":          float(rm.group(3)),
                "precision":            float(rm.group(4)),
                "recall":               float(rm.group(5)),
                "f1":                   float(rm.group(6)),
                "communication_cost_mb": float(rm.group(7)),
                "round_time_s":         float(rm.group(8)),
            })

        if not rows_data:
            continue

        df_log = (pd.DataFrame(rows_data)
                  .drop_duplicates(subset="round")
                  .sort_values("round"))
        exp_id = f"log_comp{int(round(comp_rate*100))}pct"

        # Only keep if we don't already have a structured result for this rate
        existing_rates = {r["compression_rate"] for r in records}
        if comp_rate in existing_rates:
            # Enrich existing entry with per-round data if missing
            for r in records:
                if r["compression_rate"] == comp_rate and r["exp_id"] not in metrics_by_exp:
                    metrics_by_exp[r["exp_id"]] = df_log
            continue

        bm = list(BEST_RE.finditer(text))
        if bm:
            best_acc = float(bm[-1].group(1)) / 100.0
            best_rnd = int(bm[-1].group(2))
        else:
            best_acc = df_log["global_accuracy"].max()
            best_rnd = int(df_log.loc[df_log["global_accuracy"].idxmax(), "round"])

        threshold  = 0.95 * best_acc
        above      = df_log[df_log["global_accuracy"] >= threshold]
        conv_rnd   = int(above["round"].iloc[0]) if not above.empty else int(df_log["round"].max())
        final_row  = df_log.iloc[-1]

        rec = {
            "source":              "log",
            "exp_id":              exp_id,
            "dataset":             "femnist",
            "intra_zone_alpha":    100.0,
            "inter_zone_alpha":    5.0,    # best fixed value from noniid comparison
            "compression_rate":    comp_rate,
            "comp_pct":            int(round(comp_rate * 100)),
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
            "total_training_time": float(df_log["round_time_s"].sum()),
            "average_round_time":  float(df_log["round_time_s"].mean()),
            "total_comm_mb":       float(df_log["communication_cost_mb"].sum()),
            "convergence_rounds":  conv_rnd,
            "total_rounds_run":    int(df_log["round"].max()),
        }
        metrics_by_exp[exp_id] = df_log
        records.append(rec)

# ─────────────────────────────────────────────────────────────
# 3. Build comparison DataFrame
# ─────────────────────────────────────────────────────────────
if not records:
    print("❌  No compression experiment results found. Run experiments first.")
    sys.exit(1)

df = pd.DataFrame(records).sort_values("compression_rate").reset_index(drop=True)

# Remove exact duplicates (same comp rate + dataset) keeping best_accuracy winner
df = (df.sort_values("best_accuracy", ascending=False)
        .drop_duplicates(subset=["dataset", "compression_rate"])
        .sort_values("compression_rate")
        .reset_index(drop=True))

# ── Composite scoring ────────────────────────────────────────────────────────
# Weights:  best_accuracy 40%  |  final_f1 20%  |  conv_speed 20%  |  comm_eff 20%
def norm(series, higher_is_better=True):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    n = (series - mn) / (mx - mn)
    return n if higher_is_better else 1 - n

df["_n_best_acc"] = norm(df["best_accuracy"],      higher_is_better=True)
df["_n_f1"]       = norm(df["final_f1"],           higher_is_better=True)
df["_n_conv"]     = norm(df["convergence_rounds"], higher_is_better=False)
df["_n_comm"]     = norm(df["total_comm_mb"],      higher_is_better=False)

df["composite_score"] = (
    0.40 * df["_n_best_acc"] +
    0.20 * df["_n_f1"]       +
    0.20 * df["_n_conv"]     +
    0.20 * df["_n_comm"]
)

df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# ─────────────────────────────────────────────────────────────
# 4. Save CSV
# ─────────────────────────────────────────────────────────────
EXPORT_COLS = [
    "rank", "comp_pct", "compression_rate", "dataset",
    "inter_zone_alpha", "best_accuracy", "final_accuracy",
    "final_f1", "final_loss", "convergence_rounds",
    "total_rounds_run", "total_comm_mb", "average_round_time",
    "composite_score",
]
df[EXPORT_COLS].to_csv(OUT_DIR / "comparison_table.csv", index=False)
print(f"✅  Table saved → {OUT_DIR / 'comparison_table.csv'}")

# ─────────────────────────────────────────────────────────────
# 5. Terminal table
# ─────────────────────────────────────────────────────────────
DIVIDER = "─" * 115
print(f"\n{'':=<115}")
print(f"  ContinuumFL — Compression Rate Comparison")
print(f"{'':=<115}")
header = (f"{'Rank':>4}  {'Comp%':>5}  {'BestAcc':>8}  {'FinalAcc':>8}  "
          f"{'F1':>6}  {'ConvRnd':>7}  {'CommMB':>9}  {'AvgRndT':>8}  "
          f"{'Score':>6}  Dataset")
print(header)
print(DIVIDER)
for _, row in df.iterrows():
    print(
        f"{int(row['rank']):>4}  "
        f"{int(row['comp_pct']):>4}%  "
        f"{row['best_accuracy']*100:>7.2f}%  "
        f"{row['final_accuracy']*100:>7.2f}%  "
        f"{row['final_f1']:>6.3f}  "
        f"{int(row['convergence_rounds']):>7}  "
        f"{row['total_comm_mb']:>9.1f}  "
        f"{row['average_round_time']:>8.1f}s  "
        f"{row['composite_score']:>6.3f}  "
        f"{row['dataset']}"
    )
print(DIVIDER)

# ─────────────────────────────────────────────────────────────
# 6. Sorted by comp_pct for clean plotting
# ─────────────────────────────────────────────────────────────
plot_df  = df.sort_values("compression_rate").reset_index(drop=True)
n_exp    = len(plot_df)
colors   = [PALETTE[i % len(PALETTE)] for i in range(n_exp)]
hatches  = [IEEE_HATCHES[i % len(IEEE_HATCHES)] for i in range(n_exp)]
x_labels = [f"{int(r)}%" for r in plot_df["comp_pct"]]

# Plot 1 – Best Accuracy vs Compression Rate (bar)
fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(x_labels, plot_df["best_accuracy"] * 100,
              color=colors, edgecolor="black", linewidth=0.8, zorder=3)
for bar, h, c in zip(bars, hatches, colors):
    bar.set_hatch(h); bar.set_facecolor(c)
ax.set_xlabel("Compression Rate (% of gradients kept)")
ax.set_ylabel("Best Global Accuracy (%)")
ax.set_title("Peak Accuracy vs. Compression Rate")
ax.yaxis.grid(True, zorder=0)
lo = max(0, plot_df["best_accuracy"].min() * 100 - 5)
ax.set_ylim(lo, min(100, plot_df["best_accuracy"].max() * 100 + 7))
for bar, val in zip(bars, plot_df["best_accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val*100:.2f}%", ha="center", va="bottom", fontsize=8, color="black")
fig.tight_layout()
fig.savefig(PLOT_DIR / "01_accuracy_vs_compression.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✅  Plot 1 saved → 01_accuracy_vs_compression.png")

# Plot 2 – Communication Cost vs Accuracy (scatter)
fig, ax = plt.subplots(figsize=(7, 4))
for i, (_, row) in enumerate(plot_df.iterrows()):
    ax.scatter(row["total_comm_mb"]/1000, row["best_accuracy"]*100,
               color=colors[i], marker=IEEE_MARKERS[i % len(IEEE_MARKERS)],
               s=80, edgecolors="black", linewidths=0.7, zorder=5,
               label=f"{int(row['comp_pct'])}%")
    ax.annotate(f"{int(row['comp_pct'])}%",
                (row["total_comm_mb"]/1000, row["best_accuracy"]*100),
                textcoords="offset points", xytext=(8,4), fontsize=8, color="black")
ax.set_xlabel("Total Communication Cost (GB)")
ax.set_ylabel("Best Global Accuracy (%)")
ax.set_title("Accuracy vs. Communication Cost — Compression Trade-off")
ax.yaxis.grid(True, zorder=0); ax.xaxis.grid(True, zorder=0)
ax.legend(fontsize=7, title="Comp%", ncol=2)
fig.tight_layout()
fig.savefig(PLOT_DIR / "02_accuracy_vs_comm_cost.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✅  Plot 2 saved → 02_accuracy_vs_comm_cost.png")

# Plot 3 – Learning curves
fig, ax = plt.subplots(figsize=(7, 4))
plotted = 0
for i, (_, row) in enumerate(plot_df.iterrows()):
    eid = row["exp_id"]
    if eid not in metrics_by_exp:
        continue
    dm  = metrics_by_exp[eid].copy()
    col = "global_accuracy" if "global_accuracy" in dm.columns else None
    if col is None:
        continue
    dm  = dm.sort_values("round" if "round" in dm.columns else dm.columns[0])
    acc = dm[col].values
    if acc.max() <= 1.01:
        acc = acc * 100
    rnd = dm["round"].values if "round" in dm.columns else range(len(acc))
    mev = max(1, len(acc)//8)
    ax.plot(rnd, acc, color=colors[i],
            linestyle=IEEE_LINESTYLES[i % len(IEEE_LINESTYLES)],
            marker=IEEE_MARKERS[i % len(IEEE_MARKERS)],
            markevery=mev, markersize=5, linewidth=1.5,
            label=f"{int(row['comp_pct'])}% kept")
    plotted += 1

if plotted:
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.set_title("Accuracy Learning Curves — All Compression Rates")
    ax.yaxis.grid(True)
    ax.legend(loc="lower right", fontsize=7, ncol=2, title="Comp%")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_learning_curves.png", dpi=300, bbox_inches="tight")
    print("✅  Plot 3 saved → 03_learning_curves.png")
else:
    print("⚠️   No per-round data for curves; skipping Plot 3.")
plt.close(fig)

# Plot 4 – Multi-metric normalised bar
metrics_keys  = ["_n_best_acc", "_n_f1", "_n_conv", "_n_comm"]
metric_labels = ["Best Acc\n(norm)", "F1\n(norm)", "Conv Speed\n(norm)", "Comm Eff\n(norm)"]
x     = np.arange(len(metrics_keys))
width = 0.8 / max(n_exp, 1)
fig, ax = plt.subplots(figsize=(7, 3.5))
for i, (_, row) in enumerate(plot_df.iterrows()):
    vals   = [row[k] for k in metrics_keys]
    offset = (i - n_exp/2)*width + width/2
    ax.bar(x+offset, vals, width*0.92,
           color=colors[i], hatch=IEEE_HATCHES[i % len(IEEE_HATCHES)],
           label=f"{int(row['comp_pct'])}%",
           edgecolor="black", linewidth=0.5, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(metric_labels)
ax.set_ylabel("Normalised Score (higher = better)")
ax.set_title("Multi-Metric Comparison Across Compression Rates")
ax.yaxis.grid(True, zorder=0); ax.set_ylim(0, 1.25)
ax.legend(loc="upper right", fontsize=7, ncol=3, title="Comp%")
fig.tight_layout()
fig.savefig(PLOT_DIR / "04_multi_metric_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✅  Plot 4 saved → 04_multi_metric_bar.png")

# Plot 5 – Composite score ranking (horizontal bar)
rank_df     = df.sort_values("composite_score", ascending=True)
rank_labels = [f"{int(r['comp_pct'])}% kept" for _, r in rank_df.iterrows()]
rank_scores = rank_df["composite_score"].values
rank_colors = [PALETTE[i % len(PALETTE)] for i in range(len(rank_df))]
rank_hatch  = [IEEE_HATCHES[i % len(IEEE_HATCHES)] for i in range(len(rank_df))]
fig, ax = plt.subplots(figsize=(7, max(3, len(rank_df)*0.5)))
bars = ax.barh(rank_labels, rank_scores,
               color=rank_colors, edgecolor="black", linewidth=0.7, zorder=3)
for bar, h in zip(bars, rank_hatch):
    bar.set_hatch(h)
ax.set_xlabel("Composite Score (higher = better)")
ax.set_title("Overall Ranking — Composite Score\n(40% Acc + 20% F1 + 20% Conv + 20% Comm)")
ax.xaxis.grid(True, zorder=0); ax.set_xlim(0, 1.15)
for bar, val in zip(bars, rank_scores):
    ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8, color="black")
fig.tight_layout()
fig.savefig(PLOT_DIR / "05_composite_ranking.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✅  Plot 5 saved → 05_composite_ranking.png")

# ─────────────────────────────────────────────────────────────
# 7. Recommendation
# ─────────────────────────────────────────────────────────────
best = df.iloc[0]

# Accuracy-only winner (might differ from composite winner)
acc_winner = df.loc[df["best_accuracy"].idxmax()]
# Comm-efficiency winner
comm_winner = df.loc[df["total_comm_mb"].idxmin()]

b_comp_pct    = int(best["comp_pct"])
b_dataset     = str(best["dataset"])
b_inter       = f"{best['inter_zone_alpha']:.4g}"
b_intra       = f"{best['intra_zone_alpha']:.4g}"
b_comp_rate   = f"{best['compression_rate']:.2f}"
b_best_acc    = best["best_accuracy"]
b_best_rnd    = int(best["best_accuracy_round"])
b_final_acc   = best["final_accuracy"]
b_f1          = best["final_f1"]
b_loss        = best["final_loss"]
b_conv        = int(best["convergence_rounds"])
b_comm        = best["total_comm_mb"]
b_avg_rnd     = best["average_round_time"]
b_score       = best["composite_score"]

recommendation = f"""
╔══════════════════════════════════════════════════════════════════╗
║       ContinuumFL — Compression Rate Best Setup Recommendation  ║
╚══════════════════════════════════════════════════════════════════╝

🏆  BEST CONFIGURATION  (composite score winner):
    compression_rate  : {b_comp_rate}  ({b_comp_pct}% of gradients kept)
    inter_zone_alpha  : {b_inter}
    intra_zone_alpha  : {b_intra}
    dataset           : {b_dataset}

📊  PERFORMANCE METRICS:
    Best accuracy     : {b_best_acc*100:.2f}%  (round {b_best_rnd})
    Final accuracy    : {b_final_acc*100:.2f}%
    Final F1-score    : {b_f1:.4f}
    Final loss        : {b_loss:.4f}
    Convergence round : {b_conv}
    Total comm. cost  : {b_comm:.1f} MB
    Avg round time    : {b_avg_rnd:.1f} s
    Composite score   : {b_score:.4f} / 1.000

🎯  SPECIAL WINNERS:
    Max accuracy      : {acc_winner['comp_pct']:.0f}%  ({acc_winner['best_accuracy']*100:.2f}%)
    Min comm. cost    : {comm_winner['comp_pct']:.0f}%  ({comm_winner['total_comm_mb']:.1f} MB)

📈  FULL RANKING SUMMARY:
"""
for _, row in df.iterrows():
    m = "🥇" if row["rank"]==1 else ("🥈" if row["rank"]==2 else ("🥉" if row["rank"]==3 else "  "))
    recommendation += (
        f"  {m} #{int(row['rank']):<2}  Comp={int(row['comp_pct']):>3}%  "
        f"BestAcc={row['best_accuracy']*100:5.2f}%  "
        f"F1={row['final_f1']:.3f}  "
        f"ConvRnd={int(row['convergence_rounds']):<5}  "
        f"CommMB={row['total_comm_mb']:>8.1f}  "
        f"Score={row['composite_score']:.3f}\n"
    )

recommendation += f"""
💡  INTERPRETATION:
    • compression_rate is the top-k fraction of gradients transmitted.
      1%  → extreme compression  — massive bandwidth saving, slow convergence.
      10% → strong compression   — good accuracy/bandwidth balance.
      50% → mild compression     — near-full gradient, faster convergence.
      100%→ no compression       — full gradient, highest possible accuracy ceiling.

    • The composite score balances four criteria:
        40% — Peak accuracy (higher is better)
        20% — F1-score      (higher is better)
        20% — Convergence speed (fewer rounds is better)
        20% — Communication efficiency (less bandwidth is better)

    • If your priority is accuracy only  → use compression_rate with highest BestAcc.
    • If your priority is bandwidth only → use lowest CommMB entry.
    • For the balanced optimum           → use the #1 composite-score winner.

🚀  RECOMMENDED COMMAND:
    python main.py \\
        --dataset          {b_dataset} \\
        --num_devices      {int(best['num_devices'])} \\
        --num_zones        {int(best['num_zones'])} \\
        --num_rounds       200 \\
        --inter_zone_alpha {b_inter} \\
        --intra_zone_alpha {b_intra} \\
        --enable_compression \\
        --compression_rate {b_comp_rate} \\
        --enable_early_stopping \\
        --early_stopping_patience 10

Plots saved to : {PLOT_DIR}
Table saved to : {OUT_DIR / 'comparison_table.csv'}
"""

print(recommendation)
(OUT_DIR / "recommendation.txt").write_text(recommendation)
print(f"✅  Recommendation saved → {OUT_DIR / 'recommendation.txt'}")
print(f"\n🎉  Analysis complete.  Results in: {OUT_DIR}\n")
