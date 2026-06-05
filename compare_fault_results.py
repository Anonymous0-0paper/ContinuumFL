#!/usr/bin/env python3
"""
compare_fault_results.py
========================
Compares all saved ContinuumFL fault-sweep experiments for FEMNIST and
proposes the best setup based on resilience.

Sources
-------
1. results/fault_sweep/femnist_dev*_zone*/.../summary.csv
2. results/fault_sweep/femnist_dev*_zone*.log  (fallback)

Outputs
-------
- Terminal ranking table
- results/fault_comparison/comparison_table.csv
- results/fault_comparison/plots/*.png  (5 charts)
- results/fault_comparison/recommendation.txt
"""

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
# IEEE publication style
# ─────────────────────────────────────────────────────────────
from ieee_plot_style import (IEEE_COLORS as PALETTE, IEEE_HATCHES,
                              IEEE_LINESTYLES, IEEE_MARKERS, setup_ieee_style)
setup_ieee_style()

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results" / "fault_sweep"
OUT_DIR     = ROOT / "results" / "fault_comparison"
PLOT_DIR    = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load structured results ────────────────────────────────────────────────
FOLDER_RE = re.compile(
    r"(?P<dataset>\w+)_dev(?P<dev>[\d.]+)_zone(?P<zone>[\d.]+)$"
)

records: list[dict] = []
metrics_by_exp: dict[str, pd.DataFrame] = {}

if not RESULTS_DIR.exists():
    print(f"❌  No fault-sweep results found at {RESULTS_DIR}. Run run_fault_sweep.sh first.")
    sys.exit(1)

for folder in sorted(RESULTS_DIR.iterdir()):
    if not folder.is_dir():
        continue
    m = FOLDER_RE.match(folder.name)
    if not m:
        continue

    dataset = m.group("dataset")
    dev_fail = float(m.group("dev"))
    zone_fail = float(m.group("zone"))
    exp_id = folder.name

    # Find the deepest summary.csv
    summary_files = list(folder.rglob("summary.csv"))
    if not summary_files:
        continue
    summary_path = summary_files[-1]
    metrics_path = summary_path.parent / "metrics.csv"

    try:
        df_s = pd.read_csv(summary_path).dropna(subset=["best_accuracy"])
    except Exception:
        continue
    if df_s.empty:
        continue

    row = df_s.loc[df_s["best_accuracy"].idxmax()]

    rec = {
        "source":              "structured",
        "exp_id":              exp_id,
        "dataset":             dataset,
        "dev_fail":            dev_fail,
        "zone_fail":           zone_fail,
        "config_label":        f"D:{dev_fail} Z:{zone_fail}",
        "final_accuracy":      float(row.get("final_accuracy", 0)),
        "best_accuracy":       float(row.get("best_accuracy",  0)),
        "best_accuracy_round": int(row.get("best_accuracy_round", 0)),
        "final_loss":          float(row.get("final_loss",   9999)),
        "final_f1":            float(row.get("final_f1",        0)),
        "final_precision":     float(row.get("final_precision",  0)),
        "final_recall":        float(row.get("final_recall",     0)),
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
            metrics_by_exp[exp_id] = df_m
        except Exception:
            pass

    records.append(rec)

# ── 2. Load fault_sweep log files (fallback / extra runs) ─────────────────────
ROUND_RE = re.compile(
    r"Round\s+(\d+)\s*\|.*?Acc=([\d.]+)%.*?Loss=([\d.]+).*?"
    r"P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+).*?Comm=([\d.]+)MB.*?Time=([\d.]+)s"
)
BEST_RE  = re.compile(r"best=([\d.]+)%@R(\d+)")

for log_file in sorted(RESULTS_DIR.glob("*_dev*_zone*.log")):
    lm = re.search(r"(?P<dataset>\w+)_dev(?P<dev>[\d.]+)_zone(?P<zone>[\d.]+)\.log", log_file.name)
    if not lm:
        continue
    dataset = lm.group("dataset")
    dev_fail = float(lm.group("dev"))
    zone_fail = float(lm.group("zone"))
    exp_id = f"{dataset}_dev{dev_fail:.2f}_zone{zone_fail:.2f}"

    # Skip if already loaded from structured results
    if any(r["dev_fail"] == dev_fail and r["zone_fail"] == zone_fail for r in records):
        # Still load per-round data if missing
        if exp_id not in metrics_by_exp:
            rows_data = []
            for rm in ROUND_RE.finditer(log_file.read_text(errors="replace")):
                rows_data.append({
                    "round": int(rm.group(1)),
                    "global_accuracy": float(rm.group(2)) / 100.0,
                    "global_loss": float(rm.group(3)),
                    "precision": float(rm.group(4)),
                    "recall": float(rm.group(5)),
                    "f1": float(rm.group(6)),
                    "communication_cost_mb": float(rm.group(7)),
                    "round_time_s": float(rm.group(8)),
                })
            if rows_data:
                metrics_by_exp[exp_id] = (
                    pd.DataFrame(rows_data)
                    .drop_duplicates("round").sort_values("round"))
        continue

    text      = log_file.read_text(errors="replace")
    rows_data = []
    for rm in ROUND_RE.finditer(text):
        rows_data.append({
            "round": int(rm.group(1)),
            "global_accuracy": float(rm.group(2)) / 100.0,
            "global_loss": float(rm.group(3)),
            "precision": float(rm.group(4)),
            "recall": float(rm.group(5)),
            "f1": float(rm.group(6)),
            "communication_cost_mb": float(rm.group(7)),
            "round_time_s": float(rm.group(8)),
        })
    if not rows_data:
        continue

    df_log  = (pd.DataFrame(rows_data)
               .drop_duplicates("round").sort_values("round"))
    metrics_by_exp[exp_id] = df_log

    bm = list(BEST_RE.finditer(text))
    if bm:
        best_acc = float(bm[-1].group(1)) / 100.0
        best_rnd = int(bm[-1].group(2))
    else:
        best_acc = df_log["global_accuracy"].max()
        best_rnd = int(df_log.loc[df_log["global_accuracy"].idxmax(), "round"])

    above    = df_log[df_log["global_accuracy"] >= 0.95 * best_acc]
    conv_rnd = int(above["round"].iloc[0]) if not above.empty else int(df_log["round"].max())
    final    = df_log.iloc[-1]

    records.append({
        "source": "log", "exp_id": exp_id,
        "dataset": dataset,
        "dev_fail": dev_fail, "zone_fail": zone_fail,
        "config_label": f"D:{dev_fail} Z:{zone_fail}",
        "final_accuracy":  float(final["global_accuracy"]),
        "best_accuracy":   best_acc,
        "best_accuracy_round": best_rnd,
        "final_loss":      float(final["global_loss"]),
        "final_f1":        float(final["f1"]),
        "final_precision": float(final["precision"]),
        "final_recall":    float(final["recall"]),
        "total_training_time": float(df_log["round_time_s"].sum()),
        "average_round_time":  float(df_log["round_time_s"].mean()),
        "total_comm_mb":       float(df_log["communication_cost_mb"].sum()),
        "convergence_rounds":  conv_rnd,
        "total_rounds_run":    int(df_log["round"].max()),
    })

# ── 3. Build DataFrame ────────────────────────────────────────────────────────
if not records:
    print("❌  No valid fault-sweep results found. Check the output of run_fault_sweep.sh.")
    sys.exit(1)

df = (pd.DataFrame(records)
        .sort_values(by=["zone_fail", "dev_fail"])
        .drop_duplicates(subset=["dev_fail", "zone_fail"])
        .reset_index(drop=True))

# Composite score: 40% best_acc | 20% f1 | 20% conv_speed | 20% comm_eff
def norm(s, higher=True):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    n = (s - mn) / (mx - mn)
    return n if higher else 1 - n

df["_n_acc"]  = norm(df["best_accuracy"],      True)
df["_n_f1"]   = norm(df["final_f1"],           True)
df["_n_conv"] = norm(df["convergence_rounds"], False)
df["_n_comm"] = norm(df["total_comm_mb"],      False)
df["composite_score"] = (0.40 * df["_n_acc"]  + 0.20 * df["_n_f1"] +
                         0.20 * df["_n_conv"] + 0.20 * df["_n_comm"])
df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# ── 4. Save CSV ───────────────────────────────────────────────────────────────
COLS = ["rank","dev_fail","zone_fail","config_label",
        "best_accuracy","final_accuracy","final_f1","final_loss",
        "convergence_rounds","total_comm_mb","average_round_time","composite_score"]
df[COLS].to_csv(OUT_DIR / "comparison_table.csv", index=False)
print(f"✅  Table saved → {OUT_DIR / 'comparison_table.csv'}")

# ── 5. Terminal table ─────────────────────────────────────────────────────────
DIV = "─" * 105
print(f"\n{'':=<105}")
print("  ContinuumFL — Fault Resilience Comparison  (FEMNIST)")
print(f"{'':=<105}")
print(f"{'Rank':>4}  {'Dev Fail':>9}  {'Zone Fail':>9}  "
      f"{'BestAcc':>8}  {'FinalAcc':>8}  {'F1':>6}  "
      f"{'ConvRnd':>7}  {'CommMB':>9}  {'Score':>6}")
print(DIV)
for _, r in df.iterrows():
    mk = "🥇" if r["rank"]==1 else ("🥈" if r["rank"]==2 else ("🥉" if r["rank"]==3 else "  "))
    print(f"{mk}{int(r['rank']):>3}  {r['dev_fail']:>9.2f}  {r['zone_fail']:>9.2f}  "
          f"{r['best_accuracy']*100:>7.2f}%  "
          f"{r['final_accuracy']*100:>7.2f}%  "
          f"{r['final_f1']:>6.3f}  "
          f"{int(r['convergence_rounds']):>7}  "
          f"{r['total_comm_mb']:>9.1f}  "
          f"{r['composite_score']:>6.3f}")
print(DIV)

# ── 6. Plots ──────────────────────────────────────────────────────────────────
plot_df = df.sort_values(by=["dev_fail", "zone_fail"]).reset_index(drop=True)
n       = len(plot_df)
colors  = [PALETTE[i % len(PALETTE)] for i in range(n)]
hatches = [IEEE_HATCHES[i % len(IEEE_HATCHES)] for i in range(n)]
labels  = plot_df["config_label"].tolist()

# Plot 1 – Best accuracy bar
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(labels, plot_df["best_accuracy"]*100,
              color=colors, edgecolor="black", linewidth=0.8, zorder=3)
for bar, h, c in zip(bars, hatches, colors):
    bar.set_hatch(h); bar.set_facecolor(c)
ax.set_xlabel("Configuration (Device Fail : Zone Fail)")
ax.set_ylabel("Best Global Accuracy (%)")
ax.set_title("Peak Accuracy vs. Fault Rates")
ax.yaxis.grid(True, zorder=0)
lo = max(0, plot_df["best_accuracy"].min()*100 - 5)
ax.set_ylim(lo, min(100, plot_df["best_accuracy"].max()*100 + 7))
for bar, val in zip(bars, plot_df["best_accuracy"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8, color="black", rotation=45)
fig.tight_layout()
fig.savefig(PLOT_DIR/"01_accuracy_by_config.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✅  Plot 1 saved → 01_accuracy_by_config.png")

# Plot 2 – Convergence Rounds bar
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(labels, plot_df["convergence_rounds"],
              color=colors, edgecolor="black", linewidth=0.8, zorder=3)
for bar, h, c in zip(bars, hatches, colors):
    bar.set_hatch(h); bar.set_facecolor(c)
ax.set_xlabel("Configuration (Device Fail : Zone Fail)")
ax.set_ylabel("Convergence Rounds")
ax.set_title("Convergence Speed vs. Fault Rates (Lower is Better)")
ax.yaxis.grid(True, zorder=0)
for bar, val in zip(bars, plot_df["convergence_rounds"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{int(val)}", ha="center", va="bottom", fontsize=8, color="black")
fig.tight_layout()
fig.savefig(PLOT_DIR/"02_convergence_by_config.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✅  Plot 2 saved → 02_convergence_by_config.png")

# Plot 3 – Learning curves
fig, ax = plt.subplots(figsize=(8, 4.5))
plotted = 0
for i, (_, r) in enumerate(plot_df.iterrows()):
    eid = r["exp_id"]
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
            label=r["config_label"])
    plotted += 1
if plotted:
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.set_title("Learning Curves — Fault Configurations")
    ax.yaxis.grid(True)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR/"03_learning_curves.png", dpi=300, bbox_inches="tight")
    print("✅  Plot 3 saved → 03_learning_curves.png")
else:
    print("⚠️   No per-round data; skipping Plot 3.")
plt.close(fig)

# Plot 4 – Multi-metric normalised bars
keys   = ["_n_acc","_n_f1","_n_conv","_n_comm"]
mlbls  = ["Best Acc\n(norm)","F1\n(norm)","Conv Speed\n(norm)","Comm Eff\n(norm)"]
x      = np.arange(len(keys))
width  = 0.8 / max(n, 1)
fig, ax = plt.subplots(figsize=(8, 4))
for i, (_, r) in enumerate(plot_df.iterrows()):
    offset = (i - n/2)*width + width/2
    ax.bar(x+offset, [r[k] for k in keys], width*0.92,
           color=colors[i], hatch=IEEE_HATCHES[i % len(IEEE_HATCHES)],
           label=r["config_label"],
           edgecolor="black", linewidth=0.5, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(mlbls)
ax.set_ylabel("Normalised Score"); ax.set_ylim(0,1.25)
ax.set_title("Multi-Metric Comparison — Fault Configurations")
ax.yaxis.grid(True, zorder=0)
ax.legend(loc="upper right", fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(PLOT_DIR/"04_multi_metric_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✅  Plot 4 saved → 04_multi_metric_bar.png")

# Plot 5 – Composite score ranking (horizontal bar)
rank_df = df.sort_values("composite_score", ascending=True)
rlabels = rank_df["config_label"].tolist()
rscores = rank_df["composite_score"].values
rcolors = [PALETTE[i % len(PALETTE)] for i in range(len(rank_df))]
rhatch  = [IEEE_HATCHES[i % len(IEEE_HATCHES)] for i in range(len(rank_df))]
fig, ax = plt.subplots(figsize=(7, max(3, len(rank_df)*0.55)))
bars = ax.barh(rlabels, rscores, color=rcolors, edgecolor="black", linewidth=0.7, zorder=3)
for bar, h in zip(bars, rhatch):
    bar.set_hatch(h)
ax.set_xlabel("Composite Score (higher = better)")
ax.set_title("Overall Ranking — Composite Score\n(40% Acc + 20% F1 + 20% Conv + 20% Comm)")
ax.xaxis.grid(True, zorder=0); ax.set_xlim(0, 1.15)
for bar, val in zip(bars, rscores):
    ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8, color="black")
fig.tight_layout()
fig.savefig(PLOT_DIR/"05_composite_ranking.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("✅  Plot 5 saved → 05_composite_ranking.png")

# ── 7. Recommendation ─────────────────────────────────────────────────────────
best     = df.iloc[0]
acc_best = df.loc[df["best_accuracy"].idxmax()]

rec = f"""
╔══════════════════════════════════════════════════════════════════╗
║   ContinuumFL — Fault Resilience Best Setup (FEMNIST)           ║
╚══════════════════════════════════════════════════════════════════╝

🏆  BEST CONFIGURATION  (composite score winner):
    device fail prob  : {best['dev_fail']:.2f}
    zone fail prob    : {best['zone_fail']:.2f}

📊  PERFORMANCE METRICS:
    Best accuracy     : {best['best_accuracy']*100:.2f}%  (round {int(best['best_accuracy_round'])})
    Final accuracy    : {best['final_accuracy']*100:.2f}%
    Final F1-score    : {best['final_f1']:.4f}
    Convergence round : {int(best['convergence_rounds'])}
    Total comm. cost  : {best['total_comm_mb']:.1f} MB
    Avg round time    : {best['average_round_time']:.1f} s
    Composite score   : {best['composite_score']:.4f} / 1.000

🎯  SPECIAL WINNERS:
    Highest accuracy  : Dev: {acc_best['dev_fail']:.2f} Zone: {acc_best['zone_fail']:.2f}  ({acc_best['best_accuracy']*100:.2f}%)

📈  FULL RANKING:
"""
for _, r in df.iterrows():
    mk = "🥇" if r["rank"]==1 else ("🥈" if r["rank"]==2 else ("🥉" if r["rank"]==3 else "  "))
    rec += (f"  {mk} #{int(r['rank']):<2}  "
            f"Dev: {r['dev_fail']:>4.2f} Zone: {r['zone_fail']:>4.2f}  "
            f"BestAcc={r['best_accuracy']*100:5.2f}%  "
            f"F1={r['final_f1']:.3f}  "
            f"ConvRnd={int(r['convergence_rounds']):<4}  "
            f"CommMB={r['total_comm_mb']:>7.0f}  "
            f"Score={r['composite_score']:.3f}\n")

rec += f"""
💡  INTERPRETATION:
    • Fault-free baseline usually gives the best accuracy and highest score.
    • Device failures (e.g. dropouts) reduce the amount of data contributed each round.
    • Zone failures (e.g. aggregator failure) lose the combined updates of multiple devices.
    • ContinuumFL's resilience is demonstrated by how well the accuracy is maintained
      despite these dropouts, compared to the fault-free baseline.

Plots : {PLOT_DIR}
Table : {OUT_DIR / 'comparison_table.csv'}
"""

print(rec)
(OUT_DIR / "recommendation.txt").write_text(rec)
print(f"✅  Recommendation saved → {OUT_DIR / 'recommendation.txt'}")
print(f"\n🎉  Analysis complete. Results in: {OUT_DIR}\n")
