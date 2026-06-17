"""
IEEE-quality data distribution figures for ContinuumFL.

Generates six figure types per dataset:
  1. stacked_bar   — zone & client stacked class bars  (overview)
  2. heatmap       — normalised class-proportion heatmap (zones × classes)
  3. jsd_matrix    — Jensen-Shannon divergence between every pair of zones
  4. entropy       — Shannon entropy per zone + per-client violin, coloured by zone
  5. lorenz        — Lorenz curve of sample counts across clients (Gini coefficient)
  6. combined      — single 2×3 panel for paper inclusion

Dataset parameters (from continuumfl_common.sh / run_baselines_sweep.sh):
  CIFAR-100 / FEMNIST / UCI-HAR / SpeechCommands : 100 devices, 20 zones, intra α=100, inter α=10
  Shakespeare                                     :  35 devices,  7 zones, intra α=10,  inter α=0.3

Usage (from repo root):
    python scripts/visualize_data_distribution.py
    python scripts/visualize_data_distribution.py --datasets cifar100 shakespeare
    python scripts/visualize_data_distribution.py --figures heatmap jsd_matrix lorenz
    python scripts/visualize_data_distribution.py --out_dir figures/dist --dpi 300
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── IEEE rcParams ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        150,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "lines.linewidth":   1.2,
    "patch.linewidth":   0.4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Dataset run parameters ────────────────────────────────────────────────────
DATASET_PARAMS = {
    "cifar100": {
        "label": "CIFAR-100", "num_classes": 100,
        "num_devices": 50, "num_zones": 5,
        "intra_alpha": 100, "inter_alpha": 5, "max_samples": 50000,
    },
    "femnist": {
        "label": "FEMNIST", "num_classes": 62,
        "num_devices": 50, "num_zones": 5,
        "intra_alpha": 100, "inter_alpha": 5, "max_samples": 60000,
    },
    "ucihar": {
        "label": "UCI-HAR", "num_classes": 6,
        "num_devices": 50, "num_zones": 5,
        "intra_alpha": 100, "inter_alpha": 5, "max_samples": 7352,
    },
    "speechcommands": {
        "label": "SpeechCommands", "num_classes": 35,
        "num_devices": 50, "num_zones": 5,
        "intra_alpha": 100, "inter_alpha": 5, "max_samples": 84843,
    },
    "shakespeare": {
        "label": "Shakespeare", "num_classes": 79,
        "num_devices": 50, "num_zones": 5,
        "intra_alpha": 100, "inter_alpha": 5, "max_samples": 16068,
    },
}

UCIHAR_CLASSES = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
CIFAR100_SUPER = [
    "aquatic", "fish", "flowers", "food", "fruit/veg", "elec.", "furn.",
    "insects", "lg.carni.", "lg.man-m.", "lg.nat.", "lg.omni.", "med.mamm.",
    "non-ins.", "people", "reptiles", "sm.mamm.", "trees", "veh.1", "veh.2",
]

ZONE_PALETTE = [
    "#4C72B0","#DD8452","#55A868","#C44E52","#8172B3",
    "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD",
    "#E377C2","#7F7F7F","#BCBD22","#17BECF","#AEC7E8",
    "#FFBB78","#98DF8A","#FF9896","#C5B0D5","#C49C94",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data generation (mirrors FederatedDataset.distribute_data_to_devices)
# ─────────────────────────────────────────────────────────────────────────────

def make_zones(num_devices, num_zones):
    base, extra = divmod(num_devices, num_zones)
    return {
        f"zone_{z}": [f"client_{z}_{c}"
                      for c in range(base + (1 if z < extra else 0))]
        for z in range(num_zones)
    }


def generate_distribution(num_classes, zones, inter_alpha, intra_alpha,
                           n_samples, seed=42):
    rng = np.random.default_rng(seed)
    zone_ids  = list(zones.keys())
    num_zones = len(zone_ids)

    lo = max(1, n_samples // (num_classes * 3))
    hi = max(lo + 1, n_samples // num_classes)
    class_pool = rng.integers(lo, hi, size=num_classes)

    zone_cc   = {z: np.zeros(num_classes, dtype=int) for z in zone_ids}
    client_cc = {}
    zone_of   = {}

    for c in range(num_classes):
        total = int(class_pool[c])
        if total == 0:
            continue
        z_props = rng.dirichlet([inter_alpha] * num_zones)
        counts  = {z: int(z_props[i] * total) for i, z in enumerate(zone_ids)}
        diff    = total - sum(counts.values())
        counts[zone_ids[int(np.argmax(z_props))]] += diff

        for z, devs in zones.items():
            zt = counts[z]
            zone_cc[z][c] += zt
            nd      = len(devs)
            d_props = rng.dirichlet([intra_alpha] * nd)
            dc      = {d: int(d_props[i] * zt) for i, d in enumerate(devs)}
            diff2   = zt - sum(dc.values())
            dc[devs[int(np.argmax(d_props))]] += diff2
            for d in devs:
                if d not in client_cc:
                    client_cc[d] = np.zeros(num_classes, dtype=int)
                    zone_of[d]   = z
                client_cc[d][c] += dc[d]

    return zone_cc, client_cc, zone_of


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _class_cmap(num_classes):
    cmap = plt.get_cmap("tab20" if num_classes <= 20 else "nipy_spectral")
    return [cmap(i / num_classes) for i in range(num_classes)]


def _entropy(counts):
    p = counts / (counts.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _jsd(p, q):
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / (b[mask] + 1e-12))))
    return float(np.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def _jsd_matrix(zone_cc, zone_ids):
    n = len(zone_ids)
    M = np.zeros((n, n))
    for i, zi in enumerate(zone_ids):
        for j, zj in enumerate(zone_ids):
            if i < j:
                pi = zone_cc[zi].astype(float)
                pj = zone_cc[zj].astype(float)
                v  = _jsd(pi, pj)
                M[i, j] = M[j, i] = v
    return M


def _lorenz(values):
    vals = np.sort(np.array(values, dtype=float))
    cum  = np.cumsum(vals)
    cum  = np.insert(cum, 0, 0)
    n    = len(vals)
    x    = np.linspace(0, 1, n + 1)
    y    = cum / (cum[-1] + 1e-12)
    gini = 1 - 2 * np.trapz(y, x)
    return x, y, gini


def _save(fig, path, dpi):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Stacked bar (zone + client overview)
# ─────────────────────────────────────────────────────────────────────────────

def fig_stacked_bar(params, zone_cc, client_cc, zones, class_labels, out_path, dpi):
    nc       = params["num_classes"]
    zone_ids = list(zones.keys())
    colors   = _class_cmap(nc)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))   # IEEE double-column width
    fig.suptitle(
        f"{params['label']}  —  non-IID data distribution  "
        f"($\\alpha_{{\\mathrm{{intra}}}}$={params['intra_alpha']}, "
        f"$\\alpha_{{\\mathrm{{inter}}}}$={params['inter_alpha']})",
        fontsize=9, fontweight="bold",
    )

    # zone bars
    ax = axes[0]
    bot = np.zeros(len(zone_ids))
    for c in range(nc):
        v = np.array([zone_cc[z][c] for z in zone_ids], dtype=float)
        ax.bar(range(len(zone_ids)), v, bottom=bot, color=colors[c],
               width=0.75, linewidth=0)
        bot += v
    ax.set_xticks(range(len(zone_ids)))
    ax.set_xticklabels([f"Z{i+1}" for i in range(len(zone_ids))],
                       fontsize=5 if len(zone_ids) > 12 else 7, rotation=45)
    ax.set_xlabel("Zone")
    ax.set_ylabel("No. of samples")
    ax.set_title("(a) Zone-level class distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    if nc <= 20:
        patches = [mpatches.Patch(color=colors[c], label=class_labels[c])
                   for c in range(nc)]
        ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=5, ncol=1, frameon=False)

    # client bars
    ax = axes[1]
    x_pos, x_labs, z_mids, z_bounds = [], [], [], []
    cur = 0
    for zi, z in enumerate(zone_ids):
        s = cur
        for _ in zones[z]:
            x_pos.append(cur); x_labs.append(""); cur += 1
        z_mids.append((s + cur - 1) / 2)
        z_bounds.append(cur - 0.5)
        cur += 1

    bot = np.zeros(len(x_pos))
    for c in range(nc):
        v = np.array([client_cc[d][c] for z in zone_ids for d in zones[z]], dtype=float)
        ax.bar(x_pos, v, bottom=bot, color=colors[c], width=0.9, linewidth=0)
        bot += v

    for xb in z_bounds[:-1]:
        ax.axvline(xb, color="black", lw=0.5, ls="--", alpha=0.35)

    # coloured zone strip at very bottom
    ax.set_xlim(min(x_pos) - 1, max(x_pos) + 1)
    ylim = ax.get_ylim()
    strip = (ylim[1] - ylim[0]) * 0.025
    off = 0
    for zi, z in enumerate(zone_ids):
        nd = len(zones[z])
        ax.broken_barh([(x_pos[off], nd - 0.1)], (ylim[0], strip),
                       facecolors=ZONE_PALETTE[zi % len(ZONE_PALETTE)], alpha=0.85)
        off += nd

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(z_mids)
    ax2.set_xticklabels([f"Z{i+1}" for i in range(len(zone_ids))],
                        fontsize=5 if len(zone_ids) > 12 else 7)
    ax2.tick_params(length=0)

    ax.set_xticks([])
    ax.set_xlabel("Clients  (zone colour strip below)")
    ax.set_ylabel("No. of samples")
    ax.set_title("(b) Client-level class distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    fig.tight_layout()
    _save(fig, out_path, dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Normalised class-proportion heatmap (zones × classes)
# ─────────────────────────────────────────────────────────────────────────────

def fig_heatmap(params, zone_cc, zones, class_labels, out_path, dpi):
    nc       = params["num_classes"]
    zone_ids = list(zones.keys())
    nz       = len(zone_ids)

    # matrix: rows=zones, cols=classes, values=fraction of zone's samples
    mat = np.array([
        zone_cc[z].astype(float) / (zone_cc[z].sum() + 1e-12)
        for z in zone_ids
    ])

    # collapse classes for readability when nc > 30
    if nc > 30:
        bucket = max(1, nc // 20)
        cols   = [mat[:, i:i+bucket].sum(axis=1) for i in range(0, nc, bucket)]
        mat    = np.column_stack(cols)
        col_labels = [f"{i}–{min(i+bucket-1, nc-1)}" for i in range(0, nc, bucket)]
    else:
        col_labels = class_labels

    fig_w = max(5.0, len(col_labels) * 0.28)
    fig_h = max(2.5, nz * 0.28)
    fig, ax = plt.subplots(figsize=(min(fig_w, 7.16), min(fig_h, 5.0)))

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fraction of zone samples", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=60, ha="right",
                       fontsize=5 if len(col_labels) > 15 else 7)
    ax.set_yticks(range(nz))
    ax.set_yticklabels([f"Zone {i+1}" for i in range(nz)], fontsize=6)
    ax.set_xlabel("Class")
    ax.set_ylabel("Zone")
    ax.set_title(
        f"{params['label']}  —  normalised class distribution per zone\n"
        f"($\\alpha_{{\\mathrm{{inter}}}}$={params['inter_alpha']})",
        fontsize=9,
    )

    fig.tight_layout()
    _save(fig, out_path, dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Jensen-Shannon divergence matrix (zones × zones)
# ─────────────────────────────────────────────────────────────────────────────

def fig_jsd_matrix(params, zone_cc, zones, out_path, dpi):
    zone_ids = list(zones.keys())
    nz       = len(zone_ids)
    M        = _jsd_matrix(zone_cc, zone_ids)

    fig, ax = plt.subplots(figsize=(min(0.38 * nz + 1.2, 6.0),
                                    min(0.38 * nz + 1.0, 5.5)))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() + 0.02)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jensen-Shannon divergence", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ticks = range(nz)
    labels = [f"Z{i+1}" for i in range(nz)]
    ax.set_xticks(ticks); ax.set_xticklabels(labels,
                                               fontsize=5 if nz > 12 else 7,
                                               rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(labels,
                                               fontsize=5 if nz > 12 else 7)

    # annotate cells only when matrix is small
    if nz <= 12:
        for i in range(nz):
            for j in range(nz):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if M[i,j] > 0.4 * M.max() else "black")

    mean_jsd = M[np.triu_indices(nz, k=1)].mean()
    ax.set_title(
        f"{params['label']}  —  inter-zone JS divergence\n"
        f"mean JSD = {mean_jsd:.3f}  "
        f"($\\alpha_{{\\mathrm{{inter}}}}$={params['inter_alpha']})",
        fontsize=9,
    )
    ax.set_xlabel("Zone"); ax.set_ylabel("Zone")
    fig.tight_layout()
    _save(fig, out_path, dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Shannon entropy: per-zone bar + per-client violin
# ─────────────────────────────────────────────────────────────────────────────

def fig_entropy(params, zone_cc, client_cc, zones, out_path, dpi):
    zone_ids    = list(zones.keys())
    nz          = len(zone_ids)
    zone_H      = [_entropy(zone_cc[z]) for z in zone_ids]
    max_H       = np.log2(params["num_classes"])

    # per-client entropy grouped by zone
    zone_client_H = {z: [_entropy(client_cc[d]) for d in zones[z]] for z in zone_ids}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # left: zone entropy bar
    ax = axes[0]
    ax.bar(range(nz), zone_H, color=[ZONE_PALETTE[i % len(ZONE_PALETTE)]
                                             for i in range(nz)],
                  width=0.65, edgecolor="white", linewidth=0.4)
    ax.axhline(max_H, color="gray", ls="--", lw=0.8, label=f"Max H = {max_H:.2f}")
    ax.set_xticks(range(nz))
    ax.set_xticklabels([f"Z{i+1}" for i in range(nz)],
                       fontsize=5 if nz > 12 else 7, rotation=45)
    ax.set_xlabel("Zone")
    ax.set_ylabel("Shannon entropy  (bits)")
    ax.set_ylim(0, max_H * 1.12)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("(a) Zone-level class entropy")
    ax.yaxis.grid(True, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # right: per-client entropy violin per zone
    ax = axes[1]
    data   = [zone_client_H[z] for z in zone_ids]
    parts  = ax.violinplot(data, positions=range(nz), widths=0.7,
                           showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(ZONE_PALETTE[i % len(ZONE_PALETTE)])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.0)
    for key in ("cmaxes", "cmins", "cbars"):
        parts[key].set_linewidth(0.6)
        parts[key].set_color("dimgray")

    ax.axhline(max_H, color="gray", ls="--", lw=0.8)
    ax.set_xticks(range(nz))
    ax.set_xticklabels([f"Z{i+1}" for i in range(nz)],
                       fontsize=5 if nz > 12 else 7, rotation=45)
    ax.set_xlabel("Zone")
    ax.set_ylabel("Shannon entropy  (bits)")
    ax.set_ylim(0, max_H * 1.12)
    ax.set_title("(b) Per-client class entropy by zone")
    ax.yaxis.grid(True, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"{params['label']}  —  class entropy  "
        f"($\\alpha_{{\\mathrm{{intra}}}}$={params['intra_alpha']}, "
        f"$\\alpha_{{\\mathrm{{inter}}}}$={params['inter_alpha']})",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, out_path, dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Lorenz curve + Gini coefficient (data imbalance across clients)
# ─────────────────────────────────────────────────────────────────────────────

def fig_lorenz(params, client_cc, zones, out_path, dpi):
    zone_ids = list(zones.keys())

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # overall Lorenz curve
    all_counts = np.array([client_cc[d].sum() for z in zone_ids for d in zones[z]])
    x, y, gini = _lorenz(all_counts)
    ax.plot(x, y, color="#4C72B0", lw=1.5, label=f"All clients (Gini={gini:.3f})")

    # per-zone Lorenz curves (thinner)
    for zi, z in enumerate(zone_ids):
        zc = np.array([client_cc[d].sum() for d in zones[z]])
        if len(zc) < 2:
            continue
        xz, yz, gz = _lorenz(zc)
        ax.plot(xz, yz, color=ZONE_PALETTE[zi % len(ZONE_PALETTE)],
                lw=0.7, alpha=0.55, label=f"Z{zi+1} (G={gz:.2f})")

    # equality line
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect equality")
    ax.fill_between(x, x, y, alpha=0.08, color="#4C72B0")

    ax.set_xlabel("Cumulative fraction of clients")
    ax.set_ylabel("Cumulative fraction of samples")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # legend: only overall + equality + up to 4 zones to avoid clutter
    handles, labels_ = ax.get_legend_handles_labels()
    keep = [0, -1] + list(range(1, min(len(handles)-1, 5)))
    ax.legend([handles[i] for i in keep], [labels_[i] for i in keep],
              fontsize=6, frameon=False, loc="upper left")

    ax.set_title(
        f"{params['label']}  —  Lorenz curve of client data sizes\n"
        f"Overall Gini = {gini:.3f}",
        fontsize=9,
    )
    ax.yaxis.grid(True, lw=0.4, alpha=0.5)
    ax.xaxis.grid(True, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_path, dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Combined 2×3 panel (paper-ready single figure)
# ─────────────────────────────────────────────────────────────────────────────

def fig_combined(params, zone_cc, client_cc, zones, class_labels, out_path, dpi):
    nc       = params["num_classes"]
    zone_ids = list(zones.keys())
    nz       = len(zone_ids)
    colors   = _class_cmap(nc)

    fig = plt.figure(figsize=(7.16, 8.5))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

    # ── (A) zone stacked bar ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    bot = np.zeros(nz)
    for c in range(nc):
        v = np.array([zone_cc[z][c] for z in zone_ids], dtype=float)
        ax.bar(range(nz), v, bottom=bot, color=colors[c], width=0.75, linewidth=0)
        bot += v
    ax.set_xticks(range(nz))
    ax.set_xticklabels([f"Z{i+1}" for i in range(nz)],
                       fontsize=4 if nz > 12 else 6, rotation=45)
    ax.set_xlabel("Zone"); ax.set_ylabel("Samples")
    ax.set_title("(A) Zone class distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    # ── (B) client stacked bar ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    x_pos, z_mids, z_bounds = [], [], []
    cur = 0
    for zi, z in enumerate(zone_ids):
        s = cur
        for _ in zones[z]: x_pos.append(cur); cur += 1
        z_mids.append((s + cur - 1) / 2)
        z_bounds.append(cur - 0.5); cur += 1
    bot = np.zeros(len(x_pos))
    for c in range(nc):
        v = np.array([client_cc[d][c] for z in zone_ids for d in zones[z]], dtype=float)
        ax.bar(x_pos, v, bottom=bot, color=colors[c], width=0.9, linewidth=0)
        bot += v
    for xb in z_bounds[:-1]:
        ax.axvline(xb, color="black", lw=0.4, ls="--", alpha=0.3)
    ylim = ax.get_ylim(); strip = (ylim[1]-ylim[0])*0.03; off = 0
    for zi, z in enumerate(zone_ids):
        nd = len(zones[z])
        ax.broken_barh([(x_pos[off], nd-0.1)], (ylim[0], strip),
                       facecolors=ZONE_PALETTE[zi % len(ZONE_PALETTE)], alpha=0.9)
        off += nd
    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(z_mids)
    ax2.set_xticklabels([f"Z{i+1}" for i in range(nz)],
                        fontsize=4 if nz > 12 else 6)
    ax2.tick_params(length=0)
    ax.set_xticks([]); ax.set_xlabel("Clients"); ax.set_ylabel("Samples")
    ax.set_title("(B) Client class distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    # ── (C) normalised heatmap ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    mat = np.array([zone_cc[z].astype(float) / (zone_cc[z].sum()+1e-12)
                    for z in zone_ids])
    if nc > 25:
        bucket = max(1, nc // 20)
        mat = np.column_stack([mat[:, i:i+bucket].sum(axis=1)
                               for i in range(0, nc, bucket)])
        hl  = [f"{i}" for i in range(0, nc, bucket)]
    else:
        hl = class_labels
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Fraction", fontsize=6); cb.ax.tick_params(labelsize=5)
    ax.set_xticks(range(len(hl)))
    ax.set_xticklabels(hl, rotation=60, ha="right",
                       fontsize=4 if len(hl) > 15 else 6)
    ax.set_yticks(range(nz))
    ax.set_yticklabels([f"Z{i+1}" for i in range(nz)], fontsize=5)
    ax.set_title("(C) Normalised class heatmap")
    ax.set_xlabel("Class"); ax.set_ylabel("Zone")

    # ── (D) JSD matrix ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    M  = _jsd_matrix(zone_cc, zone_ids)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 0.01))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("JSD", fontsize=6); cb.ax.tick_params(labelsize=5)
    tl = [f"Z{i+1}" for i in range(nz)]
    fs = 4 if nz > 12 else 6
    ax.set_xticks(range(nz)); ax.set_xticklabels(tl, fontsize=fs, rotation=45, ha="right")
    ax.set_yticks(range(nz)); ax.set_yticklabels(tl, fontsize=fs)
    mean_jsd = M[np.triu_indices(nz, k=1)].mean()
    ax.set_title(f"(D) Inter-zone JSD  (mean={mean_jsd:.3f})")
    ax.set_xlabel("Zone"); ax.set_ylabel("Zone")

    # ── (E) entropy violin ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    data  = [[_entropy(client_cc[d]) for d in zones[z]] for z in zone_ids]
    parts = ax.violinplot(data, positions=range(nz), widths=0.65,
                          showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(ZONE_PALETTE[i % len(ZONE_PALETTE)]); pc.set_alpha(0.75)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.0)
    for k in ("cmaxes","cmins","cbars"):
        parts[k].set_linewidth(0.5); parts[k].set_color("dimgray")
    max_H = np.log2(nc)
    ax.axhline(max_H, color="gray", ls="--", lw=0.7, label=f"Max H={max_H:.1f} bits")
    ax.set_xticks(range(nz))
    ax.set_xticklabels([f"Z{i+1}" for i in range(nz)],
                       fontsize=4 if nz > 12 else 6, rotation=45)
    ax.set_xlabel("Zone"); ax.set_ylabel("H(class)  [bits]")
    ax.set_ylim(0, max_H * 1.12)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("(E) Per-client entropy by zone")
    ax.yaxis.grid(True, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    # ── (F) Lorenz curve ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    all_counts = np.array([client_cc[d].sum()
                            for z in zone_ids for d in zones[z]])
    x, y, gini = _lorenz(all_counts)
    ax.fill_between(x, x, y, alpha=0.1, color="#4C72B0")
    ax.plot(x, y, color="#4C72B0", lw=1.5, label=f"All clients\n(Gini={gini:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=0.7, label="Equal dist.")
    for zi, z in enumerate(zone_ids[:min(nz, 5)]):
        zc = np.array([client_cc[d].sum() for d in zones[z]])
        if len(zc) < 2: continue
        xz,yz,gz = _lorenz(zc)
        ax.plot(xz,yz, color=ZONE_PALETTE[zi % len(ZONE_PALETTE)],
                lw=0.7, alpha=0.6, label=f"Z{zi+1} (G={gz:.2f})")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    ax.set_xlabel("Cumul. fraction of clients")
    ax.set_ylabel("Cumul. fraction of samples")
    ax.set_title(f"(F) Lorenz curve  (Gini={gini:.3f})")
    ax.legend(fontsize=5, frameon=False, loc="upper left")
    ax.yaxis.grid(True,lw=0.4,alpha=0.5); ax.xaxis.grid(True,lw=0.4,alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"{params['label']}  —  data heterogeneity overview  "
        f"($\\alpha_{{\\mathrm{{intra}}}}$={params['intra_alpha']}, "
        f"$\\alpha_{{\\mathrm{{inter}}}}$={params['inter_alpha']}, "
        f"{params['num_devices']} clients, {params['num_zones']} zones)",
        fontsize=9, fontweight="bold", y=1.002,
    )
    _save(fig, out_path, dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(params, zone_cc, client_cc, zones):
    zone_ids = list(zones.keys())
    M        = _jsd_matrix(zone_cc, zone_ids)
    nz       = len(zone_ids)
    mean_jsd = M[np.triu_indices(nz, k=1)].mean() if nz > 1 else 0.0

    all_counts = np.array([client_cc[d].sum()
                            for z in zone_ids for d in zones[z]])
    _, _, gini = _lorenz(all_counts)

    print(f"\n{'='*64}")
    print(f"  {params['label']:20s}  devices={params['num_devices']}  "
          f"zones={params['num_zones']}  "
          f"intra_α={params['intra_alpha']}  inter_α={params['inter_alpha']}")
    print(f"  mean inter-zone JSD = {mean_jsd:.4f}    Gini = {gini:.4f}")
    print(f"{'='*64}")
    print(f"  {'Zone':<12} {'samples':>8}  {'H(bits)':>8}  top-3 classes")
    print(f"  {'-'*52}")
    for z in zone_ids:
        cc   = zone_cc[z]
        top3 = np.argsort(cc)[::-1][:3].tolist()
        print(f"  {z:<12} {cc.sum():>8,}  {_entropy(cc):>8.3f}  {top3}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_FIGURES = ["stacked_bar", "heatmap", "jsd_matrix", "entropy", "lorenz", "combined"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASET_PARAMS.keys()))
    p.add_argument("--figures",  nargs="+", default=ALL_FIGURES,
                   help=f"Which figures to generate: {ALL_FIGURES}")
    p.add_argument("--out_dir",  default="figures/data_distribution")
    p.add_argument("--dpi",      type=int, default=300)
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    fig_set = set(args.figures)

    for ds in args.datasets:
        ds = ds.lower()
        if ds not in DATASET_PARAMS:
            print(f"[WARN] Unknown dataset '{ds}'. Choose from: {list(DATASET_PARAMS.keys())}")
            continue

        params = DATASET_PARAMS[ds]
        print(f"\n[{params['label']}]  {params['num_devices']} devices / "
              f"{params['num_zones']} zones  "
              f"intra_α={params['intra_alpha']}  inter_α={params['inter_alpha']}")

        zones = make_zones(params["num_devices"], params["num_zones"])
        zone_cc, client_cc, zone_of = generate_distribution(
            num_classes = params["num_classes"],
            zones       = zones,
            inter_alpha = params["inter_alpha"],
            intra_alpha = params["intra_alpha"],
            n_samples   = params["max_samples"],
            seed        = args.seed,
        )

        if ds == "ucihar":
            class_labels = UCIHAR_CLASSES
        elif ds == "cifar100":
            class_labels = CIFAR100_SUPER
        else:
            class_labels = [str(i) for i in range(params["num_classes"])]

        base = os.path.join(ROOT, args.out_dir, ds)

        if "stacked_bar"  in fig_set:
            fig_stacked_bar(params, zone_cc, client_cc, zones, class_labels,
                            f"{base}_stacked_bar.pdf", args.dpi)
        if "heatmap"      in fig_set:
            fig_heatmap(params, zone_cc, zones, class_labels,
                        f"{base}_heatmap.pdf", args.dpi)
        if "jsd_matrix"   in fig_set:
            fig_jsd_matrix(params, zone_cc, zones,
                           f"{base}_jsd_matrix.pdf", args.dpi)
        if "entropy"      in fig_set:
            fig_entropy(params, zone_cc, client_cc, zones,
                        f"{base}_entropy.pdf", args.dpi)
        if "lorenz"       in fig_set:
            fig_lorenz(params, client_cc, zones,
                       f"{base}_lorenz.pdf", args.dpi)
        if "combined"     in fig_set:
            fig_combined(params, zone_cc, client_cc, zones, class_labels,
                         f"{base}_combined.pdf", args.dpi)

        print_summary(params, zone_cc, client_cc, zones)

    print("\nDone.")
