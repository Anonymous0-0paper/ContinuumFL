"""
ieee_plot_style.py
==================
Shared IEEE publication-quality plot style for ContinuumFL comparison scripts.
- White background, Times New Roman serif font, 300 DPI
- Colorblind-safe palette + hatch patterns (distinguishable in B&W print)
- Distinct line styles and markers for curves
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── IEEE colorblind-safe palette (8 distinct colors) ─────────────────────────
IEEE_COLORS = [
    '#0072BD',   # blue
    '#D95319',   # burnt orange
    '#EDB120',   # mustard yellow
    '#7E2F8E',   # purple
    '#77AC30',   # green
    '#4DBEEE',   # sky blue
    '#A2142F',   # dark red
    '#000000',   # black
    '#808080',   # gray
    '#FF69B4',   # pink
]

# ── Hatch patterns (for B&W print distinguishability) ────────────────────────
IEEE_HATCHES = ['/', '\\\\', 'x', 'o', '.', '*', '+', 'O', '|', '-']

# ── Line styles ───────────────────────────────────────────────────────────────
IEEE_LINESTYLES = [
    '-',
    '--',
    '-.',
    ':',
    (0, (3, 1, 1, 1)),    # dashdotdot
    (0, (5, 1)),           # long dash
    (0, (1, 2)),           # dotted loose
    (0, (3, 5, 1, 5)),     # dashdotted loose
]

# ── Markers ───────────────────────────────────────────────────────────────────
IEEE_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p']


def setup_ieee_style():
    """Apply IEEE-compliant matplotlib rcParams (call once at script start)."""
    mpl.rcParams.update({
        # ── Font ──────────────────────────────────────────────────────────────
        'font.family':            'serif',
        'font.serif':             ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size':              10,
        'axes.labelsize':         10,
        'axes.titlesize':         10,
        'axes.titleweight':       'bold',
        'xtick.labelsize':        9,
        'ytick.labelsize':        9,
        'legend.fontsize':        8,
        'legend.title_fontsize':  8,
        # ── Background ────────────────────────────────────────────────────────
        'figure.facecolor':       'white',
        'axes.facecolor':         'white',
        'savefig.facecolor':      'white',
        # ── DPI ───────────────────────────────────────────────────────────────
        'figure.dpi':             150,
        'savefig.dpi':            300,
        # ── Axes ──────────────────────────────────────────────────────────────
        'axes.edgecolor':         'black',
        'axes.labelcolor':        'black',
        'axes.linewidth':         0.8,
        'axes.spines.top':        False,
        'axes.spines.right':      False,
        # ── Ticks ─────────────────────────────────────────────────────────────
        'xtick.color':            'black',
        'ytick.color':            'black',
        'xtick.major.width':      0.8,
        'ytick.major.width':      0.8,
        'xtick.direction':        'out',
        'ytick.direction':        'out',
        # ── Grid ──────────────────────────────────────────────────────────────
        'axes.grid':              True,
        'grid.color':             '#CCCCCC',
        'grid.linestyle':         '--',
        'grid.linewidth':         0.5,
        'grid.alpha':             0.7,
        # ── Lines / patches ───────────────────────────────────────────────────
        'lines.linewidth':        1.5,
        'lines.markersize':       5,
        'patch.linewidth':        0.8,
        # ── Text ──────────────────────────────────────────────────────────────
        'text.color':             'black',
        # ── Legend ────────────────────────────────────────────────────────────
        'legend.facecolor':       'white',
        'legend.edgecolor':       'black',
        'legend.framealpha':      0.9,
        'legend.borderpad':       0.4,
    })
