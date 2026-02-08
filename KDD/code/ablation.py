"""
KDD 2026 - Ablation Study: CER vs # of Instances
==================================================
Generates 42 individual high-quality figures (one per dataset × stream).
Each figure shows 4 curves: OLIDH_SD (full) and 3 ablation variants (-I, -D, -H).

Usage:
    python plot_ablation.py
    python plot_ablation.py --data_dir ../data/ablation --save_dir ../figures/ablation

Output:
    ../figures/ablation/{dataset}_{stream}_ablation.pdf
    ../figures/ablation/{dataset}_{stream}_ablation.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator

try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
# 1. Global RC Settings (KDD Publication Quality)
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.pad': 3,
    'ytick.major.pad': 3,
})

# ──────────────────────────────────────────────────────────────
# 2. Style Definitions
# ──────────────────────────────────────────────────────────────
VARIANTS = ['OLIDH_SD-I', 'OLIDH_SD-D', 'OLIDH_SD-H', 'OLIDH_SD']
CSV_COLNAMES = VARIANTS + ['x']

COLORS = {
    'OLIDH_SD':   '#B01C1C',
    'OLIDH_SD-I': '#3D5A1E',
    'OLIDH_SD-D': '#0E4D8B',
    'OLIDH_SD-H': '#4A0072',
}

LINESTYLES = {
    'OLIDH_SD':   '-',
    'OLIDH_SD-I': '-',
    'OLIDH_SD-D': '-',
    'OLIDH_SD-H': '-',
}

LINEWIDTHS = {
    'OLIDH_SD':   2.0,
    'OLIDH_SD-I': 1.2,
    'OLIDH_SD-D': 1.2,
    'OLIDH_SD-H': 1.2,
}

LABELS = {
    'OLIDH_SD':   r'$\mathrm{OLIDH_{SD}}$',
    'OLIDH_SD-I': r'$\mathrm{OLIDH_{SD}\text{-}I}$',
    'OLIDH_SD-D': r'$\mathrm{OLIDH_{SD}\text{-}D}$',
    'OLIDH_SD-H': r'$\mathrm{OLIDH_{SD}\text{-}H}$',
}

PLOT_ORDER = ['OLIDH_SD-I', 'OLIDH_SD-D', 'OLIDH_SD-H', 'OLIDH_SD']

# ──────────────────────────────────────────────────────────────
# 3. Dataset and Stream Definitions
# ──────────────────────────────────────────────────────────────
DATASETS = [
    'australian', 'diabetes', 'german', 'ionosphere', 'lymphoma',
    'madelon', 'splice', 'sylva', 'wbc', 'wdbc',
    'synthetic_1', 'synthetic_2', 'synthetic_3', 'synthetic_4',
]

DISPLAY_NAMES = {
    'synthetic_1': 'S1', 'synthetic_2': 'S2',
    'synthetic_3': 'G1', 'synthetic_4': 'G2',
}

STREAMS = ['cap', 'tra', 'evo']

# ──────────────────────────────────────────────────────────────
# 4. Smart Even-Tick Generator
# ──────────────────────────────────────────────────────────────
def make_even_ticks(data_max, n_divisions=4):
    """向上微调到能被 n_divisions 整除的最近值，然后等分。
    E.g. data_max=760, n=4 → [0, 190, 380, 570, 760]
    E.g. data_max=761, n=4 → [0, 191, 382, 573, 764]
    E.g. data_max=0.48, n=4 → [0, 0.12, 0.24, 0.36, 0.48]
    E.g. data_max=0.47, n=4 → step rounds up to 0.12 → [0, 0.12, 0.24, 0.36, 0.48]
    """
    if data_max <= 0:
        return np.array([0, 1]), 1

    raw_step = data_max / n_divisions

    if data_max > 100:
        # Integer domain: ceil step to integer
        step = int(np.ceil(raw_step))
    elif data_max > 1:
        # Round step up to 1 decimal
        step = np.ceil(raw_step * 10) / 10
    else:
        # Round step up to 2 decimals
        step = np.ceil(raw_step * 100) / 100

    tick_max = step * n_divisions
    ticks = np.linspace(0, tick_max, n_divisions + 1)
    ticks = np.round(ticks, 10)

    return ticks, tick_max

# ──────────────────────────────────────────────────────────────
# 5. Data Loading
# ──────────────────────────────────────────────────────────────
def load_ablation_data(csv_path):
    """Load ablation CSV: no header, 5 cols = [-I, -D, -H, OLIDH_SD, instance_id]."""
    try:
        df_raw = pd.read_csv(
            csv_path, header=None, encoding='utf-8',
            on_bad_lines='skip', engine='python',
        )
        df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
        df_raw = df_raw.dropna().reset_index(drop=True)

        if df_raw.shape[1] == 5:
            df_raw.columns = CSV_COLNAMES
        elif df_raw.shape[1] == 4:
            df_raw.columns = VARIANTS
            df_raw['x'] = np.arange(1, len(df_raw) + 1, dtype=float)
        else:
            print(f"  [WARN] {csv_path} has {df_raw.shape[1]} cols, expected 4 or 5.")
            return None

        df = df_raw.copy()
        df['x'] = df['x'].astype(float)

        # Skip leading rows where ANY variant is exactly 0
        drop_count = 0
        for row_idx in range(len(df)):
            row_vals = df[VARIANTS].iloc[row_idx]
            if (row_vals == 0).any():
                drop_count += 1
            else:
                break
        if drop_count > 0:
            df = df.iloc[drop_count:].reset_index(drop=True)

        # Downsample to ~300 points
        if len(df) > 300:
            step = len(df) // 300
            idx = list(range(0, len(df), step))
            if idx[-1] != len(df) - 1:
                idx.append(len(df) - 1)
            df = df.iloc[idx].reset_index(drop=True)

        return df

    except Exception as e:
        print(f"  [WARN] Error reading {csv_path}: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# 6. Single Figure Plotting
# ──────────────────────────────────────────────────────────────
def plot_single_ablation(df, dataset, stream, save_dir):
    """Plot one ablation figure and save as PDF + PNG."""

    display_name = DISPLAY_NAMES.get(dataset, dataset)
    title_str = f'{display_name}-{stream}'

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    # ── Plot curves ──
    for variant in PLOT_ORDER:
        if variant not in df.columns:
            continue
        ax.plot(
            df['x'], df[variant],
            color=COLORS[variant],
            linestyle=LINESTYLES[variant],
            linewidth=LINEWIDTHS[variant],
            zorder=10 if variant == 'OLIDH_SD' else 5,
            label=LABELS[variant],
        )

    # ── Labels & title ──
    ax.set_xlabel('# of Instances', fontsize=10)
    ax.set_ylabel('CER', fontsize=10)
    ax.set_title(title_str, fontsize=10, pad=4)

    # ── Compute even ticks ──
    x_data_max = df['x'].max()
    y_data_max = df[VARIANTS].max().max()

    xticks, x_lim = make_even_ticks(x_data_max, n_divisions=4)
    yticks, y_lim = make_even_ticks(y_data_max, n_divisions=4)

    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # X-axis K notation for large values
    if x_lim > 5000:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f'{v/1000:.0f}K' if v > 0 else '0')
        )

    # ── Minor ticks (1 between each major → doubles grid density) ──
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ── Grid: major (darker) + minor (medium) ──
    ax.grid(True, which='major', linestyle=':', alpha=0.8, linewidth=0.6, color='#888888')
    ax.grid(True, which='minor', linestyle=':', alpha=0.5, linewidth=0.4, color='#AAAAAA')

    # ── All four spines (black border) ──
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # ── Ticks on all four sides ──
    ax.tick_params(which='both', top=True, right=True, direction='in')

    # ── Legend ──
    ax.legend(
        loc='best',
        frameon=False,
        fontsize=7.5,
        handlelength=2.0,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    fig.tight_layout(pad=0.3)

    # ── Save ──
    base_name = f'{dataset}_{stream}_ablation'
    pdf_path = os.path.join(save_dir, f'{base_name}.pdf')
    png_path = os.path.join(save_dir, f'{base_name}.png')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)

    return pdf_path, png_path


# ──────────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────────
def plot_all_ablation(data_dir, save_dir):
    """Generate all 42 ablation figures."""
    os.makedirs(save_dir, exist_ok=True)

    total, success, skipped = 0, 0, 0

    for dataset in DATASETS:
        for stream in STREAMS:
            total += 1
            csv_name = f'{dataset}_{stream}_ablation.csv'
            csv_path = os.path.join(data_dir, csv_name)

            if not os.path.isfile(csv_path):
                print(f"  [SKIP] {csv_name} not found.")
                skipped += 1
                continue

            df = load_ablation_data(csv_path)
            if df is None or len(df) < 2:
                print(f"  [SKIP] {csv_name}: insufficient data.")
                skipped += 1
                continue

            pdf_path, png_path = plot_single_ablation(df, dataset, stream, save_dir)
            print(f"  [OK] {csv_name} → {os.path.basename(png_path)}")
            success += 1

    print(f"\n[DONE] {success}/{total} figures generated, {skipped} skipped.")
    print(f"  Output: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'ablation'))
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'figures', 'ablation'))
    args = parser.parse_args()
    plot_all_ablation(args.data_dir, args.save_dir)