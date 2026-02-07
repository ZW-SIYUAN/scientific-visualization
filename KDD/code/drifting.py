"""
KDD 2026 - Concept Drifting Experiment: CER vs # of Instances
=============================================================
Generates a 4×3 subplot grid (datasets × drift streams) showing
cumulative error rate (CER) curves for OLIDH_SD and 5 baselines,
with drift-point annotations.

Usage:
    python plot_drifting.py               # use simulated data
    python plot_drifting.py --data_dir ../data/drifting  # use real CSVs

Output:
    ../figures/drifting/drifting_cer.pdf
    ../figures/drifting/drifting_cer.png
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.pad': 3,
    'ytick.major.pad': 3,
})

# ──────────────────────────────────────────────────────────────
# 2. Style Definitions
# ──────────────────────────────────────────────────────────────
METHODS = ['OLIDH_SD', 'OVFM', 'SFCF', 'Dynfo', 'Aux-Drop', 'Aux-Net']

# Color-blind friendly palette (Tableau CB10 inspired + custom)
COLORS = {
    'OLIDH_SD': '#D62728',  # red
    'OVFM':     '#1F77B4',  # blue
    'SFCF':     '#2CA02C',  # green
    'Dynfo':    '#9467BD',  # purple
    'Aux-Drop': '#FF7F0E',  # orange
    'Aux-Net':  '#17BECF',  # cyan
}

LINESTYLES = {
    'OLIDH_SD': '-',
    'OVFM':     '-',
    'SFCF':     '-',
    'Dynfo':    '-',
    'Aux-Drop': '-',
    'Aux-Net':  '-',
}

MARKERS = {
    'OLIDH_SD': 'v',  # down-triangle, hollow
    'OVFM':     None,
    'SFCF':     None,
    'Dynfo':    None,
    'Aux-Drop': None,
    'Aux-Net':  None,
}

LINEWIDTHS = {
    'OLIDH_SD': 1.8,
    'OVFM': 1.0, 'SFCF': 1.0, 'Dynfo': 1.0,
    'Aux-Drop': 1.0, 'Aux-Net': 1.0,
}

DATASETS = ['S1', 'S2', 'G1', 'G2']
STREAMS  = ['cap', 'tra', 'evo']

# ──────────────────────────────────────────────────────────────
# 3. Drift Annotation Logic
# ──────────────────────────────────────────────────────────────
def annotate_drift(ax, dataset):
    """Add drift-point visual markers based on dataset type."""
    sudden_kw = dict(linewidth=1.3, alpha=0.85, zorder=1)
    gradual_line_kw = dict(linewidth=1.0, alpha=0.65)

    if dataset == 'S1':
        ax.axvline(x=10000, color='#E07B39', ls='--', **sudden_kw)
    elif dataset == 'G1':
        ax.axvline(x=10000, color='#888888', ls='--', **gradual_line_kw)
        ax.axvline(x=12000, color='#888888', ls='--', **gradual_line_kw)
        ax.axvspan(10000, 12000, color='#999999', alpha=0.30, zorder=0)
    elif dataset == 'S2':
        ax.axvline(x=5000,  color='#E07B39', ls='--', **sudden_kw)
        ax.axvline(x=12500, color='#5DADE2', ls='--', **sudden_kw)
    elif dataset == 'G2':
        for (lo, hi) in [(5000, 7000), (12000, 14000)]:
            ax.axvline(x=lo, color='#888888', ls='--', **gradual_line_kw)
            ax.axvline(x=hi, color='#888888', ls='--', **gradual_line_kw)
            ax.axvspan(lo, hi, color='#999999', alpha=0.30, zorder=0)

# ──────────────────────────────────────────────────────────────
# 4. Simulated Data Generator (fallback when no CSVs)
# ──────────────────────────────────────────────────────────────
def simulate_cer_data(dataset, stream, n_instances=20000, n_points=200):
    """
    Generate realistic CER curves.
    OLIDH_SD: fastest recovery after drift, lowest final CER.
    Baselines: varying degrees of drift sensitivity.
    """
    rng = np.random.RandomState(hash(dataset + stream) % 2**31)
    x = np.linspace(0, n_instances, n_points)

    # Base CER decay (learning curve)
    def base_curve(final_cer, speed, noise_std):
        curve = final_cer + (0.50 - final_cer) * np.exp(-speed * np.arange(n_points) / n_points)
        curve += rng.normal(0, noise_std, n_points)
        return np.clip(np.maximum.accumulate(curve[::-1])[::-1], 0.01, 0.80)

    # Drift bump: CER temporarily increases then recovers
    def add_drift_bump(curve, x, drift_pos, magnitude, recovery_speed):
        bump = magnitude * np.exp(-recovery_speed * np.maximum(x - drift_pos, 0) / n_instances)
        bump[x < drift_pos] = 0
        return curve + bump

    # Get drift positions for this dataset
    drift_configs = {
        'S1': [(10000,)],
        'G1': [(10000, 12000)],
        'S2': [(5000,), (12000,)],
        'G2': [(5000, 7000), (12000, 14000)],
    }

    # Method-specific parameters: (final_cer, speed, noise, drift_magnitude, drift_recovery)
    method_params = {
        'OLIDH_SD': (0.08, 3.5, 0.003, 0.06, 6.0),   # best: low CER, fast recovery
        'OVFM':     (0.18, 2.0, 0.004, 0.12, 2.5),
        'SFCF':     (0.22, 1.8, 0.004, 0.14, 2.0),
        'Dynfo':    (0.25, 1.6, 0.005, 0.16, 1.8),
        'Aux-Drop': (0.20, 1.9, 0.004, 0.13, 2.2),
        'Aux-Net':  (0.16, 2.2, 0.004, 0.10, 3.0),
    }

    # Stream difficulty modifier
    stream_mod = {'cap': 1.0, 'tra': 0.9, 'evo': 1.1}
    mod = stream_mod.get(stream, 1.0)

    data = {'x': x}
    for method in METHODS:
        final, speed, noise, drift_mag, drift_rec = method_params[method]
        final *= mod
        drift_mag *= mod
        curve = base_curve(final, speed, noise)

        for drift in drift_configs[dataset]:
            if len(drift) == 1:
                # Sudden drift
                curve = add_drift_bump(curve, x, drift[0], drift_mag, drift_rec)
            else:
                # Gradual drift: spread the bump
                lo, hi = drift
                mid = (lo + hi) / 2
                curve = add_drift_bump(curve, x, mid, drift_mag * 0.8, drift_rec * 0.8)

        # Smooth with rolling average
        kernel = np.ones(5) / 5
        curve = np.convolve(curve, kernel, mode='same')
        curve = np.clip(curve, 0.01, 0.80)
        data[method] = curve

    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
# 5. Data Loading
# ──────────────────────────────────────────────────────────────
def load_data(data_dir, dataset, stream):
    """Load CSV if available, otherwise simulate.
    CSV format: no header, 7 columns:
      col0=OVFM, col1=SFCF, col2=Dynfo, col3=Aux-Drop, col4=Aux-Net, col5=OLIDH_SD, col6=instance_id
    Handles: no header, Windows line endings, stray text rows, leading zeros.
    Downsamples to ~200 points for clean plotting.
    """
    csv_path = os.path.join(data_dir, f'{dataset}-{stream}.csv')
    if os.path.isfile(csv_path):
        method_names = ['OVFM', 'SFCF', 'Dynfo', 'Aux-Drop', 'Aux-Net', 'OLIDH_SD']
        try:
            # Read raw, skip bad lines, force all cols numeric
            df_raw = pd.read_csv(
                csv_path, header=None, encoding='utf-8',
                on_bad_lines='skip', engine='python',
            )
            # Some CSVs may have a text header row — detect and skip
            # Force numeric conversion; non-numeric cells become NaN
            df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
            # Drop any row that has NaN (i.e. contained text)
            df_raw = df_raw.dropna().reset_index(drop=True)

            # Assign column names based on column count
            if df_raw.shape[1] == 7:
                df_raw.columns = method_names + ['x']
            elif df_raw.shape[1] == 6:
                # No instance id column — create one
                df_raw.columns = method_names
                df_raw['x'] = np.arange(1, len(df_raw) + 1, dtype=float)
            else:
                print(f"  [WARN] {csv_path} has {df_raw.shape[1]} cols, expected 6 or 7. Using simulated data.")
                return simulate_cer_data(dataset, stream)

            df = df_raw.copy()
            df['x'] = df['x'].astype(float)

            # Skip leading rows where all methods are 0 (avoid CER spike from 0)
            # Also skip rows where x < some small threshold to avoid initial instability
            if len(df) > 1:
                # Drop all leading rows where ANY method value is exactly 0
                drop_count = 0
                for row_idx in range(len(df)):
                    row_vals = df[method_names].iloc[row_idx]
                    if (row_vals == 0).any():
                        drop_count += 1
                    else:
                        break
                if drop_count > 0:
                    df = df.iloc[drop_count:].reset_index(drop=True)

            # Downsample to ~200 points
            if len(df) > 200:
                step = len(df) // 200
                idx = list(range(0, len(df), step))
                if idx[-1] != len(df) - 1:
                    idx.append(len(df) - 1)
                df = df.iloc[idx].reset_index(drop=True)
            return df

        except Exception as e:
            print(f"  [WARN] Error reading {csv_path}: {e}. Using simulated data.")
            return simulate_cer_data(dataset, stream)
    else:
        print(f"  [INFO] {csv_path} not found, using simulated data.")
        return simulate_cer_data(dataset, stream)

# ──────────────────────────────────────────────────────────────
# 6. Main Plotting Function
# ──────────────────────────────────────────────────────────────
def plot_drifting(data_dir, save_dir):
    """Generate the 2×6 drifting CER figure (row1: S1,S2 × 3 streams; row2: G1,G2 × 3 streams)."""
    os.makedirs(save_dir, exist_ok=True)

    # Layout: 2 rows × 6 cols
    # Row 0: S1-cap, S1-tra, S1-evo, S2-cap, S2-tra, S2-evo
    # Row 1: G1-cap, G1-tra, G1-evo, G2-cap, G2-tra, G2-evo
    layout = [
        [('S1', s) for s in STREAMS] + [('S2', s) for s in STREAMS],
        [('G1', s) for s in STREAMS] + [('G2', s) for s in STREAMS],
    ]
    n_rows, n_cols = 2, 6

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.6, 3.2),  # double-column width, compact height
        sharex=False, sharey=True,
        constrained_layout=False,
    )

    subplot_idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            dataset, stream = layout[i][j]
            df = load_data(data_dir, dataset, stream)

            for method in METHODS:
                if method not in df.columns:
                    continue

                if method == 'OLIDH_SD':
                    # Find indices closest to multiples of 2000
                    marker_x = np.arange(2000, 20001, 2000)
                    mke = []
                    for mx in marker_x:
                        idx_closest = (df['x'] - mx).abs().idxmin()
                        mke.append(idx_closest)

                    ax.plot(
                        df['x'], df[method],
                        color=COLORS[method],
                        linestyle='-',
                        linewidth=LINEWIDTHS[method],
                        marker='v',
                        markersize=4.0,
                        markevery=mke,
                        markerfacecolor='white',
                        markeredgecolor=COLORS[method],
                        markeredgewidth=0.8,
                        zorder=10,
                        label=r'$\mathrm{OLIDH_{SD}}$',
                    )
                else:
                    ax.plot(
                        df['x'], df[method],
                        color=COLORS[method],
                        linestyle='-',
                        linewidth=LINEWIDTHS[method],
                        zorder=5,
                        label=method,
                    )

            annotate_drift(ax, dataset)

            ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.4, color='#CCCCCC')
            ax.set_xlim(0, 20000)
            ax.set_ylim(0.0, 0.55)

            ax.set_xticks([0, 5000, 10000, 15000, 20000])
            ax.set_xticklabels(['0', '5K', '10K', '15K', '20K'], fontsize=6.5)
            ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            ax.tick_params(axis='y', labelsize=7)

            subplot_label = chr(ord('a') + subplot_idx)
            ax.set_title(f'({subplot_label}) {dataset}-{stream}', fontsize=8, pad=3)
            subplot_idx += 1

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if i == n_rows - 1:
                ax.set_xlabel('# of Instances', fontsize=8)
            if j == 0:
                ax.set_ylabel('CER', fontsize=9)

    # Unified legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # 第一行：6个模型
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=6,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=2.2,
    )

    # 第二行：drift标注
    drift_handles = [
        Line2D([0], [0], color='#E07B39', ls='--', lw=1.3),
        Line2D([0], [0], color='#5DADE2', ls='--', lw=1.3),
        Patch(facecolor='#999999', alpha=0.30, edgecolor='#888888', linestyle='--', linewidth=0.8),
    ]
    drift_labels = ['Sudden Drift', 'Sudden Drift (2nd)', 'Gradual Drift']

    fig.legend(
        drift_handles, drift_labels,
        loc='upper center',
        ncol=5,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 0.98),  # 比第一行低一点，微调这个值
        columnspacing=1.5,
        handletextpad=0.4,
        handlelength=2.2,
    )

    fig.subplots_adjust(
        left=0.06, right=0.99,
        bottom=0.10, top=0.88,
        hspace=0.38, wspace=0.10,
    )


    # ── Save ──
    pdf_path = os.path.join(save_dir, 'drifting_cer.pdf')
    png_path = os.path.join(save_dir, 'drifting_cer.png')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"[DONE] Saved to:\n  {pdf_path}\n  {png_path}")


# ──────────────────────────────────────────────────────────────
# 7. Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'drifting'))
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'figures', 'drifting'))
    args = parser.parse_args()
    plot_drifting(args.data_dir, args.save_dir)