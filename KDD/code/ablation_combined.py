"""
KDD 2026 - Ablation Study: Combined 3x2 Grid Figure (Fixed Paths)
=================================================================
Plots 6 specific datasets in a 3-row x 2-column layout.
Unified legend at the top.
Now uses absolute paths relative to the script location to avoid FileNotFoundError.

Usage:
    python plot_combined_ablation.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator

# Try to use SciencePlots if installed
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
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# ──────────────────────────────────────────────────────────────
# 2. Style & Data Definitions
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
    'OLIDH_SD':   2.8,
    'OLIDH_SD-I': 1.6,
    'OLIDH_SD-D': 1.6,
    'OLIDH_SD-H': 1.6,
}

LABELS = {
    'OLIDH_SD':   r'$\mathrm{OLIDH_{SD}}$',
    'OLIDH_SD-I': r'$\mathrm{OLIDH_{SD}\text{-}I}$',
    'OLIDH_SD-D': r'$\mathrm{OLIDH_{SD}\text{-}D}$',
    'OLIDH_SD-H': r'$\mathrm{OLIDH_{SD}\text{-}H}$',
}

PLOT_ORDER = ['OLIDH_SD-I', 'OLIDH_SD-D', 'OLIDH_SD-H', 'OLIDH_SD']

# The 6 specific targets requested
# Ensure your CSV filenames match these patterns: e.g., 'synthetic_1_cap_ablation.csv'
TARGETS = [
    ('synthetic_1', 'cap', '(a) S1-cap'),
    ('synthetic_3', 'tra', '(b) S2-tra'),
    ('synthetic_2', 'evo', '(c) G1-evo'),
    ('synthetic_4', 'evo', '(d) G2-evo'),
    ('german',      'cap', '(e) germab-cap'),
    ('sylva',       'tra', '(f) sylva-tra'),
]

# ──────────────────────────────────────────────────────────────
# 3. Helper Functions (Data & Ticks)
# ──────────────────────────────────────────────────────────────
def make_even_ticks(data_max, n_divisions=4):
    """Generates clean, evenly spaced ticks."""
    if data_max <= 0:
        return np.array([0, 1]), 1
    raw_step = data_max / n_divisions
    if data_max > 100:
        step = int(np.ceil(raw_step))
    elif data_max > 1:
        step = np.ceil(raw_step * 10) / 10
    else:
        step = np.ceil(raw_step * 100) / 100
    
    tick_max = step * n_divisions
    ticks = np.linspace(0, tick_max, n_divisions + 1)
    return np.round(ticks, 10), tick_max

def load_ablation_data(csv_path):
    """Load and downsample data."""
    try:
        df_raw = pd.read_csv(csv_path, header=None, engine='python', on_bad_lines='skip')
        df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        
        # Check column count
        if df_raw.shape[1] == 5:
            df_raw.columns = CSV_COLNAMES
        elif df_raw.shape[1] == 4:
            df_raw.columns = VARIANTS
            df_raw['x'] = np.arange(1, len(df_raw) + 1, dtype=float)
        else:
            print(f"Warning: {os.path.basename(csv_path)} has {df_raw.shape[1]} columns, skipping.")
            return None

        # Clean zeros (remove initial rows where model hasn't started learning)
        drop_count = 0
        for i in range(len(df_raw)):
            if (df_raw[VARIANTS].iloc[i] == 0).any():
                drop_count += 1
            else:
                break
        df = df_raw.iloc[drop_count:].reset_index(drop=True)

        # Downsample for plotting speed/file size (keep ~500 points)
        if len(df) > 500:
            step = len(df) // 500
            df = df.iloc[::step].reset_index(drop=True)
            
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

# ──────────────────────────────────────────────────────────────
# 4. Main Plotting Function
# ──────────────────────────────────────────────────────────────
def plot_combined_grid(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create Figure: 3 rows, 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 9))
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []
    
    for i, (dataset, stream, title) in enumerate(TARGETS):
        ax = axes[i]
        
        # Construct path
        csv_name = f'{dataset}_{stream}_ablation.csv'
        csv_path = os.path.join(data_dir, csv_name)
        
        print(f"Processing [{i+1}/6]: {title} ...")
        print(f"  -> Looking for: {csv_path}")
        
        if not os.path.exists(csv_path):
             # Try fallback filename if 'synthetic' naming differs (e.g. S1_cap_ablation.csv)
            fallback_name = f'{dataset.replace("synthetic_", "S").replace("synthetic_", "G")}_{stream}_ablation.csv' # Just a guess
            if i==0: fallback_name = f'S1_{stream}_ablation.csv'
            # (You can add more logic here if your files are named S1, S2 etc.)
            
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center', fontsize=12)
            print(f"  [ERROR] File not found: {csv_path}")
            continue

        df = load_ablation_data(csv_path)
        
        if df is None or df.empty:
            ax.text(0.5, 0.5, 'Data Error', ha='center', va='center')
            continue

        # ── Plot Curves ──
        for variant in PLOT_ORDER:
            if variant in df.columns:
                line, = ax.plot(
                    df['x'], df[variant],
                    color=COLORS[variant],
                    linestyle=LINESTYLES[variant],
                    linewidth=LINEWIDTHS[variant],
                    zorder=10 if variant == 'OLIDH_SD' else 5,
                    label=LABELS[variant]
                )
                if i == 0: # Collect legend info from first plot
                    legend_handles.append(line)
                    legend_labels.append(LABELS[variant])

        # ── Axis Formatting ──
        # Ticks
        x_max = df['x'].max()
        y_max = df[VARIANTS].max().max()
        xticks, x_lim = make_even_ticks(x_max, 4)
        yticks, y_lim = make_even_ticks(y_max, 4)
        
        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_lim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        
        # Bold Tick Labels
        plt.setp(ax.get_xticklabels(), fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontweight='bold')
        
        # Format large X numbers (K notation)
        if x_lim > 5000:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f'{v/1000:.0f}K' if v > 0 else '0')
            )

        # Grids & Spines
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which='major', linestyle=':', alpha=0.8, linewidth=0.6, color="#4B4A4A")
        ax.grid(True, which='minor', linestyle=':', alpha=0.5, linewidth=0.4, color="#5F5F5F")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')

        # ── Titles (In bold) ──
        ax.set_title(title, fontweight='bold', fontsize=13, pad=6)

        # ── Labels (Selective) ──
        if i % 2 == 0:
            ax.set_ylabel('CER', fontsize=13, fontweight='bold')
        if i >= 4:
            ax.set_xlabel('# of Instances', fontsize=13, fontweight='bold')

    # ── Global Legend (Top) ──
    if legend_handles:
        fig.legend(
            legend_handles, 
            legend_labels,
            loc='upper center', 
            bbox_to_anchor=(0.5, 0.98), 
            ncol=4,         
            frameon=False,
            prop={'weight': 'bold', 'size': 13},
            handlelength=2.5,
            handletextpad=0.5,
            columnspacing=1.5
        )

    # ── Layout Adjustment ──
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0, w_pad=1.5)

    # ── Save ──
    out_pdf = os.path.join(save_dir, 'Combined_Ablation_3x2.pdf')
    out_png = os.path.join(save_dir, 'Combined_Ablation_3x2.png')
    
    print(f"\nSaving to:\n  {out_pdf}\n  {out_png}")
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()

if __name__ == '__main__':
    # ── FIX: Automatically determine paths relative to THIS script file ──
    # This ensures it works regardless of where you run the command from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Assuming standard structure:
    #   .../KDD/code/ablation_single.py (this file)
    #   .../KDD/data/ablation/*.csv
    #   .../KDD/figures/combined/
    default_data_dir = os.path.join(script_dir, '../data/ablation')
    default_save_dir = os.path.join(script_dir, '../figures/ablation_combined')

    # Resolve '..' to make path look cleaner in print output
    default_data_dir = os.path.abspath(default_data_dir)
    default_save_dir = os.path.abspath(default_save_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=default_data_dir)
    parser.add_argument('--save_dir', type=str, default=default_save_dir)
    args = parser.parse_args()
    
    print(f"Script location: {script_dir}")
    print(f"Data Directory:  {args.data_dir}")
    print(f"Save Directory:  {args.save_dir}")
    print("-" * 60)
    
    plot_combined_grid(args.data_dir, args.save_dir)