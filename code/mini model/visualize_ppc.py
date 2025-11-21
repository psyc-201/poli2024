"""
Visual Comparison: Original vs Simulated Data
Posterior Predictive Check Visualization (Part 1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Load original data
data_orig = pd.read_csv("../original model/Roris_nostd.csv", header=None)
data_s_orig = pd.read_csv("../original model/Roris_smiley.csv")
data_s_orig["subj"] = data_s_orig["subj"] + 73
data_orig.columns = data_s_orig.columns
data_orig = pd.concat([data_s_orig, data_orig], ignore_index=True)

# Load simulated data
data_sim = pd.read_csv("sim_data.csv")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Original vs Simulated Raw Data',
             fontsize=16, fontweight='bold')

# 1. Looking Time (Dwell) - Raw Scale
ax = axes[0, 0]
ax.hist(data_orig['dwell'].dropna(), bins=50, alpha=0.6,
        label='Original', color='steelblue', density=True)
ax.hist(data_sim['dwell'].dropna(), bins=50, alpha=0.6,
        label='Simulated', color='coral', density=True)
ax.set_xlabel('Dwell Time (ms)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Looking Time Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2. Looking Time - Z-scored
ax = axes[0, 1]
ltime_orig_z = zscore(data_orig['dwell'].values, nan_policy="omit")
ltime_sim_z = zscore(data_sim['dwell'].values, nan_policy="omit")
ax.hist(ltime_orig_z, bins=50, alpha=0.6,
        label='Original', color='steelblue', density=True)
ax.hist(ltime_sim_z, bins=50, alpha=0.6,
        label='Simulated', color='coral', density=True)
ax.set_xlabel('Looking Time (z-scored)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Looking Time (Standardized)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 3. Saccadic Latency - Raw Scale
ax = axes[1, 0]
ax.hist(data_orig['slat'].dropna(), bins=50, alpha=0.6,
        label='Original', color='steelblue', density=True)
ax.hist(data_sim['slat'].dropna(), bins=50, alpha=0.6,
        label='Simulated', color='coral', density=True)
ax.set_xlabel('Saccadic Latency (ms)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Saccadic Latency Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 4. Saccadic Latency - Z-scored
ax = axes[1, 1]
slat_orig_z = zscore(data_orig['slat'].values, nan_policy="omit")
slat_sim_z = zscore(data_sim['slat'].values, nan_policy="omit")
ax.hist(slat_orig_z, bins=50, alpha=0.6,
        label='Original', color='steelblue', density=True)
ax.hist(slat_sim_z, bins=50, alpha=0.6,
        label='Simulated', color='coral', density=True)
ax.set_xlabel('Saccadic Latency (z-scored)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Saccadic Latency (Standardized)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Calculate look-away rates for summary table
orig_rate = data_orig['event'].mean()
sim_rate = data_sim['event'].mean()

# Print summary statistics
print(f"\n{'Measure':<25} {'Original':<15} {'Simulated':<15}")
print(f"{'Looking Time (mean)':<25} {np.nanmean(ltime_orig_z):>14.3f} {np.nanmean(ltime_sim_z):>14.3f}")
print(f"{'Looking Time (std)':<25} {np.nanstd(ltime_orig_z):>14.3f} {np.nanstd(ltime_sim_z):>14.3f}")
print(f"{'Saccadic Latency (mean)':<25} {np.nanmean(slat_orig_z):>14.3f} {np.nanmean(slat_sim_z):>14.3f}")
print(f"{'Saccadic Latency (std)':<25} {np.nanstd(slat_orig_z):>14.3f} {np.nanstd(slat_sim_z):>14.3f}")
print(f"{'Look-Away (rate)':<25} {orig_rate:>14.3f} {sim_rate:>14.3f}")

plt.tight_layout()
plt.savefig('ppc_comparison_2.png', dpi=150, bbox_inches='tight')
print("\nSaved: ppc_comparison_2.png")

# Additional: Q-Q plots for normality check
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Q-Q Plots: Checking Distribution Similarity',
              fontsize=14, fontweight='bold')

# Q-Q plot for Looking Time
ax = axes2[0]
orig_sorted = np.sort(ltime_orig_z[~np.isnan(ltime_orig_z)])
sim_sorted = np.sort(ltime_sim_z[~np.isnan(ltime_sim_z)])
# Match lengths
min_len = min(len(orig_sorted), len(sim_sorted))
ax.scatter(orig_sorted[:min_len], sim_sorted[:min_len],
           alpha=0.5, s=10, color='steelblue')
ax.plot([orig_sorted.min(), orig_sorted.max()],
        [orig_sorted.min(), orig_sorted.max()],
        'r--', lw=2, label='Perfect match')
ax.set_xlabel('Original (quantiles)', fontsize=11)
ax.set_ylabel('Simulated (quantiles)', fontsize=11)
ax.set_title('Looking Time Q-Q Plot', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Q-Q plot for Saccadic Latency
ax = axes2[1]
orig_sorted = np.sort(slat_orig_z[~np.isnan(slat_orig_z)])
sim_sorted = np.sort(slat_sim_z[~np.isnan(slat_sim_z)])
min_len = min(len(orig_sorted), len(sim_sorted))
ax.scatter(orig_sorted[:min_len], sim_sorted[:min_len],
           alpha=0.5, s=10, color='coral')
ax.plot([orig_sorted.min(), orig_sorted.max()],
        [orig_sorted.min(), orig_sorted.max()],
        'r--', lw=2, label='Perfect match')
ax.set_xlabel('Original (quantiles)', fontsize=11)
ax.set_ylabel('Simulated (quantiles)', fontsize=11)
ax.set_title('Saccadic Latency Q-Q Plot', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ppc_qq_plots.png', dpi=150, bbox_inches='tight')
print("Saved: ppc_qq_plots.png")
