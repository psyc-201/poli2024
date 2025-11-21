"""
Posterior Predictive Check Part One! 
This script simulates new data from fitted model parameters (og data using ADVI)

1. Loads posterior parameter estimates from the fitted model
2. Uses the original data structure (predictors) 
3. Simulates new outcome variables (dwell, slat, event) based on model equations
4. Outputs simulated data in the same format as original data (Roris) files
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, poisson
from scipy.stats import zscore

# Set random seed for reproducibility
np.random.seed(42)

# Load posterior summary statistics
summary = pd.read_csv("gen_summary_rep.csv", index_col=0)

# Load original data to get predictor structure  
data = pd.read_csv("../original model/Roris_nostd.csv", header=None)
# Smiley file has headers
data_s = pd.read_csv("../original model/Roris_smiley.csv")

# Change subject number to avoid overlap
data_s["subj"] = data_s["subj"] + 73

# Match column structure (from original model approach)
data.columns = data_s.columns

# Concatenate
data = pd.concat([data_s, data], ignore_index=True)

print(f"Original data shape: {data.shape}")
print(f"Number of subjects: {data['subj'].nunique()}")
print(f"Number of observations: {len(data)}")

# Extract subject indices
subj_idx = data.subj.values.astype(int) - 1
nsubj = data['subj'].nunique()

# Standardize predictors (same as in og model)
kl_raw = data.D.values
ent_raw = data.H.values
surp_raw = data.I.values
ntrialseq_raw = data.ntrialseq.values

kl = zscore(kl_raw, nan_policy="omit")
ent = zscore(ent_raw, nan_policy="omit")
surp = zscore(surp_raw, nan_policy="omit")
ntrialseq = zscore(ntrialseq_raw, nan_policy="omit")

# Handle NaNs
kl = np.nan_to_num(kl, nan=0.0)
ent = np.nan_to_num(ent, nan=0.0)
surp = np.nan_to_num(surp, nan=0.0)
ntrialseq = np.nan_to_num(ntrialseq, nan=0.0)

# Extract individual subject parameters (use posterior means)
LT0 = np.array([summary.loc[f'LT0[{i}]', 'mean'] for i in range(nsubj)])
LT1 = np.array([summary.loc[f'LT1[{i}]', 'mean'] for i in range(nsubj)])
LT2 = summary.loc['LT2', 'mean']
LT3 = summary.loc['LT3', 'mean']
LT4 = summary.loc['LT4', 'mean']

# Extract fitted error/noise parameters
eps_LT = summary.loc['eps_LT', 'mean']
eps_SL = summary.loc['eps_SL', 'mean']

SL0 = np.array([summary.loc[f'SL0[{i}]', 'mean'] for i in range(nsubj)])
SL1 = np.array([summary.loc[f'SL1[{i}]', 'mean'] for i in range(nsubj)])
SL2 = summary.loc['SL2', 'mean']
SL3 = summary.loc['SL3', 'mean']

lambda0 = np.array([summary.loc[f'lambda0[{i}]', 'mean'] for i in range(nsubj)])
beta_LA = np.array([summary.loc[f'beta_LA[{i}]', 'mean'] for i in range(nsubj)])
LA2 = summary.loc['LA2', 'mean']
LA3 = summary.loc['LA3', 'mean']

# Simulate Looking Time (LT) - Student's T distribution with nu=15
est_LT = LT0[subj_idx] + LT1[subj_idx] * kl + LT2 * ntrialseq + LT3 * surp + LT4 * ent
ltime_sim = student_t.rvs(df=15, loc=est_LT, scale=eps_LT, size=len(data))

# Simulate Saccadic Latency (SL) - Student's T distribution with nu=15
est_SL = SL0[subj_idx] + SL1[subj_idx] * ent + SL2 * ntrialseq + SL3 * surp
slat_sim = student_t.rvs(df=15, loc=est_SL, scale=eps_SL, size=len(data))

# Simulate Look-Away events (LA) - Poisson distribution
# Proportional hazard model: lambda = exp(beta*X) * lambda0
log_rate = beta_LA[subj_idx] * kl + LA2 * ntrialseq + LA3 * surp
rate = np.exp(log_rate) * lambda0[subj_idx]
lookaway_sim = poisson.rvs(rate, size=len(data))
# Convert to binary (0/1) since original is binary
lookaway_sim = (lookaway_sim > 0).astype(int)

print("\nSimulated data statistics:")
print(f"  Looking time: mean={ltime_sim.mean():.3f}, std={ltime_sim.std():.3f}")
print(f"  Saccadic latency: mean={slat_sim.mean():.3f}, std={slat_sim.std():.3f}")
print(f"  Look-away rate: {lookaway_sim.mean():.3f} (proportion of 1s)")

print("\nOriginal data statistics (z-scored):")
ltime_orig = zscore(data.dwell.values, nan_policy="omit")
slat_orig = zscore(data.slat.values, nan_policy="omit")
print(f"  Looking time: mean={np.nanmean(ltime_orig):.3f}, std={np.nanstd(ltime_orig):.3f}")
print(f"  Saccadic latency: mean={np.nanmean(slat_orig):.3f}, std={np.nanstd(slat_orig):.3f}")
print(f"  Look-away rate: {data.event.mean():.3f}")

# Convert simulated z-scores back to raw scale (approximate)
# Using original data mean and std
dwell_mean = np.nanmean(data.dwell.values)
dwell_std = np.nanstd(data.dwell.values)
slat_mean = np.nanmean(data.slat.values)
slat_std = np.nanstd(data.slat.values)

dwell_sim = ltime_sim * dwell_std + dwell_mean
slat_sim_raw = slat_sim * slat_std + slat_mean

# Set negative dwell values to NaN (invalid observations)
dwell_sim[dwell_sim < 0] = np.nan

# Preserve NaN values from original data
dwell_sim[np.isnan(data.dwell.values)] = np.nan
slat_sim_raw[np.isnan(data.slat.values)] = np.nan

# Create simulated dataset with same structure as original
data_sim = data.copy()
data_sim['dwell'] = dwell_sim
data_sim['slat'] = slat_sim_raw
data_sim['event'] = lookaway_sim

# Save combined simulated data
data_sim.to_csv("sim_data.csv", index=False)
