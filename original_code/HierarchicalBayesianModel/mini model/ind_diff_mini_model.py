# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:45:35 2021
Adapted for PyMC v5 on October 29, 2025 by @thenness-y

@author: maxhin, frapol, tomghi
"""
import numpy as np
import pymc as pm  # Changed from pymc3
import pytensor.tensor as pt  # Changed from theano.tensor as T
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid plt.show() blocking
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

# Choose sampling method (ADVI vs MCMC)
useADVI = True  # Set to True to use ADVI, False for MCMC NUTS sampling

# Import data first study
data = pd.read_csv("Roris_nostd.csv", sep=",")
# Import smiley data
data_s = pd.read_csv("Roris_smiley.csv", sep=",")
# Change subject number to avoid overlap
data_s["subj"] = data_s["subj"] + 73

# Concatenate the two dataframes
data.columns = data_s.columns
data = pd.concat([data_s, data], ignore_index=True)

# FOR MINIMAL TEST: Use only first 3 subjects, comment out for full analysis
# data = data[data.subj.isin([1, 2, 3])]

############### Convert data to a pymc-friendly version ######################
# total number of subjects
nsubj = len(data.subj.unique())
# max number of sequences in the task
nseq = len(data.nseq.unique())
# max number of trials in a sequence
ntrial = len(data.ntrialseq.unique())
# index of each subject and each sequence for each trial
subj_idx = data.subj.values.astype(int) - 1
seq_idx = seq_vect = data.nseq.values.astype(int) - 1  # subjective sequence number
trial_vect = data.ntrialseq.values - 1

# Convert to PyTensor shared variables (replaces theano.shared)
trial_vect = pt.as_tensor_variable(trial_vect.astype("float64"))
seq_vect = pt.as_tensor_variable(seq_vect.astype("float64"))

################ Dependent Variables ##########################################
# Convert to regular arrays for PyMC v5 compatibility
ltime = zscore(data.dwell.values, nan_policy="omit")
slat = zscore(data.slat.values, nan_policy="omit") 
lookaway = data.event.values  # look-away from the screen (0 for no look-away, 1 for look-away)
ntrialseq = zscore(data.ntrialseq.values, nan_policy="omit")

# Convert to PyTensor tensors
ntrialseq = pt.as_tensor_variable(ntrialseq.astype("float64"))


################ Independent Variables ########################################
kl = zscore(data.D.values)
kl = pt.as_tensor_variable(kl.astype("float64"))

ent = zscore(data.H.values)
ent = pt.as_tensor_variable(ent.astype("float64"))

surp = zscore(data.I.values)
surp = pt.as_tensor_variable(surp.astype("float64"))


############################# markasgood #####################################
# this vector specifies the original participant number of the subjects

# Import markasgood first study
markasgood = pd.read_csv("markasgood.csv", sep=",", header=None)
# Import markasgood smiley data
markasgood_s = pd.read_csv("markasgood_smiley.csv", sep=",", header=None)
# concatenate them
markasgood = pd.concat([markasgood_s, markasgood], ignore_index=True)

# Note: markasgood filtering not needed since we use data.subj.unique() later

################ Bayesian Model ###############################################

with pm.Model() as model:
    # =============================================================================
    # Looking time and saccadic latency
    # =============================================================================
    ########################### Hyper-Priors ######################################
    # Define mu and sigma for ß0 Looking Time
    mu_LT0 = pm.Normal("mu_LT0", mu=0, sigma=1)
    sigma_LT0 = 1

    # Define mu and sigma ß0 for Saccadic Latency
    mu_SL0 = pm.Normal("mu_SL0", mu=0, sigma=1)
    sigma_SL0 = 1

    # Define mu and sigma ß1 for Looking Time
    mu_LT1 = pm.Normal("mu_LT1", mu=0, sigma=1)
    sigma_LT1 = 1

    # Define mu and sigma ß1 for Saccadic Latency
    mu_SL1 = pm.Normal("mu_SL1", mu=0, sigma=1)
    sigma_SL1 = 1

    ################################# Priors ######################################
    # Define beta0, beta1 and error for Looking Time
    LT0 = pm.Normal(
        "LT0", mu=mu_LT0, sigma=sigma_LT0, shape=nsubj
    )  # processing speed, how long it takes to encode
    LT1 = pm.Normal("LT1", mu=mu_LT1, sigma=sigma_LT1, shape=nsubj)  # curiosity
    LT2 = pm.Normal("LT2", mu=0, sigma=1)  # effect of time
    LT3 = pm.Normal("LT3", mu=0, sigma=1)  # surprise
    LT4 = pm.Normal("LT4", mu=0, sigma=1)  # entropy
    eps_LT = pm.HalfCauchy("eps_LT", beta=1)

    # Define beta0, beta1 and error for Saccadic Latency
    SL0 = pm.Normal(
        "SL0", mu=mu_SL0, sigma=sigma_SL0, shape=nsubj
    )  # reaction time, more like a motor thing
    SL1 = pm.Normal(
        "SL1", mu=mu_SL1, sigma=sigma_SL1, shape=nsubj
    )  # learning progress
    SL2 = pm.Normal("SL2", mu=0, sigma=1)  # effect of time
    SL3 = pm.Normal("SL3", mu=0, sigma=1)  # surprise
    eps_SL = pm.HalfCauchy("eps_SL", beta=1)

    ######################### Estimates and Likelihood ############################
    # Linear regression between Looking Time and KL-Divergence
    est_LT = (
        LT0[subj_idx] + LT1[subj_idx] * kl + LT2 * ntrialseq + LT3 * surp + LT4 * ent
    )
    LT_like = pm.StudentT(
        "LT_like", nu=15, mu=est_LT, sigma=eps_LT, observed=ltime
    )  # noise

    # Linear regression between Saccadic Latency and Entropy
    est_SL = SL0[subj_idx] + SL1[subj_idx] * ent + SL2 * ntrialseq + SL3 * surp
    SL_like = pm.StudentT(
        "SL_like", nu=15, mu=est_SL, sigma=eps_SL, observed=slat
    )  # noise

    # =============================================================================
    # Look Away
    # =============================================================================
    ########################### Hyper-Priors ######################################
    # Define kappa and theta
    kappa = pm.Gamma("kappa", alpha=1, beta=1)  # Updated Gamma parameterization
    theta = pm.Gamma("theta", alpha=1, beta=1)  # Updated Gamma parameterization

    # Define mu and sigma for ß1 Look Away
    mu_LA = pm.Normal("mu_LA", mu=0, sigma=1)
    sigma_LA = 1

    ################################# Priors ######################################
    # Define baseline hazard and beta1 for Look-Away
    lambda0 = pm.Gamma(
        "lambda0", alpha=kappa, beta=theta, shape=nsubj
    )  # baseline attention -sticky fixation- not a good thing,
    beta_LA = pm.Normal(
        "beta_LA", mu_LA, sigma=sigma_LA, shape=nsubj
    )  # also related to curiosity, how much you disengage related to information gain
    LA2 = pm.Normal("LA2", mu=0, sigma=1)  # effect of time
    LA3 = pm.Normal("LA3", mu=0, sigma=1)  # surprise

    ######################### Estimates and Likelihood ############################
    # proportional hazard model (https://docs.pymc.io/notebooks/survival_analysis.html)
    # The piecewise-constant proportional hazard model is closely related to a Poisson regression model, hence:
    LA_like = pm.Poisson(
        "LA_like",
        pt.exp(beta_LA[subj_idx] * kl + LA2 * ntrialseq + LA3 * surp)
        * lambda0[subj_idx],
        observed=lookaway,
    )

# Estimate posterior distributions
# Using either ADVI or MCMC NUTS sampling; the mini model uses ADVI
with model:
    # Using Variational Inference
    if useADVI:
        advi = pm.ADVI()
        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,
            std=advi.approx.std.eval,
        )
        approx = pm.fit(n=30000, method=advi, callbacks=[tracker])
        trace = approx.sample(draws=50000)
    # Using MCMC NUTS sampling
    else:
        # More reasonable default settings for faster testing
        trace = pm.sample(
            1000, chains=2, cores=2, tune=500, 
            init="adapt_diag", target_accept=0.8
        )

# check ELBO for ADVI
if useADVI:
    plt.figure(figsize=(10, 6))
    plt.plot(tracker["mean"], label="ADVI mean", alpha=0.7)
    plt.plot(tracker["std"], label="ADVI std", alpha=0.7)
    plt.legend()
    plt.ylabel("ADVI tracker")
    plt.xlabel("iteration")
    plt.title("ADVI Convergence")
    plt.savefig("advi_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking
    print("ADVI convergence plot saved to advi_convergence.png")

# Model goodness of fit
# Note: WAIC/LOO commented out - not compatible with ADVI or current PyTensor set up in the mini model
print("⚠️  WAIC/LOO calculations skipped")
print("   Model completed successfully without WAIC/LOO")

# # Get WAIC estimates
# waic_results = {}
# loo_results = {}
# waic_results["LT"] = az.waic(trace, var_name="LT_like")
# print("Looking Time WAIC:", waic_results["LT"])

# waic_results["SL"] = az.waic(trace, var_name="SL_like") 
# print("Saccadic Latency WAIC:", waic_results["SL"])

# waic_results["LA"] = az.waic(trace, var_name="LA_like")
# print("Look Away WAIC:", waic_results["LA"])

# # Get LOO estimates
# loo_results["LT"] = az.loo(trace, var_name="LT_like")
# print("Looking Time LOO:", loo_results["LT"])

# loo_results["SL"] = az.loo(trace, var_name="SL_like")
# print("Saccadic Latency LOO:", loo_results["SL"])

# loo_results["LA"] = az.loo(trace, var_name="LA_like")
# print("Look Away LOO:", loo_results["LA"])

############################### PLOTTING #####################################
# plot the parameters and save the plots

az.plot_forest(trace, kind="forestplot", var_names=["mu_LT1"], hdi_prob=0.89)
plt.savefig("forest_mu_LT1.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["mu_LT0"], hdi_prob=0.89)
plt.savefig("forest_mu_LT0.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["mu_SL1"], hdi_prob=0.89)
plt.savefig("forest_mu_SL1.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["mu_SL0"], hdi_prob=0.89)
plt.savefig("forest_mu_SL0.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["kappa"], hdi_prob=0.89)
plt.savefig("forest_mu_kappa.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["theta"], hdi_prob=0.89)
plt.savefig("forest_mu_theta.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["lambda0"], hdi_prob=0.89)
plt.savefig("forest_lambda0.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["beta_LA"], hdi_prob=0.89)
plt.savefig("forest_betaLA.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["LT0"], hdi_prob=0.89)
plt.savefig("forest_LT0.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["LT1"], hdi_prob=0.89)
plt.savefig("forest_LT1.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["SL0"], hdi_prob=0.89)
plt.savefig("forest_SL0.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["SL1"], hdi_prob=0.89)
plt.savefig("forest_SL1.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["lambda0", "kappa"], hdi_prob=0.89)
plt.savefig("2forest_lambda0.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["LT0", "mu_LT0"], hdi_prob=0.89)
plt.savefig("2forest_LT0.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["LT1", "mu_LT1"], hdi_prob=0.89)
plt.savefig("2forest_LT1.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["SL0", "mu_SL0"], hdi_prob=0.89)
plt.savefig("2forest_SL0.png")
plt.show()

az.plot_forest(trace, kind="forestplot", var_names=["SL1", "mu_SL1"], hdi_prob=0.89)
plt.savefig("2forest_SL1.png")
plt.show()

az.plot_trace(trace, var_names=["lambda0"])
plt.savefig("lambda0.png")
plt.show()

az.plot_trace(trace, var_names=["beta_LA"])
plt.savefig("beta_LA.png")
plt.show()

az.plot_trace(trace, var_names=["LT0"])
plt.savefig("LT0.png")
plt.show()

az.plot_trace(trace, var_names=["LT1"])
plt.savefig("LT1.png")
plt.show()

az.plot_trace(trace, var_names=["SL0"])
plt.savefig("SL0.png")
plt.show()

az.plot_trace(trace, var_names=["SL1"])
plt.savefig("SL1.png")
plt.show()

axes = az.plot_forest(
    trace,
    kind="ridgeplot",
    var_names=["lambda0"],
    hdi_prob=0.89,
    combined=True,
    ridgeplot_overlap=3,
    colors="white",
    figsize=(9, 7),
)
axes[0].set_title("lambda0")
plt.savefig("forest2_lambda0.png")
plt.show()

az.plot_trace(trace, var_names=["mu_LT0", "mu_LT1"])
plt.savefig("mu_LT.png")
plt.show()

az.plot_trace(trace, var_names=["mu_SL0", "mu_SL1"])
plt.savefig("mu_SL.png")
plt.show()

az.plot_trace(trace, var_names=["kappa", "theta", "mu_LA"])
plt.savefig("hyper_LA.png")
plt.show()

# Save summary statistics for the model parameters
summary = az.summary(
    trace,
    var_names=[
        "LT0",
        "LT1", 
        "LT2",
        "LT3",
        "LT4",
        "SL0",
        "SL1",
        "SL2",
        "SL3",
        "lambda0",
        "beta_LA",
        "LA2",
        "LA3",
        "mu_LT0",
        "mu_LT1",
        "mu_SL0",
        "mu_SL1",
        "kappa",
        "theta",
        "mu_LA",
    ],
    round_to=3,
    hdi_prob=0.89,
)
summary.to_csv("gen_summary_rep.csv")

# Extract posterior median values for each subject
# Put parameter values for each subject in dataframe
posterior = pd.DataFrame()
posterior["subjnum"] = markasgood.values.reshape(-1)
posterior["LT0"] = np.median(trace.posterior["LT0"].values, axis=(0, 1))
posterior["LT1"] = np.median(trace.posterior["LT1"].values, axis=(0, 1))
posterior["SL0"] = np.median(trace.posterior["SL0"].values, axis=(0, 1))
posterior["SL1"] = np.median(trace.posterior["SL1"].values, axis=(0, 1))
posterior["lambda0"] = np.median(trace.posterior["lambda0"].values, axis=(0, 1))
posterior["beta_LA"] = np.median(trace.posterior["beta_LA"].values, axis=(0, 1))
# ...and save them
posterior.to_csv("posterior_median_rep.csv")

print("Results saved to gen_summary_rep.csv and posterior_median_rep.csv")