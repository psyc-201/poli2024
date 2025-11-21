# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:45:35 2021

@author: maxhin, frapol, tomghi
"""
# import os
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore

# Choose sampling method (ADVI vs MCMC)
useADVI = False

# Import data first study
data = pd.read_csv("Roris_nostd.csv", sep=",")
# Import smiley data
data_s = pd.read_csv("Roris_smiley.csv", sep=",")
# Change subject number to avoid overlap
data_s["subj"] = data_s["subj"] + 73

# Concatenate the two dataframes
data.columns = data_s.columns
data = pd.concat([data_s, data], ignore_index=True)

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

# use theano instead of numpy (better for pymc3)
trial_vect = theano.shared(trial_vect.astype("float64"))
seq_vect = theano.shared(seq_vect.astype("float64"))


################ Dependent Variables ##########################################
# If nans are present, variables must be masked to function in theano
ltime = np.ma.masked_invalid(
    zscore(data.dwell.values, nan_policy="omit")
)  # looking time to the target (standardized across participants)
slat = np.ma.masked_invalid(
    zscore(data.slat.values, nan_policy="omit")
)  # saccadic latency (standardized across participants)
lookaway = (
    data.event.values
)  # look-away from the screen (0 for no look-away, 1 for look-away)
ntrialseq = np.ma.masked_invalid(zscore(data.ntrialseq.values, nan_policy="omit"))


################ Independent Variables ########################################
kl = zscore(data.D.values)
kl = theano.shared(kl.astype("float64"))

ent = zscore(data.H.values)
ent = theano.shared(ent.astype("float64"))

surp = zscore(data.I.values)
surp = theano.shared(surp.astype("float64"))


############################# markasgood #####################################
# this vector specifies the original participant number of the subjects
# Import markasgood first study
markasgood = pd.read_csv("markasgood.csv", sep=",", header=None)
# Import markasgood smiley data
markasgood_s = pd.read_csv("markasgood_smiley.csv", sep=",", header=None)
# concatenate them
markasgood = pd.concat([markasgood_s, markasgood], ignore_index=True)


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
    # Creates individual parameters for each infant
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
    )  # learning efficiency
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
    # Define kappa and thetaa
    kappa = pm.Gamma("kappa", 1, 1)
    theta = pm.Gamma("theta", 1, 1)

    # Define mu and sigma for ß1 Look Away
    mu_LA = pm.Normal("mu_LA", mu=0, sigma=1)
    sigma_LA = 1

    ################################# Priors ######################################
    # Define baseline hazard and beta1 for Look-Away
    lambda0 = pm.Gamma(
        "lambda0", kappa, theta, shape=nsubj
    )  # baseline attention -sitcky fixation- not a good thing,
    beta_LA = pm.Normal(
        "beta_LA", mu_LA, sigma=sigma_LA, shape=nsubj
    )  # also related to curiosity, how much you disengage related to information gain
    LA2 = pm.Normal("LA2", mu=0, sigma=1)  # effect of time
    LA3 = pm.Normal("LA3", mu=0, sigma=1)  # surprise

    ######################### Estimates and Likelihood ############################
    # proportional hazard model (https://docs.pymc.io/notebooks/survival_analysis.html)
    # The piecewise-constant proportional hazard model is closely related to a Poisson regression model, hence:
    # LA_like = pm.Poisson("LA_like", T.exp(beta_LA[subj_idx] * kl + LA2*ntrialseq + LA3*surp)*lambda0[subj_idx], observed=lookaway)
    LA_like = pm.Poisson(
        "LA_like",
        T.exp(beta_LA[subj_idx] * kl + LA2 * ntrialseq + LA3 * surp)
        * lambda0[subj_idx],
        observed=lookaway,
    )

# Estimate posterior
with model:
    # Using Variational Inference
    if useADVI:
        inference = pm.ADVI()
        approx = pm.fit(n=30000, method=inference)
        trace = approx.sample(draws=50000)
    # Using MCMC NUTS sampling
    else:
        trace = pm.sample(
            10000, chains=2, cores=64, tune=490000, init="advi+adapt_diag"
        )

# check ELBO for ADVI
if useADVI:
    plt.plot(-inference.hist, label="new ADVI", alpha=0.3)
    plt.plot(approx.hist, label="old ADVI", alpha=0.3)
    plt.legend()
    plt.ylabel("ELBO")
    plt.xlabel("iteration")

# change trace format for further analysis
idata = az.from_pymc3(trace)

# Nan values are treated as zero likelihood by pymc, so masked (nan) values must be removed before computing model goodness of fit
SL_like_nan = np.nan_to_num(
    idata.log_likelihood.SL_like, copy=True, nan=0
)  # mean_value)
idata.log_likelihood["SL_like"] = (idata.log_likelihood["SL_like"].dims, SL_like_nan)

LT_like_nan = np.nan_to_num(
    idata.log_likelihood.LT_like, copy=True, nan=0
)  # mean_value)
idata.log_likelihood["LT_like"] = (idata.log_likelihood["LT_like"].dims, LT_like_nan)

LA_like_nan = np.nan_to_num(
    idata.log_likelihood.LA_like, copy=True, nan=0
)  # mean_value)
idata.log_likelihood["LA_like"] = (idata.log_likelihood["LA_like"].dims, LA_like_nan)

# Get LOO estimates
model_waic1 = az.loo(idata, var_name="LT_like")
print(model_waic1)
model_waic2 = az.loo(idata, var_name="SL_like")
print(model_waic2)
model_waic3 = az.loo(idata, var_name="LA_like")
print(model_waic3)

# Get WAIC estimates
model_waic1 = az.waic(idata, var_name="LT_like")
print(model_waic1)
model_waic2 = az.waic(idata, var_name="SL_like")
print(model_waic2)
model_waic3 = az.waic(idata, var_name="LA_like")
print(model_waic3)

############################### PLOTTING #####################################
# plot the parameters and save the plots
az.plot_forest(trace, kind="forestplot", var_names=["mu_LT1"], hdi_prob=0.89)
plt.savefig("forest_mu_LT1.png")

az.plot_forest(trace, kind="forestplot", var_names=["mu_LT0"], hdi_prob=0.89)
plt.savefig("forest_mu_LT0.png")

az.plot_forest(trace, kind="forestplot", var_names=["mu_SL1"], hdi_prob=0.89)
plt.savefig("forest_mu_SL1.png")

az.plot_forest(trace, kind="forestplot", var_names=["mu_SL0"], hdi_prob=0.89)
plt.savefig("forest_mu_SL0.png")

az.plot_forest(trace, kind="forestplot", var_names=["kappa"], hdi_prob=0.89)
plt.savefig("forest_mu_kappa.png")

az.plot_forest(trace, kind="forestplot", var_names=["theta"], hdi_prob=0.89)
plt.savefig("forest_mu_theta.png")


az.plot_forest(trace, kind="forestplot", var_names=["lambda0"], hdi_prob=0.89)
plt.savefig("forest_lambda0.png")
az.plot_forest(trace, kind="forestplot", var_names=["beta_LA"], hdi_prob=0.89)
plt.savefig("forest_betaLA.png")

az.plot_forest(trace, kind="forestplot", var_names=["LT0"], hdi_prob=0.89)
plt.savefig("forest_LT0.png")
az.plot_forest(trace, kind="forestplot", var_names=["LT1"], hdi_prob=0.89)
plt.savefig("forest_LT1.png")

az.plot_forest(trace, kind="forestplot", var_names=["SL0"], hdi_prob=0.89)
plt.savefig("forest_SL0.png")
az.plot_forest(trace, kind="forestplot", var_names=["SL1"], hdi_prob=0.89)
plt.savefig("forest_SL1.png")


az.plot_forest(trace, kind="forestplot", var_names=["lambda0", "kappa"], hdi_prob=0.89)
plt.savefig("2forest_lambda0.png")

az.plot_forest(trace, kind="forestplot", var_names=["LT0", "mu_LT0"], hdi_prob=0.89)
plt.savefig("2forest_LT0.png")
az.plot_forest(trace, kind="forestplot", var_names=["LT1", "mu_LT1"], hdi_prob=0.89)
plt.savefig("2forest_LT1.png")

az.plot_forest(trace, kind="forestplot", var_names=["SL0", "mu_SL0"], hdi_prob=0.89)
plt.savefig("2forest_SL0.png")
az.plot_forest(trace, kind="forestplot", var_names=["SL1", "mu_SL1"], hdi_prob=0.89)
plt.savefig("2forest_SL1.png")


az.plot_trace(trace, var_names=["lambda0"])
plt.savefig("lambda0.png")
az.plot_trace(trace, var_names=["beta_LA"])
plt.savefig("beta_LA.png")

az.plot_trace(trace, var_names=["LT0"])
plt.savefig("LT0.png")
az.plot_trace(trace, var_names=["LT1"])
plt.savefig("LT1.png")

az.plot_trace(trace, var_names=["SL0"])
plt.savefig("SL0.png")
az.plot_trace(trace, var_names=["SL1"])
plt.savefig("SL1.png")

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

az.plot_trace(trace, var_names=["mu_LT0", "mu_LT1"])
plt.savefig("mu_LT.png")
az.plot_trace(trace, var_names=["mu_SL0", "mu_SL1"])
plt.savefig("mu_SL.png")
az.plot_trace(trace, var_names=["kappa", "theta", "mu_LA"])
plt.savefig("hyper_LA.png")

# Save summary statistics for the model parameters
with model:
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
    summary.to_csv("gen_summary.csv")


# put parameter values for each subject in dataframe...
posterior = pd.DataFrame()
posterior["subjnum"] = markasgood.values.reshape(
    -1,
)  # np.array(pd.read_csv('markasgood.csv', sep=',', header=None))[:,0]
# posterior["alpha"]=np.mean(trace["alpha"], axis=0)#[:,0]
posterior["LT0"] = np.median(trace["LT0"], axis=0)
posterior["LT1"] = np.median(trace["LT1"], axis=0)
posterior["SL0"] = np.median(trace["SL0"], axis=0)
posterior["SL1"] = np.median(trace["SL1"], axis=0)
posterior["lambda0"] = np.median(trace["lambda0"], axis=0)
posterior["beta_LA"] = np.median(trace["beta_LA"], axis=0)
# ...and save them
posterior.to_csv("posterior_median.csv")
