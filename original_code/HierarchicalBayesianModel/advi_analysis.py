"""
Analyze significance of individual subject coefficients from the model.
Based on 89% credible intervals from gen_summary_rep.csv

"""
import pandas as pd

# load the summary statistics
summary = pd.read_csv("gen_summary_rep.csv", index_col=0)


# function to check if 89% CI exludes zero for a given parameter
def analyze_parameter(param_name):
    params = summary[summary.index.str.startswith(f"{param_name}[")]

    if len(params) == 0:
        print(f"No parameters found for {param_name}")
        return

    # check if 89% CI excludes zero
    # significant if lower bound > 0 OR upper bound < 0
    sig_positive = (params["hdi_5.5%"] > 0).sum()
    sig_negative = (params["hdi_94.5%"] < 0).sum()
    n_total = len(params)
    n_significant = sig_positive + sig_negative
    n_nonsig = n_total - n_significant

    # calculate percentages
    pct_sig = (n_significant / n_total) * 100
    pct_pos = (sig_positive / n_total) * 100
    pct_neg = (sig_negative / n_total) * 100
    pct_nonsig = (n_nonsig / n_total) * 100

    # print results
    print(f"\n{param_name} (n={n_total} subjects):")
    print(f"  Significantly positive: {sig_positive} ({pct_pos:.1f}%)")
    print(f"  Significantly negative: {sig_negative} ({pct_neg:.1f}%)")
    print(f"  Total significant: {n_significant} ({pct_sig:.1f}%)")
    print(f"  Non-significant (CI includes 0): {n_nonsig} ({pct_nonsig:.1f}%)")

    return {
        'parameter': param_name,
        'n_total': n_total,
        'sig_positive': sig_positive,
        'sig_negative': sig_negative,
        'total_significant': n_significant,
        'pct_significant': pct_sig
    }


# Analyze each subject-level parameter
# LT0, LT1 - looking time intercept and slope per subject
# SL0, SL1 - saccade latency intercept and slope per subject
# lambda0 - baseline look-away rate per subject
# beta_LA - look away effect on looking time per subject
results = []
for param in ["LT0", "LT1", "SL0", "SL1",
              "lambda0", "beta_LA"]:
    result = analyze_parameter(param)
    if result:
        results.append(result)

# create summary df for the six subject level parameters
summary_stats = pd.DataFrame(results)
print(summary_stats.to_string(index=False))

summary_stats.to_csv("advi_analysis.csv", index=False)
