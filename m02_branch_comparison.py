#compare branches, which is slower

import numpy as np
from scipy import stats
import data

a,b = data.two_branches()
print(f" a len {len(a)} mean {a.mean():.2f} median {a.median():.2f}")
print(f" b len {len(a)} mean {b.mean():.2f} median {b.median():.2f}")

""" t test (Welch) calculates a 2-sample independent t-test to check if two separate groups of data have significantly different means
 statistic - the t value 0 (the groups are identical) or large number (the means are far apart)
 pavalue - probability between 0 and 1 
"""
t = stats.ttest_ind(b,a,equal_var=False)
print(f"stats {t.statistic:.2f} p value {t.pvalue:.5f}")
ci = t.confidence_interval(0.95)
print(f"b is slower by {ci.low:.2f} to {ci.high:.2f} minutes")

"""
 Two-sample Kolmogorov-Smirno
Unlike a t-test (which only compares the means), the KS test compares the entire shape of two distributions—including their
 center, spread, and shape. It checks if two samples come from the same underlying distribution.
"""
ks = stats.ks_2samp(a, b)
print(f"stats ks {ks.statistic:.2f} p value {ks.pvalue:.5f}")

# ------------------------------------------------------------- bootstrap
#Bootstrapping is used to estimate how stable or precise a metric (like a mean, median, or correlation) is.
# no formula exists for this, so we resample the data many times
def median_diff(x, y):
    return np.median(y) - np.median(x)

res = stats.bootstrap((a, b), median_diff, n_resamples=5000,
                      rng=np.random.default_rng(1))
print(f"median difference = {median_diff(a, b):.2f} min")
print(f"95% CI = [{res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f}]")
print("(the median is noisier than the mean: if the CI includes 0, the median")
print(" difference alone is not enough evidence)")

# -------------------------------------------------------- permutation test
#A permutation test is used to see if two groups are truly different (an alternative to the T-test or KS-test).
# idea: if the branch label doesn't matter, shuffling labels shouldn't change anything
def mean_diff(x, y):
    return np.mean(y) - np.mean(x)

perm = stats.permutation_test((a, b), mean_diff, n_resamples=5000,
                              alternative="greater", rng=np.random.default_rng(2))
print(f"observed difference = {perm.statistic:.2f} min, p = {perm.pvalue:.4f}")
print("(no assumption about the shape of the data is needed)")