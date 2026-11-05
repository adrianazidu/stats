"""
Module 5 - Order volume (DISCRETE distribution)
Question: do the orders per hour follow a Poisson model? If yes, we can use it
to plan the kitchen.

Covers: discrete distribution (poisson: pmf, cdf, sf, ppf, rvs),
        chisquare goodness-of-fit test
"""
import numpy as np
from scipy import stats
import data

orders = data.orders_per_hour()
lam = orders.mean()                       # Poisson has one parameter: the mean
print(f"{len(orders)} hours observed, average = {lam:.2f} orders/hour")

model = stats.poisson(lam)                # frozen discrete distribution

# -------------------------------------------------- pmf, cdf, sf, ppf, rvs
print("\n=== 1. Using the Poisson model ===")
print(f"P(exactly 7 orders)   = {model.pmf(7):.3f}   (pmf: probability of ONE value)")
print(f"P(at most 5 orders)   = {model.cdf(5):.3f}   (cdf)")
print(f"P(more than 10)       = {model.sf(10):.3f}   (sf)")
print(f"95% of hours have at most {model.ppf(0.95):.0f} orders  (ppf)")
print(f"a simulated evening (6 hours): {model.rvs(6, random_state=0)}")

# pmf of several values at once
ks = np.arange(0, 15)
print("pmf for 0..14:", model.pmf(ks).round(3))

# -------------------------------------------- chi-square goodness of fit
print("\n=== 2. Chi-square goodness-of-fit: does Poisson fit the data? ===")
# group the counts into bins: <=3, 4, 5, ..., 10, >=11
# (chi-square needs about 5+ expected observations per bin)
edges = list(range(4, 11))                # 4..10 get their own bin
observed = [np.sum(orders <= 3)]
observed += [np.sum(orders == k) for k in edges]
observed += [np.sum(orders >= 11)]

expected = [model.cdf(3)]
expected += [model.pmf(k) for k in edges]
expected += [model.sf(10)]
expected = np.array(expected) * len(orders)

print("bin:      <=3  " + "  ".join(f"{k:4d}" for k in edges) + "  >=11")
print("observed:", np.array(observed))
print("expected:", expected.round(1))

# ddof=1 because we estimated one parameter (lambda) from the data
res = stats.chisquare(observed, expected, ddof=1)
print(f"chi2 = {res.statistic:.2f}, p = {res.pvalue:.3f}")
print("-> Poisson fits well" if res.pvalue > 0.05 else "-> Poisson does NOT fit")
