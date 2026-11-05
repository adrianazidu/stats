"""
Story 1 - "Is the online shop keeping its delivery promise?"
An electronics shop promises: "average delivery under 48 hours".
Data: 5,000 shipments from last quarter (hours from order to doorstep).
"""
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
from style import save, BLUE, ORANGE, AQUA, INK, INK2

rng = np.random.default_rng(1)
hours = rng.gamma(shape=6, scale=8.3, size=4985)          # typical: mean ~50 h
lost = rng.uniform(300, 700, size=15)                     # 15 parcels lost for weeks
hours = np.concatenate([hours, lost])

print("=== Step 1: describe ===")
d = stats.describe(hours)
print(f"n={d.nobs} min={d.minmax[0]:.1f} max={d.minmax[1]:.1f} mean={d.mean:.2f} "
      f"var={d.variance:.1f} std={np.sqrt(d.variance):.1f} skew={d.skewness:.2f} kurt={d.kurtosis:.2f}")
print(f"median={np.median(hours):.2f}  p25,p75={np.percentile(hours, [25, 75]).round(1)} "
      f"IQR={stats.iqr(hours):.1f}")

print("\n=== Step 2: outliers ===")
print(f"plain mean={hours.mean():.2f}")
print(f"trim_mean 1%={stats.trim_mean(hours, 0.01):.2f}")
w = winsorize(hours, limits=(0, 0.01))
print(f"winsorized top 1% mean={w.mean():.2f}, new max={w.max():.1f}")
print(f"number above 200 h: {(hours > 200).sum()}")
clean = hours[hours < 200]
print(f"clean n={len(clean)} mean={clean.mean():.2f} skew={stats.skew(clean):.2f} "
      f"kurt={stats.kurtosis(clean):.2f}")

print("\n=== Step 3: normality ===")
small = rng.choice(clean, 500, replace=False)
print(f"shapiro (500 sample) p={stats.shapiro(small).pvalue:.2e}")
print(f"normaltest p={stats.normaltest(clean).pvalue:.2e}")
print(f"skewtest p={stats.skewtest(clean).pvalue:.2e}  stat={stats.skewtest(clean).statistic:.1f}")
print(f"kurtosistest p={stats.kurtosistest(clean).pvalue:.2e}")
print(f"jarque_bera p={stats.jarque_bera(clean).pvalue:.2e}")

print("\n=== Step 4+5: fit candidates and KS ===")
cands = {"gamma": stats.gamma, "lognorm": stats.lognorm,
         "weibull_min": stats.weibull_min, "norm": stats.norm}
fits = {}
for name, dist in cands.items():
    params = dist.fit(clean) if name == "norm" else dist.fit(clean, floc=0)
    fits[name] = dist(*params)
    ks = stats.kstest(clean, fits[name].cdf)
    print(f"{name:12s} params={np.round(params, 3)}  D={ks.statistic:.4f} p={ks.pvalue:.3g}")
best = fits["gamma"]
# honest p-value: fit on half, test on the other half
half = rng.permutation(clean)
a, b = half[:2500], half[2500:]
g_half = stats.gamma(*stats.gamma.fit(a, floc=0))
print(f"gamma fitted on half A, tested on half B: p={stats.kstest(b, g_half.cdf).pvalue:.3f}")

x = np.linspace(0, 160, 400)
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(clean, bins=70, density=True, color="#cde2fb", edgecolor="white", lw=0.4, label="data")
ax.plot(x, fits["gamma"].pdf(x), color=BLUE, label="gamma fit")
ax.plot(x, fits["lognorm"].pdf(x), color=ORANGE, ls="--", label="lognormal fit")
ax.plot(x, fits["norm"].pdf(x), color=INK2, ls=":", label="normal fit")
ax.set_xlabel("hours from order to delivery")
ax.set_ylabel("density")
ax.set_title("Which curve describes our delivery times?")
ax.legend()
save(fig, "s1_fits.png")

print("\n=== Step 6: tails ===")
print(f"P(>72h) model={best.sf(72):.4f} data={np.mean(clean > 72):.4f}")
print(f"P(>96h) model={best.sf(96):.5f} data={np.mean(clean > 96):.5f}")
for q in [0.5, 0.9, 0.95, 0.99]:
    print(f"ppf({q}) model={best.ppf(q):.1f}  empirical={np.quantile(clean, q):.1f}")

print("\n=== Step 7: one-sample t-test vs 48 ===")
t = stats.ttest_1samp(hours, popmean=48, alternative="greater")
ci = t.confidence_interval(0.95)
print(f"all data: t={t.statistic:.2f} p={t.pvalue:.2e} lower bound={ci.low:.2f} mean={hours.mean():.2f}")
t2 = stats.ttest_1samp(hours, popmean=48)
ci2 = t2.confidence_interval(0.95)
print(f"two-sided CI: [{ci2.low:.2f}, {ci2.high:.2f}]  p={t2.pvalue:.2e}")
tc = stats.ttest_1samp(clean, popmean=48, alternative="greater")
print(f"clean data: t={tc.statistic:.2f} p={tc.pvalue:.2e} lower={tc.confidence_interval().low:.2f}")
# small sample: only first week (80 parcels)
wk = hours[:80]
tw = stats.ttest_1samp(wk, 48, alternative="greater")
print(f"one week (n=80): mean={wk.mean():.2f} t={tw.statistic:.2f} p={tw.pvalue:.3f}")
