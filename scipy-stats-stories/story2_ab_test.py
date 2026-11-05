"""
Story 2 - "Which checkout page should the online shop use?" (an A/B/n test)
Visitors are randomly sent to the current checkout (control) or one of 3 new designs.
Two outcomes: (a) did the visitor buy? (yes/no)  (b) how much did a buyer spend?
"""
import numpy as np
from scipy import stats
from scipy.stats.contingency import association, odds_ratio
import matplotlib.pyplot as plt
from style import save, BLUE, ORANGE, AQUA, INK, INK2

rng = np.random.default_rng(2)
N = 8000
names = ["control", "design A", "design B", "design C"]
conv_true = [0.040, 0.041, 0.048, 0.039]
bought = [rng.random(N) < p for p in conv_true]

print("=== Part 1: conversion (chi-square) ===")
table = np.array([[b.sum(), N - b.sum()] for b in bought])
for nm, row in zip(names, table):
    print(f"{nm:9s} bought={row[0]:4d} not={row[1]:5d} rate={row[0] / N:.2%}")
chi = stats.chi2_contingency(table)
print(f"4x2 table: chi2={chi.statistic:.2f} dof={chi.dof} p={chi.pvalue:.4f}")
print("expected bought per group:", chi.expected_freq[:, 0].round(1))
print(f"Cramer's V={association(table):.3f}")
two = table[[0, 2]]
c2 = stats.chi2_contingency(two)
print(f"control vs B: chi2={c2.statistic:.2f} p={c2.pvalue:.4f}  (Yates correction on by default for 2x2)")
orr = odds_ratio(two[::-1])
ci = orr.confidence_interval()
print(f"odds ratio B vs control={orr.statistic:.2f} CI=[{ci.low:.2f},{ci.high:.2f}]")
rB, rC = two[1, 0] / N, two[0, 0] / N
print(f"risk difference={rB - rC:.4f} relative lift={(rB / rC - 1):.1%}")

print("\n=== Part 2: basket value of buyers ===")
# skewed basket values (euro); design B baskets slightly bigger, design A same, design C smaller
mean_true = [62, 62, 68, 59]
baskets = [rng.gamma(2.2, m / 2.2, size=b.sum()) for b, m in zip(bought, mean_true)]
for nm, x in zip(names, baskets):
    print(f"{nm:9s} n={len(x)} mean={x.mean():.2f} median={np.median(x):.2f} std={x.std(ddof=1):.1f}")
ctrl, B = baskets[0], baskets[2]

t = stats.ttest_ind(B, ctrl, equal_var=False)
ci = t.confidence_interval()
print(f"Welch B vs control: t={t.statistic:.2f} p={t.pvalue:.4f} CI=[{ci.low:.2f},{ci.high:.2f}] df={t.df:.0f}")
t_eq = stats.ttest_ind(B, ctrl)
print(f"Student (equal_var=True): p={t_eq.pvalue:.4f}")
mw = stats.mannwhitneyu(B, ctrl)
print(f"Mann-Whitney: p={mw.pvalue:.4f}")

def med_diff(x, y, axis):
    return np.median(x, axis=axis) - np.median(y, axis=axis)
bs = stats.bootstrap((B, ctrl), med_diff, n_resamples=5000, rng=np.random.default_rng(0))
print(f"bootstrap median diff={med_diff(B, ctrl, None):.2f} CI=[{bs.confidence_interval.low:.2f},"
      f"{bs.confidence_interval.high:.2f}] SE={bs.standard_error:.2f}")

def mean_diff(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
pt = stats.permutation_test((B, ctrl), mean_diff, n_resamples=9999,
                            alternative="two-sided", rng=np.random.default_rng(1))
print(f"permutation mean diff={pt.statistic:.2f} p={pt.pvalue:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].hist(bs.bootstrap_distribution, bins=60, color=BLUE, edgecolor="white", lw=0.3)
axes[0].axvline(bs.confidence_interval.low, color=ORANGE, lw=2)
axes[0].axvline(bs.confidence_interval.high, color=ORANGE, lw=2, label="95% CI limits")
axes[0].axvline(0, color=INK, lw=1, ls="--", label="no difference")
axes[0].set_title("Bootstrap: 5,000 re-computed median differences")
axes[0].set_xlabel("median basket B − control (euro)")
axes[0].set_yticks([]); axes[0].legend(loc="upper right")
axes[1].hist(pt.null_distribution, bins=60, color=BLUE, edgecolor="white", lw=0.3)
axes[1].axvline(pt.statistic, color=ORANGE, lw=2, label="real difference")
axes[1].set_title("Permutation: differences after shuffling labels")
axes[1].set_xlabel("mean basket difference (euro)")
axes[1].set_yticks([]); axes[1].legend(loc="upper left")
save(fig, "s2_boot_perm.png")

print("\n=== Part 3: Dunnett (3 designs vs control) ===")
res = stats.dunnett(*baskets[1:], control=ctrl)
cis = res.confidence_interval()
for i, nm in enumerate(names[1:]):
    print(f"{nm} vs control: diff={baskets[i + 1].mean() - ctrl.mean():.2f} p={res.pvalue[i]:.4f} "
          f"CI=[{cis.low[i]:.2f},{cis.high[i]:.2f}]")
naive = [stats.ttest_ind(x, ctrl).pvalue for x in baskets[1:]]
print("uncorrected t-test p:", np.round(naive, 4))

print("\n=== Part 4: tiny segment, Fisher + odds ratio ===")
# visitors using a smart-TV browser: very few
tv = np.array([[7, 23],    # design B: 7 bought of 30
               [1, 27]])   # control: 1 bought of 28
f = stats.fisher_exact(tv)
print(f"Fisher p={f.pvalue:.4f}")
chi_tv = stats.chi2_contingency(tv)
print("expected:", chi_tv.expected_freq.round(1), f"chi2 p={chi_tv.pvalue:.4f}")
o = odds_ratio(tv)
ci = o.confidence_interval()
print(f"odds ratio={o.statistic:.2f} CI=[{ci.low:.2f},{ci.high:.1f}]  simple (7/23)/(1/27)={(7/23)/(1/27):.1f}")

print("\n=== Part 5: 20 countries, multiple testing ===")
# design A has NO true effect anywhere; test it in 20 countries
pvals = []
for c in range(20):
    a = rng.gamma(2.2, 62 / 2.2, 300)
    b = rng.gamma(2.2, 62 / 2.2, 300)
    pvals.append(stats.ttest_ind(a, b, equal_var=False).pvalue)
pvals = np.array(pvals)
print("raw p < 0.05:", np.sum(pvals < 0.05), "smallest p:", np.sort(pvals)[:3].round(4))
adj = stats.false_discovery_control(pvals)
print("BH-adjusted smallest:", np.sort(adj)[:3].round(3), "significant after BH:", np.sum(adj < 0.05))
print("Bonferroni smallest:", np.minimum(np.sort(pvals)[:3] * 20, 1).round(3))
# probability of >=1 false alarm with 20 tests
print(f"P(at least one false alarm in 20 tests)={1 - 0.95 ** 20:.2f}")
