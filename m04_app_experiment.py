"""
Module 4 - App redesign experiment
Questions:
  a) Do the two new checkout designs raise the average order value vs the current app?
  b) Does the new app convert more visitors into customers?
  c) In a tiny coupon test, does the coupon bring customers back?

Covers: dunnett, chi2_contingency, fisher_exact, odds_ratio
"""
from scipy import stats
from scipy.stats.contingency import odds_ratio
import data

# ----------------------------------------------------------------- Dunnett
print("=== a) Dunnett's test: each new design vs the control ===")
control, design_1, design_2 = data.app_experiment()
print(f"control  mean = {control.mean():.2f} euro")
print(f"design 1 mean = {design_1.mean():.2f} euro")
print(f"design 2 mean = {design_2.mean():.2f} euro")

# Dunnett compares every group to ONE control, correcting for multiple comparisons
res = stats.dunnett(design_1, design_2, control=control)
for name, p in zip(["design 1", "design 2"], res.pvalue):
    verdict = "better/different" if p < 0.05 else "no real effect"
    print(f"{name} vs control: p = {p:.4f}  -> {verdict}")

# ------------------------------------------------------------ chi-square
print("\n=== b) Chi-square test of independence (contingency table) ===")
table = data.conversion_table()
print("            ordered  left")
print(f"old app     {table[0, 0]:6d}  {table[0, 1]:5d}   ({table[0, 0] / table[0].sum():.1%})")
print(f"new app     {table[1, 0]:6d}  {table[1, 1]:5d}   ({table[1, 0] / table[1].sum():.1%})")

chi = stats.chi2_contingency(table)
print(f"chi2 = {chi.statistic:.2f}, p = {chi.pvalue:.4f}, dof = {chi.dof}")
print("counts we would expect if the app made NO difference:")
print(chi.expected_freq.round(1))

# --------------------------------------------------------- Fisher + odds
print("\n=== c) Small coupon test: Fisher's exact test + odds ratio ===")
small = data.coupon_test()
print("              came back  didn't")
print(f"coupon          {small[0, 0]:5d}  {small[0, 1]:6d}")
print(f"no coupon       {small[1, 0]:5d}  {small[1, 1]:6d}")
print("(counts this small make the chi-square test unreliable -> use Fisher)")

f = stats.fisher_exact(small)
print(f"Fisher exact p = {f.pvalue:.4f}")

orr = odds_ratio(small)
ci = orr.confidence_interval(0.95)
print(f"odds ratio = {orr.statistic:.1f}  (95% CI {ci.low:.1f} to {ci.high:.1f})")
print("-> with a coupon, the odds of coming back are this many times higher")
