"""
Module 7 - What drives lateness? (CORRELATION)
Questions:
  a) Does a longer distance mean a longer delivery?   (straight-line relation)
  b) Does the hour of day matter?                     (U-shaped relation)

Covers: pearsonr, spearmanr, kendalltau, multiscale_graphcorr (MGC)
"""
from scipy import stats
import data

distance, t_dist, hour, t_hour = data.lateness_drivers()

def show_all(x, y):
    p = stats.pearsonr(x, y)
    s = stats.spearmanr(x, y)
    k = stats.kendalltau(x, y)
    print(f"  pearsonr   r   = {p.statistic:+.2f}, p = {p.pvalue:.4f}  (straight line)")
    print(f"  spearmanr  rho = {s.statistic:+.2f}, p = {s.pvalue:.4f}  (ranks, any steady up/down trend)")
    print(f"  kendalltau tau = {k.statistic:+.2f}, p = {k.pvalue:.4f}  (ranks, pairs that agree)")

print("=== a) distance vs delivery time ===")
show_all(distance, t_dist)
ci = stats.pearsonr(distance, t_dist).confidence_interval(0.95)
print(f"  Pearson 95% CI: [{ci.low:.2f}, {ci.high:.2f}]")

print("\n=== b) hour of day vs delivery time (slow at lunch AND dinner) ===")
show_all(hour, t_hour)
print("  -> the three coefficients see almost nothing: up and down cancel out!")

# MGC finds ANY kind of dependence, including U-shapes.
# It uses a permutation test internally, so it takes a few seconds.
print("\n  multiscale_graphcorr (MGC), please wait...")
mgc = stats.multiscale_graphcorr(hour, t_hour, reps=1000, random_state=0)
print(f"  MGC statistic = {mgc.statistic:.2f}, p = {mgc.pvalue:.4f}")
print("  -> MGC detects that the hour DOES matter" if mgc.pvalue < 0.05
      else "  -> MGC finds no dependence")
