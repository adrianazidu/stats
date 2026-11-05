"""
Story 4 - "Keeping a fraud-detection model healthy"
A bank uses a machine-learning model that gives each card payment a risk score.
The data team must check every month:
  1. Has the INPUT data changed since training? (data drift)
  2. Is today's number of fraud alerts normal?
  3. Was last night's batch of scores consistent with what the model usually produces?
  4. Which of three candidate new models is best?
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from style import save, BLUE, ORANGE, AQUA, INK, INK2

rng = np.random.default_rng(4)
n_train, n_now = 20000, 20000

print("=== 1. Data drift: ks_2samp vs t-test ===")
train = {
    "amount":   rng.lognormal(3.5, 0.9, n_train),
    "hour":     rng.normal(15, 4, n_train),
    "distance": rng.gamma(2, 10, n_train),
    "age":      rng.normal(42, 12, n_train),
}
now = {
    "amount":   rng.lognormal(3.5, 0.9, n_now),          # unchanged
    "hour":     rng.normal(15, 4, n_now),                # unchanged
    "distance": rng.gamma(2, 10.6, n_now),               # small shift of the mean (+6%)
    # same MEAN, different SHAPE: a new group of very young and very old customers
    "age":      np.concatenate([rng.normal(42, 12, 14000),
                                rng.choice([22.0, 62.0], 6000) + rng.normal(0, 2, 6000)]),
}
for f in train:
    ks = stats.ks_2samp(train[f], now[f])
    tt = stats.ttest_ind(train[f], now[f], equal_var=False)
    print(f"{f:9s} mean train={train[f].mean():7.2f} now={now[f].mean():7.2f}  "
          f"t-test p={tt.pvalue:.3g}  KS D={ks.statistic:.4f} p={ks.pvalue:.3g} "
          f"location={ks.statistic_location:.1f}")

x = np.sort(train["age"]); y = np.sort(now["age"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].hist(train["age"], bins=80, density=True, color=BLUE, alpha=0.5, label="training data")
axes[0].hist(now["age"], bins=80, density=True, color=ORANGE, alpha=0.5, label="this month")
axes[0].set_title("Customer age: same average, new shape")
axes[0].set_xlabel("age"); axes[0].set_yticks([]); axes[0].legend()
axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), color=BLUE, label="training data")
axes[1].plot(y, np.arange(1, len(y) + 1) / len(y), color=ORANGE, label="this month")
ks = stats.ks_2samp(train["age"], now["age"])
loc = ks.statistic_location
axes[1].axvline(loc, color=INK, ls="--", lw=1)
axes[1].annotate(f"biggest gap D = {ks.statistic:.3f}\nat age {loc:.0f}", (loc, 0.2),
                 xytext=(loc + 8, 0.12), color=INK)
axes[1].set_title("KS compares the two cumulative curves")
axes[1].set_xlabel("age"); axes[1].set_ylabel("cumulative share"); axes[1].legend(loc="upper left")
save(fig, "s4_drift.png")

print("\n=== 2. Alerts per hour: Poisson anomaly check ===")
normal_rate = 4.0
alerts = stats.poisson(normal_rate)
for seen in [7, 9, 12]:
    print(f"seen {seen}: P(>= {seen}) = {alerts.sf(seen - 1):.5f}")
print(f"alarm threshold for 1 false alarm per ~1000 hours: more than {alerts.ppf(0.999):.0f}")
print(f"P(exactly 0 alerts in an hour)={alerts.pmf(0):.4f}")

print("\n=== 3. Last night's scores vs the usual model (monte_carlo_test) ===")
usual = stats.beta(2, 18)                   # usual risk scores: mean 0.10
last_night = stats.beta(2, 14).rvs(60, random_state=11)   # secretly higher
print(f"usual mean={usual.mean():.3f}, last night mean={last_night.mean():.3f} (n=60)")
mc = stats.monte_carlo_test(last_night, lambda size: usual.rvs(size=size, random_state=rng),
                            np.mean, alternative="greater", n_resamples=9999)
print(f"monte_carlo_test p={mc.pvalue:.4f}")
mc90 = stats.monte_carlo_test(last_night,
                              lambda size: usual.rvs(size=size, random_state=rng),
                              lambda x, axis=-1: np.quantile(x, 0.9, axis=axis),
                              alternative="greater", n_resamples=9999)
print(f"same test with the 90th percentile: real={np.quantile(last_night, 0.9):.3f} "
      f"usual={usual.ppf(0.9):.3f} p={mc90.pvalue:.4f}")
normal_night = usual.rvs(60, random_state=5)
mcn = stats.monte_carlo_test(normal_night, lambda size: usual.rvs(size=size, random_state=rng),
                             np.mean, alternative="greater", n_resamples=9999)
print(f"a normal night: mean={normal_night.mean():.3f} p={mcn.pvalue:.3f}")

print("\n=== 4. Three candidate models on 24 monthly back-tests (Friedman) ===")
months = 24
difficulty = rng.normal(0, 0.03, months)    # some months are harder for everybody
auc_A = 0.86 + difficulty + rng.normal(0, 0.006, months)
auc_B = 0.87 + difficulty + rng.normal(0, 0.006, months)
auc_C = 0.861 + difficulty + rng.normal(0, 0.006, months)
for nm, a in [("A", auc_A), ("B", auc_B), ("C", auc_C)]:
    print(f"model {nm}: mean AUC={a.mean():.4f}  min={a.min():.3f} max={a.max():.3f}")
print("first 4 months:\n", np.column_stack([auc_A, auc_B, auc_C])[:4].round(3))
ranks = stats.rankdata(-np.column_stack([auc_A, auc_B, auc_C]), axis=1)
print("mean rank (1=best):", ranks.mean(axis=0).round(2))
fr = stats.friedmanchisquare(auc_A, auc_B, auc_C)
print(f"Friedman stat={fr.statistic:.2f} p={fr.pvalue:.2e}")
print(f"WRONG kruskal (ignores pairing) p={stats.kruskal(auc_A, auc_B, auc_C).pvalue:.3f}")
pairs = [("B vs A", auc_B, auc_A), ("B vs C", auc_B, auc_C), ("A vs C", auc_A, auc_C)]
raw = np.array([stats.wilcoxon(x, y).pvalue for _, x, y in pairs])
adj = stats.false_discovery_control(raw)
for (nm, _, _), p, q in zip(pairs, raw, adj):
    print(f"wilcoxon {nm}: raw p={p:.2e} adjusted={q:.2e}")
print(f"B - A mean AUC diff={np.mean(auc_B - auc_A):.4f}")
