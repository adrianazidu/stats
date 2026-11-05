"""
Story 3 - "Quality control in a juice factory"
Three filling machines fill 1-litre (1000 ml) bottles. Questions:
  1. Is each machine filling 1000 ml on average?
  2. Is one machine less CONSISTENT than the others?
  3. Does maintenance change the fill (same machine, before/after)?
  4. Are defects per hour random (Poisson) or is something odd going on?
  5. Do defects depend on the work shift?
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from style import save, BLUE, ORANGE, AQUA, INK, INK2

rng = np.random.default_rng(3)
m1 = rng.normal(1000.3, 2.0, 120)
m2 = rng.normal(999.8, 2.2, 120)
m3 = rng.normal(1000.1, 3.6, 120)      # erratic machine
machines = {"machine 1": m1, "machine 2": m2, "machine 3": m3}

print("=== Q1: mean = 1000 ml? ===")
for nm, x in machines.items():
    t = stats.ttest_1samp(x, 1000)
    ci = t.confidence_interval()
    print(f"{nm}: mean={x.mean():.2f} std={x.std(ddof=1):.2f} t={t.statistic:.2f} p={t.pvalue:.3f} "
          f"CI=[{ci.low:.2f},{ci.high:.2f}]")

print("\n=== Q2a: normality per machine ===")
for nm, x in machines.items():
    print(f"{nm}: shapiro p={stats.shapiro(x).pvalue:.3f}")

print("\n=== Q2b: equal variances ===")
print(f"bartlett p={stats.bartlett(m1, m2, m3).pvalue:.2e}")
print(f"levene   p={stats.levene(m1, m2, m3).pvalue:.2e}")
print(f"fligner  p={stats.fligner(m1, m2, m3).pvalue:.2e}")
print(f"m1 vs m2 only: bartlett p={stats.bartlett(m1, m2).pvalue:.3f} levene p={stats.levene(m1, m2).pvalue:.3f} "
      f"fligner p={stats.fligner(m1, m2).pvalue:.3f}")

print("\n=== Q2c: a sensor glitch (3 crazy readings) in machine 1 ===")
m1g = m1.copy()
m1g[:3] = [1012.0, 987.0, 1013.5]
print(f"m1 std with glitch={m1g.std(ddof=1):.2f}")
print(f"m1g vs m2: bartlett p={stats.bartlett(m1g, m2).pvalue:.4f} levene p={stats.levene(m1g, m2).pvalue:.3f} "
      f"fligner p={stats.fligner(m1g, m2).pvalue:.3f}")
print(f"shapiro m1g p={stats.shapiro(m1g).pvalue:.2e}")

fig, ax = plt.subplots(figsize=(7, 3.2))
bp = ax.boxplot([m1, m2, m3], tick_labels=list(machines), patch_artist=True, widths=0.5,
                medianprops=dict(color=INK, lw=2))
for patch in bp["boxes"]:
    patch.set_facecolor("#cde2fb"); patch.set_edgecolor(BLUE)
ax.axhline(1000, color=ORANGE, ls="--", lw=1.5, label="target 1000 ml")
ax.set_ylabel("fill (ml)")
ax.set_title("Same average, different consistency")
ax.legend(loc="upper left")
save(fig, "s3_boxes.png")

print("\n=== Q3: before/after maintenance on machine 3 (same 25 bottle slots) ===")
before = rng.normal(1001.0, 3.0, 25)
after = before - 0.9 + rng.normal(0, 0.8, 25)       # each slot drops ~0.9 ml
print(f"before mean={before.mean():.2f} after mean={after.mean():.2f}")
print(f"WRONG ttest_ind p={stats.ttest_ind(before, after).pvalue:.3f}")
tr = stats.ttest_rel(before, after)
print(f"ttest_rel p={tr.pvalue:.2e} CI=[{tr.confidence_interval().low:.2f},{tr.confidence_interval().high:.2f}]")
print(f"wilcoxon p={stats.wilcoxon(before, after).pvalue:.2e}")

print("\n=== Q4: defects per hour ===")
defects = rng.poisson(2.4, 400)
lam = defects.mean()
model = stats.poisson(lam)
print(f"mean={lam:.2f} var={defects.var(ddof=1):.2f} ratio={defects.var(ddof=1) / lam:.2f}")
vals = np.arange(0, 6)
observed = [np.sum(defects == k) for k in vals] + [np.sum(defects >= 6)]
expected = list(model.pmf(vals) * len(defects)) + [model.sf(5) * len(defects)]
print("observed:", observed)
print("expected:", np.round(expected, 1))
gof = stats.chisquare(observed, expected, ddof=1)
print(f"chisquare p={gof.pvalue:.3f}")
print(f"P(>=7 defects in an hour)={model.sf(6):.4f}  -> about once every {1 / model.sf(6):.0f} hours")
print(f"alarm level (99.9%) = more than {model.ppf(0.999):.0f} defects")
# a bursty process for comparison (negative binomial, same mean)
bursty = rng.negative_binomial(n=1.2, p=1.2 / (1.2 + 2.4), size=400)
lamb = bursty.mean()
mb = stats.poisson(lamb)
ob = [np.sum(bursty == k) for k in vals] + [np.sum(bursty >= 6)]
eb = list(mb.pmf(vals) * 400) + [mb.sf(5) * 400]
print(f"bursty: mean={lamb:.2f} var={bursty.var(ddof=1):.2f} observed={ob} expected={np.round(eb, 1)} "
      f"chisquare p={stats.chisquare(ob, eb, ddof=1).pvalue:.2e}")

print("\n=== Q5: defects by shift ===")
shift = np.array([[38, 2962],    # morning: defective, ok
                  [41, 2959],    # afternoon
                  [66, 2934]])   # night
r = stats.chi2_contingency(shift)
print("rates:", (shift[:, 0] / shift.sum(axis=1)).round(4))
print(f"chi2={r.statistic:.2f} dof={r.dof} p={r.pvalue:.4f}")
print("expected defects:", r.expected_freq[:, 0].round(1))
contrib = (shift - r.expected_freq) ** 2 / r.expected_freq
print("contributions:", contrib.round(2))
from scipy.stats.contingency import association
print(f"Cramer's V={association(shift):.3f}")
