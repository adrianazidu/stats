"""Figures and small numeric examples for Part A (foundations)."""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from style import save, BLUE, ORANGE, AQUA, INK, INK2, MUTED

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- A2: shapes
sym = rng.normal(50, 10, 4000)
skw = rng.gamma(2, 12, 4000) + 10
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, x, title in [(axes[0], sym, "Symmetric (skewness ≈ 0)"),
                     (axes[1], skw, "Right-skewed (skewness > 0)")]:
    ax.hist(x, bins=50, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(x), color=ORANGE, lw=2, label=f"mean = {np.mean(x):.0f}")
    ax.axvline(np.median(x), color=INK, lw=2, ls="--", label=f"median = {np.median(x):.0f}")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.legend(loc="upper right")
axes[0].set_ylabel("how many values")
save(fig, "a_shapes.png")
print("sym skew", stats.skew(sym).round(2), "skw skew", stats.skew(skw).round(2))

# ------------------------------------------------- A2: mean vs median example
salaries = np.array([1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 45000])
print("salaries mean", salaries.mean(), "median", np.median(salaries),
      "trim_mean 10%", stats.trim_mean(salaries, 0.1))
print("std", salaries.std(ddof=1).round(0), "IQR", stats.iqr(salaries))

# --------------------------------------------- A3: pdf / cdf / sf / ppf picture
d = stats.gamma(a=4, scale=8)
x = np.linspace(0, 100, 400)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
ax = axes[0]
ax.plot(x, d.pdf(x), color=BLUE)
xs = x[x <= 40]
ax.fill_between(xs, d.pdf(xs), color=BLUE, alpha=0.25, label=f"cdf(40) = {d.cdf(40):.2f}")
xs = x[x >= 40]
ax.fill_between(xs, d.pdf(xs), color=ORANGE, alpha=0.35, label=f"sf(40) = {d.sf(40):.2f}")
ax.axvline(40, color=INK2, lw=1)
ax.set_title("pdf: the shape. Areas are probabilities")
ax.set_xlabel("delivery time (min)")
ax.set_ylabel("density")
ax.legend()
ax = axes[1]
ax.plot(x, d.cdf(x), color=BLUE)
q = d.ppf(0.9)
ax.plot([0, q], [0.9, 0.9], color=ORANGE, lw=1.5, ls="--")
ax.plot([q, q], [0, 0.9], color=ORANGE, lw=1.5, ls="--")
ax.annotate(f"ppf(0.9) = {q:.0f} min", (q, 0.05), xytext=(q + 4, 0.25), color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2))
ax.set_title("cdf: share of values below x")
ax.set_xlabel("delivery time (min)")
ax.set_ylabel("cumulative share")
save(fig, "a_pdf_cdf.png")
print("cdf40", d.cdf(40), "sf40", d.sf(40), "ppf0.9", d.ppf(0.9), "mean", d.mean())

# ----------------------------------------------- A3: common distributions
fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
x = np.linspace(-4, 4, 300)
axes[0].plot(x, stats.norm.pdf(x), color=BLUE); axes[0].set_title("Normal")
x = np.linspace(0, 40, 300)
axes[1].plot(x, stats.gamma(3, scale=4).pdf(x), color=BLUE); axes[1].set_title("Gamma")
axes[2].plot(x, stats.lognorm(0.7, scale=8).pdf(x), color=BLUE); axes[2].set_title("Lognormal")
k = np.arange(0, 16)
axes[3].bar(k, stats.poisson(5).pmf(k), color=BLUE, width=0.7); axes[3].set_title("Poisson (counts)")
for a in axes:
    a.set_yticks([])
save(fig, "a_distributions.png")

# ------------------------------------------------ A4: central limit theorem
pop = stats.expon(scale=30)
means5 = pop.rvs((20000, 5), random_state=1).mean(axis=1)
means50 = pop.rvs((20000, 50), random_state=2).mean(axis=1)
fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
axes[0].hist(pop.rvs(20000, random_state=3), bins=60, color=BLUE, edgecolor="white", lw=0.4)
axes[0].set_title("Single values (very skewed)")
axes[1].hist(means5, bins=60, color=BLUE, edgecolor="white", lw=0.4)
axes[1].set_title("Means of 5 values")
axes[2].hist(means50, bins=60, color=BLUE, edgecolor="white", lw=0.4)
axes[2].set_title("Means of 50 values (bell-shaped)")
for a in axes:
    a.set_yticks([]); a.set_xlim(0, 120); a.set_xlabel("minutes")
save(fig, "a_clt.png")
print("SE theory n=50", 30 / np.sqrt(50), "observed", means50.std())

# --------------------------------------------- A5: p-value picture
t = np.linspace(-4, 4, 400)
obs = 2.3
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(t, stats.t.pdf(t, 40), color=BLUE)
m = t >= obs
ax.fill_between(t[m], stats.t.pdf(t[m], 40), color=ORANGE, alpha=0.5)
m = t <= -obs
ax.fill_between(t[m], stats.t.pdf(t[m], 40), color=ORANGE, alpha=0.5)
ax.axvline(obs, color=INK, lw=1.5)
p = 2 * stats.t.sf(obs, 40)
ax.annotate(f"our result: t = {obs}", (obs, 0.3), xytext=(obs + 0.2, 0.33), color=INK)
ax.annotate(f"orange area = p-value = {p:.3f}", (3, 0.02), xytext=(1.1, 0.15), color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2))
ax.set_title("What results look like when nothing is going on (H0 true)")
ax.set_xlabel("test statistic")
ax.set_yticks([])
save(fig, "a_pvalue.png")
print("p two-sided", p)

# ------------------------------------------ A5: power vs sample size (simulated)
def power(n, diff=2, sd=10, reps=2000):
    a = rng.normal(0, sd, (reps, n))
    b = rng.normal(diff, sd, (reps, n))
    return np.mean(stats.ttest_ind(a, b, axis=1).pvalue < 0.05)
for n in [20, 100, 400, 1000]:
    print("power n per group", n, power(n))

# ------------------------------------------- correlation shapes
n = 300
x = rng.uniform(0, 10, n)
ys = [2 * x + rng.normal(0, 3, n),
      np.exp(x / 2) + rng.normal(0, 5, n),
      (x - 5) ** 2 + rng.normal(0, 2, n)]
titles = ["Straight line", "Curved but always rising", "U-shape"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for a, y, ttl in zip(axes, ys, titles):
    a.scatter(x, y, s=8, color=BLUE, alpha=0.6)
    r = stats.pearsonr(x, y).statistic
    rho = stats.spearmanr(x, y).statistic
    a.set_title(f"{ttl}\nPearson r = {r:.2f}, Spearman ρ = {rho:.2f}")
    a.set_xticks([]); a.set_yticks([])
save(fig, "a_corr_shapes.png")
