"""
Story 5 - "Understanding the customers of a food-delivery app"
Questions:
  1. Do customers who open the app more often spend more?          (Pearson vs Spearman)
  2. Does inviting friends go together with spending?               (Kendall, many ties)
  3. Does the weather (temperature) affect the number of orders?    (U-shape -> MGC)
  4. At what times of day do people order?                          (1-D KDE)
  5. Where in the city are the customers?                           (2-D KDE)
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from style import save, BLUE, ORANGE, AQUA, INK, INK2

rng = np.random.default_rng(5)
n = 3000

print("=== 1. Sessions vs spending (with 'whales') ===")
sessions = rng.poisson(6, n) + 1
spend = 4 * sessions * rng.lognormal(0, 0.45, n)
whales = rng.choice(n, 15, replace=False)
spend[whales] += rng.uniform(1500, 4000, 15)     # 15 corporate accounts
p = stats.pearsonr(sessions, spend)
s = stats.spearmanr(sessions, spend)
print(f"with whales: pearson r={p.statistic:.3f} p={p.pvalue:.2e} CI=[{p.confidence_interval().low:.3f},"
      f"{p.confidence_interval().high:.3f}]  spearman rho={s.statistic:.3f}")
mask = np.ones(n, bool); mask[whales] = False
p2 = stats.pearsonr(sessions[mask], spend[mask])
s2 = stats.spearmanr(sessions[mask], spend[mask])
print(f"without whales: pearson r={p2.statistic:.3f} spearman rho={s2.statistic:.3f}")
print(f"r^2 with whales={p.statistic ** 2:.3f}, without={p2.statistic ** 2:.3f}")
small = rng.choice(np.flatnonzero(mask), 40, replace=False)
ps = stats.pearsonr(sessions[small], spend[small])
print(f"only 40 customers (no whales): r={ps.statistic:.2f} CI=[{ps.confidence_interval().low:.2f},{ps.confidence_interval().high:.2f}]")

print("\n=== 2. Friends invited (0-4, lots of ties) vs spending ===")
friends = np.minimum(rng.poisson(0.25 + sessions / 12), 4)
print("friends value counts:", np.bincount(friends))
k = stats.kendalltau(friends, spend[:n])
s3 = stats.spearmanr(friends, spend)
print(f"kendall tau={k.statistic:.3f} p={k.pvalue:.2e}   spearman rho={s3.statistic:.3f}")

print("\n=== 3. Temperature vs daily orders (U-shape) ===")
days = 365
temp = rng.uniform(-5, 38, days)
orders = 900 + 1.6 * (temp - 16) ** 2 + rng.normal(0, 90, days)
print(f"pearson r={stats.pearsonr(temp, orders).statistic:.3f} p={stats.pearsonr(temp, orders).pvalue:.3f}")
print(f"spearman rho={stats.spearmanr(temp, orders).statistic:.3f} p={stats.spearmanr(temp, orders).pvalue:.3f}")
print(f"kendall tau={stats.kendalltau(temp, orders).statistic:.3f} p={stats.kendalltau(temp, orders).pvalue:.3f}")
mgc = stats.multiscale_graphcorr(temp, orders, reps=1000, random_state=0)
print(f"MGC stat={mgc.statistic:.3f} p={mgc.pvalue:.4f} opt_scale={mgc.mgc_dict['opt_scale']}")
noise = rng.normal(0, 1, days)
mgc0 = stats.multiscale_graphcorr(temp, noise, reps=1000, random_state=0)
print(f"MGC on pure noise: p={mgc0.pvalue:.3f}")
cold = temp < 16
print(f"split: cold days rho={stats.spearmanr(temp[cold], orders[cold]).statistic:.2f}, "
      f"warm days rho={stats.spearmanr(temp[~cold], orders[~cold]).statistic:.2f}")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.scatter(temp, orders, s=10, color=BLUE, alpha=0.7)
ax.set_xlabel("temperature (°C)"); ax.set_ylabel("orders that day")
ax.set_title(f"Pearson r = {stats.pearsonr(temp, orders).statistic:.2f}, yet a clear pattern")
save(fig, "s5_temp_u.png")

print("\n=== 4. Order times (1-D KDE) ===")
lunch = rng.normal(12.7, 0.7, 1300)
dinner = rng.normal(20.0, 1.1, 2200)
late = rng.uniform(22, 24.0, 150)
t = np.concatenate([lunch, dinner, late])
t = t[(t > 9) & (t < 24)]
kde = stats.gaussian_kde(t)
grid = np.linspace(9, 24, 600)
dens = kde(grid)
peaks = [grid[i] for i in range(1, len(grid) - 1) if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]]
print("peaks at:", [f"{int(p)}:{int(p % 1 * 60):02d}" for p in peaks])
print(f"P(order between 15:00 and 17:00) KDE={kde.integrate_box_1d(15, 17):.3f} data={np.mean((t > 15) & (t < 17)):.3f}")
print(f"density at 3:40 pm={kde(15.67)[0]:.4f}, at 8 pm={kde(20.0)[0]:.4f}")
nf = stats.norm(*stats.norm.fit(t))
print(f"normal fit puts its peak at {nf.mean():.1f} h (= {int(nf.mean())}:{int(nf.mean() % 1 * 60):02d})")
print(f"kde factor (Scott)={kde.factor:.3f}, bandwidth in hours={kde.factor * t.std(ddof=1):.2f}")
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(t, bins=60, density=True, color="#cde2fb", edgecolor="white", lw=0.4, label="data")
ax.plot(grid, dens, color=BLUE, label="KDE (automatic bandwidth)")
ax.plot(grid, stats.gaussian_kde(t, bw_method=0.6)(grid), color=ORANGE, ls="--", label="bandwidth too wide")
ax.plot(grid, stats.gaussian_kde(t, bw_method=0.015)(grid), color=INK2, ls=":", lw=1.4, label="bandwidth too narrow")
ax.plot(grid, nf.pdf(grid), color=AQUA, lw=1.6, label="single normal curve (wrong)")
ax.set_xlabel("hour of day"); ax.set_ylabel("density")
ax.set_title("When do people order? Two peaks a single formula cannot show")
ax.legend(fontsize=8, loc="upper left")
save(fig, "s5_kde_hours.png")

print("\n=== 5. Customer map (2-D KDE) ===")
centre = rng.normal([0, 0], [1.2, 1.0], (1800, 2))
campus = rng.normal([4, 2], [0.5, 0.5], (700, 2))
offices = rng.normal([-3.5, -2.5], [0.5, 0.4], (500, 2))
pts = np.vstack([centre, campus, offices])
kde2 = stats.gaussian_kde(pts.T)
xs, ys = np.meshgrid(np.linspace(-6, 7, 150), np.linspace(-4, 6, 150))
Z = kde2(np.vstack([xs.ravel(), ys.ravel()])).reshape(xs.shape)
for nm, loc in [("centre", (0, 0)), ("campus", (4, 2)), ("offices", (-3.5, -2.5)), ("suburb", (5, -3))]:
    print(f"density at {nm} {loc}: {kde2(loc)[0]:.4f}")
i = np.argmax(Z)
print(f"highest density at ({xs.ravel()[i]:.1f}, {ys.ravel()[i]:.1f})")
print("3 simulated customers:", kde2.resample(3, seed=1).T.round(2).tolist())
fig, ax = plt.subplots(figsize=(6.2, 4.4))
cs = ax.contourf(xs, ys, Z, levels=14, cmap="Blues")
ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color=INK, alpha=0.25)
fig.colorbar(cs, ax=ax, label="customer density")
for nm, (a, b) in [("centre", (0, 0)), ("campus", (4, 2)), ("offices", (-3.5, -2.5))]:
    ax.annotate(nm, (a, b), xytext=(a + 0.6, b - 1.0), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK))
ax.set_xlabel("km east"); ax.set_ylabel("km north"); ax.grid(False)
ax.set_title("Where do the customers live?")
save(fig, "s5_map.png")
