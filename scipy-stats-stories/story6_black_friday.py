"""
Story 6 - "Planning Black Friday in a warehouse"
An online retailer's warehouse must decide how many pickers (people who collect the items
of each order from the shelves) to hire for Black Friday, and when their shifts start.
We build a simulator out of probability distributions, run it thousands of times,
and then search for the cheapest good plan with Quasi-Monte Carlo.
"""
import numpy as np
from scipy import stats
from scipy.stats import qmc
from scipy.stats.sampling import NumericalInversePolynomial
import matplotlib.pyplot as plt
from style import save, BLUE, ORANGE, AQUA, INK, INK2

rng = np.random.default_rng(6)

print("=== 1. Frozen distribution with shape, loc, scale ===")
pick = stats.gamma(a=3, loc=1.5, scale=1.2)       # minutes to pick one order
print(f"mean={pick.mean():.2f} min, std={pick.std():.2f}, median={pick.median():.2f}, "
      f"minimum (support)={pick.support()[0]}")
print(f"P(order takes > 8 min)={pick.sf(8):.4f}  90th percentile={pick.ppf(0.9):.2f} min")
print(f"orders per picker per hour = 60 / mean = {60 / pick.mean():.1f}")
print(f"same thing by hand: gamma(3).mean()*1.2 + 1.5 = {stats.gamma(3).mean() * 1.2 + 1.5:.2f}")

print("\n=== 2. Broadcasting: what-if table ===")
scales = np.array([[1.0], [1.2], [1.5]])        # experienced / normal / new temporary staff
limits = np.array([6, 8, 10, 12])
tab = stats.gamma.sf(limits, a=3, loc=1.5, scale=scales)
print("P(order > limit)     6 min   8 min  10 min  12 min")
for nm, row in zip(["experienced", "normal", "new temps"], tab):
    print(f"{nm:18s}" + "".join(f"{p:8.3f}" for p in row))
means = stats.gamma.mean(a=3, loc=1.5, scale=scales.ravel())
print("mean minutes:", means.round(2), "orders/hour:", (60 / means).round(1))

print("\n=== 3. New random-variable API ===")
Gamma = stats.make_distribution(stats.gamma)
pick_new = 1.2 * Gamma(a=3) + 1.5
print(f"new API mean={pick_new.mean():.2f} ccdf(8)={pick_new.ccdf(8):.4f} icdf(0.9)={pick_new.icdf(0.9):.2f}")
productivity = stats.Normal(mu=1.0, sigma=0.08)          # good/bad day for the whole team
print(f"P(team works >10% slower than usual) = {productivity.cdf(0.9):.4f}")
print("5 sampled days of productivity:", productivity.sample(5, rng=np.random.default_rng(0)).round(3))

print("\n=== 4. Custom demand curve: NumericalInversePolynomial ===")
class BlackFridayDemand:
    """Orders per hour of the day: an early rush at 10:00 when deals open,
    a bigger rush at 21:00 after work, and a low base level all day."""
    def pdf(self, h):
        return (0.15 + np.exp(-(h - 10) ** 2 / 2.0)
                + 1.4 * np.exp(-(h - 21) ** 2 / 3.0))

demand = NumericalInversePolynomial(BlackFridayDemand(), domain=(0, 24), center=21,
                                    random_state=rng)
sample_h = demand.rvs(200000)
counts, _ = np.histogram(sample_h, bins=24, range=(0, 24))
print("share of orders per hour (%):", (counts / counts.sum() * 100).round(1))
print(f"busiest hour gets {counts.max() / counts.sum():.1%} of the day's orders")
print(f"20:00-22:00 share = {np.mean((sample_h >= 20) & (sample_h < 22)):.3f}")

print("\n=== 5. Monte Carlo: simulate many Black Fridays ===")
MIN = np.arange(24 * 60)

def simulate_day(pickers, start, rng, extra_evening=0, return_detail=False):
    """One random day. Pickers work 16 hours from `start` (two shifts back to back);
    `extra_evening` temporary pickers work 17:00-24:00 and are 25% slower.
    Two trucks leave: 15:00 (takes orders placed before 12:00) and 23:30 (orders placed
    12:00-22:00). Orders after 22:00 are for tomorrow's trucks and are not counted.
    Returns the share of today's orders that miss their truck ("late")."""
    n_orders = stats.poisson(16000).rvs(random_state=rng)
    arr_min = np.sort(demand.rvs(n_orders) * 60)
    prod = productivity.sample(rng=rng)
    absent = stats.binom(pickers, 0.06).rvs(random_state=rng)
    per_min = (pickers - absent) * prod / pick.mean()
    on_shift = (MIN >= start * 60) & (MIN < (start + 16) * 60)
    cap = np.where(on_shift, per_min, 0.0)
    cap = cap + np.where(MIN >= 17 * 60, extra_evening * prod / (pick.mean() * 1.25), 0.0)
    arrivals = np.histogram(arr_min, bins=24 * 60, range=(0, 24 * 60))[0]
    done = np.zeros(24 * 60)
    backlog = 0.0
    total = 0.0
    for m in range(24 * 60):                        # first come, first served
        backlog += arrivals[m]
        work = min(backlog, cap[m])
        backlog -= work
        total += work
        done[m] = total
    k = np.arange(1, n_orders + 1)
    finish = np.searchsorted(done, k - 1e-9)          # minute order k is finished
    finish = np.where(finish >= 24 * 60, 30 * 60, finish)   # unfinished by midnight
    deadline = np.where(arr_min < 12 * 60, 15 * 60, 23.5 * 60)
    today = arr_min < 22 * 60
    late = np.mean(finish[today] > deadline[today])
    if return_detail:
        return late, arrivals, done
    return late

for pk in [100, 120, 140, 160]:
    sims = np.array([simulate_day(pk, 7, rng) for _ in range(400)])
    print(f"{pk} pickers, start 07:00: mean late share={sims.mean():.3%}  "
          f"P(late share > 2%)={np.mean(sims > 0.02):.1%}  worst day={sims.max():.1%}")

risk = []
staff = np.arange(100, 181, 5)
for pk in staff:
    sims = np.array([simulate_day(pk, 7, rng) for _ in range(300)])
    risk.append(np.mean(sims > 0.02))
risk = np.array(risk)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
hrs = np.linspace(0, 24, 400)
d = BlackFridayDemand().pdf(hrs)
axes[0].plot(hrs, d / np.trapezoid(d, hrs), color=BLUE)
axes[0].set_title("Our hand-made demand curve")
axes[0].set_xlabel("hour of day"); axes[0].set_yticks([])
axes[1].plot(staff, risk * 100, color=BLUE, marker="o", ms=4)
axes[1].axhline(5, color=ORANGE, ls="--", lw=1.5, label="acceptable risk: 5%")
axes[1].set_title("Risk of a bad day (>2% of orders miss the truck)")
axes[1].set_xlabel("pickers hired (shift starts 07:00)"); axes[1].set_ylabel("% of simulated days")
axes[1].legend()
save(fig, "s6_demand_risk.png")
for pk, r in zip(staff, risk):
    print(f"  {pk} pickers -> risk {r:.1%}")
se = np.sqrt(0.05 * 0.95 / 300)
print(f"Monte Carlo uncertainty for a risk near 5% with 300 runs: +/- {1.96 * se:.1%}")

print("\n=== 6. QMC sweep: cheapest plan with risk <= 5% ===")
# inputs: pickers (80-200), shift start hour (5-10), evening temps (0-60)
WAGE, TEMP_WAGE = 16 * 18, 7 * 15            # euro per picker per day (16 h), per temp (7 h)
MAX_RISK = 0.05

def evaluate(pickers, start, temps, n_days=40, seed=0):
    """Staff cost (euro) and estimated risk of a bad day (>2% of orders miss their truck)."""
    r = np.random.default_rng(seed)            # same random days for every plan (fair comparison)
    lates = np.array([simulate_day(int(pickers), start, r, int(temps)) for _ in range(n_days)])
    return pickers * WAGE + temps * TEMP_WAGE, np.mean(lates > 0.02)

def tidy(p):
    p[:, 0] = np.round(p[:, 0]); p[:, 2] = np.round(p[:, 2]); p[:, 1] = np.round(p[:, 1] * 2) / 2
    return p

lo, hi = [80, 5, 0], [200, 10, 60]
for nm, pts in [("random", np.random.default_rng(1).random((64, 3))),
                ("latin hypercube", qmc.LatinHypercube(d=3, rng=np.random.default_rng(1)).random(64)),
                ("sobol", qmc.Sobol(d=3, rng=np.random.default_rng(1)).random(64))]:
    print(f"discrepancy {nm:16s} = {qmc.discrepancy(pts):.5f}")

plans = tidy(qmc.scale(qmc.Sobol(d=3, rng=np.random.default_rng(1)).random(64), lo, hi))
res = np.array([evaluate(*p) for p in plans])
ok = res[:, 1] <= MAX_RISK
print(f"{ok.sum()} of 64 plans meet the risk limit")
idx = np.flatnonzero(ok)[np.argsort(res[ok, 0])]
print("cheapest safe plans:  pickers  start  temps   cost(euro)  P(bad day)")
for i in idx[:5]:
    print(f"                     {plans[i, 0]:7.0f}  {plans[i, 1]:5.1f}  {plans[i, 2]:5.0f}  {res[i, 0]:10.0f}  {res[i, 1]:8.0%}")
cheap_unsafe = np.flatnonzero(~ok)[np.argsort(res[~ok, 0])][:2]
for i in cheap_unsafe:
    print(f"cheap but unsafe: {plans[i]} cost={res[i, 0]:.0f} risk={res[i, 1]:.0%}")
best = plans[idx[0]]
zl = np.maximum(best - [20, 1, 15], lo); zh = np.minimum(best + [20, 1, 15], hi)
z = tidy(qmc.scale(qmc.Sobol(d=3, rng=np.random.default_rng(2)).random(32), zl, zh))
rz = np.array([evaluate(*p) for p in z])
okz = rz[:, 1] <= MAX_RISK
j = np.flatnonzero(okz)[np.argmin(rz[okz, 0])]
print(f"zoom (32 more plans) best safe: pickers={z[j, 0]:.0f} start={z[j, 1]:.1f} temps={z[j, 2]:.0f} "
      f"cost={rz[j, 0]:.0f} risk={rz[j, 1]:.0%}")
final = evaluate(*z[j], n_days=300, seed=99)       # re-check on new, unseen random days
print(f"winner re-checked on 300 new days: cost={final[0]:.0f} risk={final[1]:.1%}")
simple = evaluate(150, 7, 0, n_days=300, seed=99)
print(f"simple plan (150 pickers from 07:00, no temps): cost={simple[0]:.0f} risk={simple[1]:.1%}")
grid_n = (200 - 80 + 1) * 11 * 61
print(f"a full grid of every whole-number plan would need {grid_n:,} evaluations x 40 days")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, pts, ttl in [(axes[0], np.random.default_rng(1).random((128, 2)), "128 random points"),
                     (axes[1], qmc.Sobol(d=2, rng=np.random.default_rng(1)).random(128), "128 Sobol points")]:
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=BLUE)
    ax.set_title(f"{ttl}\ndiscrepancy = {qmc.discrepancy(pts):.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
save(fig, "s6_qmc_points.png")
