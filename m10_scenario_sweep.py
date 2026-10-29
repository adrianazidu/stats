"""
Module 10 - Scenario sweep with Quasi-Monte Carlo
Question: what mix of riders, kitchen staff and oven size gives the best
profit on a Friday night?

We have a (made-up) profit formula with 3 inputs. Trying every combination
on a full grid is wasteful; QMC picks points that cover the space evenly,
so few points are enough.

Covers: qmc.Sobol, qmc.LatinHypercube, qmc.discrepancy, qmc.scale
"""
import numpy as np
from scipy.stats import qmc

def profit(riders, cooks, ovens):
    """Profit (euro) of one Friday night. Too few staff -> lost orders,
    too many -> wages eat the profit."""
    capacity = np.minimum(riders * 15, np.minimum(cooks * 25, ovens * 40))
    orders = np.minimum(capacity, 150)              # demand is ~150 orders
    return orders * 15 - riders * 70 - cooks * 90 - ovens * 50

lower = [3, 2, 1]       # min riders, cooks, ovens
upper = [20, 12, 8]     # max riders, cooks, ovens

# ---------------------------------------------- 1. compare sampling methods
print("=== 1. How evenly do the points cover the space? ===")
print("(discrepancy: lower = more even coverage)")
n = 64
random_pts = np.random.default_rng(0).random((n, 3))
sobol_pts = qmc.Sobol(d=3, seed=0).random(n)          # n should be a power of 2
lhs_pts = qmc.LatinHypercube(d=3, seed=0).random(n)
print(f"plain random    : {qmc.discrepancy(random_pts):.5f}")
print(f"Latin hypercube : {qmc.discrepancy(lhs_pts):.5f}")
print(f"Sobol           : {qmc.discrepancy(sobol_pts):.5f}")

# ---------------------------------------------- 2. sweep the scenarios
print("\n=== 2. Testing 64 Sobol scenarios ===")
# QMC points are between 0 and 1 -> scale them to our real ranges
scenarios = qmc.scale(sobol_pts, lower, upper)
scenarios = np.round(scenarios).astype(int)           # you can't hire half a cook
p = profit(scenarios[:, 0], scenarios[:, 1], scenarios[:, 2])

best = np.argsort(p)[::-1][:5]
print("riders  cooks  ovens   profit")
for i in best:
    r, c, o = scenarios[i]
    print(f"{r:6d}  {c:5d}  {o:5d}   {p[i]:6.0f} euro")

# ---------------------------------------------- 2b. zoom in around the best one
print("\n=== 2b. Zoom in: 64 more Sobol points around the best scenario ===")
centre = scenarios[best[0]]
zoom_lower = np.maximum(centre - 4, lower)
zoom_upper = np.minimum(centre + 4, upper)
zoom = qmc.scale(qmc.Sobol(d=3, seed=1).random(64), zoom_lower, zoom_upper)
zoom = np.round(zoom).astype(int)
pz = profit(zoom[:, 0], zoom[:, 1], zoom[:, 2])
j = np.argmax(pz)
print(f"best after zooming: {zoom[j, 0]} riders, {zoom[j, 1]} cooks, "
      f"{zoom[j, 2]} ovens -> {pz[j]:.0f} euro")

# ---------------------------------------------- 3. check against brute force
print("\n=== 3. Check: full grid search (every combination) ===")
R, C, O = np.meshgrid(np.arange(3, 21), np.arange(2, 13), np.arange(1, 9), indexing="ij")
P = profit(R, C, O)
i = np.unravel_index(np.argmax(P), P.shape)
print(f"true best: {R[i]} riders, {C[i]} cooks, {O[i]} ovens -> {P[i]:.0f} euro")
print(f"the full grid needed {P.size} evaluations;")
print(f"QMC reached {pz.max() / P.max():.0%} of the best profit with only {2 * n}")
