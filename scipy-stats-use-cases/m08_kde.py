"""
Module 8 - Where and when (KERNEL DENSITY ESTIMATION)
Questions:
  a) What does the delivery-time curve look like, without assuming a formula?
  b) Where do our customers live? (2-D heat map -> where to open the next shop)

Covers: gaussian_kde (univariate + multivariate), bandwidth choice
Saves two pictures: kde_delivery_times.png and kde_customer_map.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")            # draw to files, no window needed
import matplotlib.pyplot as plt
from scipy import stats
import data

# ------------------------------------------------------------ univariate
print("=== a) Delivery-time curve ===")
times = data.delivery_times()
times = times[times < 80]

kde = stats.gaussian_kde(times)                 # bandwidth chosen automatically
kde_wide = stats.gaussian_kde(times, bw_method=0.5)   # smoother
kde_narrow = stats.gaussian_kde(times, bw_method=0.05)  # bumpier

grid = np.linspace(0, 80, 300)
print(f"density at 30 min = {kde(30)[0]:.4f}")
print(f"P(25 < time < 35) = {kde.integrate_box_1d(25, 35):.1%}")
print(f"most likely time  = {grid[np.argmax(kde(grid))]:.1f} min")

plt.figure(figsize=(7, 4))
plt.hist(times, bins=30, density=True, alpha=0.3, color="gray", label="data")
plt.plot(grid, kde(grid), lw=2, label="KDE (auto bandwidth)")
plt.plot(grid, kde_wide(grid), "--", label="bw 0.5 (too smooth)")
plt.plot(grid, kde_narrow(grid), ":", label="bw 0.05 (too bumpy)")
plt.xlabel("delivery time (min)")
plt.ylabel("density")
plt.legend()
plt.tight_layout()
plt.savefig("kde_delivery_times.png", dpi=120)
print("saved kde_delivery_times.png")

# ---------------------------------------------------------- multivariate
print("\n=== b) Customer map (2-D KDE) ===")
pts = data.customer_locations()                 # shape (400, 2)
kde2 = stats.gaussian_kde(pts.T)                # gaussian_kde wants shape (2, n)

xs, ys = np.meshgrid(np.linspace(-3, 5, 120), np.linspace(-3, 4, 120))
density = kde2(np.vstack([xs.ravel(), ys.ravel()])).reshape(xs.shape)

i = np.argmax(density)
print(f"busiest spot: x = {xs.ravel()[i]:.1f} km, y = {ys.ravel()[i]:.1f} km")
print(f"density at the shop (0,0) = {kde2([0, 0])[0]:.4f}")
print(f"density at the campus (3,2) = {kde2([3, 2])[0]:.4f}")
print("-> the campus is busier than the area around our shop: a good spot for a 2nd shop")

# kde objects can also generate NEW fake customers that look like the real ones
new_customers = kde2.resample(5, seed=0).T
print("5 simulated new customers:\n", new_customers.round(2))

plt.figure(figsize=(6, 5))
filled = plt.contourf(xs, ys, density, levels=20, cmap="viridis")
plt.scatter(pts[:, 0], pts[:, 1], s=3, c="white", alpha=0.4)
plt.plot(0, 0, "r^", markersize=12, label="our shop")
plt.colorbar(filled, label="customer density")
plt.xlim(-3, 5)
plt.ylim(-3, 4)
plt.xlabel("km east")
plt.ylabel("km north")
plt.legend()
plt.tight_layout()
plt.savefig("kde_customer_map.png", dpi=120)
print("saved kde_customer_map.png")
