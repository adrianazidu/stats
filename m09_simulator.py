"""
Module 9 - Friday-night simulator
Question: how many riders do we need on a Friday evening (18:00-23:00)?

We build the night out of probability distributions and simulate it many times.

Covers: frozen distributions, shape / loc / scale (shifting and scaling),
        broadcasting, the NEW random-variable API (Normal, make_distribution),
        scipy.stats.sampling (NumericalInversePolynomial), monte_carlo_test
"""
import numpy as np
from scipy import stats
from scipy.stats.sampling import NumericalInversePolynomial

rng = np.random.default_rng(0)

# ---------------------------------------------- 1. old API: frozen + loc/scale
print("=== 1. Classic distributions: shape, loc, scale, freezing ===")
# Round trip of a rider: gamma shape a=6, stretched by scale=4, shifted by loc=5
# (loc=5 means "never less than 5 minutes": time to park, ring, pay...)
trip = stats.gamma(a=6, loc=5, scale=4)      # "frozen": parameters fixed once
print(f"trip time: mean = {trip.mean():.1f} min, std = {trip.std():.1f} min")
print(f"support (min, max) = {trip.support()}   <- shifted by loc")
print(f"same as unshifted gamma + 5? {trip.mean():.1f} == {stats.gamma(6).mean() * 4 + 5:.1f}")

# ---------------------------------------------- 2. broadcasting
print("\n=== 2. Broadcasting: many questions in one call ===")
# P(trip > threshold) for 3 thresholds (columns) and 3 traffic levels (rows)
thresholds = np.array([30, 40, 50])
scales = np.array([[3.5], [4.0], [5.0]])     # light / normal / heavy traffic
table = stats.gamma.sf(thresholds, a=6, loc=5, scale=scales)
print("P(trip longer than)    30 min   40 min   50 min")
for name, row in zip(["light traffic", "normal", "heavy traffic"], table):
    print(f"{name:20s}" + "".join(f"{p:9.1%}" for p in row))

# ---------------------------------------------- 3. new random-variable API
print("\n=== 3. New random-variable API (scipy >= 1.15) ===")
# kitchen preparation time: normal, 12 +- 3 minutes
prep = stats.Normal(mu=12, sigma=3)
print(f"prep: mean = {prep.mean()}, P(prep > 15) = {prep.ccdf(15):.3f}")  # ccdf = sf

# turn any classic distribution into a new-style one with make_distribution
Gamma = stats.make_distribution(stats.gamma)
trip_new = 4 * Gamma(a=6) + 5                # shift and scale with plain arithmetic!
print(f"trip (new API): mean = {trip_new.mean():.1f}, "
      f"P(trip > 40) = {trip_new.ccdf(40):.3f}  (same as old: {trip.sf(40):.3f})")
print(f"median = {trip_new.median():.1f}, 90th percentile = {trip_new.icdf(0.9):.1f}")

# ---------------------------------------------- 4. custom sampling
print("\n=== 4. Sampling order times from our OWN demand curve ===")
# Orders between 18:00 and 23:00 have two peaks: 20:00 and 21:30.
# No scipy distribution looks like this, so we write the pdf ourselves
# and let NumericalInversePolynomial build a fast sampler from it.
class FridayDemand:
    def pdf(self, x):
        return np.exp(-(x - 20) ** 2 / 0.5) + 0.7 * np.exp(-(x - 21.5) ** 2 / 0.3)

    def support(self):
        return (18, 23)

sampler = NumericalInversePolynomial(FridayDemand(), domain=(18, 23), center=20,
                                     random_state=rng)
order_hours = sampler.rvs(10)
print("10 sampled order times:", [f"{int(h)}:{int(h % 1 * 60):02d}" for h in sorted(order_hours)])

# ---------------------------------------------- 5. simulate many nights
print("\n=== 5. Simulating 1000 Friday nights ===")
minutes = np.arange(18 * 60, 23 * 60 + 60)       # every minute from 18:00 to 24:00

def riders_needed_one_night():
    n_orders = stats.poisson(70).rvs(random_state=rng)          # how busy is the night
    start = sampler.rvs(n_orders) * 60                           # order time (minutes)
    start = start + prep.sample(n_orders, rng=rng)               # + kitchen time
    end = start + trip.rvs(n_orders, random_state=rng)           # + round trip
    # broadcasting again: (minutes x orders) -> is order j on the road at minute i?
    on_road = (start[None, :] <= minutes[:, None]) & (minutes[:, None] < end[None, :])
    return on_road.sum(axis=1).max()                             # peak riders busy

peaks = np.array([riders_needed_one_night() for _ in range(1000)])
print(f"riders needed at peak: average {peaks.mean():.1f}, "
      f"worst night {peaks.max()}")
for r in [14, 16, 18, 20, 22]:
    print(f"with {r} riders, nights where nobody waits for a rider: {np.mean(peaks <= r):.0%}")

# ---------------------------------------------- 6. Monte Carlo test
print("\n=== 6. Monte Carlo test: was last Friday normal? ===")
# Last Friday, 40 trips were recorded. Are they consistent with our trip model?
last_friday = stats.gamma(a=6, loc=5, scale=4.8).rvs(40, random_state=1)  # secretly slower
print(f"last Friday mean trip = {last_friday.mean():.1f} min (model says {trip.mean():.1f})")

# monte_carlo_test simulates many fake "last Fridays" from the model and
# checks how unusual the real one is
res = stats.monte_carlo_test(last_friday, trip.rvs, np.mean,
                             alternative="greater", n_resamples=5000)
print(f"p = {res.pvalue:.4f}")
print("-> last Friday was unusually slow" if res.pvalue < 0.05 else "-> a normal Friday")
