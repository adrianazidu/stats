"""
Fake data for the pizzeria chain "Pizza Veloce".

Every function returns simulated data with KNOWN properties, so you can
check whether each statistical test finds what we planted in the data.
The seed is fixed, so you always get the same numbers.
"""
import numpy as np

rng = np.random.default_rng(42)


def delivery_times(n=300):
    """Delivery times (minutes) for branch A.
    Mostly gamma-shaped (right-skewed) around 32 min,
    plus a few extreme outliers (rider stuck in traffic)."""
    times = rng.gamma(shape=9, scale=3.5, size=n)          # mean = 9 * 3.5 = 31.5
    outliers = rng.uniform(90, 150, size=6)                 # 6 disasters
    return np.concatenate([times, outliers])


def two_branches():
    """Branch A (old) and branch B (new, ~4 minutes slower)."""
    a = rng.gamma(shape=9, scale=3.5, size=200)             # mean 31.5
    b = rng.gamma(shape=9, scale=3.95, size=180)            # mean ~35.5
    return a, b


def rider_times():
    """Delivery times by vehicle. Same average, different spread:
    scooters are much more erratic than bikes and cars."""
    bike = rng.normal(30, 4, size=60)
    car = rng.normal(30, 5, size=60)
    scooter = rng.normal(30, 9, size=60)
    return bike, car, scooter


def app_experiment():
    """Checkout redesign: average order value (euro) per user group.
    Control = current app, design_1 = no real effect, design_2 = +2 euro."""
    control = rng.normal(24, 6, size=150)
    design_1 = rng.normal(24.3, 6, size=150)
    design_2 = rng.normal(26, 6, size=150)
    return control, design_1, design_2


def conversion_table():
    """Did the visitor place an order? rows = app version, cols = [ordered, left]."""
    return np.array([
        [120, 880],   # old app:  12% conversion
        [158, 842],   # new app: ~16% conversion
    ])


def coupon_test():
    """Small coupon test (few people -> Fisher's exact test).
    rows = [got coupon, no coupon], cols = [came back, did not come back]"""
    return np.array([
        [9, 3],
        [3, 9],
    ])


def orders_per_hour(hours=200):
    """Number of orders in each hour: Poisson with on average 7 orders/hour."""
    return rng.poisson(lam=7, size=hours)


def pizza_ratings(n_customers=25):
    """The same 25 customers rate 3 pizzas from 1 to 10.
    Diavola is liked a bit more than the others."""
    base = rng.normal(6.5, 1.2, size=n_customers)            # each person's mood
    margherita = base + rng.normal(0, 0.8, n_customers)
    quattro = base + rng.normal(0, 0.8, n_customers)
    diavola = base + 1.0 + rng.normal(0, 0.8, n_customers)
    clip = lambda x: np.clip(np.round(x), 1, 10)
    return clip(margherita), clip(quattro), clip(diavola)


def lateness_drivers(n=120):
    """What makes deliveries slow?
    - distance (km): linear effect on time
    - hour of day: U-shaped effect (slow at lunch AND dinner peaks, fast in between).
      Correlation coefficients miss this, MGC finds it."""
    distance = rng.uniform(0.5, 8, size=n)
    hour = rng.uniform(11, 23, size=n)
    time_vs_distance = 15 + 3 * distance + rng.normal(0, 3, n)
    time_vs_hour = 25 + 0.6 * (hour - 17) ** 2 + rng.normal(0, 3, n)  # min at 17:00
    return distance, time_vs_distance, hour, time_vs_hour


def customer_locations(n=400):
    """(x, y) position of customers in km from the shop.
    Two hot spots: city centre and a university campus."""
    centre = rng.normal([0, 0], [1.0, 1.0], size=(n // 2, 2))
    campus = rng.normal([3, 2], [0.6, 0.6], size=(n // 2, 2))
    return np.vstack([centre, campus])
