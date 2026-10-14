"""
file creating fake data

using rn distributions
rng.uniform - same number
rng.normal - classic bell shape
poisson - isolated vertical bars
gamma - asymmetric curve, A skewed curve that starts at zero, rises quickly to a peak, and then drops into a long tail to the right.
"""

import numpy as np
rng = np.random.default_rng(42)

def delivery_times(n=300):
    #mostly gamma shaped arounf 32 minutes plus outliers (traffic)
    times = rng.gamma(shape=9,scale=3.5,size=n) # mean = 9 * 3.5 = 31.5
    outliers=rng.uniform(90,150,size=6)         #6 outliers
    return np.concatenate([times,outliers])

def two_branches():
    #branch a old, branch b new 4 minites slower
    a = rng.gamma(shape=9, scale=3.5, size=200) # mean 31.5
    b = rng.gamma(shape=9, scale=3.95, size=200) # mean 35.5
    return a,b

def rider_times():
    #delivery times by vehicle, same average different spread
    bike = rng.normal(30,4,size=60)
    car = rng.normal(30,5,size=60)
    scooter = rng.normal(30,9,size=60)
    return bike,car,scooter
    


