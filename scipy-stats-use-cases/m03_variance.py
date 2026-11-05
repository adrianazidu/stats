#compute variance
import numpy as np
from scipy import stats
import data

"""

scipy.stats.bartlett 	
-distribution Must be strictly normal	
-Very poor (Highly sensitive to non-normality)for outliers
-best for Clean, perfectly bell-curved (Gaussian) data.

scipy.stats.levene		
-Any distribution	
-Good (Robust to moderate non-normality) for outliers
-best for	The industry default for general, real-world data.

scipy.stats.fligner	
-	Any distribution	
- Excellent (Most robust against heavy skew/outliers) with outliers
-	best for Messy, skewed data with heavy tails or outliers
"""

bike, car, scooter = data.rider_times()
for name, x in [("bike", bike), ("car", car), ("scooter", scooter)]:
    print(f"{name:8s} mean = {x.mean():.1f} min, std = {x.std(ddof=1):.1f} min")

print("\nNull hypothesis: all three groups have the same variance.")

# Bartlett: most powerful, but only valid if the data is normal
print(f"bartlett  p = {stats.bartlett(bike, car, scooter).pvalue:.6f}  (assumes normal data)")

# Levene: robust, works fine with non-normal data
print(f"levene    p = {stats.levene(bike, car, scooter).pvalue:.6f}  (robust)")

# Fligner-Killeen: non-parametric (uses ranks), the most robust to outliers
print(f"fligner   p = {stats.fligner(bike, car, scooter).pvalue:.6f}  (rank based)")

print("\nNow only bike vs car (similar spread):")
print(f"levene    p = {stats.levene(bike, car).pvalue:.3f}  -> no evidence of a difference")