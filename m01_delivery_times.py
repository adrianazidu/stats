"""
delivery time profiles
"""

import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
import data

times = data.delivery_times()

#describe
print(" === 1.Descriptive statistics==")

d = stats.describe(times,nan_policy='omit')

# n obs number of obersvations
print(f"n= {d.nobs} min = {d.minmax[0]:.1f} max = {d.minmax[1]:.1f}")
print(f"mean= {d.mean:.2f} variance = {d.variance:.1f} std = {np.sqrt(d.variance):.1f}")
print(f"skewness= {d.skewness:.2f}") #(> 0 means a long tail of late deliveries)")
print(f"kurtosis= {d.kurtosis:.2f}") # (> 0 means more extreme values than a normal curve)")

"""
variance, skewness, and kurtosis are the three key statistical metrics (known as the 2nd, 3rd, and 4th moments)
 that describe how your data spreads, leans, and behaves around that center.

Variance	The overall spread of the data. How far data points fluctuate from the center.
######
STD = square root of variance (std_dev = np.std(data, ddof=1) where ddof = delta degrees of freedom 0 take all, 1 take samples
coef of variance = std/mean
cv < 0.15 (15%) => low variance , data is packed
cv > 0.30 (30%) => high variance , data is spread far apart from the average

Skewness	The asymmetry (lopsidedness) of the data distribution
#######
sk = 0 -> perfectly symmetric distribution
sk > 0 -> the curve has a long tail to the right,• Most data points are grouped on the left (low values), but a few exceptionally high values pull the tail to the right (e.g., household income).
sk < 0 -> the opposite

Kurtosis	The weight of the tails and the presence of extreme values (outliers).
######
k = 0 -> mesokurtic, normal dostrbution
k > 0 -> leptokurtic, tails are fat and heavy and there is a peak at the center
k < 0 -> platykurtic, tails are thin and light and the peak is flat or rounded

"""

print(" === 2.Outliers ==")

# trimming = throw away the most extreme 5% on each side
print(f"trimmed mean (5% each side) = {stats.trim_mean(times, 0.05):.2f}")
trimmed = stats.trimboth(times, 0.05)
print(f"trimboth kept {len(trimmed)} of {len(times)} values")

#winsorization 
win = winsorize(times, limits=[0.05,0.05])
print(f"winsorized mean= {win.mean():.2f}, max win {win.max():.2f}")

"""
Winsorization, Trimmed Mean, and Winsorized Mean are three statistical techniques used to handle outliers (extreme, anomalous values) in a dataset. They prevent these rare extreme values from distorting your overall data analysis.
The core difference lies in how they handle those extremes: a trimmed mean deletes the data points, while winsorization replaces (caps) them with less extreme values.


Trimmed Mean	Completely removes a fixed percentage of the smallest and largest values, then calculates the mean on what is left.

Winsorize	Takes values beyond a chosen threshold (e.g., the top and bottom 5%) and replaces them with the maximum and minimum acceptable values.
Used when you cannot afford to delete rows of data but want to mute the impact of crazy spikes.

Winsorized Mean	Calculates a standard arithmetic mean after the dataset has gone through Winsorization.

Both Winsorization and percentile clipping do the exact same thing: they replace extreme values with the threshold values (the percentiles themselves). Winsorization does NOT use the mean value.
"""

low, high = np.percentile(times, [5,95]) #inpandas df["col"].quantile(0.05) 
#replace extreme outliers  with lower and upper threshold values (below 5% and over 95% quartile)
clipped = np.clip(times,low,high)

#clean values
clean = times[times < 80]

print("\n=== 3. Is the data normally distributed? (p < 0.05 -> NO) ===")
#Checks the correlation between your data and ideal normal scores (small datasets)
print(f"shapiro      p = {stats.shapiro(clean).pvalue:.4f}")

#Combines skewtest and kurtosistest into a single omnibus score. (big datasets)
print(f"normaltest   p = {stats.normaltest(clean).pvalue:.4f}")

#Testing if the data leans too far left or right
print(f"skewtest     p = {stats.skewtest(clean).pvalue:.4f}   (is the skewness real?)")

#Testing if the data has too many extreme outliers
print(f"kurtosistest p = {stats.kurtosistest(clean).pvalue:.4f}")

#Also combines skewness and kurtosis, good for BIG datasets, but uses a different mathematical formula, used in finance/economics
print(f"jarque_bera  p = {stats.jarque_bera(clean).pvalue:.4f}")

"""
The p-value is a number between 0 and 1
p > 0.05 data is probably normal
p <= 0.05 data is not normal
"""

print("\n=== 4. Fit a gamma distribution + Kolmogorov-Smirnov test ===")

"""
This line performs Maximum Likelihood Estimation (MLE) to find the optimal statistical parameters that best describe
 your clean data.
 It looks at your actual dataset (clean) and mathematically calculates the specific mathematical parameters 
(shape and scale) required to reconstruct a Gamma curve that matches your data as closely as possible.
Force the distribution to start at 0
"""
shape, loc, scale = stats.gamma.fit(clean,floc=0) # floc = 0 times start at 0
print(f"fitted gamma: shape = {shape:.2f}, scale = {scale:.2f}")
fitted = stats.gamma(shape,loc,scale) 
"""
This line creates what SciPy calls a frozen distribution object named fitted.
Instead of having to pass the shape, loc, and scale values into functions over and over again,
 you now have a single object that knows its own parameters


What can you do with the fitted object next?
- Generate Matching Simulation Data: new_samples = fitted.rvs(size=1000)

-calculate probabilities prov = fitted.cdf(10) #Cumulative Distribution Function
Real-World Example: Imagine you fitted a Gamma distribution to customer wait times. You want to know how many customers wait less than 10 minutes.

- find quantiles thr95 = fitted.ppf(0.95)      #Percent Point Function
• Real-World Example: You want to give a service guarantee to your customers. You want to find a maximum wait time threshold so that 95% of all customers are served within that limit.

ppf and cdf are perfect mirror opposites ppf(x) = y _> cdf(y) = x
"""

"""
kolmogorov-smirnov (KS) test for distribution fit
This tests how well your data matches the Gamma distribution you just fitted. A large p-value (e.g., > 0.05)
 means the distance is very small, confirming that the Gamma distribution is an excellent fit for your data.
"""
ks = stats.kstest(clean,fitted.cdf)
print(f"KS test vs fitted gamma : p = {ks.pvalue:.3f}  (large p = gamma fits well)")

# This tests if your data fits a standard Normal distribution. 
ks_norm = stats.kstest(clean,stats.norm(clean.mean(),clean.std()).cdf)
print(f"KS test vs normal       : p = {ks_norm.pvalue:.3f}")


#  one-sample t-test (Hypothesis Testing)
print("\n=== 5. One-sample t-test: is the mean above 30 minutes? ===")

t = stats.ttest_1samp(clean, popmean=30, alternative="greater")

print(f"t = {t.statistic:.2f}, p = {t.pvalue:.4f}")
print("-> we are NOT keeping the promise" if t.pvalue < 0.05 else "-> no evidence we are late")

ci = t.confidence_interval(0.95)
print(f"95% lower bound of the mean: {ci.low:.2f} min")

#  tails of the model
print("\n=== 6. Tails: how often are we very late? ===")
# sf survival function The Survival Function (sf) calculates \(1 - \text{CDF}\).
print(f"P(time > 45 min) = {fitted.sf(45):.1%}   (sf = 1 - cdf)")
print(f"P(time > 60 min) = {fitted.sf(60):.2%}")

print(f"90% of deliveries arrive within {fitted.ppf(0.90):.1f} min  (ppf = inverse cdf)")
print(f"to promise 95% on time, promise {fitted.ppf(0.95):.0f} minutes")

"""
The fundamental purpose of computing a fitted distribution is to transition from descriptive statistics (the past) to predictive modeling (the future).
When you look at raw data (clean), you only see what already happened. When you fit a distribution (like your Gamma model), you smooth out the random noise and uncover the underlying mathematical law governing your system.
Here are the four primary reasons why this is done in data science and operations:

1. Risk Assessment in the "Tails" (Predicting Rare Events)

If you only have 1,000 delivery records, your raw data might not contain a single delivery that took exactly 62 minutes. However, a 62-minute delivery could happen tomorrow.
• Without a fitted model: You cannot calculate the probability of events that haven't occurred yet in your sample.
• With a fitted model: The Survival Function (sf) lets you calculate the exact probability of extreme delays (e.g., P(time > 60 min)), allowing you to quantify operational risk and prepare for worst-case scenarios.

2. Setting Smart Guarantees (SLA Design)

If you want to advertise a delivery guarantee to your customers (e.g., "We deliver in X minutes or your next order is free"), you need to choose a threshold that protects your profit margins.
• Using fitted.ppf(0.95) tells you the exact minute threshold required to ensure that 95% of all future deliveries will automatically arrive on time based on the mathematical shape of your performance history.

3. Simulation and Stress Testing (Monte Carlo)

If you want to simulate how your business will handle a massive surge in demand next month, you need to generate realistic synthetic data.
• You cannot just reuse your past 1,000 data points repeatedly. With your fitted object, you can run fitted.rvs(size=100000) to instantly generate 100,000 new, completely unique, but mathematically perfect delivery times to stress-test your logistics pipeline.
When doing this in finance (for example, simulating stock prices over time), you don't just simulate a single point; you simulate sequential paths

4. Data Compression and Generalization

Raw datasets can be massive, noisy, and difficult to share. A fitted distribution acts as a form of mathematical compression.
• Instead of storing or passing around a .csv file filled with millions of rows of clean data, you can compress that entire operational reality into just two numbers: shape = 2.00 and scale = 3.00. Anyone with those two parameters can perfectly reconstruct your entire delivery timeline's behavior.

"""