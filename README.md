# Pizza Veloce: a guided tour of `scipy.stats`

This project uses a fake pizzeria chain, **Pizza Veloce**, to try almost everything in the
[SciPy statistics tutorial](https://docs.scipy.org/doc/scipy/tutorial/stats.html).

The chain has three problems: it promises delivery in 30 minutes, it has just opened a
second branch, and it doesn't know how many riders to hire for Friday night. Each script
answers one business question with one group of statistical tools.

All data is **simulated** in `data.py`, and we know the truth behind it. For example, we
made branch B 4 minutes slower on purpose and scooters more erratic on purpose. This
lets you check whether each test finds what we hid in the data. With real data you never
know the truth, so this is the best way to learn what the tests can and cannot see.

---

## Contents

1. [Setup and how to run](#1-setup-and-how-to-run)
2. [Ideas you need for every module](#2-ideas-you-need-for-every-module)
3. [Module 1: Delivery-time profile (one sample)](#module-1--delivery-time-profile-one-sample)
4. [Module 2: Branch A vs branch B (two samples)](#module-2--branch-a-vs-branch-b-two-samples)
5. [Module 3: Rider consistency (equal variances)](#module-3--rider-consistency-equal-variances)
6. [Module 4: App redesign experiment](#module-4--app-redesign-experiment)
7. [Module 5: Order volume (discrete distributions)](#module-5--order-volume-discrete-distributions)
8. [Module 6: Customer ratings (repeated measures)](#module-6--customer-ratings-repeated-measures)
9. [Module 7: What drives lateness (correlation)](#module-7--what-drives-lateness-correlation)
10. [Module 8: Where and when (kernel density estimation)](#module-8--where-and-when-kernel-density-estimation)
11. [Module 9: Friday-night simulator](#module-9--friday-night-simulator)
12. [Module 10: Scenario sweep (Quasi-Monte Carlo)](#module-10--scenario-sweep-quasi-monte-carlo)
13. [Cheat sheet: which test should I use?](#cheat-sheet-which-test-should-i-use)
14. [Experiments to try](#experiments-to-try)
15. [Glossary](#glossary)

---

## 1. Setup and how to run

```bash
pip install "scipy>=1.15" numpy matplotlib
```

Tested with SciPy 1.18.1, NumPy 2.5 and Python 3.13. Version 1.15 or newer is needed for
the new random-variable API used in module 9.

Run the scripts from inside this folder, because they import `data.py`:

```bash
python m01_delivery_times.py     # one module
python run_all.py                # all ten, one after the other
```

| File | What it contains |
|---|---|
| `data.py` | Functions that generate the fake data. Seed is fixed, so results are the same every run. |
| `m01_ … m10_*.py` | One business question each. Read top to bottom: no classes, no tricks. |
| `run_all.py` | Runs every module in order. |
| `kde_*.png` | Charts created by module 8. |

The numbers quoted in this README come from a real run. Two results change slightly
every run (the Monte Carlo test in module 9 and the random sample of order times),
because they use fresh random draws.

---

## 2. Ideas you need for every module

Almost every test in this project follows the same recipe. If you understand this
section, you can read any output in the project.

### Null hypothesis (H0)

A **null hypothesis** is the "boring" explanation: *nothing is going on*. For example,
"branch A and branch B are equally fast" or "the app redesign changed nothing".
A test never proves that something is true. It measures **how surprising the data
would be if the boring explanation were true**.

### p-value

The **p-value** is the probability of seeing data at least as extreme as yours,
**if H0 were true**.

- **Small p (usually < 0.05)**: the data would be very unusual in a "nothing is going on"
  world. We **reject H0** and say the result is *statistically significant*.
- **Large p**: the data is compatible with H0. We **cannot reject it**.
  This does **not** prove H0 is true. It may just mean we have too little data.

Common mistakes to avoid:

- p = 0.03 does **not** mean "3% chance that H0 is true".
- p does **not** tell you how big or important the effect is. With huge samples, a
  0.1-second difference can have p = 0.0001 and still be useless in practice.
- 0.05 is a convention, not a law of nature. Medicine and physics often use stricter limits.

### Significance level (alpha)

The limit you choose **before** looking at the data, usually **alpha = 0.05**. If you test with
alpha = 0.05 when nothing is going on, you will still get a false alarm about 1 time in 20.
This is why running many tests at once needs a correction (see Dunnett's test in module 4).

### Confidence interval (CI)

A **95% confidence interval** is a range of plausible values for the true quantity.
Example from module 2: "B is slower by 0.28 to 4.74 minutes". It is often **more useful
than the p-value**, because it shows both *whether* there is an effect (does the range
include 0?) and *how large* it might be.

### One-sided vs two-sided

- **Two-sided** (`alternative="two-sided"`, the default) asks: "Is there a difference in either direction?"
- **One-sided** (`alternative="greater"` or `"less"`) asks: "Is it bigger?" (or smaller).
  Use it only when the other direction truly doesn't interest you, and decide this before
  looking at the data.

### Parametric vs non-parametric

- **Parametric** tests (t-test, Bartlett, Pearson) assume a shape for the data, usually the normal
  bell curve. When that assumption holds, they are the most powerful.
- **Non-parametric** tests (KS, Fligner, Friedman, Spearman, Kendall) use ranks or the raw data
  and assume much less. They are safer with skewed data and outliers, at the cost of a little power.
- **Resampling** methods (bootstrap, permutation, Monte Carlo) get the answer by simulation
  instead of a formula. They are flexible and assume very little.

### Every result is an object

In modern SciPy, tests return an object. You read its fields by name:

```python
res = stats.ttest_1samp(data, popmean=30)
res.statistic             # the test statistic
res.pvalue                # the p-value
res.confidence_interval() # available on many tests
```

---

## Module 1: Delivery-time profile (one sample)

**File:** `m01_delivery_times.py`
**Business question:** *Are we keeping our promise of delivery in 30 minutes, and how often are we very late?*
**Data:** 306 delivery times from branch A. Most follow a gamma curve with a mean of 31.5 min, plus
6 disasters between 90 and 150 min.

### 1.1 Descriptive statistics: `stats.describe`

**What it is.** A one-line summary of a sample: size, min and max, mean, variance, skewness and kurtosis.

**Why.** Before any test, you must *look* at your data. Many wrong conclusions come from
skipping this step.

**Our result:** mean 33.5 min, variance 267, skewness 3.59, kurtosis 19.1.

**How to interpret:**

- **Mean**: the average. It is very sensitive to outliers (see 1.2).
- **Variance / standard deviation**: how spread out the values are. The standard deviation is the
  square root of the variance and has the same unit as the data (minutes). Here it is about 16 min.
- **Skewness**: whether the distribution is lopsided.
  - 0 means symmetric, like a bell curve.
  - **Above 0** means a long tail to the right: many normal values and a few very large ones.
    Delivery times, incomes and house prices typically look like this.
  - Below 0 means a long tail to the left, like exam scores when the exam is easy.
  - Rough guide: |skew| < 0.5 is mild, more than 1 is strong. Ours (3.59) is strong because of the 6 disasters.
- **Kurtosis**: SciPy reports *excess* kurtosis, so a normal curve gives 0.
  - **Above 0** means more extreme values than a normal curve would produce ("heavy tails").
  - Our 19.1 is a clear sign of outliers.

**Real-life uses:** the first step of every analysis. Typical examples are website load
times, insurance claim sizes, hospital waiting times, and sensor readings in a factory.

### 1.2 Outliers: trimming and winsorization

**What it is.** Two ways to reduce the influence of extreme values.

| Method | What it does | Function used |
|---|---|---|
| **Trimming** | **Removes** the most extreme X% of values on each side | `stats.trim_mean`, `stats.trimboth` |
| **Winsorization** | **Keeps** every value, but replaces the extremes with the value at the X% / (100−X)% limit | `scipy.stats.mstats.winsorize` or `np.clip` with `np.percentile` |

Example with the limits at 5% and 95%: a 149-minute delivery is replaced by 55.2 minutes
(winsorization), or it disappears (trimming).

**Our result:** plain mean 33.50, trimmed mean 31.72, winsorized mean 32.11. Six bad deliveries
out of 306 pulled the mean up by about 2 minutes.

**Why.** One extreme value can move the mean a lot. If your goal is to describe a
*typical* delivery, the extremes hide the picture.

**How to interpret:**

- If the plain mean and the trimmed mean are **close**, outliers don't matter much.
- If they are **far apart** (as here), report both, and investigate the outliers. They are often
  the most important information you have: here, 6 customers waited over an hour and a half!
- Trimming reduces the sample size (276 values kept of 306). Winsorization keeps it the same.
- **Never** remove outliers just to get a "nicer" p-value. Decide the rule before the analysis.

**Real-life uses:**

- Sport judging: Olympic diving drops the highest and lowest judge scores (trimming).
- Economics: average salary with CEO pay winsorized, so one huge income doesn't distort it.
- Finance: winsorized returns, so a single crash day doesn't dominate a risk model.
- Data cleaning: handling sensor glitches such as a thermometer reading 999 °C.

### 1.3 Normality tests: `shapiro`, `normaltest`, `skewtest`, `kurtosistest`, `jarque_bera`

**What it is.** Tests of H0: *"this data comes from a normal (bell-shaped) distribution"*.

| Test | What it looks at | Notes |
|---|---|---|
| `shapiro` (Shapiro-Wilk) | Overall shape | Usually the most powerful. The common first choice. |
| `skewtest` | Only the skewness | Is the lopsidedness real or just noise? |
| `kurtosistest` | Only the tails | Are the tails heavier or lighter than normal? |
| `normaltest` (D'Agostino-Pearson) | Skewness and kurtosis combined | Needs at least about 20 values. |
| `jarque_bera` | Skewness and kurtosis combined | Popular in economics. Reliable only with large samples. |

**Our result:** every p-value is below 0.05, so the data is **not normal**. `skewtest`
(p ≈ 0) shows that the right tail is real and not luck.

**Why.** Many classic methods (t-test, Bartlett, Pearson's confidence interval) assume normal data.
These tests tell you if that assumption is reasonable.

**How to interpret:**

- p < 0.05 means "not normal": consider a robust or non-parametric method, or a transformation such as the log.
- p ≥ 0.05 means "no evidence against normality". This does **not** prove the data is normal.
- **The sample size paradox:** with small samples these tests almost never reject; with huge samples they
  reject even tiny, harmless deviations. Always look at a histogram as well.
- The t-test is fairly robust to non-normality when samples are large (about 30+ per group).

**Real-life uses:** checking assumptions before a t-test or ANOVA, quality control
(are the bolt lengths normally distributed?), and finance (stock returns have "fat tails",
and `jarque_bera` and `kurtosistest` show it immediately).

### 1.4 Fitting a distribution: `stats.gamma.fit`

**What it is.** Finding the parameters of a theoretical distribution that best match the data
(maximum likelihood estimation).

**Why gamma?** Delivery times can't be negative, they cluster around a typical value, and they have a
right tail. The gamma distribution has exactly this shape. The lognormal and Weibull distributions are
common alternatives for the same kind of data.

**Our result:** shape = 9.67, scale = 3.28. The true values we used in `data.py` were 9 and 3.5, so the
fit is close. `floc=0` tells SciPy that the times start at 0, instead of letting it guess a shift.

**What you get.** A **model** you can ask questions to: "How likely is a 60-minute delivery?"
This works even for values you have rarely or never observed (see 1.7).

**Real-life uses:**

- **Gamma**: waiting times, rainfall amounts, insurance claims.
- **Weibull**: lifetime of machines and components, wind speeds.
- **Lognormal**: incomes, file sizes, city populations.
- **Exponential**: time between customer arrivals or radioactive decays.

### 1.5 Kolmogorov-Smirnov test, one sample: `stats.kstest`

**What it is.** A test of H0: *"the data comes from this specific distribution"*. It compares the
**cumulative curve** of the data (the share of values below each point) with the cumulative curve
of the model, and measures the **largest vertical gap** (statistic D).

**Our result:** vs the fitted gamma, p = 0.77 (good fit). vs a fitted normal, p = 0.28.

**How to interpret:**

- Large p means the model is compatible with the data.
- Small p means the model does not describe the data.
- **Important caution:** the KS test is fairly *weak*. Here it doesn't reject the normal curve,
  even though Shapiro-Wilk clearly does. This happens because KS looks mostly at the middle of the
  distribution, and is not very sensitive in the tails.
- Strictly speaking, if you estimate the parameters from the same data you then test, the p-value
  is too optimistic. For a strict test, use `stats.goodness_of_fit`, which corrects for this.

**Real-life uses:** checking that a random number generator is uniform, validating a simulation
model against real measurements, and checking that sensor data follows the expected distribution.

### 1.6 One-sample t-test: `stats.ttest_1samp`

**What it is.** A test of whether the **mean** of one sample differs from a fixed reference value.
H0: *"the true mean delivery time is 30 minutes"*.
We use `alternative="greater"` because we only care whether we are *slower* than promised.

**Our result:** t = 2.92, p = 0.0019. The 95% one-sided lower bound for the mean is 30.75 minutes.

**How to interpret:**

- **t** is the distance between the sample mean and 30, measured in units of uncertainty (standard errors).
  |t| > 2 is roughly "significant".
- p = 0.0019 means that if the true mean were 30 minutes, data like ours would be very rare.
  **Conclusion: on average we are slower than promised.**
- The confidence bound adds the *size*: we are at least 0.75 minutes too slow on average. This is real,
  but small. Whether it matters is a business decision, not a statistical one.

**Real-life uses:**

- Manufacturing: is the average fill of these 500 ml bottles really 500 ml?
- Service level agreements: is our average response time under the 2 seconds we promised?
- Health: is the average cholesterol of this group different from the national reference value?

### 1.7 Tails of the distribution: `sf` and `ppf`

**What it is.** Two questions you can ask a fitted (frozen) distribution:

| Method | Question | Our result |
|---|---|---|
| `cdf(x)` | P(value ≤ x) | |
| `sf(x)` = 1 − cdf(x) ("survival function") | P(value > x) | P(> 45 min) = 10.4%, P(> 60 min) = 1.0% |
| `ppf(q)` (inverse of cdf, "quantile") | Which value has q% of the data below it? | 90% arrive within 45.3 min. 95% within 50 min. |

**Why.** The mean says nothing about the *bad* cases, and customers remember the bad cases.
Tail probabilities turn the model into business numbers.

**How to interpret:** "1 delivery in 10 takes more than 45 minutes." "If we want to keep the
promise 95% of the time, we must promise 50 minutes, not 30."

**Real-life uses:**

- Service level agreements: "99% of requests answered in under X ms" (the p99 latency).
- Finance: **Value at Risk** is the `ppf` of the loss distribution.
- Engineering: building a dam for the "100-year flood" (the `ppf` at 0.99 of yearly maximum water levels).
- Inventory: how much stock covers 95% of demand?

---

## Module 2: Branch A vs branch B (two samples)

**File:** `m02_branch_comparison.py`
**Business question:** *The new branch B seems slower. Is it really, or is it bad luck?*
**Data:** 200 deliveries from A (true mean 31.5) and 180 from B (true mean about 35.5).

Four methods answer the same question in different ways. Comparing them is the point of this module.

### 2.1 Two-sample t-test (Welch): `stats.ttest_ind(b, a, equal_var=False)`

**What it is.** A test of H0: *"the two groups have the same mean"*.
`equal_var=False` gives **Welch's version**, which does not assume equal spread in the two groups.
Use it as your default: it costs almost nothing when the spreads are equal and protects you when they aren't.

**Our result:** t = 2.22, p = 0.027. B is slower by **0.28 to 4.74 minutes** (95% CI).

**How to interpret:** p < 0.05, so the difference is probably real. The CI is wide: B could be
barely slower, or almost 5 minutes slower. More data would make the range narrower.

**Real-life uses:** A/B tests (average spend with version A vs B of a website),
clinical trials (blood pressure with drug vs placebo), comparing two machines on a production line.

**Note:** use `ttest_rel` instead when the two samples are **paired**, meaning the same subjects
measured twice (e.g. weight before and after a diet).

### 2.2 Two-sample Kolmogorov-Smirnov: `stats.ks_2samp`

**What it is.** A test of H0: *"the two samples come from the same distribution"*.
It compares the **whole shape** (centre, spread, tails), not only the mean. D is the largest gap
between the two cumulative curves.

**Our result:** D = 0.148, p = 0.027. The distributions differ.

**How to interpret:** a t-test can miss differences where the means are equal but the shapes are
different (e.g. B is equally fast on average but much more unpredictable). KS can catch these.
Its weakness: it is less powerful than the t-test when the real difference *is* just a shift of the mean.

**Real-life uses:** detecting **data drift** in machine learning (does this month's input data look
like the training data?), comparing a simulation with reality, and checking whether two production
batches have the same quality profile.

### 2.3 Bootstrap: `stats.bootstrap`

**What it is.** A way to get a confidence interval for **any** statistic, including ones with no
formula, by resampling. The idea:

1. Draw a new sample of the same size from your data, *with replacement* (some values appear twice, some not at all).
2. Calculate your statistic on it (here: the median of B minus the median of A).
3. Repeat thousands of times (here 5000).
4. The spread of those 5000 results shows how uncertain your statistic is.

**Our result:** median difference = 1.85 min, 95% CI = [−0.92, 5.09].

**How to interpret:** the interval **includes 0**, so the *median* difference alone is not
convincing evidence, even though the mean difference was. This is not a contradiction: the
median uses less information than the mean and is noisier. With skewed data you often care about the
median ("a typical customer waits..."), so it is worth knowing how uncertain it is.

**Real-life uses:** confidence intervals for medians, ratios, percentiles (e.g. the p95 latency),
correlation coefficients, and machine learning model accuracy. In general: any time you don't know
the formula.

### 2.4 Permutation test: `stats.permutation_test`

**What it is.** A test built by shuffling. If the branch didn't matter (H0), the labels "A" and "B"
would be arbitrary. So we:

1. Shuffle the labels randomly and recompute the difference of means.
2. Repeat thousands of times, which gives the distribution of differences "under H0".
3. Check how often a shuffled difference is at least as large as the real one (2.51 min).

**Our result:** p = 0.014. Shuffled labels almost never produce a difference this large.

**How to interpret:** the same as any p-value, but it makes **no assumption about the shape of the
data**. Its conclusion agrees with the t-test, which gives us more confidence.

**Real-life uses:** A/B tests with unusual metrics, small experiments where normality is
doubtful, genetics, and ecology.

---

## Module 3: Rider consistency (equal variances)

**File:** `m03_rider_variance.py`
**Business question:** *All vehicles are equally fast on average, but are scooter riders more unpredictable?*
**Data:** 60 trips each. Mean 30 minutes for all; standard deviation 4 (bike), 5 (car) and 9 (scooter).

**Why variance matters.** Customers hate unpredictability. A rider who always takes 30 minutes
is better than one who takes 15 or 45 at random, even with the same average.
Equal variance is also an **assumption** of some other tests (classic t-test, ANOVA).

All three tests check H0: *"all groups have the same variance"*.

| Test | Assumption | When to use it | Our p |
|---|---|---|---|
| `bartlett` | Data is normal | Most powerful **if** the data is really normal. Misleading otherwise. | ≈ 0 |
| `levene` | Robust | **Good default.** Works well with moderately non-normal data. | ≈ 0 |
| `fligner` (Fligner-Killeen) | Rank-based, non-parametric | Data with outliers or very skewed. | ≈ 0 |

**Our result:** all three reject H0: at least one vehicle is more erratic (the scooter, std 9.2).
Comparing only bike and car gives Levene p = 0.13: no evidence of a difference.

**How to interpret:**

- The test says *that* the variances differ, not *which* group. Look at the standard deviations,
  or compare pairs of groups (as we did for bike vs car).
- If Bartlett rejects but Levene doesn't, suspect non-normal data rather than a real difference in spread.

**Real-life uses:**

- Manufacturing: is the new machine more **consistent** than the old one? (The famous Six Sigma question.)
- Finance: is one investment more volatile (risky) than another?
- Education: are grades more spread out in one school than in another?
- Checking assumptions before a classic ANOVA.

---

## Module 4: App redesign experiment

**File:** `m04_app_experiment.py`

### 4.1 Dunnett's test: `stats.dunnett`

**Business question:** *We tested two new checkout designs against the current app. Does either one
increase the average order value?*
**Data:** 150 users per group. Design 1 has (almost) no effect; design 2 adds 2 euro.

**What it is.** A test that compares **several treatment groups to one control group**, and corrects
for the fact that you are doing several comparisons at once.

**Why the correction is needed (the multiple comparison problem).** Each test with alpha = 0.05 has a
5% chance of a false alarm. With 2 comparisons, the chance of *at least one* false alarm grows to
almost 10%. With 20 comparisons it's about 64%. Dunnett adjusts the p-values to keep the *overall*
false-alarm rate at 5%. It is more powerful than general corrections (like Bonferroni) because it uses
the "everyone vs one control" structure.

**Our result:** design 1 p = 0.76 (no real effect). Design 2 p = 0.0004 (better). This is exactly
what we planted.

**How to interpret:** read each p-value separately, as with a normal test. They are already adjusted.

**Real-life uses:** drug trials with several doses vs placebo, agriculture with several fertilizers
vs no fertilizer, and multi-variant (A/B/C/D) tests in marketing.

### 4.2 Chi-square test of independence: `stats.chi2_contingency`

**Business question:** *Does the new app convert more visitors into customers?*
**Data:** a **contingency table** (a table of counts):

|  | ordered | left | conversion |
|---|---|---|---|
| old app | 120 | 880 | 12.0% |
| new app | 158 | 842 | 15.8% |

**What it is.** A test of H0: *"the row variable and the column variable are independent"*, here
"whether a visitor orders doesn't depend on the app version".
It calculates the counts we would **expect** if H0 were true (139 orders for each version), and
measures how far the observed counts are from those.

**Our result:** chi² = 5.72, p = 0.017, dof = 1. The new app converts better.

**How to interpret:**

- p < 0.05 means the two variables are related.
- **dof** (degrees of freedom) = (rows − 1) × (columns − 1). It is used for the p-value; you rarely need it yourself.
- The test says *whether* there is a relation, not *how strong*. Look at the percentages (12.0% → 15.8%) for the size.
- Rule of thumb: every **expected** count should be at least 5. If not, use Fisher (next section).

**Real-life uses:** surveys ("Is the preferred product related to the age group?"), medicine
("Is smoking related to the disease?"), marketing (conversion by channel), and quality control
(defects by shift).

### 4.3 Fisher's exact test: `stats.fisher_exact`

**Business question:** *In a small test, 12 customers got a coupon and 12 didn't. Did the coupon bring them back?*

|  | came back | didn't |
|---|---|---|
| coupon | 9 | 3 |
| no coupon | 3 | 9 |

**What it is.** The same question as chi-square for a 2×2 table, but the p-value is calculated
**exactly** instead of approximated. This makes it reliable even with very small counts.

**Our result:** p = 0.039. The coupon probably works.

**When to use it:** small samples, or any expected count below 5. With large samples, chi-square and
Fisher give almost the same answer.

**Real-life uses:** rare diseases, small pilot studies, early-stage A/B tests, and the original
famous example: the "lady tasting tea" experiment.

### 4.4 Odds ratio: `scipy.stats.contingency.odds_ratio`

**What it is.** A measure of **how strong** the effect is in a 2×2 table.

- The **odds** of coming back with a coupon = 9 / 3 = 3 ("3 to 1").
- The odds without a coupon = 3 / 9 = 0.33.
- **Odds ratio** = 3 / 0.33 ≈ 9. SciPy reports 8.1, because by default it uses a more accurate
  estimator (the *conditional* maximum likelihood estimate).

**Our result:** OR = 8.1, 95% CI from 1.1 to 84.6.

**How to interpret:**

- OR = 1 means no effect. OR > 1 means the event is more likely in the first row (coupon). OR < 1 means less likely.
- The CI does not include 1, which confirms the effect, but it is **huge** (1.1 to 84.6).
  With only 24 people we know the coupon helps, but not by how much. A bigger test is needed before
  making a budget decision.
- Odds are not the same as probabilities. "8 times the odds" is not "8 times as likely".

**Real-life uses:** medicine and epidemiology (smokers have X times the odds of a disease),
and logistic regression results, which are reported as odds ratios.

---

## Module 5: Order volume (discrete distributions)

**File:** `m05_order_volume.py`
**Business question:** *Do orders per hour follow a Poisson model? If yes, we can plan the kitchen with it.*
**Data:** order counts for 200 hours, from a Poisson distribution with 7 orders per hour on average.

### 5.1 The Poisson distribution: `stats.poisson`

**What it is.** The classic model for **counts of independent events in a fixed interval**, when
events happen at a constant average rate λ (lambda). It has a single parameter: the mean λ, which is
also its variance.

**Discrete vs continuous.** Counts are whole numbers (0, 1, 2, …), so this is a **discrete**
distribution. Instead of `pdf` (density) it has `pmf` (**probability mass function**), the probability
of exactly one value. The other methods work as in module 1.

| Method | Question | Our result |
|---|---|---|
| `pmf(7)` | P(exactly 7 orders) | 14.9% |
| `cdf(5)` | P(at most 5) | 29.9% |
| `sf(10)` | P(more than 10) | 9.9% |
| `ppf(0.95)` | 95% of hours have at most... | 12 orders |
| `rvs(6)` | Simulate 6 hours | e.g. [14, 6, 7, 5, 9, 5] |

**How to interpret for the business:** about 1 hour in 10 has more than 10 orders. If the kitchen
is staffed for 12 orders per hour, it is enough 95% of the time.

**Real-life uses:** calls per minute to a call centre, website visits per second, typos per page,
accidents per month at a crossroads, goals in a football match, and server requests (capacity planning).

Other common discrete distributions: **binomial** (number of successes in n tries, e.g. 3 defective
items in a box of 50), **geometric** (number of tries until the first success), and
**negative binomial** (like Poisson, but with more variance than the mean, which is very common in real data).

### 5.2 Chi-square goodness-of-fit test: `stats.chisquare`

**What it is.** A test of H0: *"the observed counts match the counts the model expects"*.

How it works in our script:

1. Group the data into bins: ≤3 orders, 4, 5, …, 10, ≥11.
2. Count how many hours fall in each bin (**observed**).
3. Use the Poisson model to calculate how many hours *should* fall in each bin (**expected**).
4. Compare them with `chisquare(observed, expected, ddof=1)`.

**Our result:** chi² = 4.54, p = 0.72. **Poisson fits well.**

**How to interpret:**

- Large p: the model is compatible with the data. Small p: it isn't.
- Look at the bins too: they tell you *where* a model fails (e.g. too many hours with 0 orders).
- `ddof=1` because we estimated one parameter (λ) from the data itself. This removes one degree of freedom.
  Forgetting it makes the test too lenient.
- Bins should have an expected count of at least 5. This is why the extremes are grouped (≤3, ≥11).

**Real-life uses:** is a die fair? Are customers spread evenly over the days of the week?
Do the observed genetic ratios match Mendel's 3:1? Are election results consistent with the polls?

**Chi-square in two flavours:** `chisquare` (this module) compares **one** variable with a model.
`chi2_contingency` (module 4) tests whether **two** variables are related.

---

## Module 6: Customer ratings (repeated measures)

**File:** `m06_ratings.py`
**Business question:** *The same 25 customers tasted 3 pizzas and rated each from 1 to 10. Is one preferred?*

### Friedman test: `stats.friedmanchisquare`

**What it is.** A non-parametric test for **3 or more related measurements on the same subjects**.
H0: *"all pizzas are rated the same"*.

**Why not three separate tests, or an ANOVA?** The ratings are **not independent**. A grumpy customer
gives low scores to *all* pizzas, and a generous one gives high scores to all. Friedman deals with
this by **ranking the pizzas inside each customer** (1st, 2nd, 3rd), so only each person's
*preference* counts, not their general mood. Ratings from 1 to 10 are also ordinal ("8" is better than
"6", but not necessarily by the same amount as "6" vs "4"), and ranks suit this kind of data.

**Our result:** statistic = 22.9, p = 0.00001. At least one pizza is rated differently.
We planted it: the Diavola is liked more.

**How to interpret:**

- The test says *that* there is a difference, not *which* pizza. To find which one, follow up with
  pairwise tests (e.g. `stats.wilcoxon` for each pair) and a multiple-comparison correction.
- If there are only two related measurements, use `stats.wilcoxon` instead.

**Real-life uses:** wine or food tasting panels, the same patients trying several treatments,
the same users rating several app designs, and comparing machine learning models on the same datasets.

---

## Module 7: What drives lateness (correlation)

**File:** `m07_lateness_drivers.py`
**Business questions:** *(a) Does a longer distance mean a longer delivery? (b) Does the hour of day matter?*
**Data:** for (a), time grows in a straight line with distance. For (b), deliveries are slow at lunch
AND at dinner, and fast at 17:00: a **U shape**.

### 7.1 Three correlation coefficients

All three range from **−1 to +1** and come with a p-value for H0: *"no association"*.

| Function | Measures | Assumes / good for |
|---|---|---|
| `pearsonr` (r) | **Linear** (straight-line) association | Linear relation, no big outliers |
| `spearmanr` (rho) | **Monotonic** association: as x grows, does y keep growing (or keep falling)? Pearson computed on ranks. | Curved but steady relations, outliers, ordinal data |
| `kendalltau` (tau) | Share of pairs that "agree" (both values go up together) minus pairs that disagree | Small samples, many ties. Easy to explain: tau = 0.73 means pairs agree much more often than they disagree. |

**How to read a coefficient:**

- **+1**: perfect increase. **−1**: perfect decrease. **0**: no (linear or monotonic) association.
- Rough guide for |r|: < 0.3 weak, 0.3–0.7 moderate, > 0.7 strong.
- r² is the share of the variation explained: r = 0.90 → r² = 0.81, so distance explains about
  81% of the variation in delivery time.
- Kendall's tau is usually smaller in absolute value than the other two for the same data. That is normal.

**Our results:**

- (a) Distance: r = 0.90, rho = 0.91, tau = 0.73, all p ≈ 0. A strong positive relation. Pearson 95% CI: 0.86 to 0.93.
- (b) Hour of day: **all three are about 0** (p > 0.5)!

**The lesson of (b).** The relationship is very strong, but it goes *down* then *up*. These
coefficients only look for one direction, so the two halves cancel out.
**A correlation of 0 does NOT mean "no relationship".** Always plot your data.

**And the classic warning:** correlation is not causation. Ice cream sales and drownings are
correlated, because both go up in summer. Here we know that distance *causes* delay because we built
the data. With real data you would need an experiment or careful reasoning.

**Real-life uses:** height vs weight, advertising budget vs sales, study hours vs grades (Spearman,
if grades are ranks), agreement between two judges (Kendall), and feature selection in machine learning.

### 7.2 Multiscale Graph Correlation (MGC): `stats.multiscale_graphcorr`

**What it is.** A modern test that detects **any kind of dependence**: linear, U-shaped, circular,
clusters, and more. It compares how distances between points in x relate to distances between
points in y, at many scales (neighbourhood sizes). Its p-value comes from a permutation test, so it
is slower (`reps=1000` permutations, a few seconds).

**Our result:** statistic 0.41, **p = 0.001**. MGC finds the dependence that all three classic
coefficients missed.

**How to interpret:**

- Small p means x and y are related in *some* way. It doesn't say which shape; plot the data to see it.
- It also works with **multi-dimensional** x and y (e.g. x = 50 genes, y = 3 symptoms).
- Use it when you suspect non-linear effects, or as a check after a "correlation ≈ 0" result.

**Real-life uses:** the effect of temperature on energy consumption (high in winter AND in summer
because of air conditioning, so a U shape), drug dose vs response (often an inverted U),
neuroscience and genomics (finding which brain regions or genes relate to a condition).

---

## Module 8: Where and when (kernel density estimation)

**File:** `m08_kde.py`
**Business questions:** *(a) What does our delivery-time curve look like, without assuming a formula?
(b) Where do our customers live, and where should we open a second shop?*
**Output:** `kde_delivery_times.png` and `kde_customer_map.png`.

### 8.1 Univariate KDE: `stats.gaussian_kde(times)`

**What it is.** A smooth estimate of the probability density, built directly from the data.
It places a small bell curve (the "kernel") on every data point and adds them up.
It is like a smooth histogram that doesn't depend on where you put the bin edges.

**Why.** In module 1 we *assumed* a gamma shape. KDE makes **no assumption**, so it can show
unexpected things, like two peaks (bimodal data), which no single standard distribution captures.

**The bandwidth** (`bw_method`) controls the width of each small bell curve. It is the one decision that matters:

- **Too wide** (0.5 in the chart): too smooth, real details disappear.
- **Too narrow** (0.05): bumpy, random noise looks like structure.
- **Automatic** (default, Scott's rule): usually a good start. The chart shows all three.

**Our result:** most likely time = 30.8 min. P(25 < time < 35) = 38.2% (calculated with `integrate_box_1d`).

**How to interpret:** the height of the curve is a *density*, not a probability. The probability is the
**area** under the curve between two values. The density at 30 min (0.040) means
that about 4% of deliveries fall in a 1-minute window around 30.

**Real-life uses:** showing distributions instead of histograms ("violin plots" use KDE),
anomaly detection (a point in a region of very low density is suspicious), and comparing groups visually.

### 8.2 Multivariate KDE: `stats.gaussian_kde(points.T)`

**What it is.** The same idea in 2 (or more) dimensions. Here: customer locations on a map,
which gives a **heat map** of where customers are.
Note: `gaussian_kde` expects an array with shape (dimensions, number of points), hence the `.T` (transpose).

**Our result:** the busiest spot is at (3.0, 2.1) km, the university campus. Its density (0.127) is
almost twice the density around our shop (0.069). **The campus is the natural place for a second shop.**

**`resample`.** The KDE can generate **new fake points** that look like the real data.
This is useful for simulations, such as testing how a second shop would perform with
1000 realistic customer locations.

**Real-life uses:** crime hot-spot maps for police patrols, choosing locations for shops,
charging stations or ambulances, heat maps of where users click on a web page, and
wildlife habitat maps from GPS-tagged animals.

---

## Module 9: Friday-night simulator

**File:** `m09_simulator.py`
**Business question:** *How many riders do we need on a Friday evening (18:00 to 23:00)?*

This module builds a complete night out of probability distributions and simulates it 1000 times.
It shows how distributions are used as **building blocks** for simulation.

### 9.1 Shape, loc, scale, and freezing

**What it is.** Every classic SciPy distribution can be adjusted with:

- **shape** parameters: change the form of the curve (e.g. `a` for gamma: low values give a strong right tail).
- **`loc`**: shifts the curve left or right (adds a constant).
- **`scale`**: stretches or squeezes it (multiplies by a constant).

`stats.gamma(a=6, loc=5, scale=4)` means: take a standard gamma with shape 6, multiply by 4, add 5.
In our model, `loc=5` means "a trip never takes less than 5 minutes" (park, ring, take payment).

**Freezing.** Calling the distribution with its parameters, `trip = stats.gamma(a=6, loc=5, scale=4)`,
creates a **frozen** distribution: the parameters are fixed once, and then you just call
`trip.mean()`, `trip.sf(40)`, `trip.rvs(100)`. This avoids repeating the parameters and avoids mistakes.

**Our result:** trip mean = 29.0 min, std = 9.8 min, minimum (support) = 5.

**Real-life uses:** shift a model when there is a fixed minimum time (processing, set-up, travel),
change units (scale = 60 converts hours to minutes), and adapt a standard model to your own data.

### 9.2 Broadcasting

**What it is.** Passing **arrays** of parameters and values to calculate many results in a single
call, following NumPy's rules for combining array shapes.

```python
thresholds = np.array([30, 40, 50])           # shape (3,)  -> columns
scales = np.array([[3.5], [4.0], [5.0]])      # shape (3, 1) -> rows
stats.gamma.sf(thresholds, a=6, loc=5, scale=scales)   # result: a 3 x 3 table
```

**Our result:** with heavy traffic, 30.1% of trips take more than 40 minutes (vs 6.7% with light traffic).

**Why.** It is shorter, faster than loops, and great for "what if" tables.

**Real-life uses:** sensitivity tables (risk under several scenarios), and calculating
probabilities for thousands of products or customers at once, each with their own parameters.

### 9.3 The new random-variable API: `stats.Normal`, `stats.make_distribution`

**What it is.** A newer, cleaner interface for distributions (SciPy 1.15+). The main differences from the classic one:

| Classic | New |
|---|---|
| `stats.norm(loc=12, scale=3)` | `stats.Normal(mu=12, sigma=3)`: natural parameter names |
| `.sf(x)`, `.ppf(q)` | `.ccdf(x)`, `.icdf(q)`: clearer names (complementary cdf, inverse cdf) |
| `loc` / `scale` arguments | Plain arithmetic: `4 * Gamma(a=6) + 5` |
| `.rvs(n, random_state=...)` | `.sample(n, rng=...)` |

`stats.make_distribution(stats.gamma)` converts a classic distribution into the new style.

**Our result:** P(prep time > 15 min) = 15.9%. The new-style trip model gives P(trip > 40) = 0.132,
exactly like the classic one, which shows that both describe the same distribution.

**Why it matters.** It is the direction SciPy is moving in. It is easier to read, transformations
(shift, scale, absolute value, exponential) are simple arithmetic, and it is often more accurate in the
extreme tails. The SciPy tutorial's *Random Variable Transition Guide* covers it in detail.

### 9.4 Sampling from your own distribution: `NumericalInversePolynomial`

**What it is.** A tool from `scipy.stats.sampling` that creates a **fast random number generator for
any distribution**. You only need to give it the density function (`pdf`), and it doesn't even need
to be normalised (the area under the curve doesn't need to be 1).

**In our project:** Friday orders have two peaks, at 20:00 and 21:30. No standard distribution has this
shape, so we write the curve ourselves (two bell curves added together) and let SciPy build a sampler.

**Why not just use `rvs`?** Standard distributions don't fit custom shapes. Writing your own sampler
correctly is difficult. `NumericalInversePolynomial` does the hard work once, and then every
sample is very fast.

**Real-life uses:** simulating customer arrivals with your own daily pattern, physics and engineering
simulations with measured (empirical) curves, synthetic data generation, and risk models with custom loss curves.

### 9.5 Monte Carlo simulation

**What it is.** Answering a question by **simulating it many times** with random draws, instead of solving
equations. Each simulated night:

1. Number of orders ~ Poisson(70).
2. Each order's time ~ the custom two-peak curve.
3. Plus kitchen time ~ Normal(12, 3).
4. Plus trip time ~ gamma (as in 9.1).
5. Count the maximum number of riders on the road at the same moment.

**Our result (1000 nights):** on average 20 riders are needed at the peak; the worst night needed 32.

| Riders | Nights where nobody waits for a rider |
|---|---|
| 14 | 4% |
| 18 | 34% |
| 20 | 57% |
| 22 | 77% |

**How to interpret:** this is a **business trade-off**, not a single answer. Each extra rider costs
money; each wait costs customer satisfaction. The table shows the price of each level of service.

**Real-life uses:** staffing in call centres and hospitals, project deadline risk ("we finish by
March with 80% probability"), investment portfolio risk, queues in supermarkets and airports, and inventory planning.

### 9.6 Monte Carlo test: `stats.monte_carlo_test`

**Business question:** *Last Friday, the 40 recorded trips felt slow. Was it a normal Friday, or did something change?*

**What it is.** A test where the "H0 world" is **simulated** from a model you specify:

1. H0: *"last Friday's trips come from our usual trip model"*.
2. Simulate thousands of fake "last Fridays" with 40 trips each from the model, and calculate each one's mean.
3. p = the share of fake Fridays with a mean at least as high as the real one.

**Our result:** real mean 34.1 min vs model 29.0, **p ≈ 0.001** (it varies slightly every run).
Last Friday was unusually slow. In the data, we secretly made it slower.

**Why.** It works for any model you can simulate and any statistic you can calculate, even when no
formula exists for the p-value.

**Real-life uses:** checking whether today's data fits the usual model (fraud detection, unusual
machine behaviour), testing goodness of fit for complex models, and physics and astronomy
(is a signal real or random noise?).

---

## Module 10: Scenario sweep (Quasi-Monte Carlo)

**File:** `m10_scenario_sweep.py`
**Business question:** *Which combination of riders (3–20), cooks (2–12) and ovens (1–8) gives the highest
profit on a Friday night?*
**Data:** a made-up profit formula. With too few staff we lose orders; with too many, wages eat the profit.

### 10.1 Why Quasi-Monte Carlo?

When a model is slow to run (a real simulation can take minutes per scenario), you can't try every
combination. You choose a **sample of scenarios**. There are three ways to choose:

| Method | Idea | Problem |
|---|---|---|
| Plain random | Pick points at random | Clusters and empty gaps appear by chance |
| **Latin Hypercube** (`qmc.LatinHypercube`) | Each range is cut into n slices and each slice is used exactly once | Better, but still some gaps |
| **Sobol sequence** (`qmc.Sobol`) | A mathematical sequence designed to fill the space as evenly as possible | Works best when n is a power of 2 (64, 128, ...) |

**Quasi-random** means: it looks random, but it is built to be evenly spread.

### 10.2 Discrepancy: `qmc.discrepancy`

**What it is.** A number that measures **how unevenly** points cover the space. **Lower is better.**

**Our result (64 points in 3 dimensions):**

| Method | Discrepancy |
|---|---|
| plain random | 0.0144 |
| Latin hypercube | 0.0013 |
| Sobol | **0.0004** |

Sobol covers the space about 30 times more evenly than random points.

### 10.3 Scaling: `qmc.scale`

QMC points are always between 0 and 1. `qmc.scale` converts them into real ranges (3–20 riders, etc.).
Then we round, because you can't hire half a cook.

### 10.4 Our results

- 64 Sobol scenarios: the best was 7 riders, 5 cooks, 3 ovens → 485 euro.
- 64 more Sobol points "zoomed in" around that best scenario: 10 riders, 6 cooks, 5 ovens → 760 euro.
- The full grid (1584 evaluations) finds the true best: 10 riders, 6 cooks, 4 ovens → 810 euro.
- **QMC reached 94% of the best profit with 128 evaluations instead of 1584.**

**How to interpret:** QMC doesn't guarantee the exact best answer, but it gets close with a fraction of the work.
This "wide search, then zoom in" approach is common in practice. When each evaluation takes an hour, it is the
difference between 5 days of computing and 2 months.

**Real-life uses:**

- **Engineering design**: testing car or aircraft designs in expensive simulations (Latin Hypercube is a standard tool).
- **Finance**: pricing complex options. QMC gives more accurate answers than Monte Carlo with the same number of samples.
- **Machine learning**: hyperparameter search (learning rate × layers × batch size).
- **Computer graphics**: rendering light in 3D scenes.
- **Uncertainty analysis**: which input parameters matter most in a climate or epidemic model?

---

## Cheat sheet: which test should I use?

| I want to... | Normal-ish data | Skewed data / outliers / ranks | Any data (simulation) |
|---|---|---|---|
| Compare 1 mean to a reference value | `ttest_1samp` | `wilcoxon` (vs the reference) | `bootstrap` CI |
| Compare 2 independent groups | `ttest_ind(equal_var=False)` | `mannwhitneyu`, `ks_2samp` | `permutation_test`, `bootstrap` |
| Compare 2 paired measurements | `ttest_rel` | `wilcoxon` | `permutation_test` |
| Compare 3+ independent groups | `f_oneway` (ANOVA) | `kruskal` | `permutation_test` |
| Compare 3+ groups vs one control | `dunnett` | | |
| Compare 3+ repeated measurements | (repeated measures ANOVA, not in SciPy) | `friedmanchisquare` | |
| Compare spreads (variances) | `bartlett` | `levene`, `fligner` | |
| Check normality | `shapiro` (+ histogram!) | | |
| Check fit to any distribution | `kstest`, `goodness_of_fit` | | `monte_carlo_test` |
| Counts vs expected counts | `chisquare` | | |
| Relation between 2 categorical variables | `chi2_contingency` (large counts) | `fisher_exact` (small counts, 2×2) | |
| Strength of a 2×2 effect | `odds_ratio` | | |
| Relation between 2 numeric variables | `pearsonr` | `spearmanr`, `kendalltau` | `multiscale_graphcorr` (any shape) |
| Estimate a density without a formula | | `gaussian_kde` | |

Functions in this table that are not used in the project (`wilcoxon`, `mannwhitneyu`, `ttest_rel`,
`f_oneway`, `kruskal`, `goodness_of_fit`) all exist in `scipy.stats` and follow the same style.

---

## Experiments to try

Change one number in `data.py`, re-run the matching module, and see whether the tests notice.

| Change in `data.py` | Run | What should happen |
|---|---|---|
| `two_branches`: make B as fast as A (`scale=3.5`) | m02 | p-values rise above 0.05; the CIs include 0 |
| `two_branches`: use `size=30` for both | m02 | Same true difference, but less data: p-values grow, CIs get wider |
| `rider_times`: give scooters `std=4` | m03 | No test finds a difference in spread |
| `app_experiment`: give design 1 a mean of `27` | m04 | Design 1 becomes significant too |
| `coupon_test`: multiply every count by 5 | m04 | p drops a lot; the odds ratio CI gets much narrower |
| `orders_per_hour`: `rng.integers(0, 15, size=hours)` | m05 | Chi-square rejects the Poisson model |
| `pizza_ratings`: remove the `+ 1.0` from Diavola | m06 | Friedman finds no preference |
| `lateness_drivers`: replace `** 2` with `** 1` | m07 | Now Pearson & co. find the hour effect too |
| `delivery_times`: remove the `outliers` | m01 | Skewness and kurtosis drop; plain and trimmed means get close |

The second experiment is the most instructive: it shows that **"not significant" often means
"not enough data"**, not "no effect".

---

## Glossary

| Term | Meaning |
|---|---|
| **Sample** | The data you actually have (e.g. 306 deliveries). |
| **Population** | Everything you want to draw conclusions about (all past and future deliveries). |
| **Distribution** | The pattern of how likely each value is. |
| **pdf / pmf** | Probability density (continuous) / probability mass (discrete): the height of the distribution curve. |
| **cdf** | Cumulative distribution function: P(value ≤ x). |
| **sf / ccdf** | Survival function: P(value > x) = 1 − cdf. |
| **ppf / icdf** | Percent point function / inverse cdf: the value below which q% of the data falls (a quantile). |
| **Quantile / percentile** | The value below which a given share of the data falls. The median is the 50th percentile. |
| **Frozen distribution** | A distribution with its parameters fixed, ready to be queried. |
| **Statistic** | A number calculated from the sample (a mean, t, chi², D...). |
| **Null hypothesis (H0)** | The "nothing is going on" explanation a test tries to reject. |
| **p-value** | How surprising the data would be if H0 were true. Small means surprising. |
| **alpha** | The p-value limit chosen in advance (usually 0.05). |
| **Confidence interval** | A range of plausible values for the true quantity. |
| **Effect size** | How big a difference or relation is (e.g. 2.5 minutes, OR = 8, r = 0.9), as opposed to *whether* it exists. |
| **Degrees of freedom (dof)** | Roughly: the number of independent pieces of information left after estimating parameters. |
| **Parametric** | Assumes a specific shape for the data (usually normal). |
| **Non-parametric** | Assumes little about the shape. Often based on ranks. |
| **Resampling** | Getting answers by re-drawing from the data many times (bootstrap, permutation). |
| **Monte Carlo** | Getting answers by simulating random scenarios many times. |
| **Quasi-Monte Carlo (QMC)** | Like Monte Carlo, but with points spread evenly instead of at random. |
| **Outlier** | A value far from the rest. It can be an error or the most important observation you have. |