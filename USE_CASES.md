# Use cases: the Pizza Veloce techniques in big data

The [README](README.md) explains each technique using the pizzeria. This file takes the same
techniques out of the pizzeria and puts them into **big-data situations**: millions of rows,
event logs, A/B tests on large platforms, sensor streams, fraud monitoring.

Each technique follows the same four steps:

- **Situation**: the problem a data team has.
- **Why this technique**: why it fits better than the alternatives.
- **How**: the code (usually a few lines), plus any big-data adjustment.
- **Interpreting the result**: what the output means and what to do next.

---

## Contents

0. [Before you start: what changes when n is huge](#0-before-you-start-what-changes-when-n-is-huge)
1. [Describing one sample (module 1)](#1-describing-one-sample-module-1)
2. [Comparing two groups (module 2)](#2-comparing-two-groups-module-2)
3. [Comparing spread (module 3)](#3-comparing-spread-module-3)
4. [Experiments and categorical data (module 4)](#4-experiments-and-categorical-data-module-4)
5. [Counts and discrete distributions (module 5)](#5-counts-and-discrete-distributions-module-5)
6. [Repeated measures (module 6)](#6-repeated-measures-module-6)
7. [Correlation and dependence (module 7)](#7-correlation-and-dependence-module-7)
8. [Density estimation (module 8)](#8-density-estimation-module-8)
9. [Simulation (module 9)](#9-simulation-module-9)
10. [Scenario sweeps with Quasi-Monte Carlo (module 10)](#10-scenario-sweeps-with-quasi-monte-carlo-module-10)
11. [Quick map: situation → technique](#11-quick-map-situation--technique)

---

## 0. Before you start: what changes when n is huge

Read this first. These rules apply to every use case below.

1. **Everything is "significant".** With 10 million rows, a difference of 0.01 seconds gets
   p < 0.001. The p-value answers *"is there any difference at all?"*. With big data the answer
   is almost always yes. The useful question is *"is the difference big enough to matter?"*.
   **Always report an effect size and a confidence interval next to the p-value.** Decide in
   advance the smallest difference that matters to the business.
2. **Bias does not shrink with n.** A million rows from a biased sample give you a very precise
   wrong answer. Check how the data was collected before you test anything.
3. **Rows are often not independent.** Events from the same user, server or day are
   correlated. Tests that assume independence will give p-values that are too small. Aggregate
   to the right unit (one row per user, per session, per day) or resample whole clusters.
4. **Some functions do not scale.** `shapiro` warns above 5,000 rows. `fisher_exact`,
   `permutation_test`, `bootstrap`, `multiscale_graphcorr` and `gaussian_kde` get slow on very
   large inputs. The usual fix is a **random sample**: 10,000 to 100,000 rows are enough for
   almost any distribution-level question.
5. **Many tests means many false alarms.** If you run a test on 1,000 metrics or 1,000
   segments at alpha = 0.05, you will get about 50 false positives by chance. Correct with
   `scipy.stats.false_discovery_control` (Benjamini-Hochberg) or Bonferroni.

---

## 1. Describing one sample (module 1)

### 1.1 `stats.describe`: first look at a new table

**Situation.** A streaming platform has a new table of 800 million playback sessions. Before
building a dashboard, the analytics team wants to know what "session length" looks like.

**Why this technique.** `describe` returns count, min/max, mean, variance, skewness and
kurtosis in one call. It is the fastest way to spot problems: negative durations, a maximum of
72 hours (a player left on overnight), or extreme skewness.

**How.**

```python
sample = df["session_minutes"].sample(1_000_000, random_state=0).to_numpy()
d = stats.describe(sample, nan_policy="omit")
```

On a cluster, compute the same moments with the engine (Spark, BigQuery) and use SciPy on a sample.

**Interpreting the result.**

- `minmax` shows impossible values: these are data-quality bugs, not statistics.
- Skewness > 1 means a long right tail. The mean will be pulled up by a few very long sessions,
  so report the **median** on the dashboard.
- Kurtosis much larger than 0 (SciPy reports *excess* kurtosis) means heavy tails: expect
  outliers, and expect tests that assume normality to work badly.

---

### 1.2 Trimming and winsorization: robust KPIs

**Situation.** An e-commerce site reports "average order value" every day. Some days, one B2B
customer orders €400,000 of stock, and the daily average jumps by 30%. The KPI swings and
nobody trusts it.

**Why this technique.** Deleting "outliers" by hand is arbitrary and hides real revenue.
`trim_mean` ignores the top and bottom x% for the **summary only**. `winsorize` caps extreme
values at a percentile, so they still count but cannot dominate. Both rules are fixed and
reproducible.

**How.**

```python
stats.trim_mean(order_values, 0.01)                 # ignore 1% at each end
capped = winsorize(order_values, limits=(0, 0.01))  # cap the top 1%
```

**Interpreting the result.**

- If the trimmed mean is close to the plain mean, outliers are not a problem.
- If they differ a lot, report both: the trimmed mean for the "typical customer" trend, the
  plain sum for finance.
- Winsorize **before** a model or a t-test. Never use it on the revenue total itself.

---

### 1.3 Normality tests: `shapiro`, `normaltest`, `skewtest`, `kurtosistest`, `jarque_bera`

**Situation.** A fintech team wants to use a model that assumes normally distributed daily
returns to compute risk (Value at Risk) on 20 years of data from 3,000 stocks.

**Why this technique.** If returns are not normal (they usually have fat tails), a normal
model will underestimate the probability of a crash. The tests check this assumption
formally. `skewtest` and `kurtosistest` also tell you **how** the data is not normal:
asymmetry, or fat tails.

**How.**

```python
r = returns_one_stock            # ~5,000 days
stats.jarque_bera(r).pvalue      # skewness + kurtosis together, works with large n
stats.kurtosistest(r).pvalue     # are the tails fatter than normal?
```

`jarque_bera` and `normaltest` work well with large n. `shapiro` is best for n < 5,000.

**Interpreting the result.**

- With big data these tests **always** reject (see rule 1), because nothing real is perfectly
  normal. A small p-value says "not exactly normal", not "too far from normal to use".
- Combine the test with a picture (histogram, Q-Q plot) and the actual skewness and kurtosis
  numbers. A kurtosis of 0.1 is harmless. A kurtosis of 8 (typical for stock returns) means
  the normal model will badly underestimate extreme losses.
- Next step: use a fat-tailed distribution (Student t), or a method that does not assume
  normality (bootstrap, permutation tests, rank tests).

---

### 1.4 Fitting a distribution: `stats.gamma.fit`

**Situation.** A cloud provider wants to model API response times across 50 regions to set
timeouts and plan capacity. Response times are positive and right-skewed.

**Why this technique.** A fitted distribution compresses millions of observations into 2–3
parameters. You can then compute probabilities, compare regions by their parameters, and
simulate. The gamma distribution is a natural choice for positive, skewed waiting times
(lognormal and Weibull are the other usual candidates).

**How.**

```python
shape, loc, scale = stats.gamma.fit(latency_sample, floc=0)  # latency starts at 0
fitted = stats.gamma(shape, loc, scale)
```

Fit on a random sample per region. 50,000 points give stable parameters.

**Interpreting the result.**

- `shape` controls the form: a small shape means a strong skew, a large shape looks almost
  normal. `scale` stretches the times. The mean is `shape * scale`.
- Region-by-region parameter tables show which regions have a heavier tail, not only a
  higher average.
- A fit is only a candidate. Check it with KS (1.5) and with a plot before you rely on it.

---

### 1.5 One-sample Kolmogorov-Smirnov: `stats.kstest`

**Situation.** The cloud team from 1.4 must choose between gamma, lognormal and Weibull for
the latency model.

**Why this technique.** KS compares the whole shape of the data with a candidate distribution,
not only the mean. Its statistic D (the largest gap between the two CDFs) is an easy number to
compare across candidates.

**How.**

```python
for name, dist in {"gamma": stats.gamma, "lognorm": stats.lognorm,
                   "weibull": stats.weibull_min}.items():
    params = dist.fit(sample, floc=0)
    print(name, stats.kstest(sample, dist(*params).cdf).statistic)
```

**Interpreting the result.**

- With big data the p-value will reject every candidate (rule 1). **Compare the D statistics
  instead**: the smallest D is the best fit. D = 0.01 means the model CDF is never more than
  1 percentage point away from the data.
- The p-value is also too optimistic when you fit the parameters on the same data. For a
  correct p-value, fit on one half and test on the other, or use `stats.goodness_of_fit`.

---

### 1.6 One-sample t-test: `stats.ttest_1samp`

**Situation.** A logistics company signs an SLA promising an average delivery time below 48
hours. A regulator audits 2 million shipments from last quarter.

**Why this technique.** The question is about one mean compared with a fixed target.
With a large sample, the central limit theorem makes the mean approximately normal even when
the data is skewed, so the t-test is valid.

**How.**

```python
res = stats.ttest_1samp(hours, popmean=48, alternative="greater")
res.pvalue, res.confidence_interval(0.95)
```

If shipments are grouped by warehouse and day, first average per warehouse-day (rule 3).

**Interpreting the result.**

- Small p: the true average is above 48 hours, so the SLA is breached.
- Look at the confidence interval. "48.02 to 48.05 hours" is a breach by about 2 minutes:
  formally true, commercially irrelevant. "51 to 53 hours" is a real problem.
- Remember the mean hides the tail: an SLA met on average can still have 10% of shipments
  over 72 hours. See 1.7.

---

### 1.7 Tails: `sf` and `ppf`

**Situation.** The same cloud provider must set a timeout so that only 1 request in 1,000 is
cut off, and estimate how often a request will take longer than 2 seconds.

**Why this technique.** These are tail questions. With a fitted distribution, `sf` (survival
function) answers "what share is above x?" and `ppf` (inverse CDF) answers "below what value
are p% of the data?". The model can estimate extreme percentiles where you have few
observations.

**How.**

```python
fitted.sf(2.0)        # share of requests slower than 2 s
fitted.ppf(0.999)     # timeout so that 99.9% of requests finish
```

**Interpreting the result.**

- `sf(2.0) = 0.004` means 0.4% of requests, which is 4,000 per million.
- With big data, compare the model's p99.9 with the **empirical** one (`np.quantile`). If they
  disagree, the fitted distribution is wrong in the tail, and the tail is exactly where the
  decision lives. Trust the empirical value when you have plenty of data.

---

## 2. Comparing two groups (module 2)

### 2.1 Welch two-sample t-test: `stats.ttest_ind(..., equal_var=False)`

**Situation.** A ride-hailing company runs a pricing change in city A but not in city B and
wants to know whether average revenue per ride changed. Each city has 3 million rides.

**Why this technique.** Two independent groups and a question about the mean. Welch's version
does not assume equal variances, which is almost never true in real data, and costs nothing
when variances are equal. It should be the default.

**How.**

```python
res = stats.ttest_ind(treated, control, equal_var=False)
res.confidence_interval(0.95)    # CI for the difference in means
```

**Interpreting the result.**

- Read the confidence interval for the difference, in euros. "+€0.30 to +€0.34 per ride"
  multiplied by the ride volume gives a business number.
- Cities are not a randomized experiment: they differ in weather, events and population. A
  significant difference shows that the cities differ, not that the price caused it. For a
  causal answer, randomize users (see section 4) or use difference-in-differences.

---

### 2.2 Two-sample Kolmogorov-Smirnov: `stats.ks_2samp`

**Situation.** A machine-learning team retrains a fraud model monthly. Before deploying, they
check whether each input feature has **drifted**: is this month's distribution of
"transaction amount" different from the training data?

**Why this technique.** Drift often changes the shape (a new tail, a second peak) without
changing the mean. A t-test would miss that. KS compares the full distributions, needs no
assumption about their form, and is quick on hundreds of features.

**How.**

```python
for col in features:
    res = stats.ks_2samp(train[col], current[col])
    drift[col] = res.statistic
```

**Interpreting the result.**

- With millions of rows every feature will be "significant". Rank features by the **D
  statistic** and set an operational threshold (for example D > 0.1 means investigate).
- `res.statistic_location` tells you where the biggest gap is: at the low end, the high end,
  or the middle. That points to the cause (a new currency, a new merchant type, a bug).

---

### 2.3 Bootstrap: `stats.bootstrap`

**Situation.** A marketplace wants a confidence interval for the **difference in median**
basket size between mobile and desktop users. There is no simple textbook formula for the
CI of a difference of medians.

**Why this technique.** The bootstrap gives a confidence interval for **any** statistic
(median, p95, ratio, Gini coefficient) by resampling the data. It makes no assumption about
the distribution.

**How.**

```python
def median_diff(x, y, axis):
    return np.median(x, axis=axis) - np.median(y, axis=axis)

res = stats.bootstrap((mobile, desktop), median_diff, n_resamples=2000,
                      vectorized=True, method="BCa", random_state=0)
res.confidence_interval
```

With hundreds of millions of rows, bootstrap a random sample, or resample whole users or days
(cluster bootstrap) to respect rule 3.

**Interpreting the result.**

- A CI of (−€3.10, −€2.40) means mobile baskets have a median €2.40 to €3.10 lower. The CI does
  not contain 0, so the difference is real at about the 95% level.
- `res.standard_error` is the uncertainty of the statistic. It is useful for dashboards and
  for planning how much data you need.

---

### 2.4 Permutation test: `stats.permutation_test`

**Situation.** A security team sees that compromised accounts have a higher average number of
distinct login IPs per week than normal accounts. There are only 300 compromised accounts and
the distribution is extremely skewed.

**Why this technique.** The permutation test builds the null distribution from the data itself
by shuffling the group labels. It is exact, works for any statistic, and does not need
normality. That matters here because one group is small and the data is heavy-tailed.

**How.**

```python
def mean_diff(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

stats.permutation_test((compromised, normal_sample), mean_diff,
                       n_resamples=10_000, alternative="greater", vectorized=True)
```

Use a random sample of the normal accounts (for example 50,000). That is plenty.

**Interpreting the result.**

- p = 0.0002 means: if the labels did not matter, only 2 shuffles in 10,000 would show a
  difference this large. The feature carries real signal, so it is worth adding to the
  detection rules.
- The smallest p you can get is about 1 / n_resamples. To report a very small p, raise
  n_resamples.

---

## 3. Comparing spread (module 3)

### `stats.bartlett`, `stats.levene`, `stats.fligner`

**Situation.** A CDN operator runs three cache configurations across data centers. Average
latency is the same for all three, but customers complain about **jitter** (unpredictable
latency). Which configuration is the most consistent?

**Why this technique.** The question is about spread, not about level. These tests check
whether variances are equal across groups.

- `levene` is the robust default.
- `fligner` is rank-based and best for latency data, which has heavy tails.
- `bartlett` is only safe when data is normal, which latency never is.

**How.**

```python
stats.levene(cfg_a, cfg_b, cfg_c, center="median")   # Brown-Forsythe variant
stats.fligner(cfg_a, cfg_b, cfg_c)
```

**Interpreting the result.**

- A small p says at least one configuration has a different spread. It does not say which.
  Compare standard deviations or IQRs, or test pairs and correct for multiple tests.
- If Bartlett rejects but Levene and Fligner do not, the "difference" comes from non-normal
  tails, not a real difference in spread.
- With big data, rejection is likely (rule 1). Decide whether the gap matters: an IQR of 12 ms
  vs 14 ms may not matter, while a p99 of 40 ms vs 400 ms does.

---

## 4. Experiments and categorical data (module 4)

### 4.1 Dunnett's test: `stats.dunnett`

**Situation.** A news site tests four new homepage layouts against the current one
(control) on 5 million visitors. The metric is time on page.

**Why this technique.** Running four separate t-tests against the control raises the chance
of a false winner (rule 5). Dunnett compares every variant with **the control only** and
corrects for that, and it is more powerful than a general correction like Tukey's, which also
compares variants with each other.

**How.**

```python
res = stats.dunnett(layout_1, layout_2, layout_3, layout_4, control=current)
res.pvalue
res.confidence_interval(0.95)
```

**Interpreting the result.**

- One p-value per variant, already corrected. A variant with p < 0.05 differs from the control.
- Read the CIs as effect sizes: "+4 to +6 seconds" for layout 3 is a candidate to ship. Then
  check guardrail metrics (bounce rate, ad revenue) before deciding.

---

### 4.2 Chi-square test of independence: `stats.chi2_contingency`

**Situation.** A telecom company has 12 million customers in a table of **plan type × churned
(yes/no)**. Does churn depend on the plan?

**Why this technique.** Both variables are categorical. The chi-square test checks whether the
row and column variables are independent, using only the count table. That table is tiny even
when the raw data is huge, so it is very cheap.

**How.**

```python
table = pd.crosstab(df["plan"], df["churned"]).to_numpy()  # or GROUP BY in SQL
res = stats.chi2_contingency(table)
res.pvalue, res.expected_freq
```

**Interpreting the result.**

- A small p means churn rates differ between plans. With 12 million rows this is almost
  certain, so look at the **churn rate per plan** and at Cramér's V
  (`scipy.stats.contingency.association(table)`) for strength. V < 0.1 is weak.
- Compare `observed` with `expected_freq` cell by cell to find **which** plans drive the
  dependence.

---

### 4.3 Fisher's exact test: `stats.fisher_exact`

**Situation.** A pharmacovigilance team searches a large adverse-event database. For one rare
side effect, they have 6 cases among 2,000 patients on drug X and 1 case among 3,000 on drug Y.

**Why this technique.** The database is big, but the cell counts for this event are tiny.
Chi-square is unreliable when expected counts are below about 5. Fisher's test computes the
exact probability, so it is correct for any sample size. This is the usual situation in big
data when you drill into rare events or small segments.

**How.**

```python
table = [[6, 1994], [1, 2999]]
stats.fisher_exact(table, alternative="greater")
```

**Interpreting the result.**

- p = 0.02 means: if the drugs had the same risk, a split this lopsided would be rare.
  This is a **signal for review**, not a proof, because teams scan thousands of
  drug-event pairs (rule 5).
- Combine it with the odds ratio (4.4) to see how much higher the risk is.

---

### 4.4 Odds ratio: `scipy.stats.contingency.odds_ratio`

**Situation.** Continuing 4.3: medical reviewers need to know how much higher the risk is on
drug X, with a range of uncertainty.

**Why this technique.** A p-value only says "there is a difference". The odds ratio measures
how big it is, and its confidence interval shows how precisely we know it. It is also the
standard effect size in medicine and in logistic regression, so readers understand it.

**How.**

```python
res = odds_ratio([[6, 1994], [1, 2999]])
res.statistic, res.confidence_interval(0.95)
```

**Interpreting the result.**

- OR = 9 means the odds of the side effect are 9 times higher on drug X.
- A CI of (1.1, 415) is very wide: the risk is higher, but we do not know by how much. Collect
  more data before quoting a number.
- For rare events, odds ratio ≈ risk ratio, so "about 9 times the risk" is a fair summary.

---

## 5. Counts and discrete distributions (module 5)

### 5.1 The Poisson distribution: `stats.poisson`

**Situation.** A payments company receives on average 1,200 transactions per second at peak.
How many servers are needed so that the system is overloaded (above 1,350 tx/s) less than
0.1% of the seconds? And how many fraud alerts per hour are normal?

**Why this technique.** Poisson models counts of independent events in a fixed interval when
you know the average rate. One parameter (λ) gives the full distribution, so you can compute
tail probabilities for capacity and alerting.

**How.**

```python
peak = stats.poisson(1200)
peak.sf(1350)         # probability of more than 1,350 tx in one second
peak.ppf(0.999)       # capacity needed for 99.9% of seconds
stats.poisson(4).sf(11)   # alerts: P(more than 11 in an hour) when 4 is normal
```

**Interpreting the result.**

- `ppf(0.999) = 1,308` means servers sized for 1,308 tx/s cope with 999 seconds in 1,000.
- If you see 12 fraud alerts in an hour when the normal rate is 4, and `sf(11)` is tiny, that
  hour is anomalous. This is a simple, explainable anomaly detector.
- Check the Poisson assumption first (5.2). Real traffic is often **overdispersed** (variance >
  mean, because of bursts). Then Poisson underestimates the peaks, and a negative binomial is
  safer.

---

### 5.2 Chi-square goodness-of-fit: `stats.chisquare`

**Situation.** Before trusting the Poisson model in 5.1, the team checks it against 30 days of
per-second transaction counts.

**Why this technique.** It compares the observed frequency of each count value with the
frequency the model expects. It works for any discrete distribution.

**How.**

```python
observed = np.bincount(counts_binned)
expected = model.pmf(values) * len(counts)   # rescale so the sums match
stats.chisquare(observed, expected, ddof=1)  # ddof=1: we estimated lambda
```

Merge tail bins so that every expected count is at least 5.

**Interpreting the result.**

- A small p means Poisson does not fit. With millions of seconds it will reject even for small
  deviations, so also compare the **variance-to-mean ratio**: close to 1 is Poisson-like, 3 is
  strongly overdispersed.
- Look at which bins have the largest (observed − expected)² / expected. If they are in the
  right tail, you have bursts, and the tail is what matters for capacity.

---

## 6. Repeated measures (module 6)

### Friedman test: `stats.friedmanchisquare`

**Situation.** A recommendation team compares three ranking algorithms. Each of 400 benchmark
query sets is scored by all three algorithms (NDCG@10). Query sets differ a lot in difficulty.

**Why this technique.** The same items (query sets) are measured under every condition, so the
samples are **paired**, not independent. Friedman ranks the algorithms **within** each query
set, which removes the difficulty effect. It is non-parametric, which suits bounded scores
like NDCG.

**How.**

```python
stats.friedmanchisquare(scores_algo_a, scores_algo_b, scores_algo_c)
```

The arrays must be aligned: position i is the same query set in all three.

**Interpreting the result.**

- A small p means at least one algorithm ranks consistently differently from the others.
- To find which, run pairwise Wilcoxon signed-rank tests (`stats.wilcoxon`) with a correction,
  and report the mean rank of each algorithm. This is the standard way to compare ML models
  across many datasets (Demšar, 2006).

---

## 7. Correlation and dependence (module 7)

### 7.1 Pearson, Spearman, Kendall: `pearsonr`, `spearmanr`, `kendalltau`

**Situation.** A mobile-game studio has telemetry for 20 million players. Which behaviours in
the first week relate to spending in the first month: sessions played, friends added, levels
completed?

**Why this technique.** Correlation gives one number per pair of variables, so you can screen
many candidate features quickly.

- **Pearson** measures straight-line relationships, but outliers (the "whales" who spend
  thousands) dominate it.
- **Spearman** uses ranks: it captures any monotonic relationship and resists outliers. It is
  usually the right default for spending data.
- **Kendall** is also rank-based, easier to interpret as a probability, and better when there
  are many ties (for example "friends added", mostly 0, 1 or 2).

**How.**

```python
s = sample.sample(200_000, random_state=0)
stats.spearmanr(s["friends_week1"], s["spend_month1"])
stats.pearsonr(s["sessions_week1"], s["spend_month1"]).confidence_interval(0.95)
```

**Interpreting the result.**

- Read the coefficient, not the p-value: with 200,000 rows, r = 0.02 is "significant" but
  explains 0.04% of the variance (r²).
- If Spearman is much higher than Pearson, the relationship is monotonic but not linear, or
  Pearson is being distorted by outliers.
- Correlation is not causation: players who add friends may spend more because they were
  already more engaged. Use an experiment to test whether *encouraging* friend invites raises
  spending.

---

### 7.2 Multiscale Graph Correlation: `stats.multiscale_graphcorr`

**Situation.** An energy provider has smart-meter data. Engineers suspect that consumption
depends on outdoor temperature, but the shape is a **U** (heating in winter, air conditioning
in summer). Pearson and Spearman both report almost 0.

**Why this technique.** MGC detects **any** kind of dependence (U-shapes, cycles, clusters),
including relationships between multivariate variables. Linear and rank correlations miss
non-monotonic patterns.

**How.**

```python
idx = rng.choice(len(df), 2000, replace=False)   # MGC is O(n²): sample!
res = stats.multiscale_graphcorr(temp[idx], kwh[idx], reps=1000, random_state=0)
res.pvalue, res.stat
```

**Interpreting the result.**

- A small p means temperature and consumption are dependent, even though Pearson says 0.
  Then **plot** the data to see the shape.
- `res.mgc_dict["opt_scale"]` tells you whether the dependence is global (a large scale) or
  local (a small scale, for example only at extreme temperatures).
- MGC says *that* there is dependence, not its form. Model the form next (splines, gradient
  boosting).

---

## 8. Density estimation (module 8)

### 8.1 Univariate KDE: `stats.gaussian_kde`

**Situation.** A bank looks at the times of day of card transactions for one customer segment.
They want a smooth profile to compare with each new transaction: one at 3:40 a.m. in a
segment that never spends at night is suspicious.

**Why this technique.** A histogram depends on where you put the bins. KDE gives a smooth
density with no assumption about its shape, and it shows several peaks (lunch, evening) that
a single fitted distribution would hide.

**How.**

```python
kde = stats.gaussian_kde(hour_of_day_sample)   # sample ~50,000 points
kde(3.67)                                      # density at 3:40 a.m.
```

**Interpreting the result.**

- The peaks show when the segment is active. A low density at the time of a new transaction
  is one input to a fraud score.
- The bandwidth controls smoothness. Too wide hides real peaks, too narrow shows noise. Try a
  few `bw_method` values and compare them.
- KDE is slow to **evaluate** on big data (each evaluation touches every stored point). Fit on
  a sample, or evaluate on a grid once and look values up.

---

### 8.2 Multivariate KDE: `stats.gaussian_kde(points.T)`

**Situation.** A scooter-sharing operator has 50 million trip start points (latitude,
longitude). Where should they put the next 200 charging stations?

**Why this technique.** A 2-D KDE turns scattered points into a smooth **hotspot map** and
shows clusters of any shape without choosing a number of clusters in advance (unlike k-means).

**How.**

```python
pts = trips[["lon", "lat"]].sample(100_000, random_state=0).to_numpy()
kde2 = stats.gaussian_kde(pts.T)               # shape (2, n)
density = kde2(np.vstack([grid_x.ravel(), grid_y.ravel()]))
```

**Interpreting the result.**

- High-density areas are demand hotspots. Rank grid cells by density and place stations at
  the top ones, then check against constraints (permits, existing stations).
- Compare maps from weekdays and weekends, or from mornings and evenings. The differences show
  how demand moves through the city.

---

## 9. Simulation (module 9)

### 9.1 Frozen distributions: shape, loc, scale

**Situation.** A data-platform team models job durations in a nightly ETL pipeline: each job
has a minimum setup time of 5 minutes and then a skewed processing time.

**Why this technique.** `loc` shifts a distribution (the fixed setup time) and `scale`
stretches it (slower hardware). Freezing (`stats.gamma(a, loc, scale)`) fixes the parameters
once, so the same object is reused for probabilities and for sampling, and you cannot pass
inconsistent parameters by mistake.

**How.**

```python
job = stats.gamma(a=4, loc=5, scale=3)
job.mean(), job.sf(30)      # average and P(job > 30 min)
```

**Interpreting the result.** `sf(30) = 0.03` means 3% of jobs exceed 30 minutes. If 200 jobs
run each night, expect about 6 slow jobs per night. That number goes into the alerting design.

---

### 9.2 Broadcasting

**Situation.** The ETL team wants to know how the share of jobs above the SLA changes for 20
hardware options (scales) and 10 SLA thresholds, without writing loops.

**Why this technique.** SciPy distributions broadcast like NumPy: pass arrays of parameters
and get a whole table of results in one vectorised call. This is fast and short.

**How.**

```python
scales = np.linspace(2, 6, 20)[:, None]        # column
thresholds = np.arange(20, 70, 5)[None, :]     # row
table = stats.gamma.sf(thresholds, a=4, loc=5, scale=scales)   # 20 × 10
```

**Interpreting the result.** Each cell is "P(job > threshold) on that hardware". Read along a
row to choose an SLA for given hardware, or down a column to choose hardware for a given SLA.

---

### 9.3 The new random-variable API: `stats.Normal`, `stats.make_distribution`

**Situation.** A team builds a reusable library of distributions for forecasting and
needs clean objects that support arithmetic (for example "total time = prep + delivery").

**Why this technique.** The new API (`stats.Normal(mu=..., sigma=...)`) uses clear parameter
names, supports transformations such as shifting and scaling (`2 * X + 3`), and
`make_distribution` turns existing distributions into this style. It gives cleaner, more
readable modelling code than the older API.

**How.**

```python
prep = stats.Normal(mu=12, sigma=3)
Gamma = stats.make_distribution(stats.gamma)
delivery = Gamma(a=6)
prep.cdf(15), delivery.mean()
```

**Interpreting the result.** The numbers are the same as in the classic API. The benefit is
maintainability: in a large codebase, `Normal(mu=12, sigma=3)` is clearer than
`norm(12, 3)`, and transformations are explicit.

---

### 9.4 Custom samplers: `NumericalInversePolynomial`

**Situation.** A ticketing platform has an **empirical** demand curve for an on-sale event
(a sharp peak at 10:00, a long tail after). They need to generate millions of synthetic
arrival times for load testing.

**Why this technique.** No standard distribution matches that curve. `NumericalInversePolynomial`
builds a fast and accurate sampler from any density function you can write. After setup,
sampling millions of values is almost as fast as `np.random`.

**How.**

```python
class Demand:
    def pdf(self, t):
        return np.exp(-((t - 10) ** 2) / 0.02) + 0.1 * np.exp(-(t - 10))
sampler = NumericalInversePolynomial(Demand(), domain=(9.5, 14), center=10, random_state=rng)
arrivals = sampler.rvs(5_000_000)
```

**Interpreting the result.** Plot a histogram of `arrivals` over your target density. If they
match, the synthetic traffic is realistic, and you can replay it against a staging system to
find the breaking point before the real event.

---

### 9.5 Monte Carlo simulation

**Situation.** A warehouse company wants the probability that it misses next-day delivery for
more than 2% of orders on Black Friday. The outcome depends on random order volume, random
picking times and random truck delays all together, and no formula combines them.

**Why this technique.** Simulate the whole process thousands of times with random inputs from
the fitted distributions, then count how often the bad outcome happens. It handles any
complexity: queues, capacity limits, dependencies.

**How.**

```python
misses = []
for _ in range(10_000):
    n = stats.poisson(180_000).rvs(random_state=rng)
    pick = picking.rvs(n, random_state=rng)
    misses.append(np.mean(pick + truck_delay.rvs(random_state=rng) > cutoff))
np.mean(np.array(misses) > 0.02)
```

**Interpreting the result.**

- The share of simulated days with more than 2% misses is the estimated risk, for example 18%.
- Its precision grows with the number of runs (standard error ≈ √(p(1−p)/runs)). Run enough
  to make the estimate stable before you compare options.
- Re-run with +10% pickers to see how much the risk falls. This turns the simulator into a
  decision tool.

---

### 9.6 Monte Carlo test: `stats.monte_carlo_test`

**Situation.** The ETL team from 9.1 has a model of normal job durations. Last night's 40 jobs
look slow. Is that consistent with the model, or did something change?

**Why this technique.** It tests data against a **fully specified** model by simulating many
datasets from the model and checking where the real statistic falls. It works for any
statistic and any distribution, including ones with no textbook test.

**How.**

```python
res = stats.monte_carlo_test(last_night, job.rvs, np.mean,
                             alternative="greater", n_resamples=9999)
```

**Interpreting the result.**

- p = 0.003 means the model would produce an average this high only 3 times in 1,000 nights.
  Something changed (data volume, a slow node, a bad deployment), so investigate.
- If you run this every night, expect a false alarm about once every 1/alpha nights. Choose
  alpha with that in mind.

---

## 10. Scenario sweeps with Quasi-Monte Carlo (module 10)

### 10.1 `qmc.Sobol` and `qmc.LatinHypercube`

**Situation.** An airline's revenue team has a pricing simulator with 6 inputs (demand level,
competitor price, fuel cost, no-show rate, overbooking limit, discount share). Each run takes
30 seconds, so they can afford about 1,000 runs. Which combinations of inputs lead to losses?

**Why this technique.** Plain random sampling leaves gaps and clusters in a 6-dimensional
space. A full grid is impossible (10 values per input means 1 million runs). Sobol and Latin
hypercube spread points **evenly**, so 1,000 runs cover the space as well as several thousand
random ones.

- **Sobol** gives the most even coverage. Use a power of 2 for n.
- **Latin hypercube** guarantees each input is evenly covered on its own. It is simple and
  works for any n.

**How.**

```python
from scipy.stats import qmc
unit = qmc.Sobol(d=6, seed=0).random(1024)
```

---

### 10.2 `qmc.discrepancy`

**Situation.** Before spending 8 hours of compute, the team wants to check that the chosen
design really covers the space better than random points.

**Why this technique.** Discrepancy is a number that measures how far a point set is from
perfectly uniform. Lower is better. It turns "looks evenly spread" into something you can
compare.

**How.**

```python
qmc.discrepancy(rng.random((1024, 6))), qmc.discrepancy(unit)
```

**Interpreting the result.** If Sobol's discrepancy is much lower than random's, the same
budget gives more reliable conclusions: averages over the scenarios converge faster, and you
are less likely to miss a dangerous region.

---

### 10.3 `qmc.scale`

**Situation.** The design from 10.1 is in [0, 1]⁶. The simulator needs real values, for
example a demand level between 0.6 and 1.4, and fuel between €500 and €1,200 per tonne.

**Why this technique.** `scale` maps the unit cube to the real ranges and keeps the even
coverage. You can also rescale a new Sobol set to a smaller box to **zoom** into an
interesting region.

**How.**

```python
scenarios = qmc.scale(unit, l_bounds=[0.6, 80, 500, 0.02, 1.00, 0.0],
                            u_bounds=[1.4, 200, 1200, 0.12, 1.15, 0.4])
```

**Interpreting the results of the sweep.**

- Run the simulator on every scenario and colour each point by profit or loss. The loss
  region shows which inputs drive the risk, for example "losses happen when demand < 0.8 **and**
  overbooking < 1.05".
- Zoom into the boundary of that region with a second, smaller Sobol design to find the exact
  thresholds for policy (for example the minimum overbooking limit for low-demand seasons).
- To rank which inputs matter most, use these designs with a sensitivity analysis (Sobol
  indices, for example the SALib library).

---

## 11. Quick map: situation → technique

| You want to... | Big-data example | Technique |
|---|---|---|
| Get a first look at a numeric column | Session lengths | `describe` (on a sample) |
| Make a KPI robust to whales | Average order value | `trim_mean`, `winsorize` |
| Check a normality assumption | Stock returns for VaR | `jarque_bera`, `kurtosistest` + plots |
| Model a skewed positive metric | API latency | `gamma.fit` + `kstest` (compare D) |
| Estimate tail risk | Timeouts, p99.9 | `sf`, `ppf` vs empirical quantiles |
| Compare a mean with a target | SLA audit | `ttest_1samp` + CI |
| Compare two means | Revenue per ride | Welch `ttest_ind` + CI |
| Detect distribution drift | ML feature drift | `ks_2samp` (rank by D) |
| CI for any statistic | Difference of medians | `bootstrap` |
| Test with a small, skewed group | Compromised accounts | `permutation_test` |
| Compare consistency | Latency jitter | `levene`, `fligner` |
| Several variants vs one control | Homepage A/B/n | `dunnett` |
| Two categorical variables | Plan × churn | `chi2_contingency` + Cramér's V |
| Rare events, small cells | Adverse drug events | `fisher_exact` + `odds_ratio` |
| Model event counts | Transactions per second | `poisson` (check overdispersion) |
| Check a discrete model | Poisson fit | `chisquare` |
| Same items under several conditions | Ranking algorithms | `friedmanchisquare` |
| Screen relationships | Telemetry vs spending | `spearmanr`, `kendalltau`, `pearsonr` |
| Find non-linear dependence | Temperature vs energy | `multiscale_graphcorr` (sampled) |
| Smooth profile / hotspot map | Fraud timing, trip starts | `gaussian_kde` |
| Simulate a complex process | Black Friday capacity | Frozen distributions + Monte Carlo |
| Generate custom synthetic data | Load testing | `NumericalInversePolynomial` |
| Test data against a simulator | Nightly ETL anomaly | `monte_carlo_test` |
| Explore an expensive simulator | Airline pricing | `qmc.Sobol` + `scale` + `discrepancy` |
