Code for "Statistics in Real Life with scipy.stats"

Requirements:  pip install "scipy>=1.15" numpy matplotlib
(the charts use the Inter font if installed; otherwise matplotlib falls back to its default font)

Run any story from inside this folder, for example:
    python story1_delivery_promise.py
Each script prints the numbers quoted in the guide and saves its charts in ../figs/.

foundations.py              Part A figures and small examples
story1_delivery_promise.py  describe, trimming, winsorizing, normality tests, fitting, kstest, sf/ppf, ttest_1samp
story2_ab_test.py           chi2_contingency, Welch t-test, bootstrap, permutation_test, dunnett, fisher_exact, odds_ratio, false_discovery_control
story3_factory.py           ttest_1samp, bartlett, levene, fligner, ttest_rel, wilcoxon, poisson, chisquare, chi2_contingency
story4_ml_monitoring.py     ks_2samp, poisson alarms, monte_carlo_test, friedmanchisquare, wilcoxon
story5_customers.py         pearsonr, spearmanr, kendalltau, multiscale_graphcorr, gaussian_kde (1-D and 2-D)
story6_black_friday.py      frozen distributions, broadcasting, Normal / make_distribution, NumericalInversePolynomial, Monte Carlo, qmc
style.py                    shared chart style
