"""
One pipeline, four tools:
  1. CalibratedClassifierCV -> FIX:      make the probabilities honest
  2. CalibrationDisplay     -> VERIFY:   check the fix worked (reliability diagram)
  3. MAPIE                  -> Uncertainty: does the model know when it doesn’t know?
  4. Fairlearn              -> AUDIT:    check that steps 1-3 hold for every slice, not just on average
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, recall_score, brier_score_loss

from mapie.classification import SplitConformalClassifier
from fairlearn.metrics import MetricFrame, count

# ---------------------------------------------------------------- data
# Four disjoint splits. Calibration and conformal prediction each need data
# the previous step never saw; reusing it breaks MAPIE's coverage guarantee.

X, y = make_classification(n_samples=8000, n_features=20, n_informative=8,
                           flip_y=0.05, random_state=0)
rng = np.random.default_rng(0)
region = rng.choice(["EU", "US", "APAC"], size=len(y), p=[0.5, 0.4, 0.1])
X[region == "APAC"] += rng.normal(0, 1.0, X[region == "APAC"].shape)  # make one slice harder

idx = np.arange(len(y))
#just lists of numbers
i_train, i_rest = train_test_split(idx, test_size=0.5, random_state=0) # 50% train
i_cal, i_rest   = train_test_split(i_rest, test_size=0.66, random_state=0) #the rest 50% split into rest (0.66%) and calibration (0.34%)
i_conf, i_test  = train_test_split(i_rest, test_size=0.5, random_state=0)  #the rest split into conformal and test
Xtr, ytr = X[i_train], y[i_train]
Xca, yca = X[i_cal],   y[i_cal]
Xcf, ycf = X[i_conf],  y[i_conf]
Xte, yte = X[i_test],  y[i_test]
region_te = region[i_test]

"""
8,000 cases
│
├─ 50% ──────────────► TRAIN        4,000   teach the model
└─ 50% (4,000)
     ├─ 34% ─────────► CALIBRATION  1,360   learn the "honesty correction"
     └─ 66% (2,640)
          ├─ 50% ────► CONFORMAL    1,320   let MAPIE set its bar
          └─ 50% ────► TEST         1,320   final exam: all the numbers you saw
"""

# ---------------------------------------------------------------- 1. FIX: calibrate

def calibrate():
    #takes the trained random forest and fixes its probabilities so they match how often it’s actually right
    #it doesnt change the model, just adds a layer on top
    calibrated = CalibratedClassifierCV(FrozenEstimator(rf), method="isotonic").fit(Xca, yca)

    p_raw = rf.predict_proba(Xte)[:, 1]         #original predictions
    p_cal = calibrated.predict_proba(Xte)[:, 1] #calibrated predictions
    print(f"Brier  raw={brier_score_loss(yte, p_raw):.4f}  calibrated={brier_score_loss(yte, p_cal):.4f}")

    #The Brier score measures how close the model’s probabilities are to what actually happened. Lower is better.
    #raw = 0.0589: the original random forest. Already good, much closer to 0 than to 0.25.
    #calibrated = 0.0481: after the calibration step. Lower, so better.

    return (calibrated,p_raw,p_cal)

# ---------------------------------------------------------------- 2. VERIFY: reliability diagram
def check_after_calibrate():
    fig, ax = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_predictions(yte, p_raw, n_bins=10, strategy="quantile", name="raw RF", ax=ax)
    CalibrationDisplay.from_predictions(yte, p_cal, n_bins=10, strategy="quantile", name="calibrated", ax=ax)
    ax.set_title("Reliability diagram (test set)")
    fig.savefig("reliability.png", dpi=120, bbox_inches="tight")

# ---------------------------------------------------------------- 3. coverage and autodecided

def confidence():
    #confidence_level decides how high the bar is for an answer to get onto the list
    mapie = SplitConformalClassifier(estimator=calibrated, confidence_level=0.90,
                                    conformity_score="lac", prefit=True)
    mapie.conformalize(Xcf, ycf)
    y_pred, y_sets = mapie.predict_set(Xte)
    sets = y_sets[:, :, 0]                              # (n_test, n_classes)

    covered   = sets[np.arange(len(yte)), yte]          # true label inside the set?
    set_size  = sets.sum(axis=1)
    decision  = np.where(set_size == 1, "auto", "human_review")   # a 2-label (or empty) set means abstain
    print(f"Coverage={covered.mean():.3f} (target 0.90)  auto-decided={np.mean(decision == 'auto'):.1%}")

    #Coverage = 0.898 means the right answer was on that list about 90% of the time,we  asked for 90% confidence, so it did what you asked.
    #Auto-decided = 93.6% means that for about 94 out of every 100 cases,the model was sure which option (0,1) is valid
    return y_pred,covered,set_size
# ---------------------------------------------------------------- 4. fairlearn

def measure_gap():
    #ECE: the average gap between confidence and accuracy
    def ece(y_true, p, n_bins=10):
        """Binary ECE on P(class=1)."""
        bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        ids = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
        return sum((ids == b).mean() * abs(p[ids == b].mean() - y_true[ids == b].mean())
                for b in range(n_bins) if (ids == b).any())

    # MetricFrame passes (y_true, y_pred) to every metric, so we feed one frame per kind of "prediction".
    hard = MetricFrame(metrics={"n": count, "accuracy": accuracy_score, "recall": recall_score},
                    y_true=yte, y_pred=y_pred, sensitive_features=region_te)
    prob = MetricFrame(metrics={"brier": brier_score_loss, "ece": ece},
                    y_true=yte, y_pred=p_cal, sensitive_features=region_te)
    conf = MetricFrame(metrics={"coverage": lambda yt, c: c.mean()},
                    y_true=yte, y_pred=covered, sensitive_features=region_te)
    auto = MetricFrame(metrics={"auto_rate": lambda yt, s: (s == 1).mean()},
                    y_true=yte, y_pred=set_size, sensitive_features=region_te)

    report = pd.concat([hard.by_group, prob.by_group, conf.by_group, auto.by_group], axis=1)
    overall = pd.concat([hard.overall, prob.overall, conf.overall, auto.overall]).rename("ALL")
    report = pd.concat([report, overall.to_frame().T]).round(3)
    print("\nPer-slice report:\n", report)

    """
    Column	    Meaning	                                                            Better when
    n	        How many test cases came from that region           	                –
    accuracy	How often the model’s answer was right	                             Higher
    recall	    Of the real fraud cases, how many the model caught	                Higher
    brier	    How honest the probabilities are 	                                Lower
    ece	        the average gap, also measures honesty	                            Lower
    coverage	How often the right answer was on MAPIE’s list (target 90%)	≥ 0.90
    auto_rate	Share of cases the model decided alone, with no human needed	     Higher
    """

# ---------------------------------------------------------------- 0. train
if __name__ == '__main__':
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    calibrated,p_raw,p_cal = calibrate()
    check_after_calibrate()
    y_pred,covered,set_size = confidence()
    measure_gap()
