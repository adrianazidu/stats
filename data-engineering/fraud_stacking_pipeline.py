import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, RobustScaler, SplineTransformer, TargetEncoder

# -----------------------------------------------------------------------------
# 1. SYNTHETIC IMBALANCED TABULAR DATA (IEEE-CIS Fraud Archetype)
# -----------------------------------------------------------------------------
from get_data import load_data
df = load_data()
y = df["isFraud"].to_numpy()
X = df.drop(columns=["isFraud","TransactionID"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

# -----------------------------------------------------------------------------
# 2. UNIFIED PREPROCESSOR FOR ALL THREE DIVERSE BASE LEARNERS
# -----------------------------------------------------------------------------
# Continuous: Median imputation + Missingness indicator + Robust scaling
num_cols = ["TransactionAmt", "C1", "V1"]
num_pipe = Pipeline([
    ("union", FeatureUnion([
        ("imputer", SimpleImputer(strategy="median")),
        ("indicator", MissingIndicator(error_on_new=False))
    ])),
    ("scaler", RobustScaler())
])

# Thresholding distance (flag presence/absence)
dist_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ("binarizer", Binarizer(threshold=0.0))
])

# Periodic Splines over timestamps
spline_pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("splines", SplineTransformer(n_knots=5, degree=3, extrapolation="linear"))
])

# Target Encoding (OOF cross-fitted, safe for RF, CatBoost & LightGBM)
cat_cols = ["ProductCD", "card4", "DeviceType"]
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("target_enc", TargetEncoder(cv=5, smooth="auto", random_state=42))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("dist", dist_pipe, ["dist1"]),
        ("spline", spline_pipe, ["TransactionDT"]),
        ("cat", cat_pipe, cat_cols)
    ],
    remainder="drop"
)

# -----------------------------------------------------------------------------
# 3. BASE ESTIMATORS (DIVERSE HYPOTHESIS SPACES)
# -----------------------------------------------------------------------------
# Estimator A: Bagging Ensemble (Parallel deep variance reducers)
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

# Estimator B: Boosting via Symmetric/Oblivious Trees
cb_model = CatBoostClassifier(
    iterations=150,
    depth=5,
    learning_rate=0.05,
    auto_class_weights="Balanced",
    verbose=0,
    random_seed=42
)

# Estimator C: Boosting via Leaf-Wise Asymmetric Trees
lgb_model = lgb.LGBMClassifier(
    n_estimators=150,
    num_leaves=31,
    learning_rate=0.05,
    is_unbalance=True,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

base_estimators = [
    ("rf", rf_model),
    ("catboost", cb_model),
    ("lightgbm", lgb_model)
]

# -----------------------------------------------------------------------------
# 4. META-LEARNER & STACKING CLASSIFIER
"""
In a Stacking Classifier, the models are divided into two distinct tiers:

Level 0 (Base Learners): Random Forest, CatBoost, and LightGBM
         -  look at your raw features to make initial predictions
         -   Multiple (usually 2 to 6+ diverse algorithms)
         - high compelxity

Level 1 (Meta-Learner): Logistic Regression 
        - looks at those predictions to make the final call.
        - Typically 1 single model
        - simle and regularized


Crucial Detail: Out-of-Fold Training (Preventing Data Leakage)

If you train base learners on X_train and immediately use their predictions on X_train to train the meta-learner, the meta-learner will simply trust whichever base model overfit the training data the worst.
To prevent this, stacking uses out-of-fold predictions (handled automatically by scikit-learn's StackingClassifier):
• X_train is split into K folds (e.g., 5 folds).
• Base models train on 4 folds and predict probabilities on the held-out 5th fold.
• Repeating this across all 5 folds gives clean, unseen predictions for every row in X_train, which safely serve as the training inputs for the meta-learner.

"""
# -----------------------------------------------------------------------------
# Meta-learner operates on out-of-fold predicted probabilities
meta_learner = LogisticRegression(
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)

stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method="predict_proba",  # Passes [P(y=0), P(y=1)] per estimator to meta-learner
    n_jobs=-1,
    passthrough=False             # Set to True to concatenate original features with meta-features
)

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("stacking", stacking_clf)
])

# -----------------------------------------------------------------------------
# 5. EXECUTION & BENCHMARKING
# -----------------------------------------------------------------------------
full_pipeline.fit(X_train, y_train)

y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Extract and inspect Meta-Learner coefficients
fitted_stacker = full_pipeline.named_steps["stacking"]
meta_model = fitted_stacker.final_estimator_
meta_weights = meta_model.coef_[0]

print("================================================================")
print("              STACKING ENSEMBLE EVALUATION RESULTS              ")
print("================================================================")
print(f"ROC-AUC Score          : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"PR-AUC (Avg Precision) : {average_precision_score(y_test, y_pred_proba):.4f}")
print("\nMeta-Learner Assigned Weights (Log-Odds Impact):")
feature_names = fitted_stacker.get_feature_names_out()
for name, weight in zip(feature_names, meta_weights):
    print(f"  {name:25s}: {weight:+.4f}")

print("\nClassification Report (0.5 Cutoff):")
print(classification_report(y_test, y_pred, digits=4))