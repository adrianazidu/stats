import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, TargetEncoder

# -----------------------------------------------------------------------------
# 1. ENSURE CHRONOLOGICAL DATA ORDERING
# -----------------------------------------------------------------------------
from get_data import load_data
df = load_data()

# Sort by timestamp to ensure TimeSeriesSplit cuts across the true temporal axis
df = df.sort_values(by="TransactionDT").reset_index(drop=True)

X = df.drop(columns=["isFraud", "TransactionID"])
y = df["isFraud"].to_numpy()

# Reserve the final 20% chronologically as the unpolluted out-of-time test set
split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# -----------------------------------------------------------------------------
# 2. TEMPORAL CROSS-VALIDATION SPLITTER
# -----------------------------------------------------------------------------
# n_splits=5 creates 5 expanding temporal evaluation windows
#replace StratifiedKFold with TimeSeriesSplit in StackingClassifier to prevent lookahead bias in chronological transaction logs
# max_train_size can optionally convert expanding windows to fixed rolling windows
time_cv = TimeSeriesSplit(
    n_splits=5,
    max_train_size=None,  # Set to an integer to use rolling instead of expanding window
    gap=0                 # Set to >0 if there is latency/delay in fraud chargeback labels
)

# -----------------------------------------------------------------------------
# 3. BASE MODELS & META-LEARNER
# -----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", RobustScaler())]), ["TransactionAmt", "C1", "V1"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("target_enc", TargetEncoder(cv=time_cv, smooth="auto"))]), ["ProductCD", "card4", "DeviceType"])
    ]
)

base_estimators = [
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
    ("catboost", CatBoostClassifier(iterations=120, depth=5, auto_class_weights="Balanced", verbose=0, random_seed=42)),
    ("lightgbm", lgb.LGBMClassifier(n_estimators=120, num_leaves=31, is_unbalance=True, verbose=-1, random_state=42))
]

meta_learner = LogisticRegression(
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)

# -----------------------------------------------------------------------------
# 4. STACKING CLASSIFIER WITH TEMPORAL CV
# -----------------------------------------------------------------------------
stacking_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("stacking", StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=time_cv,                   # <-- Injects TimeSeriesSplit into the stacking engine
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False
    ))
])

stacking_pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 5. EVALUATION ON OUT-OF-TIME (FUTURE) DATA
# -----------------------------------------------------------------------------
y_pred_proba = stacking_pipeline.predict_proba(X_test)[:, 1]

print(f"Out-of-Time Test ROC-AUC : {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Out-of-Time Test PR-AUC  : {average_precision_score(y_test, y_pred_proba):.4f}")