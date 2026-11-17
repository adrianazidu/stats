import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import Binarizer, SplineTransformer

# ==============================================================================
# 1. INGESTION & DATA PREPARATION
# ==============================================================================
from get_data import load_data
df = load_data()

# ==============================================================================
# 2. FEATURE ENGINEERING SPECIFIC TO LIGHTGBM
# ==============================================================================
# Binarize physical distance (existence indicator)
binarizer = Binarizer(threshold=0.0)
df["dist1_exists"] = binarizer.fit_transform(df[["dist1"]].fillna(-1.0))

# Periodic spline features on timestamps
splines = SplineTransformer(n_knots=5, degree=3, extrapolation="linear")
spline_features = splines.fit_transform(df[["TransactionDT"]])
for i in range(spline_features.shape[1]):
    df[f"spline_dt_{i}"] = spline_features[:, i]
df.drop(columns=["TransactionDT"], inplace=True)

# KEY LIGHTGBM STEP: Cast categoricals to Pandas 'category' dtype
# LightGBM detects these natively and applies Fisher's exact optimal partition splits
categorical_cols = ["ProductCD", "card4", "DeviceType", "dist1_exists"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Chronological split
X = df.drop(columns=["isFraud"])
y = df["isFraud"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

# ==============================================================================
# 3. LIGHTGBM DATASET API & TRAINING
# ==============================================================================
# LightGBM's Dataset constructs continuous 8-bit histogram bins in memory
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols, free_raw_data=False)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, categorical_feature=categorical_cols, free_raw_data=False)

params = {
    "objective": "binary",
    "metric": "average_precision",     # PR-AUC optimization for extreme imbalance
    "boosting_type": "gbdt",          # Or 'goss' for gradient one-sided sampling
    "is_unbalance": True,             # Automatically recalculates scale_pos_weight
    "learning_rate": 0.05,
    "num_leaves": 31,                 # Crucial: 2^(max_depth) limit to prevent overfitting
    "max_depth": -1,                  # Unconstrained depth, controlled by num_leaves
    "min_child_samples": 20,          # Minimum rows in a leaf
    "subsample": 0.8,                 # Row bagging
    "colsample_bytree": 0.8,          # Column bagging
    "random_state": 42,
    "verbose": -1
}

callbacks = [
    lgb.early_stopping(stopping_rounds=30, verbose=False),
    lgb.log_evaluation(period=50)
]

# Train booster
model = lgb.train(
    params=params,
    train_set=train_data,
    num_boost_round=300,
    valid_sets=[train_data, test_data],
    valid_names=["train", "valid"],
    callbacks=callbacks
)

# ==============================================================================
# 4. INFERENCE & METRIC EVALUATION
# ==============================================================================
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\n" + "=" * 60)
print(f"LightGBM ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"LightGBM PR-AUC : {average_precision_score(y_test, y_pred_proba):.4f}")
print("=" * 60)
print(classification_report(y_test, y_pred, digits=4))