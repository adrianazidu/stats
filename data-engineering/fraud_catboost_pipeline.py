import os
import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, SplineTransformer


# ==============================================================================
# 1. INGESTION & DATA PREPARATION
# ==============================================================================
def load_data():
    """Generates or loads relational IEEE-CIS sample."""
    np.random.seed(42)
    n = 6000
    y = np.random.choice([0, 1], size=n, p=[0.965, 0.035])

    df = pd.DataFrame({
        "TransactionAmt": np.where(y == 1, np.random.exponential(180, size=n), np.random.exponential(60, size=n)),
        "TransactionDT": np.sort(np.random.randint(86400, 86400 * 30, size=n)),
        "ProductCD": np.random.choice(["W", "C", "R", "H", "S"], size=n),
        "card4": np.random.choice(["visa", "mastercard", "discover"], size=n),
        "dist1": np.where(np.random.rand(n) > 0.45, np.random.exponential(25.0, size=n), np.nan),
        "C1": np.where(y == 1, np.random.poisson(4.0, size=n), np.random.poisson(1.1, size=n)),
        "V1": np.random.normal(0, 1, size=n),
        "DeviceType": np.random.choice(["desktop", "mobile", "missing"], size=n),
        "isFraud": y
    })
    return df


df = load_data()

# ==============================================================================
# 2. MINIMAL FEATURE ENGINEERING FOR CATBOOST
# ==============================================================================
# Binarize distance (existence flag)
binarizer = Binarizer(threshold=0.0)
df["dist1_is_present"] = binarizer.fit_transform(df[["dist1"]].fillna(-1.0))

# Periodic spline features on timestamp (replaces raw timestamp)
splines = SplineTransformer(n_knots=5, degree=3, extrapolation="linear")
spline_features = splines.fit_transform(df[["TransactionDT"]])
for i in range(spline_features.shape[1]):
    df[f"spline_dt_{i}"] = spline_features[:, i]

df.drop(columns=["TransactionDT"], inplace=True)

# Declare Categorical Features - Keep them strictly as string or category types!
categorical_cols = ["ProductCD", "card4", "DeviceType", "dist1_is_present"]
for col in categorical_cols:
    df[col] = df[col].astype(str)

# Train/Test Chronological Split
X = df.drop(columns=["isFraud"])
y = df["isFraud"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

# ==============================================================================
# 3. CATBOOST POOL & TRAINING
# ==============================================================================
# Pool objects optimize memory layout and track categorical indices
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=categorical_cols
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=categorical_cols
)

model = CatBoostClassifier(
    iterations=250,
    learning_rate=0.06,
    depth=6,
    loss_function="Logloss",
    eval_metric="PRAUC",               # Directly optimize Precision-Recall AUC
    auto_class_weights="Balanced",      # Equivalent to scale_pos_weight
    nan_mode="Min",                     # Native missing value routing
    random_seed=42,
    verbose=50
)

# Fit model with built-in early stopping
model.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=30,
    use_best_model=True
)

# ==============================================================================
# 4. PREDICTIONS & EVALUATION
# ==============================================================================
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\n" + "=" * 60)
print(f"CatBoost ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"CatBoost PR-AUC : {average_precision_score(y_test, y_pred_proba):.4f}")
print("=" * 60)
print(classification_report(y_test, y_pred, digits=4))