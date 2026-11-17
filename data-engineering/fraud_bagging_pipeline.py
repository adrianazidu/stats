import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, OrdinalEncoder, SplineTransformer
from imblearn.ensemble import BalancedRandomForestClassifier


# ==============================================================================
# 1. INGESTION & DATA GENERATION
# ==============================================================================
from get_data import load_data
df = load_data()

# ==============================================================================
# 2. FEATURE ENGINEERING SPECIFIC TO BAGGING
# ==============================================================================
# 1. Continuous pipeline: Impute NaNs + preserve missingness indicator
num_cols = ["TransactionAmt", "C1", "V1"]
num_pipeline = Pipeline([
    ("union", FeatureUnion([
        ("imputer", SimpleImputer(strategy="median")),
        ("indicator", MissingIndicator(error_on_new=False))
    ]))
])

# 2. Binarize distance flag
dist_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ("binarizer", Binarizer(threshold=0.0))
])

# 3. Spline basis expansion for time
spline_pipeline = Pipeline([
    ("splines", SplineTransformer(n_knots=5, degree=3, extrapolation="linear"))
])

# 4. Categoricals: Impute missing tokens and convert to Ordinal integers for trees
cat_cols = ["ProductCD", "card4", "DeviceType"]
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("dist", dist_pipeline, ["dist1"]),
        ("spline", spline_pipeline, ["TransactionDT"]),
        ("cat", cat_pipeline, cat_cols)
    ],
    remainder="drop"
)

# Chronological split
X = df.drop(columns=["isFraud"])
y = df["isFraud"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

# ==============================================================================
# 3. MODEL 1: STANDARD RANDOM FOREST WITH BALANCED SUBSAMPLING
# ==============================================================================
rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",  # Calculates weights per bootstrap draw
        n_jobs=-1,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)
y_rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

# ==============================================================================
# 4. MODEL 2: BALANCED RANDOM FOREST (IN-BAG UNDERSAMPLING)
# ==============================================================================
# Pre-transform matrices for BalancedRandomForest
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

brf = BalancedRandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    sampling_strategy="all",      # Balances all bootstrap samples 50:50
    replacement=True,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

brf.fit(X_train_trans, y_train)
y_brf_proba = brf.predict_proba(X_test_trans)[:, 1]

# ==============================================================================
# 5. COMPARISON & EVALUATION
# ==============================================================================
print("=" * 65)
print("                    BAGGING ENSEMBLE EVALUATION                 ")
print("=" * 65)
print(f"Random Forest (Balanced Subsample) ROC-AUC : {roc_auc_score(y_test, y_rf_proba):.4f}")
print(f"Random Forest (Balanced Subsample) PR-AUC  : {average_precision_score(y_test, y_rf_proba):.4f}")
print("-" * 65)
print(f"Balanced Random Forest (In-Bag Undersample) ROC-AUC: {roc_auc_score(y_test, y_brf_proba):.4f}")
print(f"Balanced Random Forest (In-Bag Undersample) PR-AUC : {average_precision_score(y_test, y_brf_proba):.4f}")
print("=" * 65)
print("\nBalanced Random Forest Classification Report:")
print(classification_report(y_test, (y_brf_proba >= 0.5).astype(int), digits=4))