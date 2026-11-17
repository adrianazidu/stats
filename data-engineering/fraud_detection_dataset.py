from dotenv import load_dotenv
load_dotenv()

"""
Kaggle dataset Fraud Detection Processing Pipeline
Demonstrates:
  - Relational querying & aggregation (DuckDB)
  - Missingness signaling (MissingIndicator)
  - Continuous thresholding (Binarizer)
  - Discretization & Splines (KBinsDiscretizer, SplineTransformer)
  - Automated non-linear synthesis (AutoFeat / Interaction crosses)
  - Imbalanced class resampling (SMOTE via imblearn)
  - Dimensionality reduction & filtering (PCA, SelectKBest)
"""
import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.impute import MissingIndicator,SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.preprocessing import Binarizer,OneHotEncoder,RobustScaler,SplineTransformer,TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

CLASSIFIER = os.getenv("CLASSIFIER","xgb")

def load_data(tx_path="train_transaction.csv",id_path="train_identity.csv"):

    con = duckdb.connect()

    tx_path = os.path.join("data",tx_path)
    id_path = os.path.join("data",id_path)
    
  

    if os.path.exists(tx_path) and os.path.exists(id_path):
        print("loadin datasets")

        query = f"""
            SELECT 
                t.TransactionID,
                t.isFraud,
                t.TransactionAmt,
                t.TransactionDT,
                t.ProductCD,
                t.card4,
                t.dist1,
                t.C1,
                t.V1,
                COALESCE(i.DeviceType, 'missing') AS DeviceType,
                COALESCE(i.DeviceInfo, 'unknown') AS DeviceInfo
            FROM '{tx_path}' t
            LEFT JOIN '{id_path}' i ON t.TransactionID = i.TransactionID
            LIMIT 50000
        """

        df = con.execute(query).df()
    else:
        print("Data files not detected on disk. Using built-in synthetic IEEE-CIS sample.")
        tx_df, id_df = generate_mock_ieee_data()
        con.register("tx_table", tx_df)
        con.register("id_table", id_df)
        query = """
            SELECT 
                t.TransactionID,
                t.isFraud,
                t.TransactionAmt,
                t.TransactionDT,
                t.ProductCD,
                t.card4,
                t.dist1,
                t.C1,
                t.V1,
                COALESCE(i.DeviceType, 'missing') AS DeviceType,
                COALESCE(i.DeviceInfo, 'unknown') AS DeviceInfo
            FROM tx_table t
            LEFT JOIN id_table i ON t.TransactionID = i.TransactionID
        """
        df = con.execute(query).df()
    return df

def generate_mock_ieee_data():
    """Generates a small synthetic replica matching the IEEE-CIS schema."""
    np.random.seed(42)
    n = 2000
    
    # 1. Transaction Table (parent)
    tx_ids = np.arange(1000, 1000 + n)
    transaction_df = pd.DataFrame({
        "TransactionID": tx_ids,
        "isFraud": np.random.choice([0, 1], size=n, p=[0.965, 0.035]), # ~3.5% fraud rate
        "TransactionAmt": np.random.exponential(scale=65.0, size=n),
        "TransactionDT": np.sort(np.random.randint(86400, 86400 * 30, size=n)),
        "ProductCD": np.random.choice(["W", "C", "R", "H", "S"], size=n),

        #normal distribution of n numbers ranging from 1000 to 9999
        "card1": np.random.randint(1000, 9999, size=n),
        "card4": np.random.choice(["visa", "mastercard", "discover"], size=n),

        #use exponential distribution replacing 40% of data with nan, simulate missing values
        "dist1": np.where(np.random.rand(n) > 0.4, np.random.exponential(15.0, size=n), np.nan),
        "C1": np.random.poisson(lam=1.5, size=n),
        "V1": np.random.normal(0, 1, size=n)
    })
    
    # 2. Identity Table (relational child - present for ~30% of transactions)
    id_mask = np.random.rand(n) < 0.35
    identity_df = pd.DataFrame({
        "TransactionID": tx_ids[id_mask],
        "DeviceType": np.random.choice(["desktop", "mobile"], size=sum(id_mask)),
        "DeviceInfo": np.random.choice(["Windows", "iOS Device", "MacOS", "SM-G960N"], size=sum(id_mask))
    })
    
    return transaction_df, identity_df

def build_pipeline(scale_pos_weight):

    cont_cols = ["TransactionAmt", "V1"]
    dist_cols = ["dist1"]                     # Continuous with zero-inflation / missingness
    temporal_cols = ["TransactionDT"]         # Non-linear splines
    cat_cols = ["ProductCD", "card4", "DeviceType"]

    #continuous pipeline - impute-robust scale-keep missingness indicator
    cont_pipeline = Pipeline([

         #use it here to male SimpleImputer and Missingindicator work in parallel,
         #otherwise if simpleimputer fills a value with median, the indicator won't know that it was null before
         #missingindicator creates another column that says True/false if value is missing in the applied column
        ("union", FeatureUnion([  
            ("impute",SimpleImputer(strategy="median")),
            ("indicator",MissingIndicator(error_on_new=False))
            ])),
        ("scaler",RobustScaler())
        ])

    #binarize pipeline
    binarize_pipeline = Pipeline([
        ("impute",SimpleImputer(strategy="constant",fill_value=0.99)),
        ("binarizer",Binarizer(threshold=0.0)) 
        #fills values > 0.0 with 1, and values <= 0.0 with 0
    ])

    #spline
    spline_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("splines", SplineTransformer(n_knots=5, degree=3, extrapolation="linear"))
        #takes one column, produces knots + degree -1 = 7 columns
    ])

    #categorical pipeline
    if CLASSIFIER == "LR":
        cat_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("cont", cont_pipeline, cont_cols),
                ("dist", binarize_pipeline, dist_cols),
                ("spline", spline_pipeline, temporal_cols),
                ("cat", cat_pipeline, cat_cols)
            ])

        # 5. Full imblearn Pipeline with SMOTE, PCA, and SelectKBest
        full_pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            
            # Unsupervised projection of collinear features
            ("pca", PCA(n_components=6, random_state=42)),
            
            # Synthetic minority oversampling (active ONLY during fit)
            ("smote", SMOTE(k_neighbors=3, random_state=42)),
            
            # Supervised ANOVA-F hypothesis testing
            ("select_k_best", SelectKBest(score_func=f_classif, k=4)),
            
            # Final Estimator
            ("classifier",  LogisticRegression(solver="liblinear", random_state=42))
        ])

    elif CLASSIFIER == "xgb":

        """
        changes when switching from linear model to ensemble of decision trees
        -Dropping PCA and SelectKBest - unlike linear mdels,, decision trees simply pick the best split among correlated features and ignore the redundant ones,
         Trees perform intrinsic feature selection
        -Replacing OneHotEncoder with TargetEncoder
        -Replacing SMOTE with scale_pos_weight

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("target_enc", TargetEncoder(cv=5, smooth="auto", random_state=42))
            ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("cont", cont_pipeline, cont_cols),
                ("dist", binarize_pipeline, dist_cols),
                ("spline", spline_pipeline, temporal_cols),
                ("cat", cat_pipeline, cat_cols)
            ],
            remainder="drop"
            )

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("xgb", xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="aucpr",  # Area under Precision-Recall curve
                random_state=42,
                tree_method="hist"    # Fast histogram-based tree binning
            ))
            ])
    
    return full_pipeline

if __name__ == "__main__":
    df = load_data()

    #print(df.columns.tolist())
    print(df.head())

    import sys
    #sys.exit()

    y = df["isFraud"].to_numpy()
    X = df.drop(columns=["isFraud","TransactionID"])

    #shuffle prevents data leakage from future when using dates
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    print(f"dataset shape {X.shape} mean y {y.mean():.4f}")

    if CLASSIFIER == "LR":
        scale_pos_weight=0
    elif CLASSIFIER == "xgb":
        # Dynamic calculation of scale_pos_weight to counteract 96.5:3.5 class imbalance
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = float(n_neg / n_pos)
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(X_train,y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:,1] #select all rows and only the second column


    #sort makes asc from low to high, ::-1 inverts from high to low, :5 gets first 5 rows
    top_5_highest = np.sort(probs)[::-1][:5]
    print(f"Top 5 highest predicted probabilities: {np.round(top_5_highest, 4)}")

    #Instead of returning hard 0 or 1 class predictions, it returns a 2D NumPy array with two columns—one for each class—where each row adds up to 1.0:
    print(pipeline.predict_proba(X_test))

    if CLASSIFIER == "xgb":
        # In fraud detection with skewed weights, standard 0.5 threshold can be adjusted
        y_pred = (probs >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        print("\n" + "=" * 60)
        print("                EVALUATION METRICS (TEST SET)              ")
        print("=" * 60)
        print(f"ROC-AUC Score          : {roc_auc:.4f}")
        print(f"PR-AUC (Avg Precision) : {pr_auc:.4f}")
        print(f"Brier Score Loss       : {brier:.4f} (lower is better)")
        print("\nClassification Report (Default 0.5 threshold):")
        print(classification_report(y_test, y_pred, digits=4))