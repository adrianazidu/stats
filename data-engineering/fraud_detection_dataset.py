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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.impute import MissingIndicator,SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.preprocessing import Binarizer,OneHotEncoder,RobustScaler,SplineTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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

def build_pipeline():

    cont_cols = ["TransactionAmt", "V1"]
    dist_cols = ["dist1"]                     # Continuous with zero-inflation / missingness
    temporal_cols = ["TransactionDT"]         # Non-linear splines
    cat_cols = ["ProductCD", "card4", "DeviceType"]

    #continuous pipeline - impute-robust scale-keep missingness indicator
    cont_pipeline = Pipeline([

         #use it here to male SimpleImputer and Missingindicator work in parallel,
         #otherwise if simpleimputer fills a value with median, the indicator won't know that it was null before
         #missingindicator creates another column that says True/false if value is missing in the applied column
        "union", FeatureUnion([  
            "impute",SimpleImputer(strategy="median"),
            "indicator",MissingIndicator(error_on_new=False)
            ]),
        "scaler",RobustScaler()
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
    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    # ColumnTransformer orchestrator
    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", cont_pipeline, cont_cols),
            ("dist", binarize_pipeline, dist_cols),
            ("spline", spline_pipeline, temporal_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

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
        ("classifier", LogisticRegression(solver="liblinear", random_state=42))
    ])
    
    return full_pipeline


if __name__ == "__main__":
    df = load_data()

    #print(df.columns.tolist())
    print(df.head())

    import sys
    sys.exit()

    y = df["isFraud"].to_numpy()
    X = df.drop(columns=["isFraud","TransactionID","DeviceInfo"])

    print(f"dataset shape {X.shape} mean y {y.mean():.4f}")

    pipeline = build_pipeline()
    pipeline.fit(X,y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:,1]