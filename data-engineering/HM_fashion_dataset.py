"""
H&M Personalized Fashion Recommendations Pipeline
Demonstrates:
  - Text feature extraction (TF-IDF on garment descriptions)
  - Age discretization (KBinsDiscretizer)
  - Missing value signaling (MissingIndicator)
  - Price dynamics via spline smoothing (SplineTransformer)
  - Linear/dense feature compression via PCA
  - Extreme imbalance resampling with SMOTE
  - Supervised feature filtering with SelectKBest
"""

import os
import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, RobustScaler, SplineTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



def load_hm_data(articles_p="articles.csv", customers_p="customers.csv", tx_p="transactions_train.csv"):
    """Loads and aggregates multi-table H&M data via DuckDB."""
    con = duckdb.connect()

    if os.path.exists(articles_p) and os.path.exists(customers_p) and os.path.exists(tx_p):
        print("Real H&M dataset files detected. Aggregating via DuckDB...")
        query = f"""
            SELECT 
                t.price,
                c.age,
                COALESCE(c.club_member_status, 'UNKNOWN') AS club_member_status,
                a.garment_group_name,
                COALESCE(a.detail_desc, 'No description') AS detail_desc,
                1 AS purchased
            FROM '{tx_p}' t
            JOIN '{customers_p}' c ON t.customer_id = c.customer_id
            JOIN '{articles_p}' a ON t.article_id = a.article_id
            LIMIT 50000
        """
        df = con.execute(query).df()

    return df


def build_hm_pipeline():
    # 1. Unstructured text pipeline: TF-IDF on product description strings
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15, stop_words="english"))
    ])

    # 2. Demographic discretization: Quantile-based age partitioning
    age_discretizer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("discretize", KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile"))
    ])

    # 3. Price dynamics: Spline basis functions combined with missing indicators
    price_pipeline = Pipeline([
        ("union", FeatureUnion([
            ("impute", SimpleImputer(strategy="median")),
            ("indicator", MissingIndicator(error_on_new=False))
        ])),
        ("scaler", RobustScaler()),
        ("splines", SplineTransformer(n_knots=4, degree=3, extrapolation="linear"))
    ])

    # 4. Categorical pipeline
    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    # Compose column transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, "detail_desc"),
            ("age", age_discretizer, ["age"]),
            ("price", price_pipeline, ["price"]),
            ("cat", cat_pipeline, ["club_member_status", "garment_group_name"])
        ]
    )

    # 5. Pipeline with PCA, SMOTE, and SelectKBest
    full_pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        
        # Dense projection of text + spline dimensions
        ("pca", PCA(n_components=10, random_state=42)),
        
        # Balance rare positive interactions
        ("smote", SMOTE(k_neighbors=4, random_state=42)),
        
        # Supervised feature selection
        ("k_best", SelectKBest(score_func=f_classif, k=6)),
        
        # Estimator
        ("classifier", LogisticRegression(solver="liblinear", random_state=42))
    ])

    return full_pipeline


if __name__ == "__main__":
    df = load_hm_data()
    y = df["purchased"].to_numpy()
    X = df.drop(columns=["purchased", "customer_id", "article_id"], errors="ignore")

    print(f"Data ingested: {len(X)} records | Positive purchase rate: {y.mean():.4f}")
    
    pipeline = build_hm_pipeline()
    pipeline.fit(X, y)
    
    preds = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    print("\n--- H&M Pipeline Execution Complete ---")
    print(f"Model fitted successfully over text, binned age, splines, and categories.")
    print(f"Top 5 purchase probability estimates: {np.round(probabilities[:5], 4)}")