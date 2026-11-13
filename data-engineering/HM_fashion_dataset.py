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
    else:
        print("H&M files not detected on disk. Using built-in synthetic H&M multi-modal sample.")
        df = generate_mock_hm_data()

    return df

def generate_mock_hm_data():
    """Generates synthetic replica matching H&M's transactions, customers, and articles."""
    np.random.seed(42)
    n = 2500

    customers = [f"cust_{i:04d}" for i in range(200)]
    articles = [f"art_{j:04d}" for j in range(50)]

    descriptions = [
        "Slim fit cotton jeans with regular waist.",
        "Oversized printed hoodie in soft sweat fabric.",
        "Leather shoulder bag with metallic buckle closure.",
        "Ribbed knit sweater with high collar.",
        "Regular fit casual t-shirt with crew neck."
    ]

    df = pd.DataFrame({
        "customer_id": np.random.choice(customers, size=n),
        "article_id": np.random.choice(articles, size=n),
        "price": np.random.exponential(scale=0.03, size=n) + 0.005,

        #generates an array of n random numbers drawn from a normal distribution, while randomly injecting NaN (missing) values roughly 15% of the time
        #realistic missing data
        "age": np.where(np.random.rand(n) > 0.15, np.random.normal(34, 11, size=n), np.nan),
        "club_member_status": np.random.choice(["ACTIVE", "PRE-CREATE", "LEFT CLUB", np.nan], size=n),
        "garment_group_name": np.random.choice(["Jersey Basic", "Knitwear", "Trousers", "Accessories"], size=n),
        "detail_desc": np.random.choice(descriptions, size=n),
        # Target: 1 = purchase in test week, 0 = non-purchase (~5% purchase rate)
        "purchased": np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    })
    return df


def build_hm_pipeline():
    # 1. Unstructured text pipeline: TF-IDF on product description strings
    # converts raw text documents into a numerical matrix of up to 15 important words, while automatically ignoring common English filler words.
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15, stop_words="english"))
    ])

    # 2. Demographic discretization: Quantile-based age partitioning
    age_discretizer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),

        #takes continuous numerical columns and splits them into 4 equal-sized buckets (quartiles)
        #Converts the bin assignments into 4 separate binary (0.0 or 1.0) columns per input feature 
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

    print(df.head())
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