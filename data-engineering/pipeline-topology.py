"""
To understand how Pipeline, ColumnTransformer, and FeatureUnion combine, it helps to understand their structural topology:

-Pipeline (Sequential - Series): Chains operations step-by-step. Step $N$ receives the exact output of Step $N-1$.  
-ColumnTransformer (Parallel - Feature Routing): Splits input data by columns. Each branch receives distinct column subsets; outputs are horizontally concatenated.
-FeatureUnion (Parallel - Representation Branching): Applies multiple transformers in parallel to the same incoming data, then horizontally concatenates their output matrices side-by-side.   

Since each of these implements the scikit-learn estimator interface (fit, transform, and optionally predict), 
any of them can be nested inside any other.

Because both ColumnTransformer and FeatureUnion implement .fit() and .transform(), Scikit-Learn treats them as interchangeable LEGO bricks:

Passing a ColumnTransformer into Pipeline works.(example 1)
Passing a FeatureUnion into Pipeline works.(example 3)
Passing a Pipeline inside another Pipeline works.

The Only Practical Constraint: Data Compatibility
While Python allows either one in the main Pipeline, your input data dictates which one actually works without crashing:
Is your raw input a mixed DataFrame (strings, categories, numbers)?
 ├── YES ──> You MUST start with ColumnTransformer.
 │           (If you pass numbers/categories/raw strings (that have not been TfidfVectorizeed)
              directly into a top-level FeatureUnion,
 │            numeric branches like PCA or Scaler will crash immediately.)
 │
 └── NO (All columns are already clean numbers or all text)
         ──> You can pass FeatureUnion directly into the main Pipeline.

Use FeatureUnion at the top level ONLY when your entire table is homogeneous (e.g., every column is a clean number, or the entire input is a single text column).
Use ColumnTransformer at the top level whenever your columns have different types (numbers + categories + text), so it can act as a traffic cop and keep "Rome" away from the math scalers, and keep 19.99 away from the text encoders.

how to make clean numbers
- cast & stripped $45.20 becomes 45.20
- imputed - NaN is replaced
- bounded - infinite is clipped or imputed
- encoded - categrorical ids (90210) converted with ineHotEncoder into true mathematical quantities

how to clean text
- numbers stored as strings - strip formatting and cast to nunmbers
- messy categories - targetEncoder/onehotencoder
- natural text - Unicoded.,strip punctuation, pass to TfidfVectorizer
"""

#topology 1--------------------------------------------------------------------------------
# Main Pipeline -> Column Transformer -> steps as Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

#apply to dedicated columns
#why do we have transformers key here but not in example 2? In Python, func(x=10) and func(10), and transformers is the forst argument
#but often it is not specified (like in example 2) if it's the only argument
#here we also gave remainder drop after, so we specify it
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, ["age", "fare"]),
        ("cat", cat_pipeline, ["embarked", "sex"])
    ],
    remainder="drop"
)

# Canonical outer pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(random_state=42))
])

#topology 2--------------------------------------------------------------------------------
## Main Pipeline -> Column Transformer -> step as featureUnion

from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Multi-perspective text featurizer: word and sub-word n-grams simultaneously
text_union = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2))),
    ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)))
])

preprocessor = ColumnTransformer([
    ("title_features", text_union, "product_title"), #use string for TfidfVectorizer that needs 1D Series
    ("price_scaled", RobustScaler(), ["price"])      #2D matrix for scalers/imputers/encoders
])

full_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

#topology 3--------------------------------------------------------------------------------
#main pipeline - feature union with 2 pipelines

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

dense_branch = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
    ("pca", PCA(n_components=10, random_state=42))
])

rbf_branch = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", RobustScaler()),
    ("rbf", RBFSampler(n_components=50, random_state=42))
])

#applied to all columns
feature_extractor = FeatureUnion([
    ("linear_dense", dense_branch),
    ("kernel_approx", rbf_branch)
])

full_pipeline = Pipeline([
    ("features", feature_extractor),
    ("selection", SelectKBest(score_func=f_classif, k=20)),
    ("classifier", LogisticRegression(solver="liblinear"))
])


#topology 4 --------------------------------------------------------------------
#feature union only to one specific column -> nested into columntransformer as one of its branches

# Both branches receive ONLY ["dist1"] and execute in parallel, outputting 2 columns:
# [imputed_dist1, was_dist1_missing_flag]
dist_union = FeatureUnion([
    ("imputed_val", SimpleImputer(strategy="median")),
    ("missing_flag", MissingIndicator(error_on_new=False))
])

amt_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("dist_branch", dist_union, ["dist1"]),         # FeatureUnion runs ONLY on dist1
        ("amt_branch", amt_pipeline, ["TransactionAmt"]),
        ("cat_branch", cat_pipeline, ["ProductCD"])
    ],
    remainder="drop"
)

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])