from flaml import AutoML
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from get_data import load_data
df = load_data()

# 1. Prepare X and y (convert string columns to 'category' dtype for native tree support)
y = df["isFraud"].to_numpy()
X = df.drop(columns=["isFraud", "TransactionID"]).copy()

cat_cols = ["ProductCD", "card4", "DeviceType", "DeviceInfo"]
for col in cat_cols:
    X[col] = X[col].astype("category")

# 2. Standard 4-variable split (shuffle=False preserves chronological order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False
)

# 3. Configure and run FLAML
automl = AutoML()
automl.fit(
    X_train=X_train,
    y_train=y_train,
    task="classification",
    metric="ap",              # Average Precision (PR-AUC), ideal for imbalanced fraud data
    time_budget=60,           # Time limit in seconds
    eval_method="holdout",    # Uses a single chronological validation split instead of random K-fold
    split_type="time",        # Prevents future data leakage on time-ordered rows
    estimator_list=["lgbm", "xgboost", "catboost", "rf"],
    seed=42,
)

# 4. Inspect the winning model and evaluate
print(f"Best model: {automl.best_estimator}")
print(f"Best hyperparameters: {automl.best_config}")

probs = automl.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
print(f"Test PR-AUC:  {average_precision_score(y_test, probs):.4f}")

"""
-get all possible packages to test
from flaml.automl.task.factory import task_factory
print(sorted(task_factory("classification").estimators.keys()))
-if estimator_list= "auto" it uses all modles installed
-You can add your own string key by inheriting from flaml.automl.model.SKLearnEstimator (or BaseEstimator) and registering it with automl.add_learner(learner_name="my_model", learner_class=MyCustomEstimator)


AutoML supported packages
1. Gradient Boosted Trees & Tree Ensembles (Top Tabular Performers)

• lgbm: LightGBM (LGBMClassifier / LGBMRegressor) — Microsoft's fast leaf-wise gradient boosted decision trees.
• xgboost: XGBoost (XGBClassifier / XGBRegressor) — Standard depth-wise gradient boosted trees with unrestricted depth tuning.
• xgb_limitdepth: Depth-Constrained XGBoost — XGBoost restricted to shallower trees (max_depth), making it faster and less prone to overfitting on smaller datasets.
• catboost: CatBoost (CatBoostClassifier / CatBoostRegressor) — Yandex's symmetric gradient boosted trees, particularly strong on categorical columns.
• rf: Random Forest (RandomForestClassifier / RandomForestRegressor from scikit-learn) — Bagging ensemble of decision trees.
• extra_tree: Extra Trees / Extremely Randomized Trees (ExtraTreesClassifier / ExtraTreesRegressor from scikit-learn) — Similar to Random Forest, but picks split thresholds randomly rather than searching for the optimal split, making it faster and higher-variance-reducing.
• histgb: Histogram-Based Gradient Boosting (HistGradientBoostingClassifier / HistGradientBoostingRegressor from scikit-learn) — Scikit-learn's native LightGBM-inspired gradient boosting implementation.

2. Linear, Kernel & Distance Models (Fast Baselines)

• lrl1: Logistic Regression with L1 Penalty (Lasso) — Linear classifier that drives weak feature weights to exact zero (built-in feature selection).
• lrl2: Logistic Regression with L2 Penalty (Ridge) — Standard regularized linear classifier.
• enet: Elastic Net (ElasticNet from scikit-learn) — Linear regression combining both L1 and L2 penalties (used for regression tasks).
• lassolars: Lasso Lars (LassoLars from scikit-learn) — L1-regularized linear regression fitted via Least Angle Regression (used for regression tasks).
• sgd: Stochastic Gradient Descent (SGDClassifier / SGDRegressor from scikit-learn) — Fast linear model trained iteratively via gradient descent, useful for very large datasets.
• svc: Support Vector Classifier (SVC from scikit-learn) — Kernel-based classifier effective on small-to-medium datasets with complex decision boundaries, though slow on large datasets.
• kneighbor: K-Nearest Neighbors (KNeighborsClassifier / KNeighborsRegressor from scikit-learn) — Distance-based model that predicts based on the closest training rows.

3. Apache Spark Distributed Estimators (*_spark)

These run across distributed PySpark / SynapseML clusters rather than single-machine pandas/NumPy arrays (only used when you pass a Spark DataFrame to FLAML):
• lgbm_spark: Distributed LightGBM via SynapseML on Spark.
• rf_spark: Spark MLlib distributed Random Forest.
• gbt_spark: Spark MLlib distributed Gradient Boosted Trees.
• lr_spark: Spark MLlib distributed Logistic Regression (or Linear Regression).
• glr_spark: Spark MLlib Generalized Linear Regression (supports Poisson, Gamma, Tweedie distributions, etc.).
• aft_spark: Spark MLlib Accelerated Failure Time model (used for survival / time-to-event analysis).
• nb_spark: Spark MLlib distributed Naive Bayes classifier.
• svc_spark: Spark MLlib distributed Linear Support Vector Classifier.
"""