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
