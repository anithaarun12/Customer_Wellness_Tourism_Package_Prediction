import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import load_dataset
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism_project-training-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/Xtest.csv"
ytrain_path = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/ytrain.csv"
ytest_path = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define base XGBoost model
model = xgb.XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False
)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}
# ======================================================
# Hyperparameter Tuning + Experiment Tracking
# ======================================================
with mlflow.start_run(run_name="XGBoost_GridSearch"):

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log all tuned parameters
    results = grid_search.cv_results_

    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_score", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_score", results["std_test_score"][i])

    # Log best params
    mlflow.log_params(grid_search.best_params_)
    
# ======================================================
# Evaluate Best Model
# ======================================================
best_model = grid_search.best_estimator_

y_pred = best_model.predict(Xtest)

accuracy = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred)
recall = recall_score(ytest, y_pred)
f1 = f1_score(ytest, y_pred)

mlflow.log_metrics({
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
})

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# ======================================================
# Save Best Model Locally
# ======================================================
model_path = "best_tourism_model.joblib"
joblib.dump(best_model, model_path)
print("Model saved locally")

 # Log the model artifact
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face
repo_id = "anithajk/tourism_model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("Customer_Wellness_Tourism_Package_Prediction", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="best_tourism_model.joblib",
    path_in_repo="best_tourism_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
