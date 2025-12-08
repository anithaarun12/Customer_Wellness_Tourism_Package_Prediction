# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# for model serialization
import joblib
# for creating a folder
import os
# Hugging Face authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Authenticate HF
api = HfApi(token=os.getenv("HF_TOKEN"))

# Paths to processed files
Xtrain_path = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/Xtrain.csv"
Xtest_path  = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/Xtest.csv"
ytrain_path = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/ytrain.csv"
ytest_path  = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/ytest.csv"

# Load processed files
Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest  = pd.read_csv(ytest_path)

# Feature definitions (same as prep.py)
numeric_features = [
    "Age",
    "DurationOfPitch",
    "NumberOfFollowups",
    "MonthlyIncome",
    "NumberOfChildrenVisiting",
    "Passport",
    "OwnCar"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "Designation"
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

# Build pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.7],
    "xgbclassifier__reg_lambda": [0.4, 0.6],
}

# Train model
print("Running GridSearchCV…")
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring="f1")
grid_search.fit(Xtrain, ytrain)

# Best model
best_pipeline = grid_search.best_estimator_

print("Best Params:", grid_search.best_params_)

# Predictions
y_pred = best_pipeline.predict(Xtest)

# Performance
print("\n=== Model Performance ===")
print(classification_report(ytest, y_pred))

# Save final model
os.makedirs("models", exist_ok=True)
model_path = "models/best_tourism_pipeline.joblib"
joblib.dump(best_pipeline, model_path)

print(f"Model saved at: {model_path}")

# Upload model to HF Hub
repo_id = "anithajk/tourism-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print("Existing model repo found.")
except RepositoryNotFoundError:
    print("Model repo not found. Creating...")
    create_repo(repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_tourism_pipeline.joblib",
    repo_id=repo_id,
    repo_type=repo_type
)

print("Model uploaded to Hugging Face successfully!")
