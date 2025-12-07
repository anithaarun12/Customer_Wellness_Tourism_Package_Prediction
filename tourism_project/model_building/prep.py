# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop Irrelevant Columns
# ===============================
tourism_dataset.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

# ===============================
# Separate Features & Target
# ===============================
X = tourism_dataset.drop("ProdTaken", axis=1)
y = tourism_dataset["ProdTaken"]

# ===============================
# Identify Column Types
# ===============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("Numeric Features:", numeric_features.tolist())
print("Categorical Features:", categorical_features.tolist())


# ======================================================
# Create Preprocessing Pipelines
# ======================================================

# Numerical Pipeline
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical Pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ======================================================
# Combine Preprocessing Steps
# ======================================================
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ======================================================
# Apply Preprocessing
# ======================================================
X_processed = preprocessor.fit_transform(X)

# ======================================================
# Convert Processed Data to DataFrame
# ======================================================
num_features = numeric_features.tolist()

cat_features = preprocessor.named_transformers_["cat"] \
    .named_steps["encoder"] \
    .get_feature_names_out(categorical_features) \
    .tolist()

feature_names = num_features + cat_features

X_processed = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
    columns=feature_names
)

# ======================================================
# Train-Test Split
# ======================================================
# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_processed, y,     # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)
# ======================================================
# Save Processed Train & Test Files
# ======================================================
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train & test CSV files saved")

# ======================================================
# Upload Files to Hugging Face Dataset Hub
# ======================================================
api = HfApi()

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="anithajk/Customer_Wellness_Tourism_Package_Prediction",
        repo_type="dataset",
    )



print("Files uploaded to Hugging Face successfully")
