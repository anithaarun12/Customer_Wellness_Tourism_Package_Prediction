# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for train/test split
from sklearn.model_selection import train_test_split
# for HuggingFace upload
from huggingface_hub import HfApi

# Authenticate HF
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define dataset path
DATASET_PATH = "hf://datasets/anithajk/Customer_Wellness_Tourism_Package_Prediction/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop irrelevant columns
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

# Define target
target = "ProdTaken"

# Define features (raw form)
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

# Feature matrix
X = df[numeric_features + categorical_features]

# Target variable
y = df[target]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train/Test files saved.")

# Upload to HuggingFace
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for f in files:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id="anithajk/Customer_Wellness_Tourism_Package_Prediction",
        repo_type="dataset"
    )

print("Files uploaded to HuggingFace dataset hub successfully!")
