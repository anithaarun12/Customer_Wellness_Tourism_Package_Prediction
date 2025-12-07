import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="anithajk/tourism_model", filename="best_tourism_model.joblib")

# Load the model
model = joblib.load(model_path)
# Streamlit UI for Tourism Package Purchase Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("Fill the details below to predict whether the customer will purchase the package.")

# Collect user input
# -------------------------------------------------------
# USER INPUT SECTION
# -------------------------------------------------------

Age = st.number_input(
    "Age (Customer's age in years)",
    min_value=18, max_value=80, value=30
)

DurationOfPitch = st.number_input(
    "Duration Of Pitch (minutes)",
    min_value=0, max_value=120, value=15
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0, max_value=10, value=2
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000, max_value=300000, value=50000
)

NumberOfChildrenVisiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0, max_value=5, value=1
)

Passport = st.selectbox("Passport (Does customer have a passport?)", ["Yes", "No"])
OwnCar = st.selectbox("Own Car (Does customer own a car?)", ["Yes", "No"])

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Student", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)


# Convert categorical inputs to match model training
# -------------------------------------------------------
# 2. CREATE INPUT DATAFRAME
# -------------------------------------------------------
input_data = pd.DataFrame({
    "Age": [Age],
    "DurationOfPitch": [DurationOfPitch],
    "NumberOfFollowups": [NumberOfFollowups],
    "MonthlyIncome": [MonthlyIncome],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "Passport": [1 if Passport == "Yes" else 0],
    "OwnCar": [1 if OwnCar == "Yes" else 0],
    "TypeofContact": [TypeofContact],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "Designation": [Designation]
})

st.write("### Input Summary")
st.dataframe(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("The customer is **LIKELY** to purchase the tourism package!")
    else:
        st.warning("The customer is **NOT likely** to purchase the package.")
