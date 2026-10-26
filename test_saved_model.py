
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("models/churn_model.pkl")

print("Saved model loaded successfully!")

# Create sample customer data
sample = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
}])

# Encode the sample using the training feature columns
sample_encoded = pd.get_dummies(
    sample,
    drop_first=True,
    dtype=int
)

# Match the columns used during training
sample_encoded = sample_encoded.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

prediction = model.predict(sample_encoded)[0]

print("Prediction:", "Churn" if prediction == 1 else "No Churn")
