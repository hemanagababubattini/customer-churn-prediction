
import gradio as gr
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/churn_model.pkl")


def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    internet_service,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges
):
    # Create customer data
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": "No phone service" if phone_service == "No" else "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No internet service" if internet_service == "No" else "No",
        "OnlineBackup": "No internet service" if internet_service == "No" else "No",
        "DeviceProtection": "No internet service" if internet_service == "No" else "No",
        "TechSupport": "No internet service" if internet_service == "No" else "No",
        "StreamingTV": "No internet service" if internet_service == "No" else "No",
        "StreamingMovies": "No internet service" if internet_service == "No" else "No",
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }])

    # Encode and align with model's training columns
    customer_encoded = pd.get_dummies(
        customer, drop_first=True, dtype=int
    )
    customer_encoded = customer_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(customer_encoded)[0]
    probabilities = model.predict_proba(customer_encoded)[0]

    result = "Likely to Churn" if prediction == 1 else "Likely to Stay"
    confidence = float(probabilities[prediction])

    return {
        result: confidence
    }


demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Female", "Male"], label="Gender"),
        gr.Radio([0, 1], label="Senior Citizen (0 = No, 1 = Yes)", value=0),
        gr.Radio(["Yes", "No"], label="Partner", value="Yes"),
        gr.Radio(["Yes", "No"], label="Dependents", value="No"),
        gr.Slider(0, 72, value=12, step=1, label="Tenure (months)"),
        gr.Radio(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contract"
        ),
        gr.Radio(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            label="Payment Method"
        ),
        gr.Number(value=70.0, label="Monthly Charges"),
        gr.Number(value=840.0, label="Total Charges")
    ],
    outputs=gr.Label(label="Churn Prediction"),
    title="Customer Churn Prediction",
    description="Enter customer details to estimate churn risk."
)

if __name__ == "__main__":
    demo.launch()
