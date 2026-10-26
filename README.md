# Customer Churn Prediction

## 1. Project Overview
This project predicts whether a telecommunications customer is likely to leave the service (churn) using customer account and service information.

## 2. Business Problem
Customer churn can affect business revenue. Identifying customers who may leave can help a company plan customer-retention strategies.

## 3. Dataset
- Dataset: IBM Telco Customer Churn
- Number of records: 7,043
- Target column: `Churn`
- Target values: `Yes` and `No`

## 4. Project Workflow
1. Load the dataset using Pandas.
2. Inspect data types, missing values, and duplicates.
3. Convert `TotalCharges` to numeric and handle missing values.
4. Perform exploratory data analysis (EDA).
5. Encode categorical features using Pandas `get_dummies()`.
6. Split the dataset into 80% training data and 20% testing data.
7. Train Logistic Regression and Random Forest classifiers.
8. Evaluate the models using accuracy, precision, recall, F1-score, and a confusion matrix.
9. Save the selected model using Joblib.
10. Build a prediction interface using Gradio.

## 5. Model Evaluation

### Logistic Regression
- Accuracy: 80.41%
- Churn precision: 0.66
- Churn recall: 0.55
- Churn F1-score: 0.60

### Random Forest
- Accuracy: 77.64%
- Churn precision: 0.57
- Churn recall: 0.66
- Churn F1-score: 0.61

Random Forest was selected as the initial model because its churn recall was higher on the test set. Further validation and preprocessing improvements may be needed before production use.

## 6. Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Gradio

## 7. Project Structure
```text
customer-churn-prediction/
├── data/
│   └── customer_churn.csv
├── models/
│   └── churn_model.pkl
├── load_data.py
├── train_model.py
├── test_saved_model.py
├── app.py
└── README.md
```

## 8. How to Run

Install dependencies:

```bash
python -m pip install pandas scikit-learn matplotlib joblib gradio
```

Train the models and save the selected model:

```bash
python train_model.py
```

Verify the saved model:

```bash
python test_saved_model.py
```

Launch the Gradio app:

```bash
python app.py
```

Open the local URL displayed in the terminal.

## 9. Limitations
- Model predictions are estimates, not guarantees.
- Model performance may change with new customer data.
- The current Gradio interface uses default values for some service features; these should be exposed as inputs for a more complete application.
- The model should be validated further before real business deployment.
