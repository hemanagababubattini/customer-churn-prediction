
import pandas as pd

# Load the dataset
df = pd.read_csv("data/customer_churn.csv")

# 1. Display first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# 2. Check number of rows and columns
print("\n--- Dataset Shape ---")
print(df.shape)

# 3. Display column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# 4. Check data types and missing values
print("\n--- Dataset Information ---")
df.info()

# 5. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# 6. Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 7. Check duplicate rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 8. Check dataset size
print("\n--- Dataset Size ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 9. Check TotalCharges data type
print("\n--- TotalCharges Before Cleaning ---")
print(df["TotalCharges"].dtype)

# 10. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 11. Check missing values after conversion
print("\n--- Missing Values After Conversion ---")
print(df["TotalCharges"].isnull().sum())

# 12. Fill missing values with median
median_value = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

# 13. Verify cleaning
print("\n--- TotalCharges After Cleaning ---")
print("Data type:", df["TotalCharges"].dtype)
print("Missing values:", df["TotalCharges"].isnull().sum())

# 14. Identify numerical columns
print("\n--- Numerical Columns ---")
print(df.select_dtypes(include=["number"]).columns.tolist())

# 15. Identify categorical columns
print("\n--- Categorical Columns ---")
print(df.select_dtypes(include=["object"]).columns.tolist())

# 16. Display unique values in selected categorical columns
print("\n--- Sample Categorical Values ---")
for col in ["gender", "Partner", "InternetService", "Contract", "Churn"]:
    if col in df.columns:
        print(f"{col}: {df[col].unique()}")
# Step 2.8: Analyze churn distribution

print("\n--- Churn Count ---")
print(df["Churn"].value_counts())

print("\n--- Churn Percentage ---")
print(df["Churn"].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt

# Count customers in each churn category
churn_counts = df["Churn"].value_counts()

# Create bar chart
churn_counts.plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Step 2.10: Tenure vs Churn

print("\n--- Average Tenure by Churn ---")
print(df.groupby("Churn")["tenure"].mean())

# Visualize tenure by churn
df.boxplot(column="tenure", by="Churn")

import matplotlib.pyplot as plt

plt.title("Tenure vs Churn")
plt.suptitle("")
plt.xlabel("Churn Status")
plt.ylabel("Tenure (Months)")
plt.show()

# Step 2.11: Monthly Charges vs Churn

print("\n--- Average Monthly Charges by Churn ---")

print(
    df.groupby("Churn")["MonthlyCharges"].mean()
)

# Create a box plot
df.boxplot(column="MonthlyCharges", by="Churn")

import matplotlib.pyplot as plt

plt.title("Monthly Charges vs Churn")
plt.suptitle("")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")
plt.show()
print("\n--- Contract Type vs Churn ---")

print(pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100)

pd.crosstab(
    df["Contract"],
    df["Churn"]
).plot(kind="bar")

import matplotlib.pyplot as plt

plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print("\n--- EDA Observations ---")

print("1. Dataset contains customer information and churn status.")
print("2. Around 26.54% of customers have churned.")
print("3. We analyzed the relationship between tenure and churn.")
print("4. We analyzed the relationship between monthly charges and churn.")
print("5. We compared churn across different contract types.")