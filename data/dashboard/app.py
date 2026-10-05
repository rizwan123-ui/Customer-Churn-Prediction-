import streamlit as st
import pandas as pd
import joblib

# ==================================================
# LOAD MODEL AND DATA
# ==================================================

model = joblib.load("models/churn_model.pkl")

df = pd.read_csv(
    "data/processed/customer_churn_features.csv"
)

predictions = pd.read_csv(
    "data/processed/churn_predictions.csv"
)

# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# TITLE
# ==================================================

st.title("📊 Customer Churn Prediction Dashboard")

st.write(
    "Customer Churn Prediction & Business Intelligence System"
)

st.divider()

# ==================================================
# EXECUTIVE OVERVIEW
# ==================================================

st.header("Executive Overview")

total_customers = len(df)

# Handle Churn Label safely whether values are Yes/No or 1/0
churn_numeric = (
    df["Churn Label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["1", "yes", "true", "churned"])
)

churn_rate = churn_numeric.mean() * 100

high_risk = (
    predictions["Risk_Level"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("high")
    .sum()
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Current Churn Rate",
    f"{churn_rate:.2f}%"
)

col3.metric(
    "High Risk Customers",
    f"{high_risk:,}"
)

st.divider()

# ==================================================
# CHURN DISTRIBUTION
# ==================================================

st.header("Customer Churn Analysis")

st.subheader("Churn Distribution")

churn_distribution = (
    df["Churn Label"]
    .astype(str)
    .str.strip()
    .value_counts()
)

st.bar_chart(churn_distribution)

# ==================================================
# CONTRACT TYPE ANALYSIS
# ==================================================

st.subheader("Churn by Contract Type")

contract_data = pd.crosstab(
    df["Contract"],
    df["Churn Label"]
)

st.bar_chart(contract_data)

# ==================================================
# PAYMENT METHOD ANALYSIS
# ==================================================

st.subheader("Churn by Payment Method")

payment_data = pd.crosstab(
    df["Payment Method"],
    df["Churn Label"]
)

st.bar_chart(payment_data)

# ==================================================
# INTERNET SERVICE ANALYSIS
# ==================================================

st.subheader("Churn by Internet Service")

internet_data = pd.crosstab(
    df["Internet Service"],
    df["Churn Label"]
)

st.bar_chart(internet_data)

# ==================================================
# TENURE ANALYSIS
# ==================================================

st.subheader("Average Tenure by Churn Status")

tenure_data = (
    df.groupby("Churn Label")["Tenure Months"]
    .mean()
    .round(2)
)

st.bar_chart(tenure_data)

# ==================================================
# MONTHLY CHARGES ANALYSIS
# ==================================================

st.subheader("Average Monthly Charges by Churn Status")

monthly_charge_data = (
    df.groupby("Churn Label")["Monthly Charges"]
    .mean()
    .round(2)
)

st.bar_chart(monthly_charge_data)

# ==================================================
# CHURN REASONS
# ==================================================

st.subheader("Top Churn Reasons")

churned_customers = df[
    churn_numeric
]

if "Churn Reason" in churned_customers.columns:
    top_reasons = (
        churned_customers["Churn Reason"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(10)
    )

    st.bar_chart(top_reasons)
else:
    st.info("Churn Reason column is not available.")

st.divider()

# ==================================================
# RISK ANALYSIS
# ==================================================

st.header("Prediction Risk Analysis")

st.subheader("Customer Risk Levels")

risk_counts = (
    predictions["Risk_Level"]
    .astype(str)
    .str.strip()
    .value_counts()
)

st.bar_chart(risk_counts)

# ==================================================
# HIGH RISK CUSTOMERS
# ==================================================

st.subheader("High Risk Customers")

high_risk_customers = predictions[
    predictions["Risk_Level"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("high")
]

st.write(
    f"Total High Risk Customers: {len(high_risk_customers):,}"
)

columns_to_show = [
    col for col in [
        "Actual_Churn",
        "Predicted_Churn",
        "Churn_Probability",
        "Risk_Level"
    ]
    if col in high_risk_customers.columns
]

if columns_to_show:
    st.dataframe(
        high_risk_customers[columns_to_show],
        use_container_width=True
    )
else:
    st.dataframe(
        high_risk_customers,
        use_container_width=True
    )

# ==================================================
# MODEL PERFORMANCE
# ==================================================

st.divider()

st.header("Model Performance")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Accuracy", "74.88%")
m2.metric("Precision", "51.75%")
m3.metric("Recall", "78.88%")
m4.metric("F1 Score", "62.50%")
m5.metric("ROC-AUC", "84.93%")

st.info(
    "Logistic Regression was selected as the final model "
    "because it achieved strong recall, allowing the system "
    "to identify a larger proportion of customers who are "
    "actually likely to churn."
)

# ==================================================
# BUSINESS INSIGHTS
# ==================================================

st.divider()

st.header("Business Insights")

st.write(
    """
    Key insights identified during the analysis:

    • Customers on month-to-month contracts show higher churn risk.

    • Customers with shorter tenure are more likely to leave.

    • Higher monthly charges are associated with increased churn.

    • Electronic check customers show relatively higher churn.

    • High-risk customers should be prioritized for retention campaigns.

    Recommended actions include offering long-term contract incentives,
    personalized discounts, improved customer support and proactive
    engagement with high-risk customers.
    """
)

# ==================================================
# PROJECT INFORMATION
# ==================================================

st.divider()

st.header("Project Information")

st.write(
    """
    **Project:** Customer Churn Prediction & Business Intelligence System

    **Dataset:** Telecom Customer Churn Dataset

    **Customers:** 7,043

    **Final Model:** Logistic Regression

    **Machine Learning Task:** Binary Classification

    **Risk Levels:** Low, Medium and High

    **Tools Used:** Python, Pandas, NumPy, Scikit-learn,
    Matplotlib, Seaborn, Streamlit
    """
)

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.success(
    "Customer Churn Dashboard loaded successfully!"
)