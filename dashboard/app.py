import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Add project src folder to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.append(str(SRC_PATH))

from config import DATA_PATH, MODEL_PATH, RESULTS_PATH, FIGURE_PATH
model = joblib.load(MODEL_PATH)


st.sidebar.title("📞 Telecom ML Dashboard")
st.sidebar.markdown("---")
# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Telecom Customer Complaint Prediction",
    page_icon="📞",
    layout="wide"
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(DATA_PATH)
dataset_stats = {
    "Tenure": df["Tenure"].mean(),
    "Service_Score": df["Service_Score"].mean(),
    "CC_Contacted_LY": df["CC_Contacted_LY"].mean(),
    "cashback": df["cashback"].mean(),
    "rev_per_month": df["rev_per_month"].mean(),
    "CC_Agent_Score": df["CC_Agent_Score"].mean()
}
st.subheader("Project Workflow")

c1,c2,c3 = st.columns(3)

with c1:
    st.success("✔ Data Cleaning")
    st.success("✔ EDA")

with c2:
    st.success("✔ Feature Engineering")
    st.success("✔ Model Training")

with c3:
    st.success("✔ Model Evaluation")
    st.success("✔ Live Prediction")
    
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 EDA",
        "🤖 Model Performance",
        "🎯 Live Prediction",
        "ℹ️ About"
    ]
)

if page == "🏠 Home":

    st.title("📞 Telecom Customer Complaint Prediction")

    st.markdown("""
    This dashboard predicts whether a telecom customer is likely to raise a complaint using machine learning.

    ### Project Workflow

    - Data Cleaning
    - Exploratory Data Analysis
    - Feature Engineering
    - Model Training
    - Model Evaluation
    - Live Prediction
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", len(df))
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Complaint Rate", f"{df['Complain_ly'].mean()*100:.2f}%")
    col4.metric("Best Model", "Random Forest")

    st.divider()

    st.subheader("📌 Machine Learning Workflow")

    st.markdown("""
    **Raw Dataset**
    ↓

    **Data Cleaning**
    ↓

    **Feature Engineering**
    ↓

    **Model Training**
    ↓

    **Model Evaluation**
    ↓

    **Prediction Dashboard**
    """)

    st.info("Use the sidebar to explore different sections of the project.")

    # =====================================================
# DATASET PAGE
# =====================================================

elif page == "📊 Dataset":

    st.title("📊 Dataset Overview")

    st.markdown("### Dataset Preview")

    st.dataframe(df.head(10), width="stretch")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.divider()

    st.subheader("Missing Values")

    missing = (
        df.isnull()
          .sum()
          .reset_index()
    )

    missing.columns = ["Column", "Missing Values"]
    
    missing = (
    df.isnull()
      .sum()
      .reset_index()
    )

    missing.columns = ["Column", "Missing Values"]

    missing = missing[missing["Missing Values"] > 0]

    st.dataframe(
    missing,
    width="stretch"
    )
    

    st.divider()

    st.subheader("Data Types")

    dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

    dtype_df.columns = ["Column", "Data Type"]

    st.dataframe(
        dtype_df,
        width="stretch"
    )

    st.divider()

    st.download_button(
        label="⬇ Download Cleaned Dataset",
        data=df.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # =====================================================
# EDA PAGE
# =====================================================

elif page == "📈 EDA":

    st.title("📈 Exploratory Data Analysis")

    st.write(
        "Explore the data using the visualizations generated during EDA."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Distributions",
            "📉 Relationships",
            "🔥 Correlation",
            "📋 Insights"
        ]
    )

    with tab1:

        st.subheader("Complaint Distribution")

        st.image(
            FIGURE_PATH / "complaint_distribution.png",
            width="stretch"
        )

        st.subheader("Service Score")

        st.image(
            FIGURE_PATH / "Service_Score_hist.png",
            width="stretch"
        )

        st.subheader("Revenue")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                FIGURE_PATH / "rev_per_month_hist.png",
                width="stretch"
            )

        with col2:
            st.image(
                FIGURE_PATH / "cashback_hist.png",
                width="stretch"
            )

    with tab2:

        st.subheader("Payment vs Complaint")

        st.image(
            FIGURE_PATH / "payment_vs_complaint.png",
            width="stretch"
        )

        st.subheader("Customer Care Contacts")

        st.image(
            FIGURE_PATH / "cc_contact_vs_complaint.png",
            width="stretch"
        )

    with tab3:

        st.subheader("Correlation Heatmap")

        st.image(
            FIGURE_PATH / "correlation_heatmap.png",
            width="stretch"
        )

    with tab4:

        st.success("""
### Key Business Insights

• Approximately 28.5% of customers raised complaints.

• Customers with lower service scores tend to complain more.

• Frequent customer care interactions are associated with complaints.

• Regular Plus customers represent the largest customer segment.

• Revenue distribution is positively skewed.
""")

# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    results = pd.read_csv(RESULTS_PATH)

    st.subheader("Model Comparison")

    st.dataframe(
        results.style.highlight_max(axis=0),
        width="stretch"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🏆 Best Model")

        best = results.iloc[0]

        st.metric(
            "Model",
            best["Model"]
        )

        st.metric(
            "Accuracy",
            f"{best['Accuracy']:.2%}"
        )

        st.metric(
            "ROC AUC",
            f"{best['ROC AUC']:.2%}"
        )

    with col2:

        st.subheader("Evaluation Metrics")

        st.bar_chart(
            results.set_index("Model")[
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score"
                ]
            ]
        )

    st.divider()

    st.subheader("Confusion Matrix")

    st.image(
        FIGURE_PATH / "confusion_matrix.png",
        width="stretch"
    )

    st.divider()

    st.subheader("ROC Curve")

    st.image(
        FIGURE_PATH / "roc_curve.png",
        width="stretch"
    )

    st.divider()

    st.subheader("Feature Importance")

    st.image(
        FIGURE_PATH / "feature_importance.png",
        width="stretch"
    )

# =====================================================
# LIVE PREDICTION
# =====================================================
# =====================================================
# LIVE PREDICTION PAGE
# =====================================================

# =====================================================
# LIVE PREDICTION PAGE
# =====================================================

elif page == "🎯 Live Prediction":

    st.title("🎯 Telecom CRM - Complaint Prediction")

    st.write(
        "Search a customer using Account ID, review the customer profile, and predict complaint risk."
    )

    st.divider()

    # --------------------------------------------
    # Search Customer
    # --------------------------------------------

    account_id = st.number_input(
        "Customer Account ID",
        min_value=int(df["AccountID"].min()),
        max_value=int(df["AccountID"].max()),
        value=int(df["AccountID"].min()),
        step=1
    )

    col1, col2 = st.columns([1, 4])

    with col1:

        if st.button("🔍 Load Customer"):

            customer = df[df["AccountID"] == account_id]

            if customer.empty:

                st.error("Customer not found.")

                if "customer" in st.session_state:
                    del st.session_state["customer"]

            else:

                st.session_state["customer"] = customer.iloc[0]

                st.success("Customer Loaded Successfully")

    # --------------------------------------------
    # Show Customer
    # --------------------------------------------

    if "customer" in st.session_state:

        customer = st.session_state["customer"]

        st.divider()

        st.subheader("📋 Customer Profile")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric("Account ID", int(customer["AccountID"]))
            st.metric("Gender", customer["Gender"])
            st.metric("Tenure", customer["Tenure"])
            st.metric("City Tier", customer["City_Tier"])
            st.metric("Login Device", customer["Login_device"])

        with c2:

            st.metric("Segment", customer["account_segment"])
            st.metric("Payment", customer["Payment"])
            st.metric("Revenue", customer["rev_per_month"])
            st.metric("Revenue Growth", customer["rev_growth_yoy"])
            st.metric("Cashback", customer["cashback"])

        with c3:

            st.metric("Service Score", customer["Service_Score"])
            st.metric("Agent Score", customer["CC_Agent_Score"])
            st.metric("Support Contacts", customer["CC_Contacted_LY"])
            st.metric("Marital Status", customer["Marital_Status"])
            st.metric("Coupons Used", customer["coupon_used_for_payment"])

        st.divider()

        # --------------------------------------------
        # Predict
        # --------------------------------------------

        if st.button("🚀 Predict Complaint Risk"):

            input_data = pd.DataFrame({

                "Tenure": [customer["Tenure"]],
                "City_Tier": [customer["City_Tier"]],
                "CC_Contacted_LY": [customer["CC_Contacted_LY"]],
                "Payment": [customer["Payment"]],
                "Gender": [customer["Gender"]],
                "Service_Score": [customer["Service_Score"]],
                "Account_user_count": [customer["Account_user_count"]],
                "account_segment": [customer["account_segment"]],
                "CC_Agent_Score": [customer["CC_Agent_Score"]],
                "Marital_Status": [customer["Marital_Status"]],
                "rev_per_month": [customer["rev_per_month"]],
                "rev_growth_yoy": [customer["rev_growth_yoy"]],
                "coupon_used_for_payment": [customer["coupon_used_for_payment"]],
                "Day_Since_CC_connect": [customer["Day_Since_CC_connect"]],
                "cashback": [customer["cashback"]],
                "Login_device": [customer["Login_device"]]

            })

            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0][1]

            st.session_state["prediction"] = prediction
            st.session_state["probability"] = probability

    # --------------------------------------------
    # Show Prediction
    # --------------------------------------------

    if "prediction" in st.session_state:

        prediction = st.session_state["prediction"]
        probability = st.session_state["probability"]

        st.divider()

        st.subheader("📊 Prediction Result")

        st.metric(
            "Complaint Probability",
            f"{probability*100:.2f}%"
        )

        if probability < 0.30:

            st.success("🟢 LOW RISK")

        elif probability < 0.70:

            st.warning("🟠 MEDIUM RISK")

        else:

            st.error("🔴 HIGH RISK")

        if prediction == 1:

            st.error(
                "Customer is likely to raise a complaint."
            )

        else:

            st.success(
                "Customer is unlikely to raise a complaint."
            )

        result = pd.DataFrame({

            "Account ID": [customer["AccountID"]],
            "Probability (%)": [round(probability * 100, 2)],
            "Prediction": [
                "Complaint"
                if prediction == 1
                else "No Complaint"
            ]

        })

        st.download_button(

            "⬇ Download Prediction",

            result.to_csv(index=False),

            "prediction.csv",

            "text/csv"

        )
    reasons = []

if customer = st.session_state["customer"]["Service_Score"] < dataset_stats["Service_Score"]:
    reasons.append(
        f"• Service Score ({customer['Service_Score']}) is below the dataset average ({dataset_stats['Service_Score']:.2f})."
    )

if customer["CC_Contacted_LY"] > dataset_stats["CC_Contacted_LY"]:
    reasons.append(
        f"• Customer contacted support {int(customer['CC_Contacted_LY'])} times last year, above the dataset average ({dataset_stats['CC_Contacted_LY']:.1f})."
    )

if customer["cashback"] < dataset_stats["cashback"]:
    reasons.append(
        f"• Cashback received is below the dataset average."
    )

if customer["Tenure"] < dataset_stats["Tenure"]:
    reasons.append(
        f"• Customer tenure is shorter than the average customer."
    )

if customer["rev_per_month"] < dataset_stats["rev_per_month"]:
    reasons.append(
        f"• Monthly revenue is below the dataset average."
    )

st.subheader("🔍 Why this prediction?")

if reasons:
    for reason in reasons:
        st.write(reason)
else:
    st.success(
        "No major risk indicators stand out compared with the overall customer base."
    )

actions = []

if customer["Service_Score"] < dataset_stats["Service_Score"]:
    actions.append("Improve service quality and follow up with the customer.")

if customer["CC_Contacted_LY"] > dataset_stats["CC_Contacted_LY"]:
    actions.append("Assign a senior support agent for future interactions.")

if customer["cashback"] < dataset_stats["cashback"]:
    actions.append("Consider a retention offer or cashback incentive.")

if customer["Tenure"] < dataset_stats["Tenure"]:
    actions.append("Provide onboarding support and proactive engagement.")

st.subheader("💡 Recommended Actions")

if actions:
    for action in actions:
        st.write(f"✅ {action}")
else:
    st.success("No specific intervention is recommended at this time.")


st.sidebar.markdown("---")
st.sidebar.caption("Developed by Amisha")

