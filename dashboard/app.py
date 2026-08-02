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

    st.title("🎯 Live Complaint Prediction")

    st.write(
        "Enter the customer information below to predict whether the customer is likely to raise a complaint."
    )

    st.divider()

    col1, col2 = st.columns(2)

    # ==========================
    # LEFT COLUMN
    # ==========================

    with col1:

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=100,
            value=10
        )

        city = st.selectbox(
            "City Tier",
            sorted(df["City_Tier"].dropna().unique())
        )

        contacted = st.number_input(
            "Customer Care Contacts Last Year",
            min_value=0,
            max_value=150,
            value=10
        )

        payment = st.selectbox(
            "Payment Method",
            sorted(df["Payment"].dropna().unique())
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["Gender"].dropna().unique())
        )

        service = st.slider(
            "Service Score",
            0,
            5,
            3
        )

        users = st.slider(
            "Account User Count",
            1,
            6,
            3
        )

        segment = st.selectbox(
            "Account Segment",
            sorted(df["account_segment"].dropna().unique())
        )

    # ==========================
    # RIGHT COLUMN
    # ==========================

    with col2:

        agent = st.slider(
            "Customer Care Agent Score",
            1,
            5,
            3
        )

        marital = st.selectbox(
            "Marital Status",
            sorted(df["Marital_Status"].dropna().unique())
        )

        revenue = st.number_input(
            "Revenue Per Month",
            value=1000.0
        )

        growth = st.number_input(
            "Revenue Growth YoY",
            value=10.0
        )

        coupon = st.number_input(
            "Coupons Used for Payment",
            min_value=0,
            max_value=100,
            value=1
        )

        days = st.number_input(
            "Days Since Last Customer Care Contact",
            min_value=0,
            max_value=365,
            value=5
        )

        cashback = st.number_input(
            "Cashback",
            value=150.0
        )

        device = st.selectbox(
            "Login Device",
            sorted(df["Login_device"].dropna().unique())
        )

    st.divider()

    # ==========================
    # Prediction
    # ==========================

    if st.button("🔮 Predict Complaint"):

        input_data = pd.DataFrame({

            "Tenure": [tenure],
            "City_Tier": [city],
            "CC_Contacted_LY": [contacted],
            "Payment": [payment],
            "Gender": [gender],
            "Service_Score": [service],
            "Account_user_count": [users],
            "account_segment": [segment],
            "CC_Agent_Score": [agent],
            "Marital_Status": [marital],
            "rev_per_month": [revenue],
            "rev_growth_yoy": [growth],
            "coupon_used_for_payment": [coupon],
            "Day_Since_CC_connect": [days],
            "cashback": [cashback],
            "Login_device": [device]

        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Complaint Probability",
            f"{probability*100:.2f}%"
        )

        if prediction == 1:

            st.error(
                "⚠️ This customer is likely to raise a complaint."
            )

        else:

            st.success(
                "✅ This customer is unlikely to raise a complaint."
            )

        st.divider()

        result = pd.DataFrame({

            "Complaint Probability (%)": [
                round(probability * 100, 2)
            ],

            "Prediction": [
                "Complaint"
                if prediction == 1
                else "No Complaint"
            ]

        })

        st.download_button(

            "⬇ Download Prediction",

            result.to_csv(index=False),

            file_name="prediction.csv",

            mime="text/csv"

        )



st.sidebar.markdown("---")
st.sidebar.caption("Developed by Amisha")

