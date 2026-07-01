import streamlit as st

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Telecom Customer Complaint Prediction",
    page_icon="📞",
    layout="wide"
)

# =====================================================
# Title
# =====================================================

st.title("📞 Telecom Customer Complaint Prediction")

st.markdown("---")

st.write(
    """
    Welcome to the Telecom Customer Complaint Prediction Dashboard.

    This application predicts whether a telecom customer is likely to raise a complaint
    using machine learning models trained on historical customer data.
    """
)

st.success("Dashboard Loaded Successfully ✅")