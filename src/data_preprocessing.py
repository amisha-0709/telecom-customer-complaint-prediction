import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "LoyaltyVision Analytics.xlsx"

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data():
    print("Loading dataset...")
    return pd.read_excel(RAW_DATA)

# --------------------------------------------------
# Clean Object Columns
# --------------------------------------------------

def clean_object_columns(df):

    garbage_tokens = ["#", "@", "$", "*", "+"]

    object_columns = df.select_dtypes(include="object").columns

    for col in object_columns:
        df[col] = df[col].replace(garbage_tokens, np.nan)

    return df

# --------------------------------------------------
# Convert Numeric Columns
# --------------------------------------------------

def convert_numeric_columns(df):

    numeric_columns = [
        "Tenure",
        "Account_user_count",
        "rev_per_month",
        "rev_growth_yoy",
        "coupon_used_for_payment",
        "Day_Since_CC_connect",
        "cashback",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# --------------------------------------------------
# Standardize Categories
# --------------------------------------------------

def standardize_categories(df):

    df["Gender"] = df["Gender"].replace(
        {
            "F": "Female",
            "M": "Male"
        }
    )

    df["account_segment"] = df["account_segment"].replace(
        {
            "Regular +": "Regular Plus",
            "Super +": "Super Plus"
        }
    )

    return df

# --------------------------------------------------
# Save Clean Dataset
# --------------------------------------------------

def save_dataset(df):

    df.to_csv(PROCESSED_DATA, index=False)

    print("\nClean dataset saved successfully!")

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    df = load_data()

    df = clean_object_columns(df)

    df = convert_numeric_columns(df)

    df = standardize_categories(df)

    save_dataset(df)

if __name__ == "__main__":
    main()