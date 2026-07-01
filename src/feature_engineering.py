import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"

TARGET = "Complain_ly"


# ============================================================
# Load Dataset
# ============================================================

def load_dataset():
    """
    Load cleaned dataset and remove rows with missing target.
    """

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=[TARGET])

    return df


# ============================================================
# Split Features & Target
# ============================================================

def split_features_target(df):
    """
    Split dataframe into X and y.
    """

    X = df.drop(columns=[TARGET, "AccountID", "Churn"])

    y = df[TARGET]

    return X, y


# ============================================================
# Create Preprocessor
# ============================================================

def create_preprocessor(X):
    """
    Create preprocessing pipeline.
    """

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# ============================================================
# Train-Test Split
# ============================================================

def split_data(X, y):
    """
    Perform train-test split.
    """

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    df = load_dataset()

    X, y = split_features_target(df)

    preprocessor = create_preprocessor(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("=" * 60)
    print("Feature Engineering Module")
    print("=" * 60)

    print(f"\nDataset Shape : {df.shape}")

    print(f"\nTraining Set : {X_train.shape}")

    print(f"Testing Set  : {X_test.shape}")

    print("\nPipeline Created Successfully")