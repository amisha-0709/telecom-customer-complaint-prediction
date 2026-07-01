from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from feature_engineering import (
    load_dataset,
    split_features_target,
    create_preprocessor,
    split_data
)

# =====================================================
# Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

# =====================================================
# Load Data
# =====================================================

df = load_dataset()

X, y = split_features_target(df)

preprocessor = create_preprocessor(X)

X_train, X_test, y_train, y_test = split_data(X, y)

# =====================================================
# Models
# =====================================================

models = {

    "Logistic Regression":
        LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )

}

results = []

best_model = None

best_accuracy = 0

# =====================================================
# Training Loop
# =====================================================

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(

        steps=[

            ("preprocessor", preprocessor),

            ("model", model)

        ]

    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc = roc_auc_score(y_test, probabilities)

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": roc

    })

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = pipeline

        joblib.dump(

            best_model,

            MODEL_DIR / "best_model.pkl"

        )

# =====================================================
# Results
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")

print(results_df)

results_df.to_csv(

    MODEL_DIR / "model_results.csv",

    index=False

)

print("\nBest model saved successfully.")