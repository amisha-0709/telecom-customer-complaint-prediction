from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from feature_engineering import (
    load_dataset,
    split_features_target,
    split_data
)

# =====================================================
# Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

FIGURE_PATH = PROJECT_ROOT / "reports" / "figures"

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Data
# =====================================================

df = load_dataset()

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = split_data(X, y)

# =====================================================
# Load Model
# =====================================================

model = joblib.load(MODEL_PATH)

predictions = model.predict(X_test)

# =====================================================
# Classification Report
# =====================================================

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig(FIGURE_PATH / "confusion_matrix.png")

plt.close()

# =====================================================
# ROC Curve
# =====================================================

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.savefig(FIGURE_PATH / "roc_curve.png")

plt.close()

# =====================================================
# Precision Recall Curve
# =====================================================

PrecisionRecallDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.savefig(FIGURE_PATH / "precision_recall_curve.png")

plt.close()

print("\nEvaluation Completed Successfully!")

import pandas as pd

if hasattr(model.named_steps["model"], "feature_importances_"):

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    importance = model.named_steps["model"].feature_importances_

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    ).head(15)

    plt.figure(figsize=(10,6))

    plt.barh(
        feature_df["Feature"],
        feature_df["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title("Top 15 Important Features")

    plt.tight_layout()

    plt.savefig(FIGURE_PATH / "feature_importance.png")

    plt.close()