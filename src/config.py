from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "LoyaltyVision Analytics.xlsx"

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

RESULTS_PATH = PROJECT_ROOT / "models" / "model_results.csv"

FIGURE_PATH = PROJECT_ROOT / "reports" / "figures"

TARGET = "Complain_ly"