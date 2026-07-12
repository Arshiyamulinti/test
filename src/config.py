from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw"

PROCESSED_DATA = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = "Churn"