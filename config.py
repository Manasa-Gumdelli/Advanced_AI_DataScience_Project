from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("diabetic_data.csv")
OUTPUT_DIR = BASE_DIR / "outputs"

TARGET_COLUMN = "readmitted"
POSITIVE_CLASS = "<30"
TOP_SPECIALTIES = 10
