import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PATH, TOP_SPECIALTIES, TARGET_COLUMN, POSITIVE_CLASS


def group_diagnosis(code: str) -> str:
    if pd.isna(code) or str(code) in {"?", "nan", "None"}:
        return "Missing"

    code = str(code)
    try:
        value = float(code)

        if 390 <= value < 460 or value == 785:
            return "Circulatory"
        if 460 <= value < 520 or value == 786:
            return "Respiratory"
        if 520 <= value < 580 or value == 787:
            return "Digestive"
        if 250 <= value < 251:
            return "Diabetes"
        if 800 <= value < 1000:
            return "Injury"
        if 710 <= value < 740:
            return "Musculoskeletal"
        if 580 <= value < 630 or value == 788:
            return "Genitourinary"
        if 140 <= value < 240:
            return "Neoplasms"

        return "Other"

    except ValueError:
        if code.startswith("V"):
            return "Supplementary"
        if code.startswith("E"):
            return "External"
        return "Other"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, na_values=["?"], low_memory=False).copy()

    # commonly removed rows in this dataset
    df = df[df["discharge_disposition_id"] != 11].copy()   # expired
    df = df[df["gender"] != "Unknown/Invalid"].copy()

    # target: readmitted within 30 days
    df["target"] = (df[TARGET_COLUMN] == POSITIVE_CLASS).astype(int)
    df["readmission_label"] = df[TARGET_COLUMN].copy()

    # grouped diagnoses
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].astype(str)
        df[f"{col}_group"] = df[col].apply(group_diagnosis)

    # grouped specialty
    specialty = df["medical_specialty"].fillna("Missing")
    top_specialties = specialty.value_counts().nlargest(TOP_SPECIALTIES).index
    df["medical_specialty_grouped"] = np.where(
        specialty.isin(top_specialties),
        specialty,
        "Other"
    )

    # feature engineering
    df["age_mid"] = df["age"].str.extract(r"\[(\d+)-").astype(float)
    df["has_weight_record"] = df["weight"].notna().astype(int)
    df["total_visits"] = (
        df["number_outpatient"] +
        df["number_emergency"] +
        df["number_inpatient"]
    )
    df["medication_load"] = df["num_medications"] / df["time_in_hospital"].replace(0, 1)

    # drop leakage / high-cardinality / weak columns
    drop_columns = [
        TARGET_COLUMN,
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
        "readmission_label",
    ]

    X = df.drop(columns=drop_columns + ["target"])
    y = df["target"]

    return df, X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor
