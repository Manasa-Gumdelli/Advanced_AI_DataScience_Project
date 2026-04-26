import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import OUTPUT_DIR
from data_preprocessing import group_diagnosis

MODEL_PATH = OUTPUT_DIR / "xgboost_model.joblib"
THRESHOLD = 0.14

TOP_SPECIALTIES = [
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
    "Cardiology", "Surgery-General", "Nephrology", "Orthopedics",
    "Orthopedics-Reconstructive", "Radiologist", "Pulmonology",
]

MED_OPTIONS = ["No", "Steady", "Up", "Down"]
MED_OPTIONS_LIMITED = ["No", "Steady"]
DIAG_GROUPS = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Supplementary",
    "External", "Other", "Missing",
]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_input_row(inputs: dict) -> pd.DataFrame:
    row = inputs.copy()

    for col in ["diag_1", "diag_2", "diag_3"]:
        row[f"{col}_group"] = row.pop(col)

    specialty = row.pop("medical_specialty")
    row["medical_specialty_grouped"] = specialty if specialty in TOP_SPECIALTIES else "Other"

    age_str = row["age"]
    row["age_mid"] = float(age_str.strip("[)").split("-")[0])

    row["has_weight_record"] = 0  # weight not collected in UI

    row["total_visits"] = (
        row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
    )
    row["medication_load"] = row["num_medications"] / max(row["time_in_hospital"], 1)

    drop = ["encounter_id", "patient_nbr", "weight", "payer_code",
            "medical_specialty", "readmitted", "readmission_label", "target"]
    for k in drop:
        row.pop(k, None)

    return pd.DataFrame([row])


def main():
    st.set_page_config(page_title="Readmission Risk Predictor", layout="wide")
    st.title("Diabetic Patient Readmission Risk Predictor")
    st.markdown("Fill in the patient details below and click **Predict** to estimate the risk of readmission within 30 days.")

    model = load_model()

    # ── Demographics ──────────────────────────────────────────────────────────
    st.header("Demographics")
    c1, c2, c3 = st.columns(3)
    race = c1.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
    gender = c2.selectbox("Gender", ["Female", "Male"])
    age = c3.selectbox("Age Group", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ], index=6)

    # ── Admission Info ────────────────────────────────────────────────────────
    st.header("Admission Info")
    c1, c2, c3 = st.columns(3)
    admission_type_id = c1.selectbox(
        "Admission Type",
        options=[1, 2, 3, 4, 5, 6, 7, 8],
        format_func=lambda x: {
            1: "1 – Emergency", 2: "2 – Urgent", 3: "3 – Elective",
            4: "4 – Newborn", 5: "5 – Not Available", 6: "6 – NULL",
            7: "7 – Trauma Center", 8: "8 – Not Mapped"
        }[x]
    )
    discharge_disposition_id = c2.selectbox(
        "Discharge Disposition",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15,
                 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28],
        format_func=lambda x: {
            1: "1 – Discharged to home", 2: "2 – Short-term hospital",
            3: "3 – SNF", 4: "4 – ICF", 5: "5 – Another inpatient care",
            6: "6 – Home with home health", 7: "7 – Left AMA",
            8: "8 – Home IV provider", 9: "9 – Admitted as inpatient",
            10: "10 – Neonate to different level", 12: "12 – Still patient",
            13: "13 – Hospice / home", 14: "14 – Hospice / medical facility",
            15: "15 – Swing bed", 16: "16 – Outpatient",
            17: "17 – Medicare swing bed", 18: "18 – Not mapped",
            19: "19 – Expired at home", 20: "20 – Expired in a medical facility",
            22: "22 – Rehab facility", 23: "23 – Long-term care hospital",
            24: "24 – Nursing facility", 25: "25 – Critical access hospital",
            27: "27 – Federal health care", 28: "28 – Unknown"
        }.get(x, str(x))
    )
    admission_source_id = c3.selectbox(
        "Admission Source",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 20, 22, 25],
        format_func=lambda x: {
            1: "1 – Physician Referral", 2: "2 – Clinic Referral",
            3: "3 – HMO Referral", 4: "4 – Transfer from hospital",
            5: "5 – Transfer from SNF", 6: "6 – Transfer from another",
            7: "7 – Emergency Room", 8: "8 – Court/Law",
            9: "9 – Not Available", 10: "10 – Transfer from critical access",
            11: "11 – Normal Delivery", 13: "13 – Sick Baby",
            14: "14 – Extramural Birth", 17: "17 – NULL",
            20: "20 – Not Mapped", 22: "22 – Transfer from outpatient surgery",
            25: "25 – Transfer from ambulatory surgery"
        }.get(x, str(x))
    )

    c1, c2 = st.columns(2)
    time_in_hospital = c1.slider("Time in Hospital (days)", 1, 14, 3)
    medical_specialty = c2.selectbox(
        "Medical Specialty",
        ["Missing"] + TOP_SPECIALTIES + ["Other"]
    )

    # ── Lab & Procedures ──────────────────────────────────────────────────────
    st.header("Lab & Procedures")
    c1, c2, c3, c4 = st.columns(4)
    num_lab_procedures = c1.slider("Lab Procedures", 1, 132, 40)
    num_procedures = c2.slider("Procedures", 0, 6, 1)
    num_medications = c3.slider("Medications", 1, 81, 15)
    number_diagnoses = c4.slider("Number of Diagnoses", 1, 16, 7)

    # ── Visit History ─────────────────────────────────────────────────────────
    st.header("Prior Visit History")
    c1, c2, c3 = st.columns(3)
    number_outpatient = c1.slider("Outpatient Visits", 0, 42, 0)
    number_emergency = c2.slider("Emergency Visits", 0, 76, 0)
    number_inpatient = c3.slider("Inpatient Visits", 0, 21, 0)

    # ── Diagnoses ─────────────────────────────────────────────────────────────
    st.header("Diagnoses")
    c1, c2, c3 = st.columns(3)
    diag_1 = c1.selectbox("Primary Diagnosis Group", DIAG_GROUPS, index=0)
    diag_2 = c2.selectbox("Secondary Diagnosis Group", DIAG_GROUPS, index=0)
    diag_3 = c3.selectbox("Additional Diagnosis Group", DIAG_GROUPS, index=10)

    # ── Lab Results ───────────────────────────────────────────────────────────
    st.header("Lab Results")
    c1, c2 = st.columns(2)
    max_glu_serum = c1.selectbox("Max Glucose Serum", ["None", ">200", ">300", "Norm"])
    A1Cresult = c2.selectbox("A1C Result", ["None", ">7", ">8", "Norm"])

    # ── Medications ───────────────────────────────────────────────────────────
    st.header("Medications")
    med_cols = st.columns(4)
    meds = {}
    med_full = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "glipizide", "glyburide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "insulin",
    ]
    for i, med in enumerate(med_full):
        meds[med] = med_cols[i % 4].selectbox(med.capitalize(), MED_OPTIONS, index=1)

    med_limited = {
        "acetohexamide": "No", "tolbutamide": "No", "troglitazone": "No",
        "tolazamide": "No", "examide": "No", "citoglipton": "No",
        "glyburide-metformin": "No", "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No", "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
    }
    with st.expander("Combination / Less-common Medications"):
        lc_cols = st.columns(4)
        for i, (med, default) in enumerate(med_limited.items()):
            opts = MED_OPTIONS if med in ["glyburide-metformin", "tolazamide"] else MED_OPTIONS_LIMITED
            med_limited[med] = lc_cols[i % 4].selectbox(med, opts, index=0)

    c1, c2 = st.columns(2)
    change = c1.selectbox("Medication Change", ["No", "Ch"], format_func=lambda x: "No change" if x == "No" else "Changed")
    diabetesMed = c2.selectbox("On Diabetes Medication", ["Yes", "No"])

    # ── Predict ───────────────────────────────────────────────────────────────
    st.divider()
    if st.button("Predict Readmission Risk", type="primary", use_container_width=True):
        inputs = {
            "race": race,
            "gender": gender,
            "age": age,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "medical_specialty": medical_specialty,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": None if max_glu_serum == "None" else max_glu_serum,
            "A1Cresult": None if A1Cresult == "None" else A1Cresult,
            **meds,
            **med_limited,
            "change": change,
            "diabetesMed": diabetesMed,
        }

        X = build_input_row(inputs)
        prob = model.predict_proba(X)[0, 1]
        flagged = prob >= THRESHOLD

        st.subheader("Prediction Result")
        col_prob, col_risk = st.columns(2)

        col_prob.metric("Readmission Probability", f"{prob:.1%}")

        if flagged:
            col_risk.error("HIGH RISK — Likely readmission within 30 days")
        else:
            col_risk.success("LOW RISK — Readmission within 30 days unlikely")

        st.progress(float(prob), text=f"Risk score: {prob:.3f}  (threshold: {THRESHOLD})")


if __name__ == "__main__":
    main()
