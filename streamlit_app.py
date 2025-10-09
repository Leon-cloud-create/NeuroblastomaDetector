# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
from datetime import datetime
import os

# ---------- Config & CSS ----------
st.set_page_config(page_title="🏥 Neuroblastoma Risk Predictor", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; color: #0b2a4a; }
    .card { background: #f8fafc; padding: 12px; border-radius: 8px; }
    div.stButton > button:first-child {
        width: 100%;
        background-color: #0b66c3;
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.85em 0;
    }
    div.stButton > button:first-child:hover { background-color: #094c8d; }
    .risk-dot { display:inline-block; width:18px; height:18px; border-radius:50%; margin-right:8px; vertical-align:middle; }
    .footer { text-align:center; color:gray; padding:10px 0; margin-top:18px; }
    .small-muted { color:#6b7280; font-size:13px; }
    </style>
    """, unsafe_allow_html=True
)

# ---------- Files & persistence ----------
PATIENTS_CSV = "patients.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# ---------- Translations ----------
translations = {
    "en": {
        "title": "🏥 Neuroblastoma Risk Predictor",
        "disclaimer": "This tool is for informational and educational purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider. Severe symptoms require emergency care.",
        "nutshell_title": "🧠 Neuroblastoma in a Nutshell",
        "nutshell_text": "Neuroblastoma is a rare childhood cancer arising from immature nerve cells of the sympathetic nervous system. It most often affects infants and young children and commonly presents with an abdominal mass, bone pain, or bruised eyes.",
        "major_symptoms": "🩺 Major Symptoms",
        "additional_symptoms": "➕ Additional Symptoms",
        "predict_button": "🔍 Predict Risk",
        "risk_low": "Low Risk",
        "risk_moderate": "Moderate Risk",
        "risk_high": "High Risk",
        "suggestions_low": "- Continue routine monitoring and regular pediatric visits.\n- If symptoms change or worsen, seek medical advice.",
        "suggestions_moderate": "- Arrange prompt clinical evaluation with a pediatrician.\n- Consider imaging or referral to a specialist if recommended.",
        "suggestions_high": "- Seek immediate medical attention; contact a pediatric specialist or emergency services.\n- Bring a full symptom timeline and request appropriate diagnostic tests.",
        "store_data": "📦 Do you want your data stored? (will appear in Past Patient Data and help future patients)",
        "feedback": "🗒️ Feedback",
        "submit_feedback": "Submit Feedback",
        "name_optional": "Name (optional)",
        "age": "Age (years)",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "other": "Other",
        "assessment_date": "Assessment Date",
        "past_patient_data": "📁 Past Patient Data",
        "download_csv": "📥 Download assessment CSV",
        "download_all_csv": "Download all stored patients (CSV)"
    },
    "es": {},
    "fr": {}
}

# ---------- Load model & scaler ----------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, load_error = load_model_and_scaler()
if load_error:
    st.error(f"Model/scaler load error: {load_error}")
    st.stop()

# ---------- helper functions ----------
def gender_to_numeric(g):
    return 1 if g.lower().startswith("m") else 0

def save_patient_row(row: dict):
    df_row = pd.DataFrame([row])
    if os.path.exists(PATIENTS_CSV):
        df_row.to_csv(PATIENTS_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(PATIENTS_CSV, index=False)

def load_patients():
    if os.path.exists(PATIENTS_CSV):
        try:
            return pd.read_csv(PATIENTS_CSV)
        except Exception:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

if "patients_df" not in st.session_state:
    st.session_state["patients_df"] = load_patients()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🌐 Language")
    lang = st.selectbox("", options=["en","es","fr"], index=0)
    t = translations[lang]

    st.markdown("---")
    st.info("📝 Please fill out patient information first.")
    st.header("🧍 Patient Information")
    patient_name = st.text_input(t["name_optional"])
    assessment_date = st.date_input(t["assessment_date"], value=datetime.now().date())
    age = st.number_input(t["age"], min_value=0, max_value=120, value=5, step=1)
    gender = st.selectbox(t["gender"], options=[t["male"], t["female"], t["other"]])

    st.markdown("---")
    st.markdown(f"### {t['past_patient_data']}")
    if not st.session_state["patients_df"].empty:
        st.dataframe(st.session_state["patients_df"].sort_values(by="Date", ascending=False).reset_index(drop=True), use_container_width=True)
        st.download_button(t["download_all_csv"], data=st.session_state["patients_df"].to_csv(index=False).encode(), file_name="patients.csv", mime="text/csv")
    else:
        st.info("No past patient data yet.")

# ---------- MAIN ----------
st.title(t["title"])
st.markdown(t["disclaimer"])
st.subheader(t["nutshell_title"])
st.write(t["nutshell_text"])
st.markdown("---")

# ---------- Symptoms ----------
st.subheader(t["major_symptoms"])
maj_col1, maj_col2 = st.columns(2)
with maj_col1:
    s_lump = st.checkbox("Large/Medium lump (usually on abdomen, chest, or neck)")
    s_abdominal_pain = st.checkbox("Abdominal pain")
    s_weight_loss = st.checkbox("Unexplained weight loss")
    s_bone_pain = st.checkbox("Bone Pain (usually followed by swelling, fever, and limping)")
with maj_col2:
    s_fatigue = st.checkbox("Fatigue / Weakness")
    s_bulging_eyes = st.checkbox("Bulging or bruised eyes")
    s_constipation = st.checkbox("Constipation")
    s_aches = st.checkbox("Aches/Pain (usually in the leg causing limping)")

st.markdown("---")
st.subheader(t["additional_symptoms"])
add_col1, add_col2 = st.columns(2)
with add_col1:
    s_fever = st.checkbox("Fever")
    s_cough = st.checkbox("Cough")
with add_col2:
    s_sore = st.checkbox("Sore Throat")
    s_runny = st.checkbox("Runny / Stuffy nose")

st.markdown("---")
predict_clicked = st.button(t["predict_button"])
results_placeholder = st.empty()

if predict_clicked:
    # Build features array dynamically
    features = [age, gender_to_numeric(gender)]
    features += [int(s) for s in [
        s_lump, s_abdominal_pain, s_weight_loss, s_fever,
        s_fatigue, s_bulging_eyes, s_constipation, s_aches,
        s_bone_pain, s_cough, s_runny, s_sore
    ]]
    features = np.array([features], dtype=float)

    try:
        features_scaled = scaler.transform(features)
        proba = model.predict_proba(features_scaled)[0]
        pred = model.predict(features_scaled)[0]
    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()

    neuro_prob = float(proba[1])
    confidence = float(np.max(proba))

    if neuro_prob <= 0.30:
        risk_level = t["risk_low"]
        dot_color = "#2ca02c"
        suggestion = t["suggestions_low"]
    elif neuro_prob <= 0.70:
        risk_level = t["risk_moderate"]
        dot_color = "#f0ad4e"
        suggestion = t["suggestions_moderate"]
    else:
        risk_level = t["risk_high"]
        dot_color = "#d62728"
        suggestion = t["suggestions_high"]

    with results_placeholder.container():
        st.markdown("### 🔬 Prediction Results")
        c1, c2 = st.columns([3,1])
        with c1:
            st.markdown(f"<span class='risk-dot' style='background:{dot_color}'></span> **{risk_level}**", unsafe_allow_html=True)
            st.write(f"**Prediction:** {'Neuroblastoma' if pred==1 else 'Not Neuroblastoma'}")
            st.write(f"**Probability:** {neuro_prob*100:.1f}%")
            st.markdown("**Suggestions:**")
            st.write(suggestion)
        with c2:
            st.markdown("**Model confidence**")
            st.markdown(f"<div style='font-weight:700; font-size:20px; color:#0b66c3'>{confidence*100:.1f}%</div>", unsafe_allow_html=True)
            st.progress(int(neuro_prob*100))

        # CSV download
        result_df = pd.DataFrame([{
            "Name": patient_name or "(no name)",
            "Date": str(assessment_date),
            "Age": age,
            "Gender": gender,
            "Prediction": int(pred),
            "Probability_%": round(neuro_prob*100,2),
            "Confidence_%": round(confidence*100,2),
            "Risk": risk_level
        }])
        csv_bytes = result_df.to_csv(index=False).encode()

        st.download_button(t["download_csv"], data=csv_bytes,
                           file_name=f"assessment_{(patient_name or 'patient')}_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

        # ---------- Store data checkbox ----------
        st.markdown("---")
        store = st.checkbox(t["store_data"], value=False)
        if store:
            row = {
                "Name": patient_name or "(no name)",
                "Date": str(assessment_date),
                "Age": age,
                "Gender": gender,
                "Prediction": int(pred),
                "Probability_%": round(neuro_prob*100,2),
                "Confidence_%": round(confidence*100,2),
                "Risk": risk_level
            }
            try:
                save_patient_row(row)
                st.success("✅ Data stored.")
                st.session_state["patients_df"] = load_patients()
            except Exception as e:
                st.error(f"Error saving data: {e}")

        # ---------- Feedback ----------
        st.markdown("---")
        st.subheader(t["feedback"])
        feedback_text = st.text_area("🗒️ Feedback (optional) — share your thoughts or report issues", height=140, placeholder="Type your feedback here...")
        if st.button(t["submit_feedback"]):
            if feedback_text.strip():
                st.success("Thanks for your feedback!")
            else:
                st.warning("Please enter feedback before submitting.")

else:
    results_placeholder.markdown("### 🔬 Prediction Results")
    results_placeholder.info("Fill symptoms and press **Predict Risk** to see results.")

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<div class='footer'>© 2025 Neuroblastoma Risk Predictor | For educational use only — Contact: <a href='mailto:leonj062712@gmail.com'>leonj062712@gmail.com</a></div>",
    unsafe_allow_html=True
)


