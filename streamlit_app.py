# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
import os
from datetime import datetime

# --------- Configuration & CSS ----------
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
    """,
    unsafe_allow_html=True
)

# --------- Files & paths ----------
PATIENTS_CSV = "patients.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# --------- Translations (English filled; minimal placeholders for es/fr) ----------
translations = {
    "en": {
        "title": "🏥 Neuroblastoma Risk Predictor",
        "disclaimer": "This tool is for informational and educational purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider. If you experience severe symptoms, seek emergency care immediately.",
        "nutshell_title": "🧠 Neuroblastoma in a Nutshell",
        "nutshell_text": "Neuroblastoma is a rare childhood cancer arising from immature nerve cells of the sympathetic nervous system. It most often affects infants and young children and commonly presents with an abdominal mass, bone pain, or periorbital bruising.",
        "major_symptoms": "🩺 Major Symptoms",
        "additional_symptoms": "➕ Additional Symptoms",
        "predict_button": "🔍 Predict Risk",
        "risk_low": "Low Risk",
        "risk_moderate": "Moderate Risk",
        "risk_high": "High Risk",
        "suggestions_low": "- Continue routine monitoring and regular pediatric visits.\n- If symptoms change or worsen, seek medical advice.",
        "suggestions_moderate": "- Arrange prompt clinical evaluation with a pediatrician.\n- Consider imaging or referral to a specialist if recommended.",
        "suggestions_high": "- Seek immediate medical attention; contact a pediatric specialist or emergency services.\n- Bring a full symptom timeline and request appropriate diagnostic tests (imaging, labs, biopsy).",
        "store_data": "📦 Do you want your data stored? (will appear in Past Patient Data)",
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
        "download_all_csv": "Download all stored patients (CSV)",
        "fill_patient_note": "📝 Please fill out patient information first."
    },
    "es": {
        "title": "🏥 Evaluador de Riesgo de Neuroblastoma",
        "disclaimer": "Esta herramienta es solo para fines informativos y educativos. No proporciona diagnóstico médico.",
        "nutshell_title": "🧠 Neuroblastoma en pocas palabras",
        "nutshell_text": "El neuroblastoma es un cáncer infantil raro...",
        "major_symptoms": "🩺 Síntomas principales",
        "additional_symptoms": "➕ Síntomas adicionales",
        "predict_button": "🔍 Predecir Riesgo",
        "risk_low": "Riesgo Bajo",
        "risk_moderate": "Riesgo Moderado",
        "risk_high": "Riesgo Alto",
        "suggestions_low": "- Continúe con el monitoreo de rutina y visitas pediátricas.",
        "suggestions_moderate": "- Consulte a un pediatra para evaluación.",
        "suggestions_high": "- Busque atención médica inmediata.",
        "store_data": "📦 ¿Desea que se guarden sus datos? (aparecerán en Pacientes Anteriores)",
        "feedback": "🗒️ Comentarios",
        "submit_feedback": "Enviar Comentarios",
        "name_optional": "Nombre (opcional)",
        "age": "Edad (años)",
        "gender": "Género",
        "male": "Masculino",
        "female": "Femenino",
        "other": "Otro",
        "assessment_date": "Fecha de evaluación",
        "past_patient_data": "📁 Datos de Pacientes Anteriores",
        "download_csv": "📥 Descargar evaluación (CSV)",
        "download_all_csv": "Descargar todos (CSV)",
        "fill_patient_note": "📝 Por favor complete la información del paciente primero."
    },
    "fr": {
        "title": "🏥 Prédicteur de Risque de Neuroblastome",
        "disclaimer": "Cet outil est à des fins d'information et d'éducation seulement. Il ne remplace pas un avis médical.",
        "nutshell_title": "🧠 Neuroblastome en bref",
        "nutshell_text": "Le neuroblastome est un cancer pédiatrique rare...",
        "major_symptoms": "🩺 Symptômes principaux",
        "additional_symptoms": "➕ Symptômes supplémentaires",
        "predict_button": "🔍 Prédire le Risque",
        "risk_low": "Risque Faible",
        "risk_moderate": "Risque Modéré",
        "risk_high": "Risque Élevé",
        "suggestions_low": "- Surveillance de routine recommandée.",
        "suggestions_moderate": "- Consulter un pédiatre.",
        "suggestions_high": "- Consulter d'urgence.",
        "store_data": "📦 Voulez-vous que vos données soient stockées ?",
        "feedback": "🗒️ Retour",
        "submit_feedback": "Envoyer",
        "name_optional": "Nom (optionnel)",
        "age": "Âge (ans)",
        "gender": "Genre",
        "male": "Homme",
        "female": "Femme",
        "other": "Autre",
        "assessment_date": "Date d'évaluation",
        "past_patient_data": "📁 Données Patients",
        "download_csv": "📥 Télécharger l'évaluation (CSV)",
        "download_all_csv": "Télécharger tout (CSV)",
        "fill_patient_note": "📝 Veuillez remplir les informations du patient d'abord."
    }
}

# --------- Load model & scaler ----------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return None, None, f"Error loading model '{MODEL_PATH}': {e}"
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        return None, None, f"Error loading scaler '{SCALER_PATH}': {e}"
    return model, scaler, None

model, scaler, load_error = load_model_and_scaler()
if load_error:
    st.error(load_error)
    st.stop()

# --------- Helpers ----------
def gender_to_numeric(g):
    # Use Male=1, Female=0, Other=0 (change if your trained model uses a different encoding)
    return 1 if str(g).lower().startswith("m") else 0

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
    return pd.DataFrame()

# initialize session storage for showing immediate changes
if "patients_df" not in st.session_state:
    st.session_state["patients_df"] = load_patients()

# --------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🌐 Language")
    lang = st.selectbox("", options=["en", "es", "fr"], index=0)
    t = translations[lang]

    st.markdown("---")
    st.info(t["fill_patient_note"])

    st.header("🧍 " + t["name_optional"].split("(")[0].strip())
    patient_name = st.text_input(t["name_optional"], value="", key="patient_name")
    assessment_date = st.date_input(t["assessment_date"], value=datetime.now().date(), key="assessment_date")
    age = st.number_input(t["age"], min_value=0, max_value=120, value=5, step=1, key="age")
    gender = st.selectbox(t["gender"], options=[t["male"], t["female"], t["other"]], key="gender")

    st.markdown("---")
    st.markdown(f"### {t['past_patient_data']}")
    if not st.session_state["patients_df"].empty:
        st.dataframe(st.session_state["patients_df"].sort_values(by="Date", ascending=False).reset_index(drop=True), use_container_width=True)
        st.download_button(t["download_all_csv"], data=st.session_state["patients_df"].to_csv(index=False).encode(), file_name="patients.csv", mime="text/csv")
    else:
        st.info("No past patient data yet.")

# --------- MAIN ----------
st.title(t["title"])
st.markdown(t["disclaimer"])
st.subheader(t["nutshell_title"])
st.write(t["nutshell_text"])
st.markdown("---")

# --------- Symptoms (checkbox lists) ----------
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

# --------- Predict ----------
predict_clicked = st.button(t["predict_button"])
results_placeholder = st.empty()

if predict_clicked:
    # Build features array in EXACT order required by your model
    features = [
        age,
        gender_to_numeric(gender),
        int(s_lump),
        int(s_abdominal_pain),
        int(s_weight_loss),
        int(s_fever),
        int(s_fatigue),
        int(s_bulging_eyes),
        int(s_constipation),
        int(s_aches),
        int(s_bone_pain),
        int(s_cough),
        int(s_runny),
        int(s_sore)
    ]
    X = np.array([features], dtype=float)

    # Scale and predict
    try:
        Xs = scaler.transform(X)
    except Exception as e:
        st.error(f"Scaler transform error: {e}")
        st.stop()

    try:
        probs = model.predict_proba(Xs)[0]
        pred = model.predict(Xs)[0]
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        st.stop()

    neuro_prob = float(probs[1])  # probability of class 1 = neuroblastoma
    confidence = float(np.max(probs))

    # Risk categorization & color/suggestions
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

    # Render results
    with results_placeholder.container():
        st.markdown("### 🔬 Prediction Results")
        c1, c2 = st.columns([3,1])
        with c1:
            st.markdown(f"<span class='risk-dot' style='background:{dot_color}'></span> **{risk_level}**", unsafe_allow_html=True)
            st.write(f"**Prediction:** {'Neuroblastoma' if pred == 1 else 'Not Neuroblastoma'}")
            st.write(f"**Probability:** {neuro_prob*100:.1f}%")
            st.markdown("**Suggestions:**")
            st.write(suggestion)
        with c2:
            st.markdown("**Model confidence**")
            st.markdown(f"<div style='font-weight:700; font-size:20px; color:#0b66c3'>{confidence*100:.1f}%</div>", unsafe_allow_html=True)
            st.progress(int(neuro_prob * 100))

        # Download CSV for this single assessment
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

        # Store data option (checkbox)
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

        # Feedback subheading + large text area
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



