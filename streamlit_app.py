# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
import os
from datetime import datetime

# ---------------- Config & CSS ----------------
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

# ---------------- Files & constants ----------------
PATIENTS_CSV = "patients.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# ---------------- Translations ----------------
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
        "suggestions_high": "- Seek immediate medical attention; contact a pediatric specialist or emergency services.\n- Possible diagnostic steps: imaging (MRI/MIBG) and biopsy. Treatments may include chemotherapy, immunotherapy, and investigational approaches (e.g., micro-patch vaccines) as determined by specialists.",
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
        "fill_patient_note": "📝 Please fill out patient information first.",
        "reset_results": "🔄 Reset Results"
    },
    "es": {
        "title": "🏥 Evaluador de Riesgo de Neuroblastoma",
        "disclaimer": "Esta herramienta es solo para fines informativos y educativos. No está destinada a proporcionar asesoramiento médico, diagnóstico o tratamiento. Consulte a un profesional de la salud.",
        "nutshell_title": "🧠 Neuroblastoma en pocas palabras",
        "nutshell_text": "El neuroblastoma es un cáncer infantil poco frecuente que surge de células nerviosas inmaduras del sistema nervioso simpático...",
        "major_symptoms": "🩺 Síntomas principales",
        "additional_symptoms": "➕ Síntomas adicionales",
        "predict_button": "🔍 Predecir Riesgo",
        "risk_low": "Riesgo Bajo",
        "risk_moderate": "Riesgo Moderado",
        "risk_high": "Riesgo Alto",
        "suggestions_low": "- Continúe con el monitoreo de rutina y visitas pediátricas.",
        "suggestions_moderate": "- Consulte a un pediatra para evaluación.",
        "suggestions_high": "- Busque atención médica inmediata; consulte a un especialista pediátrico.",
        "store_data": "📦 ¿Desea que se guarden sus datos? (aparecerán en Datos de Pacientes Anteriores)",
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
        "fill_patient_note": "📝 Por favor complete la información del paciente primero.",
        "reset_results": "🔄 Restablecer Resultados"
    },
    "fr": {
        "title": "🏥 Prédicteur de Risque de Neuroblastome",
        "disclaimer": "Cet outil est à des fins d'information et d'éducation uniquement. Il ne remplace pas un avis médical.",
        "nutshell_title": "🧠 Neuroblastome en bref",
        "nutshell_text": "Le neuroblastome est un cancer pédiatrique rare provenant de cellules nerveuses immatures du système nerveux sympathique...",
        "major_symptoms": "🩺 Symptômes principaux",
        "additional_symptoms": "➕ Symptômes supplémentaires",
        "predict_button": "🔍 Prédire le Risque",
        "risk_low": "Risque Faible",
        "risk_moderate": "Risque Modéré",
        "risk_high": "Risque Élevé",
        "suggestions_low": "- Surveillance de routine recommandée.",
        "suggestions_moderate": "- Consulter un pédiatre pour une évaluation.",
        "suggestions_high": "- Consulter d'urgence; contacter un spécialiste pédiatrique.",
        "store_data": "📦 Voulez-vous que vos données soient stockées ? (apparaîtront dans Données Patients)",
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
        "fill_patient_note": "📝 Veuillez remplir les informations du patient d'abord.",
        "reset_results": "🔄 Réinitialiser les résultats"
    }
}

# ---------------- Load model & scaler ----------------
@st.cache_resource
def load_model_and_scaler():
    # Loads model and scaler and returns (model, scaler, err)
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model file not found: {MODEL_PATH}"
    if not os.path.exists(SCALER_PATH):
        return None, None, f"Scaler file not found: {SCALER_PATH}"
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return None, None, f"Error loading model: {e}"
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        return None, None, f"Error loading scaler: {e}"
    return model, scaler, None

model, scaler, load_error = load_model_and_scaler()
if load_error:
    st.error(load_error)
    st.stop()

# ---------------- Helpers ----------------
def gender_to_numeric(g):
    # Encodes gender: Male=1, Female=0, Other=0
    if isinstance(g, str) and g.lower().startswith("m"):
        return 1
    return 0

def save_patient_row(row: dict):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(PATIENTS_CSV)
    df_row.to_csv(PATIENTS_CSV, mode="a", header=header, index=False)

def load_patients():
    if os.path.exists(PATIENTS_CSV):
        try:
            return pd.read_csv(PATIENTS_CSV)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ensure session_state keys
if "patients_df" not in st.session_state:
    st.session_state["patients_df"] = load_patients()
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "feedbacks" not in st.session_state:
    st.session_state["feedbacks"] = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🌐 Language")
    lang_display = st.selectbox("", options=["English", "Spanish", "French"], index=0)
    lang_map = {"English": "en", "Spanish": "es", "French": "fr"}
    lang = lang_map[lang_display]
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

# ---------------- Main ----------------
st.title(t["title"])
st.markdown(t["disclaimer"])
st.subheader(t["nutshell_title"])
st.write(t["nutshell_text"])
st.markdown("---")

# ------ Symptoms (checkboxes) ------
st.subheader(t["major_symptoms"])
maj_col1, maj_col2 = st.columns(2)
with maj_col1:
    s_lump = st.checkbox("Large/Medium lump (usually on abdomen, chest, or neck)", key="s_lump")
    s_abdominal_pain = st.checkbox("Abdominal pain", key="s_abdominal_pain")
    s_weight_loss = st.checkbox("Unexplained weight loss", key="s_weight_loss")
    s_bone_pain = st.checkbox("Bone Pain (usually followed by swelling, fever, and limping)", key="s_bone_pain")
with maj_col2:
    s_fatigue = st.checkbox("Fatigue / Weakness", key="s_fatigue")
    s_bulging_eyes = st.checkbox("Bulging or bruised eyes", key="s_bulging_eyes")
    s_constipation = st.checkbox("Constipation", key="s_constipation")
    s_aches = st.checkbox("Aches/Pain (usually in the leg causing limping)", key="s_aches")

st.markdown("---")
st.subheader(t["additional_symptoms"])
add_col1, add_col2 = st.columns(2)
with add_col1:
    s_fever = st.checkbox("Fever", key="s_fever")
    s_cough = st.checkbox("Cough", key="s_cough")
with add_col2:
    s_sore = st.checkbox("Sore Throat", key="s_sore")
    s_runny = st.checkbox("Runny / Stuffy nose", key="s_runny")

st.markdown("---")

# -------- Predict button --------
predict_clicked = st.button(t["predict_button"])
results_placeholder = st.empty()

def compute_and_store_result():
    # Build features in exact order:
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

    # Scale & predict
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[0]
    pred = model.predict(Xs)[0]

    neuro_prob = float(probs[1])
    confidence = float(np.max(probs))

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

    # compile result dict (including all symptoms)
    result = {
        "Name": patient_name or "(no name)",
        "Date": str(assessment_date),
        "Age": age,
        "Gender": gender,
        "Large_Medium_lump": int(s_lump),
        "Abdominal_pain": int(s_abdominal_pain),
        "Unexplained_weight_loss": int(s_weight_loss),
        "Fever": int(s_fever),
        "Fatigue_Weakness": int(s_fatigue),
        "Bulging_bruised_eyes": int(s_bulging_eyes),
        "Constipation": int(s_constipation),
        "Aches_Pain": int(s_aches),
        "Bone_Pain": int(s_bone_pain),
        "Cough": int(s_cough),
        "Runny_Stuffy_nose": int(s_runny),
        "Sore_Throat": int(s_sore),
        "Prediction": int(pred),
        "Probability_%": round(neuro_prob*100,2),
        "Confidence_%": round(confidence*100,2),
        "Risk": risk_level
    }

    # Save last result to session_state (so it persists)
    st.session_state["last_result"] = {
        "result": result,
        "dot_color": dot_color,
        "suggestion": suggestion,
        "neuro_prob": neuro_prob,
        "confidence": confidence
    }

# If user clicked predict, compute and store result
if predict_clicked:
    try:
        compute_and_store_result()
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

# Render results from session_state if present
if st.session_state.get("last_result") is not None:
    r = st.session_state["last_result"]
    res = r["result"]
    dot_color = r["dot_color"]
    suggestion = r["suggestion"]
    neuro_prob = r["neuro_prob"]
    confidence = r["confidence"]

    with results_placeholder.container():
        st.markdown("### 🔬 Prediction Results")
        c1, c2 = st.columns([3,1])
        with c1:
            st.markdown(f"<span class='risk-dot' style='background:{dot_color}'></span> **{res['Risk']}**", unsafe_allow_html=True)
            st.write(f"**Prediction:** {'Neuroblastoma' if res['Prediction']==1 else 'Not Neuroblastoma'}")
            st.write(f"**Probability:** {res['Probability_%']:.1f}%")
            st.markdown("**Suggestions:**")
            st.write(suggestion)
        with c2:
            st.markdown("**Model confidence**")
            st.markdown(f"<div style='font-weight:700; font-size:20px; color:#0b66c3'>{confidence*100:.1f}%</div>", unsafe_allow_html=True)
            st.progress(int(neuro_prob * 100))

        # Download single-assessment CSV
        single_df = pd.DataFrame([res])
        csv_bytes = single_df.to_csv(index=False).encode()
        st.download_button(t["download_csv"], data=csv_bytes,
                           file_name=f"assessment_{(res['Name'] or 'patient')}_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

        # Store data option (checkbox). If user checks, append to patients.csv and refresh sidebar list
        st.markdown("---")
        store = st.checkbox(t["store_data"], value=False, key="store_now")
        if store:
            try:
                save_patient_row(res)
                st.success("✅ Data stored.")
                st.session_state["patients_df"] = load_patients()
            except Exception as e:
                st.error(f"Error saving data: {e}")

        # Reset results button (clears only last_result)
        st.markdown("---")
        if st.button(t["reset_results"]):
            st.session_state["last_result"] = None
            # remove checkbox store_now state so it doesn't remain checked
            if "store_now" in st.session_state:
                del st.session_state["store_now"]
            results_placeholder.empty()
            st.info("Results cleared.")

        # Feedback subheading + large text area
        st.markdown("---")
        st.subheader(t["feedback"])
        feedback_text = st.text_area("🗒️ Feedback (optional) — share your thoughts or report issues", height=140, placeholder="Type your feedback here...")
        if st.button(t["submit_feedback"]):
            if feedback_text.strip():
                st.session_state["feedbacks"].append({
                    "timestamp": datetime.now().isoformat(),
                    "name": patient_name or "",
                    "feedback": feedback_text
                })
                st.success("Thanks for your feedback!")
                # optionally you can also write feedback to disk or DB here
            else:
                st.warning("Please enter feedback before submitting.")
else:
    results_placeholder.markdown("### 🔬 Prediction Results")
    results_placeholder.info("Fill symptoms and press **Predict Risk** to see results.")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<div class='footer'>© 2025 Neuroblastoma Risk Predictor | For educational use only — Contact: <a href='mailto:leonj062712@gmail.com'>leonj062712@gmail.com</a></div>",
    unsafe_allow_html=True
)
