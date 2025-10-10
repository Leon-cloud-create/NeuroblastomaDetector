# app.py (with Spanish & French + reduced top padding)
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
    .stApp { background-color: #ffffff; color: #0b2a4a; margin-top: -40px !important; } /* reduces top gap */
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
    "English": {
        "title": "🏥 Neuroblastoma Risk Predictor",
        "disclaimer": "**DISCLAIMER:** This tool is for informational and educational purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider.",
        "nutshell_title": "🧠 Neuroblastoma in a Nutshell",
        "nutshell_text": "Neuroblastoma is a rare childhood cancer arising from immature nerve cells of the sympathetic nervous system. It most often affects infants and young children and commonly presents with an abdominal mass, bone pain, or bulging eyes. Neuroblastoma is often detected at Stage 4 because its aggressive nature allows it to spread, or metastasize, to distant parts of the body, such as the bone marrow, liver, skin, and other organs, before the primary tumor grows large enough to cause noticeable local symptoms.",
        "major_symptoms": "🩺 Major Symptoms",
        "additional_symptoms": "➕ Additional Symptoms",
        "predict_button": "🔍 Predict Risk",
        "risk_low": "Low Risk",
        "risk_moderate": "Moderate Risk",
        "risk_high": "High Risk",
        "suggestions_low": "- Continue routine monitoring and regular pediatric visits.\n- If symptoms change or worsen, seek medical advice.",
        "suggestions_moderate": "- Arrange prompt clinical evaluation with a pediatrician.\n- Consider imaging or referral to a specialist.\n- Early detection is critical for treatment.",
        "suggestions_high": "- Seek immediate medical attention; contact a pediatric specialist.\n- Consider getting a CT or MRI scan.\n- Possible treatments: chemotherapy, immunotherapy, or micro-patch vaccine trials.",
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
        "risk_ref": "### 📊 Risk Levels Reference\n\n• **0–60% → Low Risk** — Generally low probability of neuroblastoma.\n• **61–80% → Moderate Risk** — May require further clinical evaluation.\n• **81–100% → High Risk** — Immediate medical assessment recommended."
    },
    "Spanish": {
        "title": "🏥 Predictor de Riesgo de Neuroblastoma",
        "disclaimer": "**DESCARGO DE RESPONSABILIDAD:** Esta herramienta es solo para fines informativos y educativos. No pretende proporcionar consejos médicos, diagnóstico o tratamiento. Siempre consulte a un profesional médico autorizado.",
        "nutshell_title": "🧠 Neuroblastoma en Resumen",
        "nutshell_text": "El neuroblastoma es un cáncer infantil poco común que surge de células nerviosas inmaduras del sistema nervioso simpático. A menudo afecta a bebés y niños pequeños, y se presenta con una masa abdominal, dolor óseo o ojos abultados. Frecuentemente se detecta en la Etapa 4 debido a su naturaleza agresiva, que le permite propagarse antes de que el tumor primario crezca lo suficiente como para causar síntomas locales evidentes.",
        "major_symptoms": "🩺 Síntomas Principales",
        "additional_symptoms": "➕ Síntomas Adicionales",
        "predict_button": "🔍 Predecir Riesgo",
        "risk_low": "Riesgo Bajo",
        "risk_moderate": "Riesgo Moderado",
        "risk_high": "Riesgo Alto",
        "suggestions_low": "- Continuar con visitas pediátricas regulares.\n- Si los síntomas cambian o empeoran, busque atención médica.",
        "suggestions_moderate": "- Evaluación clínica rápida con un pediatra.\n- Considerar imágenes médicas o derivación a un especialista.",
        "suggestions_high": "- Buscar atención médica inmediata; contactar un especialista pediátrico.\n- Posibles tratamientos: quimioterapia, inmunoterapia o vacunas experimentales.",
        "store_data": "📦 ¿Desea guardar sus datos? (aparecerán en Datos de Pacientes Anteriores)",
        "feedback": "🗒️ Comentarios",
        "submit_feedback": "Enviar Comentarios",
        "name_optional": "Nombre (opcional)",
        "age": "Edad (años)",
        "gender": "Género",
        "male": "Masculino",
        "female": "Femenino",
        "other": "Otro",
        "assessment_date": "Fecha de Evaluación",
        "past_patient_data": "📁 Datos de Pacientes Anteriores",
        "download_csv": "📥 Descargar Evaluación (CSV)",
        "download_all_csv": "Descargar todos los pacientes (CSV)",
        "fill_patient_note": "📝 Por favor complete primero la información del paciente.",
        "risk_ref": "### 📊 Niveles de Riesgo\n\n• **0–60% → Bajo** — Baja probabilidad de neuroblastoma.\n• **61–80% → Moderado** — Puede requerir evaluación médica.\n• **81–100% → Alto** — Evaluación médica inmediata recomendada."
    },
    "French": {
        "title": "🏥 Prédicteur de Risque de Neuroblastome",
        "disclaimer": "**AVERTISSEMENT :** Cet outil est uniquement destiné à des fins d'information et d'éducation. Il ne remplace pas un avis médical professionnel. Consultez toujours un médecin qualifié.",
        "nutshell_title": "🧠 Le Neuroblastome en Bref",
        "nutshell_text": "Le neuroblastome est un cancer pédiatrique rare provenant des cellules nerveuses immatures du système nerveux sympathique. Il touche principalement les nourrissons et les jeunes enfants et se manifeste souvent par une masse abdominale, des douleurs osseuses ou des yeux saillants.",
        "major_symptoms": "🩺 Symptômes Majeurs",
        "additional_symptoms": "➕ Symptômes Supplémentaires",
        "predict_button": "🔍 Prédire le Risque",
        "risk_low": "Risque Faible",
        "risk_moderate": "Risque Modéré",
        "risk_high": "Risque Élevé",
        "suggestions_low": "- Poursuivre la surveillance et les visites régulières.\n- Consulter un médecin si les symptômes changent.",
        "suggestions_moderate": "- Évaluation clinique rapide avec un pédiatre.\n- Envisager des examens d’imagerie.",
        "suggestions_high": "- Consulter immédiatement un spécialiste pédiatrique.\n- Possibles traitements : chimiothérapie, immunothérapie ou vaccins expérimentaux.",
        "store_data": "📦 Voulez-vous enregistrer les données ? (elles apparaîtront dans Données des Patients)",
        "feedback": "🗒️ Commentaires",
        "submit_feedback": "Soumettre",
        "name_optional": "Nom (optionnel)",
        "age": "Âge (années)",
        "gender": "Genre",
        "male": "Homme",
        "female": "Femme",
        "other": "Autre",
        "assessment_date": "Date d'Évaluation",
        "past_patient_data": "📁 Données des Patients",
        "download_csv": "📥 Télécharger l'Évaluation (CSV)",
        "download_all_csv": "Télécharger tous les patients (CSV)",
        "fill_patient_note": "📝 Veuillez d'abord remplir les informations du patient.",
        "risk_ref": "### 📊 Niveaux de Risque\n• **0–60 % → Faible** — Probabilité faible.\n• **61–80 % → Modéré** — Nécessite une évaluation.\n• **81–100 % → Élevé** — Consultation médicale urgente."
    }
}

# ---------------- Model loader, helpers, main code (same as yours) ----------------
# ✅ You can now paste all your remaining code (model loading, UI, predict, feedback, footer) unchanged.


# ---------------- Load model & scaler ----------------
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model file not found: {MODEL_PATH}"
    if not os.path.exists(SCALER_PATH):
        return None, None, f"Scaler file not found: {SCALER_PATH}"
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, None

model, scaler, load_error = load_model_and_scaler()
if load_error:
    st.error(load_error)
    st.stop()

# ---------------- Helpers ----------------
def gender_to_numeric(g):
    return 1 if isinstance(g, str) and g.lower().startswith("m") else 0

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

if "patients_df" not in st.session_state:
    st.session_state["patients_df"] = load_patients()
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🌐 Website Language")
    lang_display = st.selectbox("", options=["English, Spanish, French"], index=0)
    lang = "English"
    t = translations[lang]

    st.markdown("---")
    st.info(t["fill_patient_note"])
    patient_name = st.text_input(t["name_optional"], value="", key="patient_name")
    assessment_date = st.date_input(t["assessment_date"], value=datetime.now().date(), key="assessment_date")
    age = st.number_input(t["age"], min_value=0, max_value=120, value=5, step=1, key="age")
    gender = st.selectbox(t["gender"], options=[t["male"], t["female"], t["other"]], key="gender")

    st.markdown("---")
    st.markdown(f"### {t['past_patient_data']}")
    if not st.session_state["patients_df"].empty:
        st.dataframe(st.session_state["patients_df"], use_container_width=True)
        st.download_button(t["download_all_csv"], data=st.session_state["patients_df"].to_csv(index=False).encode(),
                           file_name="patients.csv", mime="text/csv")
    else:
        st.info("No past patient data yet.")

# ---------------- Main ----------------
st.title(t["title"])
st.markdown(t["disclaimer"])
st.subheader(t["nutshell_title"])
st.write(t["nutshell_text"])
st.markdown("---")

# ------ Symptoms ------
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

def compute_and_store_result():
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
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[0]
    pred = model.predict(Xs)[0]
    neuro_prob = float(probs[1])
    confidence = float(np.max(probs))

    # Updated thresholds: 0–34 low, 35–69 moderate, 70–100 high
    if neuro_prob <= 0.34:
        risk_level = t["risk_low"]
        dot_color = "#2ca02c"
        suggestion = t["suggestions_low"]
    elif neuro_prob <= 0.69:
        risk_level = t["risk_moderate"]
        dot_color = "#f0ad4e"
        suggestion = t["suggestions_moderate"]
    else:
        risk_level = t["risk_high"]
        dot_color = "#d62728"
        suggestion = t["suggestions_high"]

    result = {
        "Name": patient_name or "(no name)",
        "Date": str(assessment_date),
        "Age": age,
        "Gender": gender,
        "Prediction": int(pred),
        "Probability_%": round(neuro_prob*100,2),
        "Confidence_%": round(confidence*100,2),
        "Risk": risk_level
    }
    st.session_state["last_result"] = {
        "result": result,
        "dot_color": dot_color,
        "suggestion": suggestion,
        "neuro_prob": neuro_prob,
        "confidence": confidence
    }

if predict_clicked:
    try:
        compute_and_store_result()
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

if st.session_state.get("last_result"):
    r = st.session_state["last_result"]
    res = r["result"]
    dot_color = r["dot_color"]
    suggestion = r["suggestion"]
    neuro_prob = r["neuro_prob"]
    confidence = r["confidence"]

    with results_placeholder.container():
        # ✅ Show risk levels card first
        st.markdown(t["risk_ref"])
        st.markdown("---")

        st.markdown("### 🔬 Prediction Results")
        st.markdown(f"<span class='risk-dot' style='background:{dot_color}'></span> **{res['Risk']}**", unsafe_allow_html=True)
        st.write(f"**Prediction:** {'Neuroblastoma' if res['Prediction']==1 else 'Not Neuroblastoma'}")
        st.write(f"**Probability:** {res['Probability_%']:.1f}%")
        st.markdown("**Suggestions:**")
        st.write(suggestion)

        # Get prediction probabilities
proba = model.predict_proba(user_data)[0]
non_neuro_prob = proba[0]
neuro_prob = proba[1]

# Make prediction
prediction = model.predict(user_data)[0]

# Show confidence based on which one was predicted
if prediction == "Neuroblastoma":
    confidence = neuro_prob * 100
else:
    confidence = non_neuro_prob * 100

# Display results
st.markdown("**Model confidence:**")
st.progress(int(confidence))
st.write(f"{confidence:.2f}% confident this patient has {prediction.lower()}.")


        single_df = pd.DataFrame([res])
        st.download_button(t["download_csv"], data=single_df.to_csv(index=False).encode(),
                           file_name=f"assessment_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

        st.markdown("---")
        store = st.checkbox(t["store_data"], value=False)
        if store:
            save_patient_row(res)
            st.success("✅ Data stored.")
            st.session_state["patients_df"] = load_patients()

# ---------------- Feedback (above footer) ----------------
st.markdown("---")
st.subheader(t["feedback"])
feedback_text = st.text_area("🗒️ Feedback (optional) — share your thoughts or report issues", height=140, placeholder="Type your feedback here...")
if st.button(t["submit_feedback"]):
    if feedback_text.strip():
        st.success("Thanks for your feedback!")
    else:
        st.warning("Please enter feedback before submitting.")

st.markdown("---")
st.markdown("<div class='footer'>© 2025 Neuroblastoma Risk Predictor | Contact: <a href='mailto:leonj062712@gmail.com'>leonj062712@gmail.com</a></div>", unsafe_allow_html=True)



