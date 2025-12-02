import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
import os
from datetime import datetime
from PIL import Image  # for scan handling
from tensorflow.keras.models import load_model  # adjust if you use another framework

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
    """,
    unsafe_allow_html=True
)

# ---------------- Files & constants ----------------
PATIENTS_CSV = "patients.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# NEW: imaging model path
SCAN_MODEL_PATH = "scan_model.h5"  # TODO: change if your imaging model file has a different name

# ---------------- Translations ----------------
translations = {
    "English": {
        "title": "🏥 Neuroblastoma Risk Predictor",
        "disclaimer": "**DISCLAIMER:** This tool is for informational and educational purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider.",
        "nutshell_title": "🧠 Neuroblastoma in a Nutshell",
        "nutshell_text": "Neuroblastoma is a rare childhood cancer arising from immature nerve cells of the sympathetic nervous system. It most often affects infants and young children and commonly presents with an abdominal mass, bone pain, or bulging eyes. Neuroblastoma is often detected at Stage 4 because its aggressive nature allows it to spread, or metastasize, to distant parts of the body, such as the bone marrow, liver, skin, and other organs, causing internal and external damage. This website helps to catch Neuroblastoma at early stages by using symptoms, checking for genetic changes, and by looking at scans. We recommend that you check your child's symptoms first and then go to labs or get scans.",
        "major_symptoms": "🩺 Major Symptoms",
        "rarer_symptoms": "🎗️ Rarer Symptoms",
        "additional_symptoms": "➕ Additional Symptoms",
        "lab_results_title": "🔬 Lab Results",
        "symptom_list": {
            "lump": "Large/Medium lump (usually on abdomen, chest, or neck)",
            "abdominal_pain": "Abdominal pain",
            "weight_loss": "Unexplained weight loss",
            "bone_pain": "Bone Pain (usually followed by swelling, fever, and limping)",
            "fatigue": "Fatigue / Weakness",
            "bulging_eyes": "Bulging or bruised eyes",
            "constipation": "Constipation",
            "aches": "Aches/Pain (usually in the leg causing limping)",
            "fever": "Fever",
            "cough": "Cough",
            "runny_nose": "Runny/Stuffy nose",
            "sore_throat": "Sore Throat",
            "unexplained_pain": "Unexplained pain",
            "high_bp": "High Blood Pressure (BP)",
            "vomiting": "Vomiting",
            "mycn": "MYCN Amplification",
            "alk": "ALK Mutation",
            "deletion_11q": "11q deletion",
            "gain_17q": "17q gain"
        },
        "predict_button": "🔍 Predict Risk",
        "risk_low": "Low Risk",
        "risk_mild": "Mild Risk",
        "risk_moderate": "Moderate Risk",
        "risk_high": "High Risk",
        "suggestions_low": "- Continue routine monitoring and regular pediatric visits.\n- If symptoms change or worsen, seek medical advice.",
        "suggestions_mild": "- Monitor symptoms closely and consult pediatrician if needed.\n- Consider early clinical evaluation.",
        "suggestions_moderate": "- Arrange prompt clinical evaluation with a pediatrician.\n- Consider imaging or referral to a specialist.\n- Early detection is critical for treatment.",
        "suggestions_high": "- Seek immediate medical attention; contact a pediatric specialist.\n- Consider getting a CT or MRI scan.\n- Possible treatments: chemotherapy, immunotherapy, or micro-patch vaccine trials.",
        "store_data": "📦 Do you want your data stored? (will appear in Past Patient Data)",
        "feedback": "🗒️ Feedback",
        "submit_feedback": "Submit Feedback",
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
        "risk_ref_title": "### 📊 Risk Levels Reference",
        "risk_ref_text": """• **0–34% → Low Risk** — Generally low probability of neuroblastoma.\n• **35–50% → Mild Risk** — Monitor closely; early clinical evaluation may be considered.\n• **51–74% → Moderate Risk** — May require further clinical evaluation.\n• **75–100% → High Risk** — Immediate medical assessment recommended.""",
        "prediction_results_section": {
            "title": "🔬 Prediction Results",
            "prediction": "Prediction",
            "probability": "Probability",
            "model_confidence": "Model Confidence",
            "suggestions": "Suggestions",
            "confidence_message": "confident this patient has"
        },
        # NEW imaging strings
        "scan_section_title": "🖼️ Scan Upload (Optional)",
        "scan_uploader_label": "Upload a scan image (JPG/PNG) — experimental imaging model",
        "scan_analyze_button": "🔍 Analyze Scan",
        "scan_model_not_available": "Imaging model not available on this server.",
        "scan_probability_label": "Scan model probability of neuroblastoma:",
        "scan_prediction_label": "Scan model prediction:",
        "scan_neuro_text": "Neuroblastoma",
        "scan_non_neuro_text": "No Neuroblastoma",
        "scan_combined_title": "🧮 Combined (experimental) risk",
        "scan_combined_note": "This combined risk is experimental and for research/education only. Always discuss results with a doctor."
    },
    "Spanish": {
        "title": "🏥 Predictor de Riesgo de Neuroblastoma",
        "disclaimer": "**DESCARGO DE RESPONSABILIDAD:** Esta herramienta es solo para fines informativos y educativos. No pretende ofrecer consejos médicos, diagnósticos o tratamientos. Siempre consulte a un profesional médico autorizado.",
        "nutshell_title": "🧠 Neuroblastoma en Resumen",
        "nutshell_text": "El neuroblastoma es un cáncer infantil poco común que surge de células nerviosas inmaduras del sistema nervioso simpático. Afecta con mayor frecuencia a bebés y niños pequeños y suele presentarse con una masa abdominal, dolor óseo o ojos abultados. El neuroblastoma suele detectarse en la etapa 4 debido a su naturaleza agresiva, que le permite propagarse (hacer metástasis) a partes distantes del cuerpo antes de que el tumor principal sea lo suficientemente grande como para causar síntomas locales notorios.",
        "major_symptoms": "🩺 Síntomas Principales",
        "rarer_symptoms": "🎗️ Síntomas más raros",
        "additional_symptoms": "➕ Síntomas Adicionales",
        "lab_results_title": "🔬 Resultados de Laboratorio",
        "symptom_list": {
            "lump": "Bulto grande/mediano (generalmente en abdomen, pecho o cuello)",
            "abdominal_pain": "Dolor abdominal",
            "weight_loss": "Pérdida de peso inexplicada",
            "bone_pain": "Dolor óseo (generalmente seguido de hinchazón, fiebre y cojera)",
            "fatigue": "Fatiga / Debilidad",
            "bulging_eyes": "Ojos abultados o moreteados",
            "constipation": "Estreñimiento",
            "aches": "Dolores (generalmente en la pierna causando cojera)",
            "fever": "Fiebre",
            "cough": "Tos",
            "runny_nose": "Nariz que moquea/congestionada",
            "sore_throat": "Dolor de garganta",
            "unexplained_pain": "Dolor inexplicable",
            "high_bp": "Presión arterial alta (PA)",
            "vomiting": "Vómitos",
            "mycn": "Amplificación MYCN",
            "alk": "Mutación ALK",
            "deletion_11q": "Deleción 11q",
            "gain_17q": "Ganancia 17q"
        },
        "predict_button": "🔍 Predecir Riesgo",
        "risk_low": "Bajo Riesgo",
        "risk_mild": "Riesgo Leve",
        "risk_moderate": "Riesgo Moderado",
        "risk_high": "Riesgo Alto",
        "suggestions_low": "- Continúe con revisiones pediátricas regulares.\n- Si los síntomas cambian o empeoran, busque consejo médico.",
        "suggestions_mild": "- Monitorear síntomas de cerca y consultar al pediatra si es necesario.\n- Considerar evaluación clínica temprana.",
        "suggestions_moderate": "- Evaluación clínica rápida con un pediatra.\n- Considerar imágenes médicas o derivación a un especialista.\n- La detección temprana es crucial para el tratamiento.",
        "suggestions_high": "- Buscar atención médica inmediata; contactar a un especialista pediátrico.\n- Considerar una tomografía o resonancia magnética.\n- Posibles tratamientos: quimioterapia, inmunoterapia o vacunas experimentales.",
        "store_data": "📦 ¿Desea guardar sus datos? (aparecerán en Datos de Pacientes Anteriores)",
        "feedback": "🗒️ Comentarios",
        "submit_feedback": "Enviar Comentarios",
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
        "risk_ref_title": "### 📊 Niveles de Riesgo",
        "risk_ref_text": """• **0–34% → Bajo** — Baja probabilidad de neuroblastoma.\n• **35–50% → Leve** — Monitorear de cerca; considerar evaluación clínica temprana.\n• **51–74% → Moderado** — Puede requerir evaluación médica adicional.\n• **75–100% → Alto** — Evaluación médica inmediata recomendada.""",
        "prediction_results_section": {
            "title": "🔬 Resultados de la Predicción",
            "prediction": "Predicción",
            "probability": "Probabilidad",
            "model_confidence": "Confianza del Modelo",
            "suggestions": "Sugerencias",
            "confidence_message": "de confianza en que este paciente tiene"
        },
        # NEW imaging strings
        "scan_section_title": "🖼️ Carga de Escán (Opcional)",
        "scan_uploader_label": "Sube una imagen de escán (JPG/PNG) — modelo de imágenes experimental",
        "scan_analyze_button": "🔍 Analizar Escán",
        "scan_model_not_available": "El modelo de imágenes no está disponible en este servidor.",
        "scan_probability_label": "Probabilidad de neuroblastoma según el modelo de escán:",
        "scan_prediction_label": "Predicción del modelo de escán:",
        "scan_neuro_text": "Neuroblastoma",
        "scan_non_neuro_text": "Sin neuroblastoma",
        "scan_combined_title": "🧮 Riesgo combinado (experimental)",
        "scan_combined_note": "Este riesgo combinado es experimental y solo para investigación/educación. Siempre consulte los resultados con un médico."
    },
    "French": {
        "title": "🏥 Prédicteur de Risque de Neuroblastome",
        "disclaimer": "**AVERTISSEMENT :** Cet outil est uniquement destiné à des fins d'information et d'éducation. Il ne remplace pas un avis médical professionnel. Consultez toujours un médecin qualifié.",
        "nutshell_title": "🧠 Le Neuroblastome en Bref",
        "nutshell_text": "Le neuroblastome est un cancer pédiatrique rare provenant des cellules nerveuses immatures du système nerveux sympathique. Il touche principalement les nourrissons et les jeunes enfants et se manifeste souvent par une masse abdominale, des douleurs osseuses ou des yeux saillants.",
        "major_symptoms": "🩺 Symptômes Majeurs",
        "rarer_symptoms": "🎗️ Symptômes plus rares",
        "additional_symptoms": "➕ Symptômes Supplémentaires",
        "lab_results_title": "🔬 Résultats de Laboratoire",
        "symptom_list": {
            "lump": "Masse grande/moyenne (généralement sur l'abdomen, la poitrine ou le cou)",
            "abdominal_pain": "Douleur abdominale",
            "weight_loss": "Perte de poids inexpliquée",
            "bone_pain": "Douleur osseuse (souvent suivie de gonflement, fièvre et boiterie)",
            "fatigue": "Fatigue / Faiblesse",
            "bulging_eyes": "Yeux saillants ou contusionnés",
            "constipation": "Constipation",
            "aches": "Douleurs (souvent dans la jambe causant une boiterie)",
            "fever": "Fièvre",
            "cough": "Toux",
            "runny_nose": "Nez qui coule/bouché",
            "sore_throat": "Mal de gorge",
            "unexplained_pain": "Douleur inexpliquée",
            "high_bp": "Hypertension artérielle (TA)",
            "vomiting": "Vomissements",
            "mycn": "Amplification MYCN",
            "alk": "Mutation ALK",
            "deletion_11q": "Délétion 11q",
            "gain_17q": "Gain 17q"
        },
        "predict_button": "🔍 Prédire le Risque",
        "risk_low": "Risque Faible",
        "risk_mild": "Risque Léger",
        "risk_moderate": "Risque Modéré",
        "risk_high": "Risque Élevé",
        "suggestions_low": "- Poursuivre la surveillance et les visites régulières.\n- Consulter un médecin si les symptômes changent.",
        "suggestions_mild": "- Surveiller les symptômes de près et consulter un pédiatre si nécessaire.\n- Envisager une évaluation clinique précoce.",
        "suggestions_moderate": "- Évaluation clinique rapide avec un pédiatre.\n- Envisager des examens d’imagerie.\n- La détection précoce est cruciale pour le traitement.",
        "suggestions_high": "- Consulter immédiatement un spécialiste pédiatrique.\n- Envisager une tomodensitométrie ou une IRM.\n- Traitements possibles : chimiothérapie, immunothérapie ou vaccins expérimentaux.",
        "store_data": "📦 Voulez-vous enregistrer les données ? (elles apparaîtront dans Données des Patients)",
        "feedback": "🗒️ Commentaires",
        "submit_feedback": "Soumettre",
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
        "risk_ref_title": "### 📊 Niveaux de Risque",
        "risk_ref_text": """• **0–34 % → Faible** — Probabilité faible de neuroblastoma.\n• **35–50 % → Léger** — Surveiller de près; envisager évaluation clinique précoce.\n• **51–74 % → Modéré** — Peut nécessiter une évaluation supplémentaire.\n• **75–100 % → Élevé** — Consultation médicale urgente recommandée.""",
        "prediction_results_section": {
            "title": "🔬 Résultats de la Prédiction",
            "prediction": "Prédiction",
            "probability": "Probabilité",
            "model_confidence": "Confiance du Modèle",
            "suggestions": "Suggestions",
            "confidence_message": "confiant que ce patient présente"
        },
        # NEW imaging strings
        "scan_section_title": "🖼️ Téléversement de Scan (Optionnel)",
        "scan_uploader_label": "Téléversez une image de scan (JPG/PNG) — modèle d’imagerie expérimental",
        "scan_analyze_button": "🔍 Analyser le Scan",
        "scan_model_not_available": "Le modèle d’imagerie n’est pas disponible sur ce serveur.",
        "scan_probability_label": "Probabilité de neuroblastome selon le modèle de scan :",
        "scan_prediction_label": "Prédiction du modèle de scan :",
        "scan_neuro_text": "Neuroblastome",
        "scan_non_neuro_text": "Pas de neuroblastome",
        "scan_combined_title": "🧮 Risque combiné (expérimental)",
        "scan_combined_note": "Ce risque combiné est expérimental et uniquement pour la recherche/éducation. Discutez toujours des résultats avec un médecin."
    }
}

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

# NEW: load imaging model
@st.cache_resource
def load_scan_model():
    if not os.path.exists(SCAN_MODEL_PATH):
        return None, f"Scan model file not found: {SCAN_MODEL_PATH}"
    # TODO: change this load method if you use PyTorch or another library
    scan_model = load_model(SCAN_MODEL_PATH)
    return scan_model, None

model, scaler, load_error = load_model_and_scaler()
scan_model, scan_load_error = load_scan_model()

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

# store last scan result
if "last_scan_result" not in st.session_state:
    st.session_state["last_scan_result"] = None

# ---------------- Sidebar ----------------
with st.sidebar:
    lang_display = st.selectbox("🌐 Website Language", options=["English", "Spanish", "French"], index=0)
    lang = lang_display
    t = translations[lang]

    st.markdown("---")
    st.info(t["fill_patient_note"])
    # Name removed – privacy-friendly

    assessment_date = st.date_input(t["assessment_date"], value=datetime.now().date(), key="assessment_date")
    age = st.number_input(t["age"], min_value=0, max_value=120, value=5, step=1, key="age")
    gender = st.selectbox(t["gender"], options=[t["male"], t["female"], t["other"]], key="gender")

    st.markdown("---")
    st.markdown(f"### {t['past_patient_data']}")
    if not st.session_state["patients_df"].empty:
        st.dataframe(st.session_state["patients_df"], use_container_width=True)
        st.download_button(
            t["download_all_csv"],
            data=st.session_state["patients_df"].to_csv(index=False).encode(),
            file_name="patients.csv",
            mime="text/csv"
        )
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
    s_lump = st.checkbox(t["symptom_list"]["lump"])
    s_abdominal_pain = st.checkbox(t["symptom_list"]["abdominal_pain"])
with maj_col2:
    s_weight_loss = st.checkbox(t["symptom_list"]["weight_loss"])
    s_constipation = st.checkbox(t["symptom_list"]["constipation"])

st.markdown("---")
st.subheader(t["rarer_symptoms"])
maj_col1, maj_col2 = st.columns(2)
with maj_col1:
    s_bone_pain = st.checkbox(t["symptom_list"]["bone_pain"])
with maj_col2:
    s_bulging_eyes = st.checkbox(t["symptom_list"]["bulging_eyes"])

st.markdown("---")
st.subheader(t["additional_symptoms"])
add_col1, add_col2, add_col3 = st.columns(3)
with add_col1:
    s_fever = st.checkbox(t["symptom_list"]["fever"])
    s_fatigue = st.checkbox(t["symptom_list"]["fatigue"])
    s_cough = st.checkbox(t["symptom_list"]["cough"])
with add_col2:
    s_runny = st.checkbox(t["symptom_list"]["runny_nose"])
    s_sore = st.checkbox(t["symptom_list"]["sore_throat"])
    s_aches = st.checkbox(t["symptom_list"]["aches"])
with add_col3:
    s_unexplained_pain = st.checkbox(t["symptom_list"]["unexplained_pain"])
    s_high_bp = st.checkbox(t["symptom_list"]["high_bp"])
    s_vomiting = st.checkbox(t["symptom_list"]["vomiting"])

# ------ Lab Results ------
st.markdown("---")
st.subheader(t["lab_results_title"])

genetics_not_checked = st.checkbox(
    "Not checked for genetic changes yet",
    value=False,
    help="If checked, the model will ignore MYCN, ALK, 11q, and 17q and only use symptoms."
)

lab_col1, lab_col2 = st.columns(2)
with lab_col1:
    s_mycn = st.checkbox(t["symptom_list"]["mycn"], disabled=genetics_not_checked)
    s_alk = st.checkbox(t["symptom_list"]["alk"], disabled=genetics_not_checked)
with lab_col2:
    s_11q = st.checkbox(t["symptom_list"]["deletion_11q"], disabled=genetics_not_checked)
    s_17q = st.checkbox(t["symptom_list"]["gain_17q"], disabled=genetics_not_checked)

# ------ Scan Upload Section (NEW) ------
st.markdown("---")
st.subheader(t["scan_section_title"])

uploaded_scan = st.file_uploader(
    t["scan_uploader_label"],
    type=["jpg", "jpeg", "png"]
)

if uploaded_scan is not None:
    st.image(uploaded_scan, caption="Uploaded scan", use_container_width=True)
    if scan_load_error or scan_model is None:
        st.warning(t["scan_model_not_available"])
    else:
        if st.button(t["scan_analyze_button"]):
            try:
                img = Image.open(uploaded_scan).convert("RGB")

                # TODO: adjust preprocessing to match YOUR imaging model
                img = img.resize((224, 224))
                img_arr = np.array(img) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)  # shape (1, H, W, C)

                # For a 2-class model: [p_no_neuro, p_neuro]
                scan_probs = scan_model.predict(img_arr)[0]
                scan_prob_neuro = float(scan_probs[1])
                scan_pred = int(scan_prob_neuro >= 0.5)

                scan_pred_text = t["scan_neuro_text"] if scan_pred == 1 else t["scan_non_neuro_text"]

                st.write(f"**{t['scan_probability_label']}** {scan_prob_neuro * 100:.1f}%")
                st.write(f"**{t['scan_prediction_label']}** {scan_pred_text}")

                st.session_state["last_scan_result"] = {
                    "prob_neuro": scan_prob_neuro,
                    "pred_text": scan_pred_text
                }

            except Exception as e:
                st.error(f"Scan analysis error: {e}")

# ---------------- Prediction button and results ----------------
st.markdown("---")
predict_clicked = st.button(t["predict_button"])
results_placeholder = st.empty()

def compute_and_store_result():
    # Genetics handling: if not checked, force genetics to 0 so model effectively uses symptoms only
    if genetics_not_checked:
        mycn_val = 0
        alk_val = 0
        q11_val = 0
        q17_val = 0
    else:
        mycn_val = int(s_mycn)
        alk_val = int(s_alk)
        q11_val = int(s_11q)
        q17_val = int(s_17q)

    features = [
        age,
        gender_to_numeric(gender),
        int(s_lump),
        int(s_abdominal_pain),
        int(s_weight_loss),
        int(s_constipation),
        int(s_bone_pain),
        int(s_bulging_eyes),
        int(s_fever),
        int(s_fatigue),
        int(s_cough),
        int(s_runny),
        int(s_sore),
        int(s_aches),
        int(s_unexplained_pain),
        int(s_high_bp),
        int(s_vomiting),
        mycn_val,
        alk_val,
        q11_val,
        q17_val
    ]

    X = np.array([features], dtype=float)
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[0]
    pred = model.predict(Xs)[0]

    neuro_prob = float(probs[1])
    non_neuro_prob = float(probs[0])

    # Confidence based on predicted class
    if pred == 1:  # Neuroblastoma
        confidence = neuro_prob * 100
        prediction_text = "Neuroblastoma"
    else:
        confidence = non_neuro_prob * 100
        prediction_text = "No Neuroblastoma"

    # Risk levels
    if neuro_prob <= 0.34:
        risk_level = t["risk_low"]
        dot_color = "#2ca02c"
        suggestion = t["suggestions_low"]
    elif neuro_prob <= 0.50:
        risk_level = t["risk_mild"]
        dot_color = "#ffc107"  # yellow
        suggestion = t["suggestions_mild"]
    elif neuro_prob <= 0.74:
        risk_level = t["risk_moderate"]
        dot_color = "#f0ad4e"
        suggestion = t["suggestions_moderate"]
    else:
        risk_level = t["risk_high"]
        dot_color = "#d62728"
        suggestion = t["suggestions_high"]

    result = {
        "Date": str(assessment_date),
        "Age": age,
        "Gender": gender,
        "Prediction": pred,
        "Prediction_Text": prediction_text,
        "Probability_%": round(neuro_prob * 100, 2),
        "Confidence_%": round(confidence, 2),
        "Risk": risk_level
    }

    st.session_state["last_result"] = {
        "result": result,
        "dot_color": dot_color,
        "suggestion": suggestion,
        "confidence": confidence,
        "neuro_prob": neuro_prob
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
    confidence = r["confidence"]
    neuro_prob = r["neuro_prob"]

    with results_placeholder.container():
        st.markdown(t["risk_ref_title"])
        st.markdown(t["risk_ref_text"])
        st.markdown("---")

        st.markdown("### 🔬 Prediction Results")
        st.markdown(
            f"<span class='risk-dot' style='background:{dot_color}'></span> **{res['Risk']}**",
            unsafe_allow_html=True
        )
        st.write(f"**Prediction:** {res['Prediction_Text']}")
        st.write(f"**Probability:** {res['Probability_%']:.1f}%")

        st.markdown("**Suggestions:**")
        st.write(suggestion)

        st.markdown("**Model confidence:**")
        st.progress(int(confidence))
        st.write(f"{confidence:.2f}% confident this patient has {res['Prediction_Text'].lower()}.")

        # If a scan result exists, show combined (experimental) risk
        if st.session_state.get("last_scan_result") is not None:
            scan_res = st.session_state["last_scan_result"]
            scan_prob_neuro = scan_res["prob_neuro"]

            combined_prob = (neuro_prob + scan_prob_neuro) / 2.0
            st.markdown("---")
            st.markdown(f"### {t['scan_combined_title']}")
            st.write(f"**Combined estimated probability (experimental):** {combined_prob * 100:.1f}%")
            st.write(t["scan_combined_note"])

        # Download CSV
        single_df = pd.DataFrame([res])
        st.download_button(
            t["download_csv"],
            data=single_df.to_csv(index=False).encode(),
            file_name=f"assessment_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        st.markdown("---")
        store = st.checkbox(t["store_data"], value=False)
        if store:
            save_patient_row(res)
            st.success("✅ Data stored.")
            st.session_state["patients_df"] = load_patients()

# ---------------- Feedback (above footer) ----------------
st.markdown("---")
st.subheader(t["feedback"])
feedback_text = st.text_area(
    "🗒️ Feedback (optional) — share your thoughts or report issues",
    height=140,
    placeholder="Type your feedback here..."
)
if st.button(t["submit_feedback"]):
    if feedback_text.strip():
        st.success("Thanks for your feedback!")
    else:
        st.warning("Please enter feedback before submitting.")

st.markdown("---")
st.markdown(
    "<div class='footer'>© 2025 Neuroblastoma Risk Predictor | Contact: "
    "<a href='mailto:leonj062712@gmail.com'>leonj062712@gmail.com</a></div>",
    unsafe_allow_html=True
)


