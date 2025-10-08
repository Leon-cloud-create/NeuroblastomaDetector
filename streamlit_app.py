import streamlit as st
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from datetime import datetime
import io
import pandas as pd

# -------------------------
# Basic page setup & CSS
# -------------------------
st.set_page_config(page_title="Neuroblastoma Risk Assessment Tool", page_icon="🎗️🏥", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container { background: #ffffff; }
    .stApp { background: #ffffff; color: #0b2a4a; }
    .card { background: #f8fafc; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(16,24,40,0.06); }
    .blue-value { color: #0b66c3; font-weight: 700; font-size: 24px; }
    .small-muted { color: #6b7280; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Translations
# -------------------------
translations = {
    "en": {
        "title": "🏥 Neuroblastoma Risk Assessment Tool",
        "language_label": "Language",
        "disclaimer_title": "IMPORTANT MEDICAL DISCLAIMER",
        "disclaimer": (
            "This tool is for informational and educational purposes only. "
            "It is not intended to provide medical advice, diagnosis, or treatment. "
            "Always consult a licensed healthcare provider for medical concerns. "
            "If you experience severe symptoms, seek emergency care immediately. "
            "The AI model provides probabilistic insights and should never be interpreted as definitive."
        ),
        "nutshell_title": "Neuroblastoma in a Nutshell",
        "nutshell_text": (
            "Neuroblastoma is a rare childhood cancer arising from immature nerve cells in the "
            "sympathetic nervous system. Early awareness of key symptoms (abdominal mass, bone pain, periorbital bruising, unexplained weight loss) "
            "can prompt timely medical evaluation."
        ),
        "patient_name": "Patient Name (optional)",
        "assessment_date": "Assessment Date",
        "age": "Age (years)",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "predict": "🔍 Predict Risk",
        "symptoms_header": "Major Symptoms (select yes/no)",
        "lump": "Large/Medium lump (abdomen/chest/neck)",
        "abdominal_pain": "Abdominal pain",
        "weight_loss": "Unexplained weight loss",
        "fever": "Fever",
        "fatigue": "Fatigue / Weakness",
        "bruised_eyes": "Periorbital bruising / Bulging or bruised eyes",
        "constipation": "Constipation",
        "aches": "Aches / Bone pain",
        "cough": "Cough",
        "runny": "Runny / Stuffy nose",
        "sore": "Sore throat",
        "result_title": "Prediction Results",
        "neuroblastoma": "Neuroblastoma",
        "not_neuroblastoma": "Not Neuroblastoma",
        "risk_low": "Low Risk (0–30%)",
        "risk_moderate": "Moderate Risk (30–70%)",
        "risk_high": "High Risk (70–100%)",
        "confidence": "Model Confidence"
    }
}

# -------------------------
# Load model & scaler
# -------------------------
@st.cache_resource
def load_model_and_scaler():
    scaler_path = "scaler.pkl"
    model_path = "model.pkl"
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        return None, None, f"Missing files: {', '.join([p for p in [scaler_path, model_path] if not os.path.exists(p)])}"
    try:
        with open(scaler_path, "rb") as f:
            sc = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return sc, model, None
    except Exception as e:
        return None, None, str(e)

scaler, model, load_error = load_model_and_scaler()
if load_error:
    st.error(f"Model load error: {load_error}")
    st.stop()

# -------------------------
# Helper functions
# -------------------------
def prepare_input(age, gender, lump, abdominal_pain, weight_loss, fever, fatigue, bruised_eyes, constipation, aches, cough, runny, sore):
    return np.array([[age, gender, lump, abdominal_pain, weight_loss, fever, fatigue, bruised_eyes, constipation, aches, cough, runny, sore]])

def categorize_risk(prob):
    if prob < 0.30:
        return "Low"
    elif prob < 0.70:
        return "Moderate"
    else:
        return "High"

def risk_label_text(prob, t):
    cat = categorize_risk(prob)
    if cat == "Low":
        return t["risk_low"]
    elif cat == "Moderate":
        return t["risk_moderate"]
    else:
        return t["risk_high"]

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Patient Info")
    patient_name = st.text_input(translations["en"]["patient_name"], "")
    assessment_date = st.date_input(translations["en"]["assessment_date"], value=datetime.now().date())
    age = st.number_input(translations["en"]["age"], min_value=0, max_value=120, value=5, step=1)
    gender_text = st.selectbox(translations["en"]["gender"], options=[0,1], format_func=lambda x: translations["en"]["female"] if x==1 else translations["en"]["male"])
    st.markdown("---")
    st.write("Version: 1.0")
    st.write("Developer: You")
    st.markdown("[St. Jude Neuroblastoma Info](https://www.stjude.org/disease/neuroblastoma.html)")

# -------------------------
# Main content
# -------------------------
st.title(translations["en"]["title"])
st.markdown(f"**{translations['en']['disclaimer_title']}**")
st.info(translations["en"]["disclaimer"])
st.subheader(translations["en"]["nutshell_title"])
st.write(translations["en"]["nutshell_text"])
st.markdown("---")

# Symptoms input
st.header(translations["en"]["symptoms_header"])
col1, col2 = st.columns(2)

with col1:
    lump = st.selectbox(translations["en"]["lump"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    abdominal_pain = st.selectbox(translations["en"]["abdominal_pain"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    weight_loss = st.selectbox(translations["en"]["weight_loss"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    aches = st.selectbox(translations["en"]["aches"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    constipation = st.selectbox(translations["en"]["constipation"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")

with col2:
    fever = st.selectbox(translations["en"]["fever"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    fatigue = st.selectbox(translations["en"]["fatigue"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    bruised_eyes = st.selectbox(translations["en"]["bruised_eyes"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    cough = st.selectbox(translations["en"]["cough"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    runny = st.selectbox(translations["en"]["runny"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    sore = st.selectbox(translations["en"]["sore"], [0,1], format_func=lambda x: "Yes" if x==1 else "No")

st.markdown("---")

# Predict button
if st.button(translations["en"]["predict"]):
    X = prepare_input(age, 1 if gender_text==translations["en"]["female"] else 0, lump, abdominal_pain,
                      weight_loss, fever, fatigue, bruised_eyes, constipation, aches, cough, runny, sore)
    try:
        Xs = scaler.transform(X)
        probs = model.predict_proba(Xs)[0]
        pred = model.predict(Xs)[0]
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        st.stop()

    neuro_prob = float(probs[1])
    confidence = float(np.max(probs))
    risk_cat = categorize_risk(neuro_prob)

    st.markdown("### " + translations["en"]["result_title"])
    col_r1, col_r2 = st.columns([2,1])

    with col_r1:
        pred_text = translations["en"]["neuroblastoma"] if pred == 1 else translations["en"]["not_neuroblastoma"]
        st.write(f"**Prediction:** {pred_text}")
        st.write(f"**Probability (Neuroblastoma):** {neuro_prob*100:.1f}%")
        st.write(f"**Risk category:** {risk_label_text(neuro_prob, translations['en'])}")

        # Plotly risk bar chart
        bins = ["Low (0-30%)", "Moderate (30-70%)", "High (70-100%)"]
        values = [0,0,0]
        if neuro_prob < 0.30:
            values[0] = neuro_prob*100
        elif neuro_prob < 0.70:
            values[1] = neuro_prob*100
        else:
            values[2] = neuro_prob*100

        fig = go.Figure(go.Bar(x=bins, y=values, marker_color=["#2ca02c","#f0ad4e","#d62728"]))
        fig.update_layout(title_text="Risk Summary (percent)", yaxis=dict(range=[0,100], title="Percent"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.markdown(f"<div class='small-muted'>{translations['en']['confidence']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='blue-value'>{confidence*100:.1f}%</div>", unsafe_allow_html=True)

    # CSV download
    result_df = pd.DataFrame([{
        "patient_name": patient_name,
        "date": str(assessment_date),
        "age": age,
        "gender": gender_text,
        "prediction": int(pred),
        "neuroblastoma_probability": round(neuro_prob*100,2),
        "confidence_percent": round(confidence*100,2),
        "risk_category": risk_cat
    }])
    csv_buf = io.StringIO()
    result_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode()
    st.download_button(
        label="Download assessment CSV",
        data=csv_bytes,
        file_name=f"assessment_{patient_name or 'patient'}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("Contact us: leonj062712@gmail.com")
