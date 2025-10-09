# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
from datetime import datetime

# ---------------------------
# Config & CSS
# ---------------------------
st.set_page_config(page_title="🏥 Neuroblastoma Risk Predictor", layout="wide")

# Full-width large predict button & simple hospital-white theme
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
        padding: 0.8em 0;
    }
    div.stButton > button:first-child:hover {
        background-color: #094c8d;
    }
    .footer {
        text-align:center; color:gray; padding:8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load model & scaler
# ---------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

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
    st.error(f"Model load error: {load_error}")
    st.stop()

# ---------------------------
# Session state storage
# ---------------------------
if "patients" not in st.session_state:
    st.session_state["patients"] = []  # list of dicts for past patient data

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("### 🌐 Language")
    # simple language selector placeholder (not wired to translations here)
    language = st.selectbox("", options=["English", "Español", "Français"])

    st.markdown("---")
    st.info("⚠️ Please fill out patient information first.")

    st.header("🧍 Patient Information")
    patient_name = st.text_input("Name (optional)", value="")
    assessment_date = st.date_input("Assessment Date", value=datetime.now().date())
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=5, step=1)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

    st.markdown("---")
    st.markdown("### 📁 Past Patient Data")
    if st.session_state["patients"]:
        df_past = pd.DataFrame(st.session_state["patients"])
        st.dataframe(df_past, use_container_width=True)
    else:
        st.info("No past patient data yet.")

# ---------------------------
# Main content
# ---------------------------
st.title("🏥 Neuroblastoma Risk Predictor")

# Neuroblastoma in a nutshell
st.subheader("🧠 Neuroblastoma in a Nutshell")
st.markdown(
    """
Neuroblastoma is a rare pediatric cancer that arises from immature nerve cells of the sympathetic nervous system. 
It most commonly affects infants and young children and can present with an abdominal mass, bone pain, or periorbital bruising.
Early recognition of symptoms followed by clinical evaluation and imaging can help clinicians reach a prompt diagnosis.
This tool gives an informational risk estimate based on symptoms — it is **not a diagnosis**.
"""
)

st.markdown("---")

# Symptoms - Major
st.subheader("🩺 Major Symptoms (select yes/no)")
col1, col2 = st.columns(2)

with col1:
    lump = st.selectbox("Large/Medium lump (usually on abdomen, chest, or neck)", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    abdominal_pain = st.selectbox("Abdominal pain", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    weight_loss = st.selectbox("Unexplained weight loss", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    fatigue = st.selectbox("Fatigue / Weakness", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    periorbital_bruising = st.selectbox("Bulging or bruised eyes", [0,1], format_func=lambda x: "Yes" if x==1 else "No")

with col2:
    constipation = st.selectbox("Constipation", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    aches_pain = st.selectbox("Aches/Pain (usually in the leg causing limping)", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    bone_pain = st.selectbox("Bone Pain (usually followed by swelling, fever, and limping)", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    # keep placeholders to maintain layout
    _ = st.empty()
    _ = st.empty()

st.markdown("---")

# Additional Symptoms
st.subheader("🤒 Additional Symptoms")
col3, col4 = st.columns(2)

with col3:
    fever = st.selectbox("Fever", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    cough = st.selectbox("Cough", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
with col4:
    runny_nose = st.selectbox("Runny / Stuffy nose", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    sore_throat = st.selectbox("Sore Throat", [0,1], format_func=lambda x: "Yes" if x==1 else "No")

st.markdown("---")

# Predict button (full width due to CSS above)
predict_clicked = st.button("🔍 Predict Risk")

if predict_clicked:
    # Encode gender to numeric for model input (choose mapping; adjust if your model uses different encoding)
    # We'll encode Male=1, Female=0, Other=0
    gender_encoded = 1 if gender == "Male" else 0

    # Prepare features in the requested order:
    # Age, Gender, Large/Medium lump, Abdominal pain, Unexplained weight loss,
    # Fever, Fatigue/Weakness, Bulging or bruised eyes, Constipation,
    # Aches/Pain, Bone Pain, Cough, Runny/Stuffy nose, Sore Throat
    features = np.array([[
        age,
        gender_encoded,
        int(lump),
        int(abdominal_pain),
        int(weight_loss),
        int(fever),
        int(fatigue),
        int(periorbital_bruising),
        int(constipation),
        int(aches_pain),
        int(bone_pain),
        int(cough),
        int(runny_nose),
        int(sore_throat)
    ]], dtype=float)

    # Scale & predict
    try:
        features_scaled = scaler.transform(features)
    except Exception as e:
        st.error(f"Scaler transform error: {e}")
        st.stop()

    try:
        probs = model.predict_proba(features_scaled)[0]
        pred = model.predict(features_scaled)[0]
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        st.stop()

    neuro_prob = float(probs[1])  # assumes class 1 = neuroblastoma
    confidence = float(np.max(probs))
    # Determine risk level text
    if neuro_prob <= 0.30:
        risk_text = "Low Risk"
    elif neuro_prob <= 0.70:
        risk_text = "Moderate Risk"
    else:
        risk_text = "High Risk"

    # Show results
    st.markdown("### 🔬 Prediction Results")
    col_r1, col_r2 = st.columns([3,1])
    with col_r1:
        st.write(f"**Prediction:** {'Neuroblastoma' if pred == 1 else 'Not Neuroblastoma'}")
        st.write(f"**Probability (Neuroblastoma):** {neuro_prob*100:.1f}%")
        st.write(f"**Risk category:** {risk_text}")
    with col_r2:
        st.markdown("**Model confidence**")
        st.markdown(f"<div style='font-weight:700; font-size:20px; color:#0b66c3'>{confidence*100:.1f}%</div>", unsafe_allow_html=True)

    # CSV download for this single assessment
    result_df = pd.DataFrame([{
        "patient_name": patient_name,
        "date": str(assessment_date),
        "age": age,
        "gender": gender,
        "prediction": int(pred),
        "neuroblastoma_probability": round(neuro_prob*100,2),
        "confidence_percent": round(confidence*100,2),
        "risk_category": risk_text
    }])
    csv_buf = io.StringIO()
    result_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download assessment CSV",
        data=csv_buf.getvalue().encode(),
        file_name=f"assessment_{(patient_name or 'patient')}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    # ---------------------------
    # Store choice (below results)
    # ---------------------------
    st.markdown("---")
    store_choice = st.radio("📦 Do you want your data stored?", options=["No", "Yes"], horizontal=True)
    if store_choice == "Yes":
        st.session_state["patients"].append({
            "Name": patient_name or "(no name)",
            "Date": str(assessment_date),
            "Age": age,
            "Gender": gender,
            "Risk": risk_text,
            "Probability_%": round(neuro_prob*100,2)
        })
        st.success("✅ Data stored and visible in Past Patient Data (sidebar).")

    # ---------------------------
    # Feedback box (large, above footer)
    # ---------------------------
    st.markdown("---")
    feedback = st.text_area("🗒️ Feedback (optional) — share your thoughts or report issues", height=140, placeholder="Type your feedback here...")

# If user hasn't clicked predict, show an empty results placeholder
else:
    st.info("Fill the symptoms and click 'Predict Risk' to see results.")

# Feedback (ensure feedback appears above footer if it was not shown after predict)
if not predict_clicked:
    st.markdown("---")
    st.text_area("🗒️ Feedback (optional) — share your thoughts or report issues", height=140, placeholder="Type your feedback here...")

# Footer
st.markdown("---")
st.markdown("<div class='footer'>© 2025 Neuroblastoma Risk Predictor | For educational use only — Contact: leonj062712@gmail.com</div>", unsafe_allow_html=True)


