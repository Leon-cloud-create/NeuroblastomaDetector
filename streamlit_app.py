import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize session state for patient storage
if "patients" not in st.session_state:
    st.session_state.patients = []

# Sidebar
st.sidebar.title("🧭 Navigation")

# Language Selector (Top)
language = st.sidebar.selectbox("🌐 Language", ["English", "Español", "Français"])

# Patient Info Section
st.sidebar.subheader("🧍 Patient Information")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=5)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
date = st.sidebar.date_input("Date")

# Past Patient Data (Bottom)
st.sidebar.subheader("📁 Past Patient Data")
if st.session_state.patients:
    df_past = pd.DataFrame(st.session_state.patients)
    st.sidebar.dataframe(df_past, use_container_width=True)
else:
    st.sidebar.info("No past patient data yet.")

# Main Page
st.title("🧬 Neuroblastoma Risk Predictor")

# Neuroblastoma Overview
st.subheader("🧠 Neuroblastoma in a Nutshell")
st.markdown("""
Neuroblastoma is a cancer that develops from immature nerve cells, most commonly found in and around the adrenal glands. 
It primarily affects young children and can spread to the bones, liver, and other organs. Early detection is critical for effective treatment.  
This tool predicts the **risk level** of neuroblastoma based on symptoms and patient information, helping guide further evaluation.
""")

# Symptoms Input
st.subheader("🔍 Major Symptoms")
fatigue = st.checkbox("Fatigue")
abdominal_mass = st.checkbox("Abdominal Mass")
weight_loss = st.checkbox("Weight Loss")
bone_pain = st.checkbox("Bone Pain")

st.subheader("➕ Additional Symptoms")
fever = st.checkbox("Fever")
loss_of_appetite = st.checkbox("Loss of Appetite")
swelling = st.checkbox("Swelling")
bruising = st.checkbox("Bruising")

# Predict Button
if st.button("⚡ Predict Risk", use_container_width=True):
    features = np.array([[fatigue, abdominal_mass, weight_loss, bone_pain,
                          fever, loss_of_appetite, swelling, bruising]], dtype=int)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    risk = "High Risk" if prediction == 1 else "Low Risk"
    st.success(f"🩺 Prediction: **{risk}**")

    # Option to store data
    store_choice = st.radio("📦 Do you want your data stored?", ["No", "Yes"], horizontal=True)
    if store_choice == "Yes":
        st.session_state.patients.append({
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Date": str(date),
            "Risk": risk
        })
        st.success("✅ Data stored successfully!")

# Feedback Section
st.subheader("🗒️ Feedback")
st.text_area("Share your thoughts or suggestions:", placeholder="Type your feedback here...")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>© 2025 Neuroblastoma Risk Predictor | For educational use only</div>",
    unsafe_allow_html=True
)

