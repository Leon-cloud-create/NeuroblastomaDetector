import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
import os
from datetime import datetime
from PIL import Image  # for scan handling

# Attempt to import tensorflow.keras, set flag if unavailable
try:
    from tensorflow.keras.models import load_model  # for scan handling
    TENSORFLOW_AVAILABLE = True
except ImportError:
    load_model = None
    TENSORFLOW_AVAILABLE = False

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

# Imaging model path
SCAN_MODEL_PATH = "neuro-model.keras"  # your new keras format filename

# ---------------- Translations ----------------
translations = {
    # Your translation dictionary remains unchanged...
    # ...
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

# Load imaging model with TensorFlow check
@st.cache_resource
def load_scan_model():
    if not TENSORFLOW_AVAILABLE:
        return None, "TensorFlow not installed, imaging model not available."
    if not os.path.exists(SCAN_MODEL_PATH):
        return None, f"Scan model file not found: {SCAN_MODEL_PATH}"
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

# store last scan result (only if TensorFlow available)
if "last_scan_result" not in st.session_state:
    st.session_state["last_scan_result"] = None

# ---------------- Sidebar ----------------
with st.sidebar:
    lang_display = st.selectbox("🌐 Website Language", options=["English", "Spanish", "French"], index=0)
    lang = lang_display
    t = translations[lang]

    st.markdown("---")
    st.info(t["fill_patient_note"])

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

    if not TENSORFLOW_AVAILABLE or scan_load_error or scan_model is None:
        st.warning(t["scan_model_not_available"])
    else:
        if st.button(t["scan_analyze_button"]):
            try:
                img = Image.open(uploaded_scan).convert("RGB")

                # Adjust preprocessing to match YOUR imaging model
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

        # If a scan result exists and TensorFlow is available, show combined (experimental) risk
        if TENSORFLOW_AVAILABLE and st.session_state.get("last_scan_result") is not None:
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
