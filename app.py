import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# HEARTCARE AI - PUBLIC DEMONSTRATION
# No patient records are saved or stored by this application.
# ---------------------------------------------------------

st.set_page_config(
    page_title="HeartCare AI | Heart Health Prediction",
    page_icon="❤️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heartcare_rf_pipeline.joblib"

FEATURES = [
    "age", "gender", "chestpain", "restingBP", "serumcholestrol",
    "fastingbloodsugar", "restingrelectro", "maxheartrate",
    "exerciseangia", "oldpeak", "slope", "noofmajorvessels"
]

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The model file 'heartcare_rf_pipeline.joblib' was not found. "
            "Make sure it is uploaded to the same GitHub repository folder as app.py."
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the prediction model: {exc}")
    st.stop()

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {background: #f5f8fc;}
[data-testid="stSidebar"] {background: #102a43;}
[data-testid="stSidebar"] * {color: white !important;}
.hero {
    background: linear-gradient(120deg, #12304a, #176b87);
    padding: 30px; border-radius: 18px; color: white; margin-bottom: 22px;
}
.hero h1 {color: white; font-size: 36px;}
.hero p {color: #e5f3fa; font-size: 17px;}
.main-title {font-size: 34px; font-weight: 800; color: #12304a;}
.subtitle {font-size: 16px; color: #64748b;}
.result-card {
    background: white; border-radius: 16px; padding: 24px;
    border: 1px solid #e3edf5; text-align: center;
}
div.stButton > button, div.stFormSubmitButton > button {
    background: #087e8b; color: white; border-radius: 9px;
    border: 0; font-weight: 600;
}
.footer {text-align: center; color: #718096; padding: 22px; font-size: 13px;}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def predict_patient(values):
    input_df = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(input_df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        if prediction in classes:
            probability = float(probabilities[classes.index(prediction)] * 100)

    return prediction, probability, input_df


def show_prediction(prediction, probability, input_df):
    st.divider()
    st.markdown("## Prediction Result")

    if prediction == 1:
        st.error("❤️ Model Prediction: Heart Disease")
    elif prediction == 0:
        st.success("💚 Model Prediction: No Heart Disease")
    else:
        st.warning(f"Model returned class: {prediction}")

    if probability is not None:
        st.markdown(
            f"""
            <div class="result-card">
              <h3>Probability for the predicted class</h3>
              <h1 style="color:#087e8b;font-size:44px">{probability:.2f}%</h1>
              <p>This is the model's predicted-class probability, not a clinical risk score.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability / 100, 0.0), 1.0))

    st.markdown("### Demonstration input summary")
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    st.warning(
        "This is an academic demonstration only, not a medical diagnosis. "
        "Do not use this result to make healthcare decisions. Consult a qualified "
        "healthcare professional for medical advice."
    )


# ---------------- SIDEBAR ----------------
st.sidebar.markdown("# ❤️ HeartCare AI")
st.sidebar.caption("Heart Health Prediction Demonstration")
st.sidebar.divider()
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predictor", "Understanding Health Parameters", "Heart Health Tips"],
    key="selected_page",
)
st.sidebar.divider()
st.sidebar.info(
    "Demo mode: predictions are processed for the current session only. "
    "This app does not save patient records."
)

# ---------------- HOME ----------------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
          <h1>❤️ HeartCare AI</h1>
          <p>Heart Disease Risk Prediction Demonstration</p>
          <p>A machine-learning academic project that demonstrates predictions
          from selected heart-health parameters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Welcome to HeartCare AI")
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(
            """
            ### About this project
            HeartCare AI uses a trained Random Forest machine-learning pipeline
            to demonstrate how selected health-related input values are processed
            by a classification model.

            **The demonstration includes:**
            - Patient parameter input form
            - Heart disease classification output
            - Predicted-class probability, when supported by the model
            - Explanations of the input parameters
            """
        )
        if st.button("Start Heart Disease Prediction", use_container_width=True):
            st.session_state.selected_page = "Predictor"
            st.rerun()
    with right:
        st.image(
            "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=900",
            caption="Healthcare professional",
            use_container_width=True,
        )

    st.info(
        "Use fictional/sample values only. Do not enter real patient-identifying "
        "information or use this demonstration for diagnosis."
    )

# ---------------- PREDICTOR ----------------
elif page == "Predictor":
    st.markdown(
        '<p class="main-title">❤️ Heart Disease Predictor</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Enter fictional demonstration values for the model inputs.</p>',
        unsafe_allow_html=True,
    )
    st.warning(
        "Use sample values only. Do not enter a real person's private medical "
        "information. The application does not save submitted values."
    )

    with st.form("prediction_form"):
        st.markdown("### 👤 Basic Parameters")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        with c2:
            gender_label = st.selectbox("Gender", ["Female (0)", "Male (1)"])

        st.divider()
        st.markdown("### ❤️ Heart and Test Parameters")
        c1, c2 = st.columns(2)

        with c1:
            chest_label = st.selectbox(
                "Chest Pain Type",
                [
                    "0 (Typical angina)",
                    "1 (Atypical angina)",
                    "2 (Non-anginal chest pain)",
                    "3 (Asymptomatic)",
                ],
            )
            resting_bp = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=94, max_value=200, value=120,
            )
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=126, max_value=602, value=200,
            )
            fasting_label = st.selectbox(
                "Fasting Blood Sugar", ["0 (≤120 mg/dL)", "1 (>120 mg/dL)"]
            )
            ecg_label = st.selectbox(
                "Resting ECG",
                [
                    "0 (Normal ECG)",
                    "1 (ST-T wave abnormality)",
                    "2 (Possible/definite LVH)",
                ],
            )

        with c2:
            max_heart_rate = st.number_input(
                "Maximum Heart Rate (bpm)",
                min_value=71, max_value=202, value=150,
            )
            angina_label = st.selectbox(
                "Exercise-Induced Angina", ["0 (No)", "1 (Yes)"]
            )
            oldpeak = st.number_input(
                "Oldpeak", min_value=0.0, max_value=6.2,
                value=1.0, step=0.1, format="%.1f",
            )
            slope_label = st.selectbox(
                "Slope",
                [
                    "0 (Unspecified dataset category)",
                    "1 (Upsloping)",
                    "2 (Flat)",
                    "3 (Downsloping)",
                ],
            )
            vessels_label = st.selectbox(
                "Number of Major Vessels",
                [
                    "0 (0 major vessels)",
                    "1 (1 major vessel)",
                    "2 (2 major vessels)",
                    "3 (3 major vessels)",
                ],
            )

        submitted = st.form_submit_button(
            "🔍 Predict Heart Disease", use_container_width=True
        )

    if submitted:
        values = [
            int(age),
            0 if gender_label.startswith("Female") else 1,
            int(chest_label.split()[0]),
            int(resting_bp),
            int(cholesterol),
            int(fasting_label.split()[0]),
            int(ecg_label.split()[0]),
            int(max_heart_rate),
            int(angina_label.split()[0]),
            float(oldpeak),
            int(slope_label.split()[0]),
            int(vessels_label.split()[0]),
        ]

        try:
            prediction, probability, input_df = predict_patient(values)
            show_prediction(prediction, probability, input_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the uploaded model was trained with the same 12 "
                "feature names and order used by this application."
            )

# ---------------- UNDERSTANDING PARAMETERS ----------------
elif page == "Understanding Health Parameters":
    st.markdown(
        '<p class="main-title">Understanding Health Parameters</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">The 12 input features used by the prediction model.</p>',
        unsafe_allow_html=True,
    )

    items = [
        ("1. Age", "Patient age in years.", "Enter age in years."),
        ("2. Gender", "Gender value recorded by the dataset.", "0 = Female; 1 = Male."),
        ("3. Chest Pain Type", "Chest-pain category recorded for the patient.",
         "0 = Typical angina; 1 = Atypical angina; 2 = Non-anginal pain; 3 = Asymptomatic."),
        ("4. Resting Blood Pressure", "Blood pressure recorded at rest, in mmHg.",
         "Use a measurement from a medical report. Demo input range: 94–200."),
        ("5. Serum Cholesterol", "Blood cholesterol measurement, usually mg/dL.",
         "Use a laboratory report. Demo input range: 126–602."),
        ("6. Fasting Blood Sugar", "Fasting blood sugar category used by the dataset.",
         "0 = ≤120 mg/dL; 1 = >120 mg/dL."),
        ("7. Resting ECG", "ECG records the heart's electrical activity at rest.",
         "0 = Normal; 1 = ST-T wave abnormality; 2 = Possible/definite LVH."),
        ("8. Maximum Heart Rate", "Maximum heart rate recorded during the relevant test, in bpm.",
         "Demo input range: 71–202."),
        ("9. Exercise-Induced Angina", "Indicates whether angina was recorded during exercise.",
         "0 = No; 1 = Yes."),
        ("10. Oldpeak", "ST-segment depression value recorded in an exercise test.",
         "Use the value from the relevant test report. Demo input range: 0–6.2."),
        ("11. Slope", "Slope of the peak exercise ST segment.",
         "0 = Unspecified dataset category; 1 = Upsloping; 2 = Flat; 3 = Downsloping."),
        ("12. Number of Major Vessels", "Number of major vessels recorded in the dataset.",
         "0, 1, 2, or 3; use the medical report."),
    ]

    for title, meaning, values in items:
        with st.expander(title):
            st.write("**Meaning:**", meaning)
            st.write("**Dataset coding / demo range:**", values)

    st.info(
        "The displayed category descriptions are provided for project explanation. "
        "Do not guess ECG, Oldpeak, or major-vessel values."
    )

# ---------------- HEART HEALTH TIPS ----------------
elif page == "Heart Health Tips":
    st.markdown(
        '<p class="main-title">Heart Health Tips ❤️</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">General educational information.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.image(
            "https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=900",
            caption="Balanced food choices",
            use_container_width=True,
        )
        st.markdown("### 🥗 Eat a Balanced Diet")
        st.write(
            "Include fruits, vegetables, whole grains and balanced meals. "
            "Limit excess salt and highly processed foods."
        )
        st.markdown("### 🚶 Stay Physically Active")
        st.write("Choose regular physical activity suitable for your health.")
        st.markdown("### 🚭 Avoid Smoking")
        st.write("Avoid tobacco and seek professional support to stop smoking.")

    with right:
        st.image(
            "https://images.unsplash.com/photo-1506126613408-eca07ce68773?w=900",
            caption="Healthy lifestyle",
            use_container_width=True,
        )
        st.markdown("### 😴 Get Enough Sleep")
        st.write("Maintain a regular sleep schedule and get enough rest.")
        st.markdown("### 🩺 Monitor Your Health")
        st.write(
            "Check blood pressure, blood sugar and cholesterol as advised "
            "by your healthcare professional."
        )
        st.markdown("### 🧘 Manage Stress")
        st.write("Relaxation and healthy routines can help manage everyday stress.")

    st.warning(
        "For chest pain, severe breathlessness, fainting or other emergency "
        "symptoms, seek urgent medical care."
    )

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
      <hr><b>HeartCare AI | Academic Demonstration</b><br>
      No patient information is stored by this application.
    </div>
    """,
    unsafe_allow_html=True,
)
