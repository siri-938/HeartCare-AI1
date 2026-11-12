
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from supabase import create_client

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heartcare_rf_pipeline.joblib"

FEATURES = [
    "age", "gender", "chestpain", "restingBP",
    "serumcholestrol", "fastingbloodsugar",
    "restingrelectro", "maxheartrate",
    "exerciseangia", "oldpeak", "slope",
    "noofmajorvessels"
]

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "heartcare_rf_pipeline.joblib not found."
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ---------------- SUPABASE CONNECTION ----------------

@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


try:
    supabase = get_supabase()
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    st.stop()


# Restore administrator session
if (
    st.session_state.get("access_token")
    and st.session_state.get("refresh_token")
):
    try:
        supabase.auth.set_session(
            st.session_state["access_token"],
            st.session_state["refresh_token"]
        )
    except Exception:
        for key in ["access_token", "refresh_token", "admin_id"]:
            st.session_state.pop(key, None)


# ---------------- STYLING ----------------

st.markdown("""
<style>
.stApp {
    background: #f5f8fc;
}
[data-testid="stSidebar"] {
    background: #102a43;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.hero {
    background: linear-gradient(120deg,#12304a,#176b87);
    padding: 30px;
    border-radius: 18px;
    color: white;
}
.hero h1 {
    color: white;
}
.main-title {
    font-size: 32px;
    font-weight: bold;
    color: #12304a;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL FUNCTIONS ----------------

def predict_patient(values):

    input_df = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    prediction = int(model.predict(input_df)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        if prediction in classes:
            probability = float(
                probabilities[classes.index(prediction)] * 100
            )

    return prediction, probability, input_df


def get_prediction_label(prediction):

    if prediction == 1:
        return "Heart Disease"

    elif prediction == 0:
        return "No Heart Disease"

    return f"Model class {prediction}"


def show_prediction(prediction, probability, input_df):

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("❤️ Model Prediction: Heart Disease")

    elif prediction == 0:
        st.success("💚 Model Prediction: No Heart Disease")

    if probability is not None:

        st.metric(
            "Predicted-class probability",
            f"{probability:.2f}%"
        )

        st.progress(
            min(max(probability / 100, 0), 1)
        )

    st.write("Input Summary")
    st.dataframe(input_df, use_container_width=True)

    st.warning(
        "This is an academic machine-learning prediction, "
        "not a medical diagnosis. Consult a qualified healthcare "
        "professional for medical advice."
    )


# ---------------- INPUT FORM ----------------

def health_input_form(form_name, button_text):

    with st.form(form_name):

        st.subheader("Basic Parameters")

        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Age",
                1, 120, 45,
                key=form_name + "_age"
            )

        with c2:
            gender = st.selectbox(
                "Gender",
                ["Female (0)", "Male (1)"],
                key=form_name + "_gender"
            )

        st.divider()
        st.subheader("Heart and Test Parameters")

        c1, c2 = st.columns(2)

        with c1:

            chest = st.selectbox(
                "Chest Pain Type",
                [
                    "0 (Typical angina)",
                    "1 (Atypical angina)",
                    "2 (Non-anginal pain)",
                    "3 (Asymptomatic)"
                ],
                key=form_name + "_chest"
            )

            bp = st.number_input(
                "Resting Blood Pressure",
                94, 200, 120,
                key=form_name + "_bp"
            )

            cholesterol = st.number_input(
                "Serum Cholesterol",
                126, 602, 200,
                key=form_name + "_chol"
            )

            fasting = st.selectbox(
                "Fasting Blood Sugar",
                ["0 (≤120 mg/dL)", "1 (>120 mg/dL)"],
                key=form_name + "_fasting"
            )

            ecg = st.selectbox(
                "Resting ECG",
                [
                    "0 (Normal ECG)",
                    "1 (ST-T abnormality)",
                    "2 (Possible/definite LVH)"
                ],
                key=form_name + "_ecg"
            )

        with c2:

            maxhr = st.number_input(
                "Maximum Heart Rate",
                71, 202, 150,
                key=form_name + "_maxhr"
            )

            angina = st.selectbox(
                "Exercise-Induced Angina",
                ["0 (No)", "1 (Yes)"],
                key=form_name + "_angina"
            )

            oldpeak = st.number_input(
                "Oldpeak",
                0.0, 6.2, 1.0,
                step=0.1,
                key=form_name + "_oldpeak"
            )

            slope = st.selectbox(
                "Slope",
                [
                    "0 (Unspecified)",
                    "1 (Upsloping)",
                    "2 (Flat)",
                    "3 (Downsloping)"
                ],
                key=form_name + "_slope"
            )

            vessels = st.selectbox(
                "Number of Major Vessels",
                ["0", "1", "2", "3"],
                key=form_name + "_vessels"
            )

        submitted = st.form_submit_button(
            button_text,
            use_container_width=True
        )

    if not submitted:
        return None

    values = [
        int(age),
        0 if gender.startswith("Female") else 1,
        int(chest.split()[0]),
        int(bp),
        int(cholesterol),
        int(fasting.split()[0]),
        int(ecg.split()[0]),
        int(maxhr),
        int(angina.split()[0]),
        float(oldpeak),
        int(slope.split()[0]),
        int(vessels)
    ]

    return values


# ---------------- ADMIN LOGIN ----------------

st.sidebar.title("❤️ HeartCare AI")
st.sidebar.caption("Heart Health Prediction")
st.sidebar.divider()

st.sidebar.subheader("Administrator Login")

if not st.session_state.get("admin_id"):

    with st.sidebar.form("login_form"):

        email = st.text_input("Email")

        password = st.text_input(
            "Password",
            type="password"
        )

        login = st.form_submit_button("Login")

    if login:

        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if response.user and response.session:

                st.session_state["admin_id"] = response.user.id

                st.session_state["access_token"] = (
                    response.session.access_token
                )

                st.session_state["refresh_token"] = (
                    response.session.refresh_token
                )

                st.rerun()

            else:
                st.sidebar.error("Login failed.")

        except Exception:
            st.sidebar.error(
                "Invalid administrator email or password."
            )

else:

    st.sidebar.success("Administrator logged in")

    if st.sidebar.button("Logout"):

        try:
            supabase.auth.sign_out()
        except Exception:
            pass

        for key in ["admin_id", "access_token", "refresh_token"]:
            st.session_state.pop(key, None)

        st.rerun()
        # ---------------- ADMIN SECURITY HELPER ----------------

def require_admin():

    if not st.session_state.get("admin_id"):
        st.error("Administrator login required.")
        st.stop()


def get_records():

    require_admin()

    response = (
        supabase.table("patients")
        .select("*")
        .eq("created_by", st.session_state["admin_id"])
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []




# ---------------- NAVIGATION ----------------

public_pages = [
    "Home",
    "Predictor",
    "Understanding Parameters",
    "Heart Health Tips"
]

admin_pages = [
    "Register Patient",
    "Patient Records",
    "Update Patient",
    "Delete Patient"
]

if st.session_state.get("admin_id"):
    pages = public_pages + admin_pages
else:
    pages = public_pages

page = st.sidebar.radio(
    "Navigation",
    pages,
    key="navigation"
)


# ---------------- HOME ----------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>❤️ HeartCare AI</h1>
        <h3>Heart Disease Risk Prediction</h3>
        <p>Machine Learning Based Healthcare Application</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("Welcome to HeartCare AI")

    st.write("""
    HeartCare AI uses a trained Random Forest machine-learning
    model to predict heart disease from selected health parameters.

    Features include:

    - Heart disease prediction
    - Administrator authentication
    - Patient registration
    - Automatic Patient ID generation
    - Patient record management
    - Prediction result storage
    """)

    st.info(
        "Public visitors can use the prediction feature. "
        "Only the administrator can manage saved patient records."
    )


# ---------------- PUBLIC PREDICTOR ----------------

elif page == "Predictor":

    st.markdown(
        '<p class="main-title">Heart Disease Predictor</p>',
        unsafe_allow_html=True
    )

    st.info(
        "Public predictions are not saved to the patient database."
    )

    values = health_input_form(
        "public_prediction",
        "Predict Heart Disease"
    )

    if values is not None:

        try:
            prediction, probability, input_df = predict_patient(values)

            show_prediction(
                prediction,
                probability,
                input_df
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")



# ---------------- REGISTER PATIENT ----------------

elif page == "Register Patient":

    require_admin()

    st.markdown(
        '<p class="main-title">Register Patient</p>',
        unsafe_allow_html=True
    )

    with st.form("patient_details"):

        patient_name = st.text_input("Patient Name")

        contact = st.text_input(
            "Contact Number (Optional)"
        )

        consent = st.checkbox(
            "I confirm that the patient has agreed "
            "to the collection and storage of these details."
        )

        details_submit = st.form_submit_button(
            "Continue"
        )

    if details_submit:

        if not patient_name.strip():
            st.error("Enter patient name.")

        elif not consent:
            st.error("Patient consent is required.")

        else:

            st.session_state["pending_name"] = patient_name.strip()
            st.session_state["pending_contact"] = contact.strip()

            st.success(
                "Patient details entered. Complete the health parameters."
            )

    if st.session_state.get("pending_name"):

        values = health_input_form(
            "patient_health",
            "Predict and Save Patient"
        )

        if values is not None:

            try:

                prediction, probability, input_df = predict_patient(values)

                record = {

                    "patient_name": st.session_state["pending_name"],

                    "contact_number":
                        st.session_state.get("pending_contact") or None,

                    "age": values[0],

                    "gender":
                        "Female" if values[1] == 0 else "Male",

                    "chestpain": values[2],

                    "restingbp": values[3],

                    "serumcholestrol": values[4],

                    "fastingbloodsugar": values[5],

                    "restingrelectro": values[6],

                    "maxheartrate": values[7],

                    "exerciseangia": values[8],

                    "oldpeak": values[9],

                    "slope": values[10],

                    "noofmajorvessels": values[11],

                    "prediction": prediction,

                    "prediction_label":
                        get_prediction_label(prediction),

                    "probability": probability,

                    "prediction_created_at":
                        pd.Timestamp.now(tz="UTC").isoformat(),

                    "created_by": st.session_state["admin_id"]
                }

                result = (
                    supabase.table("patients")
                    .insert(record)
                    .execute()
                )

                if result.data:

                    st.success("Patient registered successfully!")

                    st.subheader("Generated Patient ID")

                    st.code(
                        result.data[0]["patient_id"]
                    )

                    show_prediction(
                        prediction,
                        probability,
                        input_df
                    )

                    for key in ["pending_name", "pending_contact"]:
                        st.session_state.pop(key, None)

            except Exception as e:

                st.error(f"Patient registration failed: {e}")


# ---------------- VIEW PATIENT RECORDS ----------------

elif page == "Patient Records":

    require_admin()

    st.markdown(
        '<p class="main-title">Patient Records</p>',
        unsafe_allow_html=True
    )

    try:

        records = get_records()

        if records:

            df = pd.DataFrame(records)

            columns = [
                "patient_id",
                "patient_name",
                "age",
                "gender",
                "contact_number",
                "prediction_label",
                "probability",
                "created_at"
            ]

            st.dataframe(
                df[[c for c in columns if c in df.columns]],
                use_container_width=True,
                hide_index=True
            )

            st.write("Total patients:", len(records))

        else:

            st.info("No patient records available.")

    except Exception as e:

        st.error(f"Unable to load records: {e}")


# ---------------- UPDATE PATIENT ----------------

elif page == "Update Patient":

    require_admin()

    st.markdown(
        '<p class="main-title">Update Patient</p>',
        unsafe_allow_html=True
    )

    try:

        records = get_records()

        if records:

            choices = {
                f"{r['patient_name']} - {r['patient_id']}": r
                for r in records
            }

            selected_label = st.selectbox(
                "Select Patient",
                list(choices.keys())
            )

            patient = choices[selected_label]

            with st.form("update_form"):

                new_name = st.text_input(
                    "Patient Name",
                    patient["patient_name"]
                )

                new_age = st.number_input(
                    "Age",
                    1,
                    120,
                    int(patient["age"])
                )

                new_gender = st.selectbox(
                    "Gender",
                    ["Female", "Male"],
                    index=(
                        0 if patient["gender"] == "Female" else 1
                    )
                )

                new_contact = st.text_input(
                    "Contact Number",
                    patient.get("contact_number") or ""
                )

                update = st.form_submit_button(
                    "Update Patient"
                )

            if update:

                if not new_name.strip():
                    st.error("Patient name is required.")

                else:

                    (
                        supabase.table("patients")
                        .update({
                            "patient_name": new_name.strip(),
                            "age": int(new_age),
                            "gender": new_gender,
                            "contact_number": new_contact.strip() or None
                        })
                        .eq("patient_id", patient["patient_id"])
                        .eq("created_by", st.session_state["admin_id"])
                        .execute()
                    )

                    st.success("Patient updated successfully.")

                    st.rerun()

        else:

            st.info("No patient records available.")

    except Exception as e:

        st.error(f"Update failed: {e}")


# ---------------- DELETE PATIENT ----------------

elif page == "Delete Patient":

    require_admin()

    st.markdown(
        '<p class="main-title">Delete Patient</p>',
        unsafe_allow_html=True
    )

    try:

        records = get_records()

        if records:

            choices = {
                f"{r['patient_name']} - {r['patient_id']}": r
                for r in records
            }

            selected_label = st.selectbox(
                "Select Patient",
                list(choices.keys())
            )

            patient = choices[selected_label]

            confirm = st.checkbox(
                "I confirm that I want to permanently delete this record."
            )

            if st.button(
                "Delete Patient",
                disabled=not confirm
            ):

                (
                    supabase.table("patients")
                    .delete()
                    .eq("patient_id", patient["patient_id"])
                    .eq("created_by", st.session_state["admin_id"])
                    .execute()
                )

                st.success("Patient deleted successfully.")

                st.rerun()

        else:

            st.info("No patient records available.")

    except Exception as e:

        st.error(f"Delete failed: {e}")


# ---------------- UNDERSTANDING PARAMETERS ----------------

elif page == "Understanding Parameters":

    st.markdown(
        '<p class="main-title">Understanding Health Parameters</p>',
        unsafe_allow_html=True
    )

    parameters = {
        "Age": "Patient age in years.",
        "Gender": "0 = Female; 1 = Male.",
        "Chest Pain Type": "0 = Typical; 1 = Atypical; 2 = Non-anginal; 3 = Asymptomatic.",
        "Resting BP": "Blood pressure at rest.",
        "Serum Cholesterol": "Blood cholesterol measurement.",
        "Fasting Blood Sugar": "0 = ≤120 mg/dL; 1 = >120 mg/dL.",
        "Resting ECG": "ECG measurement at rest.",
        "Maximum Heart Rate": "Maximum heart rate recorded during the test.",
        "Exercise-Induced Angina": "0 = No; 1 = Yes.",
        "Oldpeak": "ST-segment depression value during excercise.",
        "Slope": "Peak exercise ST segment slope.",
        "Major Vessels": "Number of major vessels recorded."
    }

    for title, description in parameters.items():

        with st.expander(title):
            st.write(description)


# ---------------- HEART HEALTH TIPS ----------------

elif page == "Heart Health Tips":

    st.markdown(
        '<p class="main-title">Heart Health Tips ❤️</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    - Eat a balanced diet.
    - Stay physically active.
    - Avoid smoking.
    - Get enough sleep.
    - Monitor your health.
    - Manage stress.
    """)

    st.warning(
        "For chest pain, severe breathlessness or fainting, "
        "seek urgent medical care."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "HeartCare AI "
    "Machine-learning predictions are not medical diagnoses."
)


