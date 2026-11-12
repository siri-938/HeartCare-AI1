import os, sqlite3, uuid
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HeartCare AI | Heart Health Dashboard", page_icon="❤️", layout="wide")

MODEL_PATH = "/content/heartcare_rf_pipeline.joblib"
DB_DIR = "/content/drive/MyDrive/HeartCareAI"
DB_PATH = os.path.join(DB_DIR, "heartcare_patients.db")
FEATURES = ["age","gender","chestpain","restingBP","serumcholestrol","fastingbloodsugar",
            "restingrelectro","maxheartrate","exerciseangia","oldpeak","slope","noofmajorvessels"]

# ---------------- DATABASE ----------------
def connect_db():
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError("Google Drive is not mounted. Run drive.mount('/content/drive') first.")
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=30)

def init_db():
    con = connect_db()
    con.execute("""
    CREATE TABLE IF NOT EXISTS patient_records (
      record_id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id TEXT UNIQUE NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT,
      age INTEGER, gender INTEGER, chestpain INTEGER, restingBP INTEGER,
      serumcholestrol INTEGER, fastingbloodsugar INTEGER, restingrelectro INTEGER,
      maxheartrate INTEGER, exerciseangia INTEGER, oldpeak REAL, slope INTEGER,
      noofmajorvessels INTEGER, prediction INTEGER, predicted_probability REAL
    )""")
    cols = {r[1] for r in con.execute("PRAGMA table_info(patient_records)")}
    if "updated_at" not in cols:
        con.execute("ALTER TABLE patient_records ADD COLUMN updated_at TEXT")
        con.execute("UPDATE patient_records SET updated_at=created_at WHERE updated_at IS NULL")
    con.commit()
    con.close()

def save_record(d):
    con = connect_db()
    con.execute("""
    INSERT INTO patient_records
    (patient_id,created_at,updated_at,age,gender,chestpain,restingBP,serumcholestrol,
     fastingbloodsugar,restingrelectro,maxheartrate,exerciseangia,oldpeak,slope,
     noofmajorvessels,prediction,predicted_probability)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, tuple(d[k] for k in [
        "patient_id","created_at","updated_at","age","gender","chestpain","restingBP",
        "serumcholestrol","fastingbloodsugar","restingrelectro","maxheartrate",
        "exerciseangia","oldpeak","slope","noofmajorvessels","prediction","predicted_probability"]))
    con.commit()
    con.close()

def read_records(patient_id=None):
    con = connect_db()
    if patient_id:
        df = pd.read_sql_query(
            "SELECT * FROM patient_records WHERE patient_id=? ORDER BY record_id DESC",
            con, params=(patient_id.strip().upper(),))
    else:
        df = pd.read_sql_query("SELECT * FROM patient_records ORDER BY record_id DESC", con)
    con.close()
    return df

def update_record(d):
    con = connect_db()
    cur = con.execute("""
    UPDATE patient_records SET updated_at=?, age=?, gender=?, chestpain=?, restingBP=?,
      serumcholestrol=?, fastingbloodsugar=?, restingrelectro=?, maxheartrate=?,
      exerciseangia=?, oldpeak=?, slope=?, noofmajorvessels=?, prediction=?,
      predicted_probability=? WHERE patient_id=?
    """, tuple(d[k] for k in [
        "updated_at","age","gender","chestpain","restingBP","serumcholestrol",
        "fastingbloodsugar","restingrelectro","maxheartrate","exerciseangia",
        "oldpeak","slope","noofmajorvessels","prediction","predicted_probability","patient_id"]))
    con.commit()
    n = cur.rowcount
    con.close()
    return n

def delete_record(pid):
    con = connect_db()
    cur = con.execute("DELETE FROM patient_records WHERE patient_id=?", (pid.strip().upper(),))
    con.commit()
    n = cur.rowcount
    con.close()
    return n

# ---------------- MODEL AND STARTUP ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    init_db()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.info("In Colab, mount Google Drive first: from google.colab import drive; drive.mount('/content/drive')")
    st.stop()

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "navigation" not in st.session_state:
    st.session_state.navigation = "Home"
if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = "Not generated yet"

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp{background:#f5f8fc}
[data-testid="stSidebar"]{background:#102a43}
[data-testid="stSidebar"] *{color:white!important}
.hero{background:linear-gradient(120deg,#12304a,#176b87);padding:32px;border-radius:18px;color:white;margin-bottom:24px}
.hero h1{color:white;font-size:36px}.hero p{color:#e5f3fa;font-size:17px}
.main-title{font-size:35px;font-weight:800;color:#12304a}
.subtitle{font-size:16px;color:#64748b}
.info-card{background:white;padding:22px;border-radius:14px;border:1px solid #e3edf5;min-height:140px}
.info-card h3{color:#176b87}.info-card p{color:#526477}
.result-card{background:white;border-radius:16px;padding:25px;border:1px solid #e3edf5;text-align:center}
div.stButton>button,div.stFormSubmitButton>button{background:#087e8b;color:white;border-radius:9px;border:0;font-weight:600}
.footer{text-align:center;color:#718096;padding:24px;font-size:13px}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def parse_int(raw, label, low, high):
    try:
        value = int(str(raw).strip())
    except (ValueError, TypeError):
        return None, f"{label} must be a valid whole number."
    if not low <= value <= high:
        return None, f"{label} must be between {low} and {high}."
    return value, None

def predict(dataframe):
    pred = int(model.predict(dataframe)[0])
    probs = model.predict_proba(dataframe)[0]
    classes = list(model.classes_)
    return pred, float(probs[classes.index(pred)] * 100)

def display_result(result):
    st.divider()
    st.markdown("## Prediction Result")
    st.write(f"**Patient ID:** {result['patient_id']}")
    st.write(f"**Date and time:** {result['created_at']}")
    if result["prediction"] == 1:
        st.error("❤️ Model Prediction: Heart Disease")
    else:
        st.success("💚 Model Prediction: No Heart Disease")
    st.markdown(f"""
    <div class="result-card">
      <h3>Predicted Class Probability</h3>
      <h1 style="color:#087e8b;font-size:45px">{result['probability']:.2f}%</h1>
      <p>Probability corresponding to the predicted class</p>
    </div>""", unsafe_allow_html=True)
    st.progress(min(max(result["probability"]/100, 0), 1))
    st.markdown("### Patient Input Summary")
    st.dataframe(result["input"], use_container_width=True, hide_index=True)
    st.warning("This academic model output is not a medical diagnosis. Consult a qualified healthcare professional.")

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("# ❤️ HeartCare AI")
st.sidebar.caption("Heart Health Prediction System")
st.sidebar.divider()
pages = ["Home","Predictor","Patient Records","Understanding Health Parameters","Heart Health Tips"]
page = st.sidebar.radio("Navigation", pages, key="navigation")
st.sidebar.divider()
st.sidebar.markdown("### Patient Session")
st.sidebar.write(f"Latest Patient ID: {st.session_state.current_patient_id}")

# ---------------- HOME ----------------
if page == "Home":
    st.markdown("""
    <div class="hero"><h1>❤️ HeartCare AI</h1>
    <p>Intelligent Heart Health Dashboard</p>
    <p>A Machine Learning based system designed to predict heart disease using patient health parameters.</p></div>
    """, unsafe_allow_html=True)
    st.markdown("## Welcome to HeartCare AI")
    c1,c2 = st.columns([1.5,1])
    with c1:
        st.markdown("""
        ### About HeartCare AI
        ❤️
        HeartCare AI uses a trained Machine Learning model to analyze selected patient health information.

        The system accepts:
        - Age and gender
        - Chest pain information
        - Blood pressure and serum cholesterol
        - Resting ECG and maximum heart rate
        - Exercise-related information

        The application displays the model's predicted class and its probability.
        """)
        def open_predictor_page():
            st.session_state.last_result = None
            st.session_state.navigation = "Predictor"

        st.button(
            "Start Heart Disease Prediction",
            use_container_width=True,
            on_click=open_predictor_page,
        )
    with c2:
        st.image("https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=900",
                 caption="Healthcare professional", use_container_width=True)
    st.divider()
    st.markdown("## Dashboard Features")
    a,b,c = st.columns(3)
    with a:
        st.markdown('<div class="info-card"><h3>🧠 Machine Learning</h3><p>Uses a trained Random Forest model.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="info-card"><h3>📊 Health Parameters</h3><p>Analyzes patient health and medical test information.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="info-card"><h3>📈 Prediction Results</h3><p>Displays predicted class and its probability.</p></div>', unsafe_allow_html=True)
    st.info("This application is an academic project. Its predictions are not a medical diagnosis.")

# ---------------- PREDICTOR ----------------
elif page == "Predictor":
    st.markdown('<p class="main-title">❤️ Heart Disease Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter patient information and medical test values below.</p>', unsafe_allow_html=True)
    st.info("Use values from the patient's medical report. Do not guess technical test values.")

    with st.form("prediction_form"):
        st.markdown("### 👤 Patient Information")
        c1,c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 1, 120, 45)
        with c2:
            gender_label = st.selectbox("Gender", ["Female (0)","Male (1)"])
        st.divider()
        st.markdown("### ❤️ Heart and Blood Test Information")
        c1,c2 = st.columns(2)
        with c1:
            chest_label = st.selectbox("Chest Pain Type", [
                "0 (Typical angina)","1 (Atypical angina)",
                "2 (Non-anginal chest pain)","3 (Asymptomatic)"])
            bp_text = st.text_input("Resting Blood Pressure (mmHg)", "120")
            chol_text = st.text_input("Serum Cholesterol (mg/dL)", "200")
            fbs_label = st.selectbox("Fasting Blood Sugar", ["0 (≤120 mg/dL)","1 (>120 mg/dL)"])
            ecg_label = st.selectbox("Resting ECG", [
                "0 (Normal ECG)","1 (ST-T wave abnormality)","2 (Possible/definite LVH)"])
        with c2:
            hr_text = st.text_input("Maximum Heart Rate (bpm)", "150")
            angina_label = st.selectbox("Exercise-Induced Angina", ["0 (No)","1 (Yes)"])
            oldpeak = st.number_input("Oldpeak", 0.0, 6.2, 1.0, 0.1, format="%.1f")
            slope_label = st.selectbox("Slope", [
                "0 (Unspecified dataset category)","1 (Upsloping)","2 (Flat)","3 (Downsloping)"])
            vessels_label = st.selectbox("Number of Major Vessels", [
                "0 (0 major vessels)","1 (1 major vessel)","2 (2 major vessels)","3 (3 major vessels)"])
        submitted = st.form_submit_button("🔍 Predict Heart Disease", use_container_width=True)

    if submitted:
        errors = []
        bp,err = parse_int(bp_text,"Resting Blood Pressure",94,200)
        if err: errors.append(err)
        chol,err = parse_int(chol_text,"Serum Cholesterol",126,602)
        if err: errors.append(err)
        hr,err = parse_int(hr_text,"Maximum Heart Rate",71,202)
        if err: errors.append(err)
        if errors:
            for msg in errors: st.error(msg)
        else:
            vals = [
                int(age), 0 if gender_label.startswith("Female") else 1,
                int(chest_label.split()[0]), bp, chol, int(fbs_label.split()[0]),
                int(ecg_label.split()[0]), hr, int(angina_label.split()[0]),
                float(oldpeak), int(slope_label.split()[0]), int(vessels_label.split()[0])
            ]
            X = pd.DataFrame([vals], columns=FEATURES)
            try:
                pred, prob = predict(X)
                now = datetime.now().astimezone().isoformat(timespec="seconds")
                pid = "HC-" + uuid.uuid4().hex[:8].upper()
                record = dict(zip(FEATURES, vals))
                record.update({
                    "patient_id":pid, "created_at":now, "updated_at":now,
                    "prediction":pred, "predicted_probability":prob
                })
                save_record(record)
                st.session_state.current_patient_id = pid
                st.session_state.last_result = {
                    "patient_id":pid,"created_at":now,"prediction":pred,
                    "probability":prob,"input":X.copy()
                }
                st.success(f"Patient record saved successfully! Patient ID: {pid}")
            except Exception as e:
                st.error(f"Prediction or database save failed: {e}")
    if st.session_state.last_result is not None:
        display_result(st.session_state.last_result)

# ---------------- PATIENT RECORDS ----------------
elif page == "Patient Records":
    st.markdown('<p class="main-title">🗂️ Patient Records</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">View, search, update, download and delete saved patient information.</p>', unsafe_allow_html=True)
    try:
        records = read_records()
    except Exception as e:
        st.error(f"Could not read patient records: {e}")
        st.stop()
    st.metric("Total Saved Patient Records", len(records))
    st.divider()

    st.markdown("### 🔍 Search Patient")
    search_id = st.text_input("Enter exact Patient ID", placeholder="Example: HC-A1B2C3D4", key="search_id")
    shown = read_records(search_id) if search_id.strip() else records
    if shown.empty:
        st.info("No patient records found.")
    else:
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download displayed records as CSV",
            shown.to_csv(index=False).encode("utf-8"),
            "HeartCare_Patient_Records.csv","text/csv",use_container_width=True)

    st.divider()
    st.markdown("### ✏️ Update Patient Details")
    all_records = read_records()
    if all_records.empty:
        st.info("No patient records are available to update.")
    else:
        ids = all_records["patient_id"].tolist()
        selected_id = st.selectbox("Select Patient ID to update", ids, key="update_id")
        old = all_records[all_records.patient_id == selected_id].iloc[0]

        with st.form("update_patient_form"):
            st.caption("Edit details below. The same Patient ID will be retained and the model prediction will be recalculated.")
            c1,c2 = st.columns(2)
            with c1:
                e_age = st.number_input("Age (years)",1,120,int(old.age),key="e_age")
                e_gender = st.selectbox("Gender",["Female (0)","Male (1)"],int(old.gender),key="e_gender")
                e_chest = st.selectbox("Chest Pain Type",[
                    "0 (Typical angina)","1 (Atypical angina)","2 (Non-anginal chest pain)","3 (Asymptomatic)"],
                    int(old.chestpain),key="e_chest")
                e_bp = st.text_input("Resting Blood Pressure (mmHg)",str(int(old.restingBP)),key="e_bp")
                e_chol = st.text_input("Serum Cholesterol (mg/dL)",str(int(old.serumcholestrol)),key="e_chol")
                e_fbs = st.selectbox("Fasting Blood Sugar",["0 (≤120 mg/dL)","1 (>120 mg/dL)"],
                    int(old.fastingbloodsugar),key="e_fbs")
            with c2:
                e_ecg = st.selectbox("Resting ECG",[
                    "0 (Normal ECG)","1 (ST-T wave abnormality)","2 (Possible/definite LVH)"],
                    int(old.restingrelectro),key="e_ecg")
                e_hr = st.text_input("Maximum Heart Rate (bpm)",str(int(old.maxheartrate)),key="e_hr")
                e_angina = st.selectbox("Exercise-Induced Angina",["0 (No)","1 (Yes)"],
                    int(old.exerciseangia),key="e_angina")
                e_oldpeak = st.number_input("Oldpeak",0.0,6.2,float(old.oldpeak),0.1,key="e_oldpeak")
                e_slope = st.selectbox("Slope",[
                    "0 (Unspecified dataset category)","1 (Upsloping)","2 (Flat)","3 (Downsloping)"],
                    int(old.slope),key="e_slope")
                e_vessels = st.selectbox("Number of Major Vessels",[
                    "0 (0 major vessels)","1 (1 major vessel)","2 (2 major vessels)","3 (3 major vessels)"],
                    int(old.noofmajorvessels),key="e_vessels")
            do_update = st.form_submit_button("💾 Update Patient Details",use_container_width=True)

        if do_update:
            errors = []
            new_bp,err = parse_int(e_bp,"Resting Blood Pressure",94,200)
            if err: errors.append(err)
            new_chol,err = parse_int(e_chol,"Serum Cholesterol",126,602)
            if err: errors.append(err)
            new_hr,err = parse_int(e_hr,"Maximum Heart Rate",71,202)
            if err: errors.append(err)
            if errors:
                for msg in errors: st.error(msg)
            else:
                vals = [
                    int(e_age),0 if e_gender.startswith("Female") else 1,
                    int(e_chest.split()[0]),new_bp,new_chol,int(e_fbs.split()[0]),
                    int(e_ecg.split()[0]),new_hr,int(e_angina.split()[0]),
                    float(e_oldpeak),int(e_slope.split()[0]),int(e_vessels.split()[0])
                ]
                X = pd.DataFrame([vals],columns=FEATURES)
                try:
                    pred,prob = predict(X)
                    update_data = dict(zip(FEATURES,vals))
                    update_data.update({
                        "patient_id":selected_id,
                        "updated_at":datetime.now().astimezone().isoformat(timespec="seconds"),
                        "prediction":pred,"predicted_probability":prob
                    })
                    changed = update_record(update_data)
                    if changed:
                        st.session_state.last_result = None
                        st.success(f"Patient {selected_id} updated. Prediction recalculated.")
                        if pred == 1: st.error("Updated prediction: Heart Disease")
                        else: st.success("Updated prediction: No Heart Disease")
                        st.write(f"Updated predicted-class probability: {prob:.2f}%")
                        st.rerun()
                    else:
                        st.error("No matching record was updated. Refresh and try again.")
                except Exception as e:
                    st.error(f"Unable to update patient record: {e}")

    st.divider()
    st.markdown("### 🗑️ Delete Patient Record")
    delete_id = st.text_input("Enter exact Patient ID to delete",key="delete_id")
    confirm = st.checkbox("I confirm that I want to permanently delete this record.",key="confirm_delete")
    if st.button("Delete Patient Record"):
        if not delete_id.strip():
            st.error("Please enter a Patient ID.")
        elif not confirm:
            st.warning("Please confirm before deleting.")
        else:
            try:
                n = delete_record(delete_id)
                if n:
                    st.success("Patient record deleted successfully.")
                    st.rerun()
                else: st.warning("Patient ID not found.")
            except Exception as e: st.error(f"Unable to delete record: {e}")
    st.info("Update changes the existing row under the same Patient ID. This version stores the latest values, not a separate edit history.")


# ---------------- UNDERSTANDING PARAMETERS ----------------
elif page == "Understanding Health Parameters":
    st.markdown('<p class="main-title">Understanding Health Parameters</p>',unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Learn about the 12 parameters used by the prediction model.</p>',unsafe_allow_html=True)
    items = [
        ("1. Age","Patient age in years.","Enter age in years."),
        ("2. Gender","Gender value recorded by the dataset.","0 = Female; 1 = Male."),
        ("3. Chest Pain Type","Chest-pain category recorded for the patient.","0 = Typical angina; 1 = Atypical angina; 2 = Non-anginal pain; 3 = Asymptomatic."),
        ("4. Resting Blood Pressure","Blood pressure recorded at rest, in mmHg.","Use a measurement or medical report. App input range: 94–200."),
        ("5. Serum Cholesterol","Blood cholesterol measurement, usually mg/dL.","Use a laboratory report. App input range: 126–602; zero is not accepted."),
        ("6. Fasting Blood Sugar","Fasting blood sugar category used by the dataset.","0 = ≤120 mg/dL; 1 = >120 mg/dL."),
        ("7. Resting ECG","ECG records the heart's electrical activity at rest.","0 = Normal; 1 = ST-T wave abnormality; 2 = Possible/definite LVH."),
        ("8. Maximum Heart Rate","Maximum heart rate recorded during the relevant test, in bpm.","Use the test report. App input range: 71–202."),
        ("9. Exercise-Induced Angina","Indicates whether angina was recorded during exercise.","0 = No; 1 = Yes."),
        ("10. Oldpeak","ST-segment depression value recorded in an exercise test.","Use the report value. Dataset range: 0–6.2."),
        ("11. Slope","Slope of the peak exercise ST segment.","0 = Unspecified dataset category; 1 = Upsloping; 2 = Flat; 3 = Downsloping."),
        ("12. Number of Major Vessels","Number of major vessels recorded in the dataset.","0, 1, 2, or 3; use the medical report.")
    ]
    for title,meaning,values in items:
        with st.expander(title):
            st.write("**What is it?**",meaning)
            st.write("**Values / information:**",values)
    st.info("Do not guess ECG, Oldpeak, or major-vessel values. Use the patient's medical report and healthcare professional's guidance.")

# ---------------- HEART HEALTH TIPS ----------------
elif page == "Heart Health Tips":
    st.markdown('<p class="main-title">Heart Health Tips ❤️</p>',unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Simple habits that support a healthy heart.</p>',unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.image("https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=900",caption="Balanced food choices",use_container_width=True)
        st.markdown("### 🥗 Eat a Balanced Diet")
        st.write("Include fruits, vegetables, whole grains and balanced meals. Limit excess salt and highly processed foods.")
        st.markdown("### 🚶 Stay Physically Active")
        st.write("Choose regular physical activity suitable for your health.")
        st.markdown("### 🚭 Avoid Smoking")
        st.write("Avoid tobacco and seek professional support to stop smoking.")
    with c2:
        st.image("https://images.unsplash.com/photo-1506126613408-eca07ce68773?w=900",caption="Healthy lifestyle",use_container_width=True)
        st.markdown("### 😴 Get Enough Sleep")
        st.write("Maintain a regular sleep schedule and get enough rest.")
        st.markdown("### 🩺 Monitor Your Health")
        st.write("Check blood pressure, blood sugar and cholesterol as advised by your healthcare professional.")
        st.markdown("### 🧘 Manage Stress")
        st.write("Relaxation and healthy routines can help manage everyday stress.")
    st.warning("For chest pain, severe breathlessness, fainting or other emergency symptoms, seek urgent medical care.")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer"><hr><b>HeartCare AI | Heart Health Dashboard</b><br>
</div>
""",unsafe_allow_html=True)