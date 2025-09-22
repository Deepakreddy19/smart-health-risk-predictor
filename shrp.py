import streamlit as st
import pickle
import random
import plotly.express as px
import google.generativeai as genai
import os
import xgboost
import pandas as pd
from datetime import datetime
from s3_model_loader import load_model_from_s3 # <-- IMPORT THE S3 LOADER

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🩺 Smart Health Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="❤️‍🩹"
)

# --- DIRECT CSS INJECTION ---
# ... (Your existing CSS is great, no changes needed here)
custom_css = """
/* --- HEARTBEAT ANIMATION BACKGROUND --- */
@keyframes ekg-scroll {
    from { background-position-x: 0; }
    to { background-position-x: -1000px; }
}
.stApp {
    background-color: #F0F2F6;
    background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1000 100'%3e%3cpath d='M0 50 L150 50 L160 40 L170 60 L180 45 L190 55 L200 50 L800 50 L810 55 L820 45 L830 60 L840 40 L850 50 L1000 50' stroke='%23FF4B4B' stroke-width='2' fill='none' stroke-opacity='0.2'/%3e%3c/svg%3e");
    background-repeat: repeat-x;
    background-size: 1000px auto;
    background-position: center;
    animation: ekg-scroll 10s linear infinite;
}
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
}
.st-emotion-cache-1y4p8pa {
    background-color: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 10px;
    padding: 2rem;
    border: 1px solid #E0E0E0;
}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label {
    color: #31333F !important;
}
[data-testid="stTabs"] button {
    font-weight: bold;
    color: #888 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FF4B4B !important;
    border-bottom: 2px solid #FF4B4B;
}
[data-testid="stButton"] > button {
    border-radius: 8px;
    font-weight: bold;
    border: none;
    background-color: #FF4B4B;
    color: white;
    transition: background-color 0.2s;
}
[data-testid="stButton"] > button:hover {
    background-color: #E04040;
    color: white;
}
div[data-testid="stSlider"] > div:nth-child(2) > div > div > div:nth-child(2) {
    background-color: #FF4B4B;
}
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #FF4B4B;
}
[data-testid="stRadio"] input:checked + div::before {
    background-color: #FF4B4B !important;
}
"""
st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)


# ---------------- LOAD MODELS FROM S3 ----------------
@st.cache_resource
def load_all_models():
    """Loads all models from S3 and caches them."""
    models = {}
    model_files = {
        'Heart Disease': 'heart_model.pkl',
        'Diabetes': 'diabetes_model.pkl',
        'Kidney Disease': 'kidney_model.pkl'
    }
    all_models_loaded = True
    for disease, model_key in model_files.items():
        models[disease] = load_model_from_s3(model_key)
        if models[disease] is None:
            all_models_loaded = False
    if not all_models_loaded:
        st.error("A critical error occurred while loading prediction models. The application cannot continue.")
        st.stop()
    return models

models = load_all_models()


# ---------------- SIDEBAR & UI STYLING ----------------
with st.sidebar:
    st.title("🩺 Smart Health Predictor")
    st.markdown("---")
    st.info("This application uses machine learning to predict the risk of several common diseases. It is not a substitute for professional medical advice.")
    st.markdown("---")
    st.subheader("🎉 Fun Health Fact")
    fun_facts = [
        "💧 Your kidneys filter about 50 gallons of blood every day!",
        "❤️ Your heart beats about 100,000 times a day!",
        "🩸 Diabetes affects over 400 million people worldwide.",
        "🚶 Walking 30 minutes daily improves overall health.",
        "💦 Drinking water boosts kidney and brain function."
    ]
    st.success(random.choice(fun_facts))
    st.markdown("---")


# ---------------- HELPER FUNCTIONS ----------------
def show_risk_gauge(prob):
    """Displays a Plotly gauge chart for the risk probability."""
    colors = ["#FF4B4B", "#E0E0E0"] if prob > 0.5 else ["#4CAF50", "#E0E0E0"]
    fig = px.pie(
        values=[prob, 1 - prob], 
        names=["Risk", "Safe"], 
        hole=0.7,
        color_discrete_sequence=colors
    )
    fig.update_traces(textinfo='none')
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0), 
        showlegend=False,
        annotations=[dict(text=f'{prob*100:.1f}%<br>Risk', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    st.plotly_chart(fig, use_container_width=True)

def log_prediction(disease_type, features, prediction, probability):
    """Logs prediction data to a CSV file."""
    # NOTE: In a containerized environment, this log is ephemeral. For persistent logging,
    # consider a database or logging service.
    log_file = "log_data.csv"
    timestamp = datetime.now().strftime("%Y-m-%d %H:%M:%S")
    
    log_data = { "timestamp": timestamp, "disease_type": disease_type, "prediction": "High Risk" if prediction == 1 else "Low Risk", "probability": f"{probability:.2f}" }
    feature_dict = {f"feature_{i+1}": val for i, val in enumerate(features)}
    log_data.update(feature_dict)
    
    df_log = pd.DataFrame([log_data])
    
    if not os.path.exists(log_file):
        df_log.to_csv(log_file, index=False)
    else:
        df_log.to_csv(log_file, mode='a', header=False, index=False)

# --- REFACTORED PREDICTION UI FUNCTION ---
def render_prediction_ui(disease_name, icon_url, description, model, get_features_func, high_risk_text, low_risk_text):
    """
    A generic function to render the UI for a disease prediction tab.
    """
    with st.container(border=True):
        col_img, col_title = st.columns([1, 4])
        with col_img:
            st.image(icon_url, width=128)
        with col_title:
            st.header(f"{disease_name} Prediction")
            st.markdown(description)
        
        st.markdown("---")
        
        # Get features from the specific function for this disease
        features = get_features_func()

        if st.button(f"🔎 Predict {disease_name} Risk", use_container_width=True, type="primary"):
            prediction = model.predict([features])[0]
            prob = model.predict_proba([features])[0][1]
            
            log_prediction(disease_name, features, prediction, prob)

            st.markdown("---")
            st.subheader("Prediction Result")
            col1, col2 = st.columns([1, 2])
            with col1:
                show_risk_gauge(prob)
            with col2:
                if prediction == 1:
                    st.error(f"### ⚠️ High Risk of {disease_name} Detected.")
                    st.write(high_risk_text)
                else:
                    st.success(f"### ✅ Low Risk of {disease_name} Detected.")
                    st.write(low_risk_text)

# --- FEATURE INPUT FUNCTIONS ---
def get_heart_disease_features():
    """Renders sliders and radios for heart disease and returns features."""
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("👤 Age", 1, 100, 45, key="h_age")
        sex = st.radio("⚧ Sex", ["Male", "Female"], key="h_sex")
        cp = st.slider("💢 Chest Pain Type (0-3)", 0, 3, 1, key="h_cp")
        trestbps = st.slider("🩺 Resting Blood Pressure", 80, 200, 120, key="h_trestbps")
        chol = st.slider("🧪 Cholesterol", 100, 600, 200, key="h_chol")
        fbs = st.radio("🍬 Fasting Blood Sugar >120 mg/dl?", ["Yes", "No"], key="h_fbs")
    with col2:
        restecg = st.slider("🩻 Resting ECG (0-2)", 0, 2, 1, key="h_restecg")
        thalach = st.slider("🏃 Max Heart Rate", 60, 220, 150, key="h_thalach")
        exang = st.radio("💔 Exercise Induced Angina", ["Yes", "No"], key="h_exang")
        oldpeak = st.slider("📉 ST Depression", 0.0, 10.0, 1.0, key="h_oldpeak")
        slope = st.slider("📈 Slope of ST Segment (0-2)", 0, 2, 1, key="h_slope")
        ca = st.slider("🩸 Major Vessels (0-3)", 0, 3, 0, key="h_ca")
        thal = st.slider("🧬 Thalassemia (0-3)", 0, 3, 2, key="h_thal")

    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0
    return [age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]

def get_diabetes_features():
    """Renders sliders for diabetes and returns features."""
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("🤰 Pregnancies", 0, 20, 1, key="d_preg")
        glucose = st.slider("🍭 Glucose", 0, 200, 120, key="d_glucose")
        blood_pressure = st.slider("🩺 Blood Pressure", 0, 122, 70, key="d_bp")
        skin_thickness = st.slider("🧍 Skin Thickness", 0, 100, 20, key="d_skin")
    with col2:
        insulin = st.slider("💉 Insulin", 0, 846, 79, key="d_insulin")
        bmi_input = st.slider("⚖️ BMI", 0.0, 70.0, 25.0, 0.1, key="d_bmi")
        dpf = st.slider("🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01, key="d_dpf")
        age_d = st.slider("👤 Age", 0, 120, 33, key="d_age")
    return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi_input, dpf, age_d]

def get_kidney_disease_features():
    """Renders inputs for kidney disease and returns features."""
    col1, col2 = st.columns(2)
    with col1:
        age_k = st.slider("👤 Age", 1, 100, 55, key="k_age")
        bp = st.slider("🩺 Blood Pressure", 40, 200, 80, key="k_bp")
        sg = st.number_input("🧪 Specific Gravity", 1.000, 1.030, 1.020, step=0.001, format="%.3f", key="k_sg")
        al = st.slider("💧 Albumin (0-5)", 0, 5, 0, key="k_al")
        su = st.slider("🍬 Sugar (0-5)", 0, 5, 0, key="k_su")
        rbc = st.radio("🔴 Red Blood Cells", ["normal", "abnormal"], key="k_rbc")
        pc = st.radio("🟢 Pus Cell", ["normal", "abnormal"], key="k_pc")
        pcc = st.radio("🧫 Pus Cell Clumps", ["present", "notpresent"], key="k_pcc")
        ba = st.radio("🦠 Bacteria", ["present", "notpresent"], key="k_ba")
        bgr = st.slider("🩸 Blood Glucose Random", 22, 500, 121, key="k_bgr")
        bu = st.slider("🧪 Blood Urea", 1, 400, 50, key="k_bu")
        sc = st.slider("💉 Serum Creatinine", 0.4, 76.0, 1.2, step=0.1, key="k_sc")
    with col2:
        sod = st.slider("🧂 Sodium", 4, 170, 135, key="k_sod")
        pot = st.slider("🥔 Potassium", 2.5, 50.0, 4.5, step=0.1, key="k_pot")
        hemo = st.slider("🩸 Hemoglobin", 3.0, 18.0, 15.0, step=0.1, key="k_hemo")
        pcv = st.slider("🧬 Packed Cell Volume", 9, 54, 40, key="k_pcv")
        wc = st.slider("⚪ WBC Count", 2200, 26400, 7800, step=100, key="k_wc")
        rc = st.slider("🔴 RBC Count", 2.1, 8.0, 5.2, step=0.1, key="k_rc")
        htn = st.radio("💢 Hypertension", ["yes", "no"], key="k_htn")
        dm = st.radio("🩸 Diabetes Mellitus", ["yes", "no"], key="k_dm")
        cad = st.radio("❤️ Coronary Artery Disease", ["yes", "no"], key="k_cad")
        appet = st.radio("🍽️ Appetite", ["good", "poor"], key="k_appet")
        pe = st.radio("🦶 Pedal Edema", ["yes", "no"], key="k_pe")
        ane = st.radio("💉 Anemia", ["yes", "no"], key="k_ane")

    cat_map = {"normal": 0, "abnormal": 1, "present": 1, "notpresent": 0, "yes": 1, "no": 0, "good": 0, "poor": 1}
    return [ age_k, bp, sg, al, su, cat_map[rbc], cat_map[pc], cat_map[pcc], cat_map[ba], bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, cat_map[htn], cat_map[dm], cat_map[cad], cat_map[appet], cat_map[pe], cat_map[ane] ]

# ---------------- MAIN PAGE LAYOUT ----------------
st.title("Smart Health Risk Prediction Dashboard")
st.markdown("Select a tool from the tabs below to assess health risks or chat with our AI assistant.")

tab_heart, tab_diabetes, tab_kidney, tab_bmi, tab_chatbot, tab_about = st.tabs(["❤️ Heart Disease", "🩸 Diabetes", "🧬 Kidney Disease", "⚖️ BMI Calculator", "💬 AI Chatbot", "ℹ️ About"])

# ---------------- PREDICTION TABS (USING REFACTORED FUNCTION) ----------------
with tab_heart:
    render_prediction_ui(
        disease_name="Heart Disease",
        icon_url="https://cdn-icons-png.flaticon.com/512/2966/2966487.png",
        description="Fill in your details below to check your heart disease risk.",
        model=models['Heart Disease'],
        get_features_func=get_heart_disease_features,
        high_risk_text="The model predicts a significant risk based on the provided data. Please consult a cardiologist for a comprehensive evaluation and further tests.",
        low_risk_text="The model indicates a low risk. Continue to maintain a healthy lifestyle with regular check-ups to ensure long-term well-being."
    )

with tab_diabetes:
    render_prediction_ui(
        disease_name="Diabetes",
        icon_url="https://cdn-icons-png.flaticon.com/512/2873/2873204.png",
        description="Enter your health parameters to check the risk of **Diabetes**.",
        model=models['Diabetes'],
        get_features_func=get_diabetes_features,
        high_risk_text="Your profile shows indicators associated with a higher risk of diabetes. It is highly recommended to consult a healthcare professional for a formal diagnosis.",
        low_risk_text="Your profile indicates a low risk for diabetes. Keep up with healthy habits, including a balanced diet and regular exercise, to maintain your well-being."
    )

with tab_kidney:
    render_prediction_ui(
        disease_name="Kidney Disease",
        icon_url="https://cdn-icons-png.flaticon.com/512/3503/3503838.png",
        description="Provide your medical test results for kidney disease risk assessment.",
        model=models['Kidney Disease'],
        get_features_func=get_kidney_disease_features,
        high_risk_text="The analysis suggests a potential risk of CKD. We strongly recommend consulting a nephrologist for further tests and a formal diagnosis.",
        low_risk_text="The model shows no immediate signs of kidney disease. Regular monitoring and a healthy lifestyle are key to prevention."
    )

# ---------------- BMI CALCULATOR TAB (FIXED) ----------------
with tab_bmi:
    with st.container(border=True):
        col_img, col_title = st.columns([1, 4])
        with col_img:
            st.image("https://cdn-icons-png.flaticon.com/512/1078/1078490.png", width=128)
        with col_title:
            st.header("⚖️ Body Mass Index (BMI) Calculator")
            st.markdown("Calculate your BMI to understand if you are in a healthy weight range.")
        
        st.markdown("---")
        unit = st.radio("Select Units", ["Metric (kg, cm)", "Imperial (lbs, ft)"])
        
        if unit == "Metric (kg, cm)":
            height_str = st.text_input("🧍 Height (cm)", value="170")
            weight_str = st.text_input("⚖️ Weight (kg)", value="70")
        else:
            col1, col2 = st.columns(2)
            with col1:
                feet_str = st.text_input("🧍 Height (feet)", value="5")
            with col2:
                inches_str = st.text_input(" (inches)", value="9")
            weight_lbs_str = st.text_input("⚖️ Weight (lbs)", value="150")

        if st.button("Calculate BMI", use_container_width=True, type="primary"):
            try:
                if unit == "Metric (kg, cm)":
                    height = float(height_str)
                    weight = float(weight_str)
                else:
                    feet, inches, weight_lbs = int(feet_str), int(inches_str), float(weight_lbs_str)
                    height = (feet * 30.48) + (inches * 2.54)
                    weight = weight_lbs * 0.453592
                
                if height > 0:
                    height_m = height / 100
                    bmi = weight / (height_m ** 2)
                    
                    st.markdown("---")
                    st.subheader("Your BMI Result")
                    
                    if bmi < 18.5: category, st_func = "Underweight", st.warning
                    elif 18.5 <= bmi < 25: category, st_func = "Normal weight", st.success
                    elif 25 <= bmi < 30: category, st_func = "Overweight", st.warning
                    else: category, st_func = "Obese", st.error
                    st_func(f"### Your BMI is {bmi:.2f} ({category})")
                        
                    st.markdown("---")
                    st.markdown("##### BMI Categories (WHO):")
                    st.markdown("- **Below 18.5**: Underweight\n- **18.5 – 24.9**: Normal weight\n- **25.0 – 29.9**: Overweight\n- **30.0 and above**: Obese")
                else:
                    st.error("Please enter a valid height greater than 0.")
            except (ValueError, TypeError):
                st.error("Invalid input. Please enter valid numbers for height and weight.")


# ---------------- AI CHATBOT TAB ----------------
with tab_chatbot:
    with st.container(border=True):
        st.header("💬 AI Health Assistant")
        st.markdown("I'm here to answer general health questions. **I am not a doctor.** Always consult a healthcare professional for medical advice or diagnosis.")
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_session = None
            st.session_state.messages = []
            st.rerun()
        try:
            api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                st.error("Gemini API key not found. Please set it in your secrets or environment variables.")
                st.stop()
            genai.configure(api_key=api_key)
        except Exception as e:
            st.error(f"Could not configure Gemini: {e}")
            st.stop()

        SYSTEM_PROMPT = """You are a friendly and helpful AI Health Assistant. Your role is to provide general health information...""" # Truncated for brevity
        
        if "chat_session" not in st.session_state or st.session_state.chat_session is None:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", system_instruction=SYSTEM_PROMPT)
            st.session_state.chat_session = model.start_chat(history=[])

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask a health-related question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                with st.chat_message("assistant"):
                    response_stream = st.session_state.chat_session.send_message(prompt, stream=True)
                    def stream_generator(stream):
                        for chunk in stream: yield chunk.text
                    full_response = st.write_stream(stream_generator(response_stream))
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"An error occurred: {e}")

# ---------------- ABOUT TAB ----------------
with tab_about:
    with st.container(border=True):
        st.header("ℹ️ About This Project")
        st.markdown("""
        The **Smart Health Predictor** is an end-to-end machine learning project designed to provide early risk assessment for several chronic diseases. 
        This tool leverages the power of data science to help users understand potential health risks based on their own metrics.
        ### **Key Features:**
        - **Multi-Disease Prediction:** Utilizes trained XGBoost and Scikit-learn models for Heart Disease, Diabetes, and Chronic Kidney Disease.
        - **Interactive UI:** A responsive and user-friendly interface built with Streamlit.
        - **AI Health Assistant:** A Gemini-powered chatbot for general health inquiries.
        - **Data-Driven Insights:** Includes a BMI calculator as a key health indicator.
        ### **Technology Stack:**
        - **Language:** Python
        - **Machine Learning:** Scikit-learn, XGBoost, Pandas
        - **Frontend:** Streamlit, Plotly
        - **AI Integration:** Google Gemini API
        - **Cloud & CI/CD:** AWS EC2, S3, Docker, Nginx, GitHub Actions
        ### **Disclaimer:**
        This application is an educational tool and **is not a substitute for professional medical advice, diagnosis, or treatment.** The predictions are based on statistical models and should be considered as preliminary insights only. Always consult with a qualified healthcare provider for any health concerns.
        ### **Developer:**
        - **Name:** VANGOORI DEEPAK REDDY
        - **Role:** Data Scientist | ML Engineer
        - **Connect with me:** [LinkedIn](https://www.linkedin.com/in/vangoori-deepak-reddy-696890250) | [GitHub](https://github.com/Deepakreddy19)
        """)

