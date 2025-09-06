import streamlit as st
import pickle
import random
import plotly.express as px
import google.generativeai as genai
import os
import xgboost

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🩺 Smart Health Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="❤️‍🩹"
)

# --- DIRECT CSS INJECTION WITH NEW BACKGROUND ---
custom_css = """
/* --- HEARTBEAT ANIMATION BACKGROUND (FIXED) --- */
@keyframes ekg-scroll {
    from {
        background-position-x: 0;
    }
    to {
        background-position-x: -1000px; /* This value should match the width of the SVG pattern */
    }
}

.stApp {
    background-color: #F0F2F6; /* A fallback color */
    background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1000 100'%3e%3cpath d='M0 50 L150 50 L160 40 L170 60 L180 45 L190 55 L200 50 L800 50 L810 55 L820 45 L830 60 L840 40 L850 50 L1000 50' stroke='%23FF4B4B' stroke-width='2' fill='none' stroke-opacity='0.2'/%3e%3c/svg%3e");
    background-repeat: repeat-x; /* Allow the image to repeat horizontally */
    background-size: 1000px auto; /* Set the size of one repeating pattern */
    background-position: center;
    animation: ekg-scroll 10s linear infinite; /* Use the corrected animation */
}

/* Force a white background on the sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
}

/* --- MAIN CARD STYLING --- */
.st-emotion-cache-1y4p8pa {
    background-color: rgba(255, 255, 255, 0.9); /* Slightly transparent cards */
    backdrop-filter: blur(10px); /* Frosted glass effect */
    border-radius: 10px;
    padding: 2rem;
    border: 1px solid #E0E0E0;
}

/* --- FORCE DARK TEXT --- */
.st-emotion-cache-1y4p8pa,
.st-emotion-cache-1y4p8pa p, 
.st-emotion-cache-1y4p8pa li,
.st-emotion-cache-1y4p8pa label,
.st-emotion-cache-1y4p8pa h1,
.st-emotion-cache-1y4p8pa h2,
.st-emotion-cache-1y4p8pa h3,
.st-emotion-cache-1y4p8pa .st-emotion-cache-1kyxreq {
    color: #31333F !important;
}

/* --- TAB STYLING --- */
[data-testid="stTabs"] button {
    font-weight: bold;
    color: #888;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FF4B4B;
    border-bottom: 2px solid #FF4B4B;
}

/* --- BUTTON STYLING --- */
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

/* --- SLIDER STYLING --- */
div[data-testid="stSlider"] > div:nth-child(2) > div > div > div:nth-child(2) {
    background-color: #FF4B4B;
}
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #FF4B4B;
}

/* --- RADIO BUTTON STYLING --- */
[data-testid="stRadio"] input:checked + div::before {
    background-color: #FF4B4B !important;
}
"""
st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
try:
    heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
    diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
    kidney_model = pickle.load(open("models/kidney_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'heart_model.pkl', 'diabetes_model.pkl', and 'kidney_model.pkl' are in the 'models/' directory.")
    st.stop()


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
    colors = ["#FF4B4B", "#00C0F2"] if prob > 0.5 else ["#4CAF50", "#E0E0E0"]
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

# ---------------- MAIN PAGE LAYOUT ----------------
st.title("Smart Health Risk Prediction Dashboard")
st.markdown("Select a tool from the tabs below to assess health risks or chat with our AI assistant.")

tab_heart, tab_diabetes, tab_kidney, tab_bmi, tab_chatbot, tab_about = st.tabs(["❤️ Heart Disease", "🩸 Diabetes", "🧬 Kidney Disease", "⚖️ BMI Calculator", "💬 AI Chatbot", "ℹ️ About"])

# ---------------- HEART DISEASE TAB ----------------
with tab_heart:
    with st.container(border=True):
        col_img, col_title = st.columns([1, 4])
        with col_img:
            st.image("https://cdn-icons-png.flaticon.com/512/2966/2966487.png", width=128)
        with col_title:
            st.header("❤️ Heart Disease Prediction")
            st.markdown("Fill in your details below to check your heart disease risk.")
        
        st.markdown("---")

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

        if st.button("🔎 Predict Heart Disease Risk", use_container_width=True, type="primary"):
            sex_val = 1 if sex == "Male" else 0
            fbs_val = 1 if fbs == "Yes" else 0
            exang_val = 1 if exang == "Yes" else 0
            features = [[age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]]
            
            prediction = heart_model.predict(features)[0]
            prob = heart_model.predict_proba(features)[0][1]
            
            st.markdown("---")
            st.subheader("Prediction Result")
            col1, col2 = st.columns([1, 2])
            with col1:
                show_risk_gauge(prob)
            with col2:
                if prediction == 1:
                    st.error("### ⚠️ High Risk of Heart Disease Detected.")
                    st.write("The model predicts a significant risk based on the provided data. Please consult a cardiologist for a comprehensive evaluation and further tests.")
                else:
                    st.success("### ✅ Low Risk of Heart Disease Detected.")
                    st.write("The model indicates a low risk. Continue to maintain a healthy lifestyle with regular check-ups to ensure long-term well-being.")

# ---------------- DIABETES TAB ----------------
with tab_diabetes:
    with st.container(border=True):
        col_img, col_title = st.columns([1, 4])
        with col_img:
            st.image("https://cdn-icons-png.flaticon.com/512/2873/2873204.png", width=128)
        with col_title:
            st.header("🩸 Diabetes Prediction")
            st.markdown("Enter your health parameters to check the risk of **Diabetes**.")
        
        st.markdown("---")

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

        if st.button("🔎 Predict Diabetes Risk", use_container_width=True, type="primary"):
            features = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi_input, dpf, age_d]]
            prediction = diabetes_model.predict(features)[0]
            prob = diabetes_model.predict_proba(features)[0][1]
            
            st.markdown("---")
            st.subheader("Prediction Result")
            col1, col2 = st.columns([1, 2])
            with col1:
                show_risk_gauge(prob)
            with col2:
                if prediction == 1:
                    st.error("### ⚠️ High Risk of Diabetes Detected.")
                    st.write("Your profile shows indicators associated with a higher risk of diabetes. It is highly recommended to consult a healthcare professional for a formal diagnosis.")
                else:
                    st.success("### ✅ Low Risk of Diabetes Detected.")
                    st.write("Your profile indicates a low risk for diabetes. Keep up with healthy habits, including a balanced diet and regular exercise, to maintain your well-being.")

# ---------------- KIDNEY DISEASE TAB ----------------
with tab_kidney:
    with st.container(border=True):
        col_img, col_title = st.columns([1, 4])
        with col_img:
            st.image("https://cdn-icons-png.flaticon.com/512/3503/3503838.png", width=128)
        with col_title:
            st.header("🧬 Chronic Kidney Disease Prediction")
            st.markdown("Provide your medical test results for kidney disease risk assessment.")

        st.markdown("---")
        
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

        if st.button("🔎 Predict Kidney Disease Risk", use_container_width=True, type="primary"):
            cat_map = { "rbc": {"normal": 0, "abnormal": 1}, "pc": {"normal": 0, "abnormal": 1}, "pcc": {"notpresent": 0, "present": 1}, "ba": {"notpresent": 0, "present": 1}, "htn": {"no": 0, "yes": 1}, "dm": {"no": 0, "yes": 1}, "cad": {"no": 0, "yes": 1}, "appet": {"good": 0, "poor": 1}, "pe": {"no": 0, "yes": 1}, "ane": {"no": 0, "yes": 1} }
            features = [[ age_k, bp, sg, al, su, cat_map["rbc"][rbc], cat_map["pc"][pc], cat_map["pcc"][pcc], cat_map["ba"][ba], bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, cat_map["htn"][htn], cat_map["dm"][dm], cat_map["cad"][cad], cat_map["appet"][appet], cat_map["pe"][pe], cat_map["ane"][ane] ]]
            
            prediction = kidney_model.predict(features)[0]
            prob = kidney_model.predict_proba(features)[0][1]
            
            st.markdown("---")
            st.subheader("Prediction Result")
            col1, col2 = st.columns([1, 2])
            with col1:
                show_risk_gauge(prob)
            with col2:
                if prediction == 1:
                    st.error("### ⚠️ High Risk of Chronic Kidney Disease Detected.")
                    st.write("The analysis suggests a potential risk of CKD. We strongly recommend consulting a nephrologist for further tests and a formal diagnosis.")
                else:
                    st.success("### ✅ Low Risk of Kidney Disease Detected.")
                    st.write("The model shows no immediate signs of kidney disease. Regular monitoring and a healthy lifestyle are key to prevention.")

# ---------------- BMI CALCULATOR TAB ----------------
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
            height = st.number_input("🧍 Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1, format="%.1f")
            weight = st.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1, format="%.1f")
        else: # Imperial
            col1, col2 = st.columns(2)
            with col1:
                feet = st.slider("🧍 Height (feet)", 3, 8, 5)
            with col2:
                inches = st.slider(" (inches)", 0, 11, 9)
            weight_lbs = st.number_input("⚖️ Weight (lbs)", min_value=60.0, max_value=450.0, value=150.0, step=0.1, format="%.1f")
            
            height = (feet * 30.48) + (inches * 2.54)
            weight = weight_lbs * 0.453592

        if st.button("Calculate BMI", use_container_width=True, type="primary"):
            if height > 0:
                height_m = height / 100
                bmi = weight / (height_m ** 2)
                
                st.markdown("---")
                st.subheader("Your BMI Result")
                
                if bmi < 18.5:
                    category = "Underweight"
                    st.warning(f"### Your BMI is {bmi:.2f} ({category})")
                elif 18.5 <= bmi < 25:
                    category = "Normal weight"
                    st.success(f"### Your BMI is {bmi:.2f} ({category})")
                elif 25 <= bmi < 30:
                    category = "Overweight"
                    st.warning(f"### Your BMI is {bmi:.2f} ({category})")
                else: # bmi >= 30
                    category = "Obese"
                    st.error(f"### Your BMI is {bmi:.2f} ({category})")
                    
                st.markdown("---")
                st.markdown("##### BMI Categories (WHO):")
                st.markdown("- **Below 18.5**: Underweight\n- **18.5 – 24.9**: Normal weight\n- **25.0 – 29.9**: Overweight\n- **30.0 and above**: Obese")
            else:
                st.error("Please enter a valid height.")

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
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
        except Exception:
            st.error("Could not configure Gemini. Have you added your API key to the .streamlit/secrets.toml file?")
            st.stop()

        SYSTEM_PROMPT = """You are a friendly and helpful AI Health Assistant...""" # (Same as before)
        
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
                        for chunk in stream:
                            yield chunk.text
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

        ### **Disclaimer:**
        This application is an educational tool and **is not a substitute for professional medical advice, diagnosis, or treatment.** The predictions are based on statistical models and should be considered as preliminary insights only. Always consult with a qualified healthcare provider for any health concerns.

        ### **Developer:**
        - **Name:** Vangoori Deepak Reddy
        - **Connect with me:** [LinkedIn](https://www.linkedin.com/in/vangoori-deepak-reddy-696890250) | [GitHub](https://github.com/Deepakreddy19)
        """)

