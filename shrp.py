import streamlit as st
import pickle
import random
import plotly.express as px
import google.generativeai as genai
import os
import xgboost
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🩺 Smart Health Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="❤️‍🩹"
)

# --- DIRECT CSS INJECTION WITH NEW BACKGROUND ---
# ... (CSS code is the same, redacted for brevity)
custom_css = """
/* --- HEARTBEAT ANIMATION BACKGROUND (FIXED) --- */
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
/* ... rest of CSS ... */
[data-testid="stSidebar"] { background-color: #FFFFFF !important; }
.st-emotion-cache-1y4p8pa { background-color: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-radius: 10px; padding: 2rem; border: 1px solid #E0E0E0; }
.st-emotion-cache-1y4p8pa, .st-emotion-cache-1y4p8pa p, .st-emotion-cache-1y4p8pa li, .st-emotion-cache-1y4p8pa label, .st-emotion-cache-1y4p8pa h1, .st-emotion-cache-1y4p8pa h2, .st-emotion-cache-1y4p8pa h3, .st-emotion-cache-1y4p8pa .st-emotion-cache-1kyxreq { color: #31333F !important; }
[data-testid="stTabs"] button { font-weight: bold; color: #888; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #FF4B4B; border-bottom: 2px solid #FF4B4B; }
[data-testid="stButton"] > button { border-radius: 8px; font-weight: bold; border: none; background-color: #FF4B4B; color: white; transition: background-color 0.2s; }
[data-testid="stButton"] > button:hover { background-color: #E04040; color: white; }
div[data-testid="stSlider"] > div:nth-child(2) > div > div > div:nth-child(2) { background-color: #FF4B4B; }
div[data-testid="stSlider"] div[role="slider"] { background-color: #FF4B4B; }
[data-testid="stRadio"] input:checked + div::before { background-color: #FF4B4B !important; }
"""
st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------
# (Code is the same as before)
try:
    heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
    diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
    kidney_model = pickle.load(open("models/kidney_model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found...")
    st.stop()

# ---------------- SIDEBAR & UI STYLING ----------------
# (Code is the same as before)
with st.sidebar:
    st.title("🩺 Smart Health Predictor")
    st.markdown("---")
    st.info("This application uses machine learning...")
    st.markdown("---")
    st.subheader("🎉 Fun Health Fact")
    fun_facts = ["..."]
    st.success(random.choice(fun_facts))
    st.markdown("---")

# ---------------- HELPER FUNCTIONS ----------------
# (Code is the same as before)
def show_risk_gauge(prob):
    #...
    pass
def log_prediction(disease_type, features, prediction, probability):
    #...
    pass

# ---------------- MAIN PAGE LAYOUT ----------------
# (Code is the same as before)
st.title("Smart Health Risk Prediction Dashboard")
st.markdown("Select a tool from the tabs below...")
tab_heart, tab_diabetes, tab_kidney, tab_bmi, tab_chatbot, tab_about = st.tabs(["❤️ Heart Disease", "🩸 Diabetes", "🧬 Kidney Disease", "⚖️ BMI Calculator", "💬 AI Chatbot", "ℹ️ About"])

# ---------------- OTHER TABS ----------------
with tab_heart:
    # ... (code is the same)
    pass
with tab_diabetes:
    # ... (code is the same)
    pass
with tab_kidney:
    # ... (code is the same)
    pass
with tab_bmi:
    # ... (code is the same)
    pass

# ---------------- AI CHATBOT TAB (UPDATED FOR DEBUGGING) ----------------
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
            # --- THIS IS THE UPDATED DEBUGGING PART ---
            api_key = None
            
            # 1. Explicitly check the environment variable
            env_api_key = os.getenv("GEMINI_API_KEY")
            st.info(f"Attempting to read environment variable 'GEMINI_API_KEY'...")
            if env_api_key:
                st.success("Found API key in environment variable.")
                api_key = env_api_key
            else:
                st.warning("Environment variable 'GEMINI_API_KEY' not found.")

            # 2. If environment variable fails, try Streamlit secrets
            if not api_key:
                st.info("Attempting to read Streamlit secrets...")
                if "GEMINI_API_KEY" in st.secrets:
                    st.success("Found API key in Streamlit secrets.")
                    api_key = st.secrets["GEMINI_API_KEY"]
                else:
                    st.warning("API key not found in Streamlit secrets.")

            # 3. Final check and configure
            if not api_key:
                st.error("Could not find Gemini API key. Please ensure it is set correctly for your environment.")
                st.stop()
            
            genai.configure(api_key=api_key)

        except Exception as e:
            st.error(f"An error occurred during configuration: {e}")
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
            # ... (rest of chatbot logic is the same)
            pass


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

