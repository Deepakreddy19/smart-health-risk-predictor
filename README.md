🩺 Smart Health Risk Predictor
A comprehensive, end-to-end machine learning application designed to provide early risk assessment for several chronic diseases, including Heart Disease, Diabetes, and Chronic Kidney Disease. This project showcases a full-stack approach, from model training and backend logic to a feature-rich frontend and automated cloud deployment.

<!-- Optional: Add a screenshot of your app here -->

✨ Features
Multi-Disease Prediction: Utilizes trained XGBoost and Scikit-learn models to provide risk probabilities for three major health conditions.

Interactive UI: A responsive and user-friendly interface built with Streamlit, featuring data visualizations and clear result interpretations.

AI Health Assistant: A Gemini-powered chatbot for answering general health and wellness questions in a conversational manner.

BMI Calculator: An integrated tool to calculate Body Mass Index, a key health indicator.

Anonymous Data Logging: Captures anonymized prediction data for future analysis and BI dashboard integration.

CI/CD Automation: A complete GitHub Actions workflow for automatically building a Docker container and publishing it to Docker Hub on every push to the main branch.

🛠️ Technology Stack
This project leverages a modern data science and MLOps technology stack:

Backend & Machine Learning:

Language: Python

Libraries: Scikit-learn, XGBoost, Pandas, NumPy

Frontend:

Framework: Streamlit

Visualizations: Plotly Express

AI Integration:

LLM: Google Gemini API (gemini-1.5-flash-latest)

Deployment & MLOps:

Containerization: Docker

CI/CD: GitHub Actions

Cloud Hosting (Target): AWS EC2

Data Storage (Target): AWS S3 / RDS for BI data logging

🚀 Getting Started
To run this application on your local machine, please follow the steps below.

Prerequisites
Python 3.9+

A Google Gemini API Key

Docker Desktop (for containerized execution)

Local Installation
Clone the repository:

git clone [https://github.com/Deepakreddy19/smart-health-risk-predictor.git](https://github.com/Deepakreddy19/smart-health-risk-predictor.git)
cd smart-health-risk-predictor

Create and activate a virtual environment:

python -m venv venv
source venv/Scripts/activate

Install the required dependencies:

pip install -r requirements.txt

Set up your API Key:

Create a folder named .streamlit in the root of the project.

Inside that folder, create a file named secrets.toml.

Add your Gemini API key to the file in the following format:

GEMINI_API_KEY = "your_api_key_here"

Running the Application
Using Streamlit
To run the app directly with Streamlit, use the following command:

streamlit run shrp.py

Using Docker (Recommended)
To run the application in a self-contained Docker container (the same way it runs in production):

Build the Docker image:

docker build -t health-predictor-app .

Run the Docker container:

docker run -p 8501:8501 health-predictor-app
