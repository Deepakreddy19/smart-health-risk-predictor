import streamlit as st
import boto3
import pickle
import os

# --- CONSTANTS ---
BUCKET_NAME = "mycsk-health-predictor-models-2025" # The S3 bucket name

# --- AWS S3 CLIENT SETUP ---
# Use Streamlit's secrets for local development, falls back to IAM role on EC2
s3_client = boto3.client(
    's3',
    aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
    region_name='us-east-1' # IMPORTANT: Change if your bucket is in a different region
)

# --- MODEL LOADING FUNCTION ---
@st.cache_resource
def load_model_from_s3(model_key):
    """
    Downloads a model file from S3 and loads it into memory using pickle.
    Uses Streamlit's caching to avoid re-downloading on every script run.

    Args:
        model_key (str): The filename of the model in the S3 bucket (e.g., "heart_model.pkl").

    Returns:
        A loaded machine learning model object, or None if an error occurs.
    """
    try:
        # Get the object from S3
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=model_key)
        
        # Read the object's content
        model_bytes = response['Body'].read()
        
        # Load the model using pickle
        loaded_model = pickle.loads(model_bytes)
        
        print(f"Successfully loaded model '{model_key}' from S3.")
        return loaded_model

    except Exception as e:
        st.error(f"Error loading model '{model_key}' from S3: {e}")
        return None

# --- EXAMPLE USAGE (for reference, not to be run directly) ---
if __name__ == '__main__':
    # This block is for testing purposes if you run this file directly.
    # In your main app, you will import the function and use it as shown below.
    
    st.title("S3 Model Loader Test")

    # How to use in your main app.py:
    # 1. from s3_model_loader import load_model_from_s3
    # 2. Define your model files
    model_files_to_load = {
        'Heart': 'heart_model.pkl',
        'Diabetes': 'diabetes_model.pkl',
        'Kidney': 'kidney_model.pkl'
    }

    # 3. Load them into a dictionary
    models = {}
    for name, key in model_files_to_load.items():
        with st.spinner(f"Loading {name} model..."):
            models[name] = load_model_from_s3(key)
    
    if all(models.values()):
        st.success("All models loaded successfully!")
        st.write(models)
    else:
        st.error("One or more models failed to load. Check the error messages above.")

