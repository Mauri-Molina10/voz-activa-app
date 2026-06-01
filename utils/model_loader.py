import joblib
import os
import streamlit as st

@st.cache_resource
def cargar_modelo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "modelo_svm_vozactiva.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"¡El modelo no existe en: {model_path}!")
        
    return joblib.load(model_path)
