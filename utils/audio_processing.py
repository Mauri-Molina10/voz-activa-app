import whisper
import streamlit as st

# Usamos cache_resource para que el modelo se cargue una sola vez en la RAM
@st.cache_resource
def cargar_whisper():
    # Cambiamos a 'tiny' para reducir drásticamente el uso de memoria
    return whisper.load_model("tiny")

# Cargamos el modelo una vez
modelo_whisper = cargar_whisper()

def transcribir_audio(audio_path):
    # Transcribimos usando la instancia en caché
    resultado = modelo_whisper.transcribe(audio_path, language="es")
    texto = resultado["text"]
    return texto
