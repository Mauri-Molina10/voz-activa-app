import os
import tempfile
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import librosa
import librosa.display
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder
from utils.predictor import predecir_audio

# Configuración de página
st.set_page_config(page_title="Detector de Voz", page_icon="🎤", layout="wide")

# Inicialización del estado de la sesión
if "historial" not in st.session_state: st.session_state.historial = []
if "audio_bytes" not in st.session_state: st.session_state.audio_bytes = None
if "audio_extension" not in st.session_state: st.session_state.audio_extension = None

st.title("🎤 Clasificador Inteligente de Voz")

# 1. Selector de modo (La clave para no mezclar archivos)
metodo = st.radio("Seleccioná cómo ingresar el audio:", ["Subir archivo", "Grabar con micrófono"])

# Función para limpiar el estado al cambiar de modo
def limpiar_estado():
    st.session_state.audio_bytes = None

# 2. Entrada de audio según el modo seleccionado
if metodo == "Subir archivo":
    uploaded_file = st.file_uploader("📂 Subir archivo", type=["wav", "mp3", "m4a", "ogg"], on_change=limpiar_estado)
    if uploaded_file:
        st.session_state.audio_bytes = uploaded_file.read()
        st.session_state.audio_extension = uploaded_file.name.split(".")[-1]
else:
    st.markdown("## 🎙️ Grabar desde micrófono")
    audio_grabado = mic_recorder(start_prompt="🎤 Empezar", stop_prompt="⏹️ Detener", key="recorder")
    if audio_grabado:
        st.session_state.audio_bytes = audio_grabado["bytes"]
        st.session_state.audio_extension = "wav"

# 3. Procesamiento y Análisis
if st.session_state.audio_bytes is not None:
    st.audio(st.session_state.audio_bytes)
    
    # Crear archivo temporal para librosa y el predictor
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{st.session_state.audio_extension}") as tmp:
        tmp.write(st.session_state.audio_bytes)
        temp_audio_path = tmp.name

    # Visualización de onda
    try:
        y, sr = librosa.load(temp_audio_path)
        fig, ax = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)
    except:
        st.warning("No se pudo generar la visualización.")

    # Botón para ejecutar la predicción
    if st.button("🚀 Analizar Audio"):
        with st.spinner("Procesando audio..."):
            try:
                resultado = predecir_audio(temp_audio_path)
                clase = resultado["prediccion"]
                probs = resultado["probabilidades"]
                texto = resultado["texto"]

                st.success(f"✅ Predicción: {clase.upper()}")
                st.write(f"**Texto detectado:** {texto}")

                # Gráfico de probabilidades
                df_probs = pd.DataFrame({"Clase": list(probs.keys()), "Probabilidad": list(probs.values())})
                fig = px.bar(df_probs, x="Clase", y="Probabilidad", title="📊 Probabilidades")
                st.plotly_chart(fig, use_container_width=True)

                # Guardar en historial
                st.session_state.historial.append({
                    "fecha": datetime.now().strftime("%H:%M:%S"),
                    "texto": texto,
                    "prediccion": clase
                })
            except Exception as e:
                st.error(f"Error procesando: {e}")

# 4. Historial
st.markdown("---")
st.subheader("🕘 Historial de Predicciones")
if st.session_state.historial:
    st.dataframe(pd.DataFrame(st.session_state.historial), use_container_width=True)
else:
    st.info("Todavía no hay predicciones.")