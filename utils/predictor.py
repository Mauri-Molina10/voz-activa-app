import gc # Importante: recolector de basura
from sentence_transformers import SentenceTransformer
from utils.audio_processing import transcribir_audio
from utils.model_loader import cargar_modelo

# Carga única
modelo_svm = cargar_modelo()
modelo_embeddings = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
clases = ["agua", "fuego", "dolor"]

def predecir_audio(audio_path):
    # 1. Transcribir
    texto = transcribir_audio(audio_path)

    # 2. Embedding
    embedding = modelo_embeddings.encode([texto])

    # 3. Predicción
    prediccion = modelo_svm.predict(embedding)[0]
    probabilidades = modelo_svm.predict_proba(embedding)[0]

    # --- LIBERACIÓN DE MEMORIA ---
    del embedding # Borramos el objeto de memoria
    gc.collect()  # Forzamos al sistema a limpiar RAM
    # -----------------------------

    probs_dict = {clases[i]: float(probabilidades[i]) for i in range(len(clases))}

    return {
        "texto": texto,
        "prediccion": prediccion,
        "probabilidades": probs_dict
    }
