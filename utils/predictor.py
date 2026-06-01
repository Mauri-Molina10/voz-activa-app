from sentence_transformers import SentenceTransformer

from utils.audio_processing import transcribir_audio
from utils.model_loader import cargar_modelo

# ======================================
# CARGA GLOBAL
# ======================================

modelo_svm = cargar_modelo()

modelo_embeddings = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

clases = ["agua", "ayuda", "dolor"]


# ======================================
# PREDICCIÓN
# ======================================


def predecir_audio(audio_path):

    # 1. Transcribir audio
    texto = transcribir_audio(audio_path)

    # 2. Embedding
    embedding = modelo_embeddings.encode([texto])

    # 3. Predicción
    prediccion = modelo_svm.predict(embedding)[0]

    # 4. Probabilidades
    probabilidades = modelo_svm.predict_proba(embedding)[0]

    probs_dict = {
        clases[i]: float(probabilidades[i])
        for i in range(len(clases))
    }

    return {
        "texto": texto,
        "prediccion": prediccion,
        "probabilidades": probs_dict
    }