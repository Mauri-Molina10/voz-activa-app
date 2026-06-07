import gc
from sentence_transformers import SentenceTransformer
from utils.audio_processing import transcribir_audio
from utils.model_loader import cargar_modelo

# Carga única
modelo_svm = cargar_modelo()
modelo_embeddings = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def predecir_audio(audio_path):
    # 1. Transcribir
    texto = transcribir_audio(audio_path)

    # 2. Embedding
    embedding = modelo_embeddings.encode([texto])

    # 3. Predicción
    prediccion = modelo_svm.predict(embedding)[0]
    probabilidades = modelo_svm.predict_proba(embedding)[0]

    # modelo_svm.classes_ contiene el orden real que aprendió el modelo durante el entrenamiento
    clases_reales = modelo_svm.classes_ 
    
    probs_dict = {clases_reales[i]: float(probabilidades[i]) for i in range(len(clases_reales))}
    # -----------------------

    # --- LIBERACIÓN DE MEMORIA ---
    del embedding
    gc.collect() 

    return {
        "texto": texto,
        "prediccion": prediccion,
        "probabilidades": probs_dict
    }
