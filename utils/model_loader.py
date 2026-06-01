import joblib

def cargar_modelo():
    modelo = joblib.load("modelo_svm_vozactiva.pkl")
    return modelo