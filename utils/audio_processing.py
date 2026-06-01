import whisper

# Cargar modelo Whisper una sola vez
modelo_whisper = whisper.load_model("base")


def transcribir_audio(audio_path):

    resultado = modelo_whisper.transcribe(audio_path, language="es")

    texto = resultado["text"]

    return texto