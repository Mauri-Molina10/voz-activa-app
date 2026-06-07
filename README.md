# 🎙️ VozActiva  
### Tecnología al servicio de la comunicación humana

---

## 🧠 Descripción del Proyecto

**VozActiva** es un sistema inteligente de comunicación aumentativa diseñado para personas con dificultades severas del habla (como ELA, parálisis cerebral, ACVs, entre otras).

A diferencia de los métodos tradicionales rígidos, esta solución utiliza **Inteligencia Artificial** para interpretar la intención detrás de las palabras, devolviendo autonomía comunicativa a quienes más lo necesitan.

> “La máquina no obliga al paciente a adaptarse, sino que la máquina entiende al paciente.”

---

## 👥 Equipo - RGB Squad

- **Maciel Barbero** 
- **Mauricio Molina** 
- **Jonathan Molina** 

---

## 💡 Problemática

La pérdida de la capacidad de comunicación oral implica una pérdida crítica de autonomía.

Los sistemas actuales (como tableros pictográficos) suelen ser **lentos y rígidos**.

**VozActiva propone un cambio de paradigma:** interpretar la intención del usuario en lugar de obligarlo a expresarse de forma estructurada.

---

## 🛠️ Arquitectura Técnica

El sistema sigue un pipeline de procesamiento de IA optimizado:

### 🎤 1. Captura y normalización
- Audio estandarizado a **16 kHz**
- Mejora de calidad de señal

### 📝 2. Transcripción
- Uso de **Whisper (OpenAI)**
- Conversión de audio a texto preciso

### 🧠 3. Procesamiento semántico
- Modelos **Sentence Transformers**
- Generación de embeddings
- Interpretación de intención:
  - “Tengo sed”
  - “Necesito agua”
  - “Quiero tomar agua”  
  → mismas representaciones semánticas

### 🤖 4. Clasificación inteligente
- Modelo: **SVM (Support Vector Machine)**
- Clases:
  - 💧 Agua  
  - 🩺 Dolor  
  - 🔥 Fuego  

---

## 📊 Estadísticas del modelo

- 📦 Dataset: 63 muestras de entrenamiento  
- ⚡ Latencia: ~2.36 segundos por comando  
- 🎯 Precisión: >92% de exactitud interna  

---

## 🛡️ Seguridad Médica (Confianza del sistema)

Para garantizar seguridad en entornos críticos:

- Umbral de confianza: **70%**
- Si la predicción no supera ese valor:
  - El sistema rechaza la acción
  - Muestra: *"Intención no reconocida"*

Esto reduce falsos positivos en situaciones sensibles.

---

## 🔮 Futuro del Proyecto

- 📈 Ampliación del conjunto de intenciones  
- 🏥 Validación con datos clínicos reales  
- 📱 Optimización para Edge AI (móviles y hospitales)  

---

## 🧾 Cierre

> “La voz es el puente que nos conecta con el mundo. Con VozActiva, no solo optimizamos tensores y latencias; estamos construyendo tecnología para devolverle la voz a quienes el mundo dejó de escuchar.”

---

📌 Proyecto presentado como parte del **ABP - Tecnicatura Superior en Ciencia de Datos e IA**
