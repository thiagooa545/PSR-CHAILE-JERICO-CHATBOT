"""Configuracion central del proyecto.

Centraliza rutas e hiperparametros para que el script de entrenamiento
(train.py) y el servidor de red (app.py) compartan los mismos
valores y no haya desincronizacion entre lo entrenado y lo que da.
"""

from pathlib import Path

# -Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

INTENTS_PATH = BASE_DIR / "intents.json"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "chatbot_model.h5"
WORDS_PATH = MODEL_DIR / "words.pkl"
LEXICO_PATH = MODEL_DIR / "lexico.pkl"
CLASSES_PATH = MODEL_DIR / "classes.pkl"

# -RF-04: umbral de certeza
# Si la probabilidad maxima de la red no supera este valor, el bot no arriesga
# una respuesta: devuelve el mensaje generico y deriva a secretaria.
ERROR_THRESHOLD = 0.60

FALLBACK_RESPONSE = (
    "Lo siento, no logré entender tu consulta sobre el trámite. "
    "Por favor, comunicate con Secretaría al correo secretaria@almirantebrown36.edu.ar "
    "o acercate a Preceptoría de lunes a viernes de 8:00 a 12:00 hs."
)

# -Hiperparametros de la red neuronal (MLP)
HIDDEN_UNITS = (128, 64)
DROPOUT_RATE = 0.5
EPOCHS = 400
BATCH_SIZE = 8
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# -Servidor de red (RF-03 / RNF-03)
API_HOST = "0.0.0.0"
API_PORT = 5000
