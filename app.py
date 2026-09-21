"""Servidor de red del chatbot escolar (RF-03).

Ejecutar:  python app.py

Expone una API REST en el puerto 5000. El servidor solo habla JSON sobre HTTP:
no genera ni sirve HTML, de modo que la interfaz grafica queda completamente
separada de la logica del chatbot (RNF-03). La comunicacion entre el frontend
y este backend se realiza por red mediante fetch(), habilitada por CORS.

Endpoints:
    GET  /        -> informacion de la API
    GET  /health  -> estado del servicio y del modelo
    POST /chat    -> recibe {"message": "..."} y devuelve {"response": "..."}
"""

import logging

from flask import Flask, jsonify, request
from flask_cors import CORS

from chatbot import ChatbotPredictor, config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chatbot-api")

app = Flask(__name__)
CORS(app)  # Permite solicitudes desde el frontend de la escuela (otro origen)

# El modelo se carga una unica vez al levantar el servidor, no por peticion.
# Es lo que garantiza el tiempo de respuesta exigido por el RNF-02.
logger.info("Cargando modelo de red neuronal...")
predictor = ChatbotPredictor()
logger.info(
    "Modelo cargado: %d intenciones, vocabulario de %d términos.",
    len(predictor.classes),
    len(predictor.words),
)


@app.route("/", methods=["GET"])
def index():
    """Descripcion de la API para verificar que el servicio esta publicado."""
    return jsonify({
        "servicio": "API Chatbot Escuela Técnica",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": 'Recibe {"message": "texto"} y devuelve la respuesta del bot',
            "GET /health": "Estado del servicio",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    """Chequeo de salud: util para la demostracion de red con Postman o curl."""
    return jsonify({
        "status": "ok",
        "modelo_cargado": True,
        "intenciones": len(predictor.classes),
        "vocabulario": len(predictor.words),
        "umbral_certeza": config.ERROR_THRESHOLD,
    })


@app.route("/chat", methods=["POST"])
def chat():
    """Ruta principal de la API REST (RF-03).

    Recibe la pregunta del usuario en formato JSON, la procesa con la red
    neuronal y devuelve la respuesta procesada por la IA, tambien en JSON.
    """
    # silent=True evita que Flask lance una excepcion si el cuerpo no es JSON
    # valido: preferimos responder un 400 controlado antes que un error 500.
    data = request.get_json(silent=True)

    if data is None or not isinstance(data, dict):
        return jsonify({
            "response": "La solicitud debe ser un JSON con el campo 'message'.",
            "error": "invalid_json",
        }), 400

    user_message = data.get("message", "")

    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({
            "response": "No enviaste ningún mensaje.",
            "error": "empty_message",
        }), 400

    # Limite defensivo: evita procesar cargas desmedidas desde la red.
    if len(user_message) > 500:
        return jsonify({
            "response": "Tu consulta es demasiado larga. Resumila en pocas palabras.",
            "error": "message_too_long",
        }), 413

    resultado = predictor.responder(user_message)

    logger.info(
        'Consulta: "%s" -> tag=%s conf=%.2f fallback=%s (%.1f ms)',
        user_message[:60],
        resultado["tag"],
        resultado["confidence"],
        resultado["fallback"],
        resultado["elapsed_ms"],
    )

    return jsonify(resultado), 200


@app.errorhandler(404)
def no_encontrado(_error):
    """Devuelve JSON en lugar del HTML por defecto de Flask."""
    return jsonify({"response": "Endpoint inexistente.", "error": "not_found"}), 404


@app.errorhandler(405)
def metodo_no_permitido(_error):
    return jsonify({
        "response": "Método HTTP no permitido para esta ruta. Usá POST en /chat.",
        "error": "method_not_allowed",
    }), 405


@app.errorhandler(500)
def error_interno(error):
    """Ultima barrera: el servidor nunca debe caerse ante una entrada rara."""
    logger.error("Error interno: %s", error)
    return jsonify({
        "response": config.FALLBACK_RESPONSE,
        "error": "internal_error",
    }), 500


if __name__ == "__main__":
    logger.info("Servidor escuchando en http://localhost:%d", config.API_PORT)
    app.run(host=config.API_HOST, port=config.API_PORT, debug=False)
