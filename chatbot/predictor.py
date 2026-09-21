"""Motor de inferencia del chatbot.

Encapsula la carga del modelo entrenado y la logica de decision. Esta clase no
sabe nada de HTTP ni de Flask: esa separacion es la que cumple el RNF-03
(arquitectura desacoplada) y permite reutilizar el motor desde la consola,
desde tests o desde cualquier otro servidor.
"""

import json
import pickle
import random
import time

import numpy as np

from . import config
from .preprocessing import bolsa_de_palabras


class ChatbotPredictor:
    """Carga el modelo una sola vez y resuelve consultas en memoria.
    """

    def __init__(self):
        import tensorflow as tf

        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {config.MODEL_PATH}. "
                "Ejecutá primero: python train.py"
            )

        self.model = tf.keras.models.load_model(config.MODEL_PATH)

        with open(config.WORDS_PATH, "rb") as f:
            self.words = pickle.load(f)
        with open(config.CLASSES_PATH, "rb") as f:
            self.classes = pickle.load(f)
        with open(config.LEXICO_PATH, "rb") as f:
            self.lexico = pickle.load(f)
        with open(config.INTENTS_PATH, "r", encoding="utf-8") as f:
            self.intents = json.load(f)

        # Diccionario tag
        self.respuestas = {
            intent["tag"]: intent["responses"] for intent in self.intents["intents"]
        }

    def predecir(self, texto: str) -> list:
        vector, coincidencias, total = bolsa_de_palabras(texto, self.words, self.lexico)
        if coincidencias == 0:
            return []
        probabilidades = self.model(
            vector.reshape(1, -1), training=False
        ).numpy()[0]
        cobertura = coincidencias / total

        resultados = [
            {
                "tag": self.classes[i],
                "probabilidad": float(p),
                "certeza": float(p) * cobertura,
            }
            for i, p in enumerate(probabilidades)
            if float(p) * cobertura > config.ERROR_THRESHOLD
        ]
        resultados.sort(key=lambda r: r["certeza"], reverse=True)
        return resultados

    def responder(self, texto: str) -> dict:
        inicio = time.perf_counter()
        resultados = self.predecir(texto)

        if resultados:
            tag = resultados[0]["tag"]
            confianza = resultados[0]["certeza"]
            respuesta = random.choice(self.respuestas[tag])
            fallback = False
        else:
            # RF-04: respuesta por defecto con derivacion institucional.
            tag = None
            confianza = 0.0
            respuesta = config.FALLBACK_RESPONSE
            fallback = True

        return {
            "response": respuesta,
            "tag": tag,
            "confidence": round(confianza, 4),
            "fallback": fallback,
            "elapsed_ms": round((time.perf_counter() - inicio) * 1000, 2),
        }
