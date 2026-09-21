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

    Cargar los artefactos en el constructor (y no en cada peticion) es lo que
    mantiene la respuesta muy por debajo de los 2 segundos exigidos por el
    RNF-02.
    """

    def __init__(self):
        # Se importa TensorFlow aca dentro para que el arranque del modulo sea
        # liviano y el error sea claro si falta la dependencia.
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

        # Diccionario tag -> lista de respuestas, para resolver en O(1).
        self.respuestas = {
            intent["tag"]: intent["responses"] for intent in self.intents["intents"]
        }

    def predecir(self, texto: str) -> list:
        """Devuelve las intenciones que superan el umbral, de mayor a menor.

        Cada elemento es un dict con 'tag', 'probabilidad' (salida cruda de la
        red) y 'certeza' (probabilidad ponderada por cobertura lexica). La
        lista vacia significa que el bot no esta seguro de ninguna intencion.
        """
        vector, coincidencias, total = bolsa_de_palabras(texto, self.words, self.lexico)

        # Guarda anti-ruido: si no se reconocio ninguna palabra del dominio
        # (por ejemplo "asdasd" o una pregunta sobre otro tema), no tiene
        # sentido consultar a la red; el resultado seria arbitrario.
        if coincidencias == 0:
            return []

        # Se invoca al modelo directamente en lugar de usar model.predict().
        # predict() esta pensado para lotes grandes y agrega cerca de 2 s de
        # sobrecarga en cada llamada individual, lo que incumpliria el RNF-02;
        # la llamada directa resuelve en pocos milisegundos.
        probabilidades = self.model(
            vector.reshape(1, -1), training=False
        ).numpy()[0]

        # Cobertura lexica: que porcentaje de la consulta pertenece al dominio
        # escolar. Una frase como "quiero pedir una pizza" comparte palabras
        # sueltas con el dominio ("quiero pedir") y la red la clasificaria con
        # alta probabilidad; ponderar por la cobertura evita ese falso positivo
        # y hace que el filtro de confianza del RF-04 sea realmente efectivo.
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
        """Resuelve una consulta completa y devuelve la respuesta y su metadata."""
        inicio = time.perf_counter()
        resultados = self.predecir(texto)

        if resultados:
            tag = resultados[0]["tag"]
            confianza = resultados[0]["certeza"]
            respuesta = random.choice(self.respuestas[tag])
            fallback = False
        else:
            # RF-04: respuesta por defecto amable con derivacion institucional.
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
