"""Paquete del asistente virtual escolar.

Modulos:
    config        -> rutas e hiperparametros compartidos.
    preprocessing -> pipeline de PLN (RF-01).
    predictor     -> motor de inferencia desacoplado del servidor (RNF-03).
"""

from .predictor import ChatbotPredictor

__all__ = ["ChatbotPredictor"]
__version__ = "1.0.0"
