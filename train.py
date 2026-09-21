"""Script de entrenamiento de la red neuronal del chatbot.

Ejecutar:  python train.py

Toma la base de conocimiento intents.json, preprocesa el texto (tokenizacion y
lematizacion), arma el set de entrenamiento con la tecnica de Bolsa de Palabras
(Bag of Words), entrena una red neuronal secuencial densa (MLP) y guarda en
disco el modelo junto con las estructuras de datos que necesita el servidor.

Artefactos generados en model/:
    chatbot_model.h5  -> red neuronal entrenada
    words.pkl         -> vocabulario (raices ordenadas)
    classes.pkl       -> etiquetas de intencion ordenadas
    lexico.pkl        -> palabras completas, para corregir errores de tipeo
"""

import sys

# En Windows la consola usa cp1252 por defecto y rompe las tildes del castellano.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from chatbot import config
from chatbot.preprocessing import lematizar, tokens_significativos


def cargar_intents() -> dict:
    """Lee y valida la base de conocimiento."""
    with open(config.INTENTS_PATH, "r", encoding="utf-8") as f:
        intents = json.load(f)

    if not intents.get("intents"):
        raise ValueError("intents.json no contiene intenciones.")

    for intent in intents["intents"]:
        if not intent.get("patterns") or not intent.get("responses"):
            raise ValueError(
                f"La intención '{intent.get('tag')}' no tiene patrones o respuestas."
            )
    return intents


def construir_corpus(intents: dict):
    """Aplica el pipeline de PLN y arma vocabulario, clases y documentos.

    Returns:
        words: vocabulario ordenado y sin duplicados (raices lematizadas).
        classes: lista ordenada de tags.
        documents: pares (tokens_procesados, tag) usados como ejemplos.
        lexico: palabras completas sin lematizar; el servidor las usa para
            corregir los errores de tipeo del usuario antes de lematizar.
    """
    words, classes, documents, lexico = [], [], [], []

    for intent in intents["intents"]:
        tag = intent["tag"]
        if tag not in classes:
            classes.append(tag)

        for pattern in intent["patterns"]:
            crudos = tokens_significativos(pattern)   # tokeniza + quita stopwords
            if not crudos:
                continue
            tokens = lematizar(crudos)                # reduce a las raices
            lexico.extend(crudos)
            words.extend(tokens)
            documents.append((tokens, tag))

    words = sorted(set(words))
    classes = sorted(set(classes))
    lexico = sorted(set(lexico))
    return words, classes, documents, lexico


def construir_dataset(words: list, classes: list, documents: list):
    """Vectoriza los documentos con Bolsa de Palabras y codifica las salidas.

    Cada ejemplo de entrada es un vector binario del largo del vocabulario y
    cada salida es un vector one-hot del largo de la cantidad de clases.
    """
    indice = {palabra: i for i, palabra in enumerate(words)}
    salida_vacia = [0] * len(classes)
    training = []

    for tokens, tag in documents:
        bolsa = [0] * len(words)
        for token in tokens:
            bolsa[indice[token]] = 1

        fila_salida = list(salida_vacia)
        fila_salida[classes.index(tag)] = 1
        training.append((bolsa, fila_salida))

    random.shuffle(training)
    train_x = np.array([t[0] for t in training], dtype=np.float32)
    train_y = np.array([t[1] for t in training], dtype=np.float32)
    return train_x, train_y


def construir_modelo(n_entradas: int, n_salidas: int) -> Sequential:
    """Red neuronal secuencial densa (MLP) para clasificacion multiclase.

    Arquitectura: 128 -> Dropout -> 64 -> Dropout -> softmax
    - ReLU en las capas ocultas: converge rapido y evita el desvanecimiento
      del gradiente.
    - Dropout del 50%: regularizacion necesaria porque el dataset es chico.
    - Softmax en la salida: devuelve una distribucion de probabilidad sobre
      las intenciones, lo que permite aplicar el umbral de certeza del RF-04.
    - Perdida categorical_crossentropy: la adecuada para salidas one-hot.
    """
    model = Sequential([
        Input(shape=(n_entradas,), name="bolsa_de_palabras"),
        Dense(config.HIDDEN_UNITS[0], activation="relu"),
        Dropout(config.DROPOUT_RATE),
        Dense(config.HIDDEN_UNITS[1], activation="relu"),
        Dropout(config.DROPOUT_RATE),
        Dense(n_salidas, activation="softmax", name="intenciones"),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def main():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    print("=" * 62)
    print(" ENTRENAMIENTO DEL ASISTENTE VIRTUAL ESCOLAR")
    print("=" * 62)

    intents = cargar_intents()
    words, classes, documents, lexico = construir_corpus(intents)

    print(f"\n[1/4] Preprocesamiento")
    print(f"      Intenciones (clases) : {len(classes)}")
    print(f"      Patrones de ejemplo  : {len(documents)}")
    print(f"      Vocabulario (raíces) : {len(words)}")
    print(f"      Léxico (palabras)    : {len(lexico)}")

    train_x, train_y = construir_dataset(words, classes, documents)
    print(f"\n[2/4] Dataset vectorizado con Bolsa de Palabras")
    print(f"      Entradas: {train_x.shape}  Salidas: {train_y.shape}")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.WORDS_PATH, "wb") as f:
        pickle.dump(words, f)
    with open(config.CLASSES_PATH, "wb") as f:
        pickle.dump(classes, f)
    with open(config.LEXICO_PATH, "wb") as f:
        pickle.dump(lexico, f)

    model = construir_modelo(train_x.shape[1], train_y.shape[1])
    print(f"\n[3/4] Arquitectura de la red")
    model.summary()

    print(f"\n[4/4] Entrenando {config.EPOCHS} épocas...")
    historial = model.fit(
        train_x,
        train_y,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.15,
        verbose=0,
    )

    acc = historial.history["accuracy"][-1]
    val_acc = historial.history["val_accuracy"][-1]
    loss = historial.history["loss"][-1]

    model.save(config.MODEL_PATH)

    print("\n" + "=" * 62)
    print(f" Precisión entrenamiento : {acc:.2%}")
    print(f" Precisión validación    : {val_acc:.2%}")
    print(f" Pérdida final           : {loss:.4f}")
    print(f"\n Modelo guardado en      : {config.MODEL_PATH}")
    print(f" Vocabulario guardado en : {config.WORDS_PATH}")
    print(f" Clases guardadas en     : {config.CLASSES_PATH}")
    print(f" Léxico guardado en      : {config.LEXICO_PATH}")
    print("=" * 62)
    print("\n¡Modelo entrenado y guardado con éxito!")


if __name__ == "__main__":
    main()
