"""Procesamiento de Lenguaje Natural (RF-01).
Etapas:
    1. Normalizacion  -> minusculas, sin tildes, sin signos de puntuacion.
    2. Tokenizacion   -> separacion en palabras por expresion regular.
    3. Filtrado       -> descarte de palabras vacias (stopwords) sin valor semantico.
    4. Lematizacion   -> reduccion a la raiz mediante el algoritmo Snowball en espanol
                         ("constancias", "constancia" -> "constanc").
    5. Correccion difusa -> los errores de tipeo se aproximan al termino mas
                         parecido del vocabulario aprendido (difflib).
"""

import re
import unicodedata
from difflib import get_close_matches
from functools import lru_cache

import numpy as np
from nltk.stem import SnowballStemmer

_STEMMER = SnowballStemmer("spanish")
STOPWORDS = {
    "a", "al", "algo", "ante", "aqui", "asi", "cada", "como", "con", "cual",
    "cuales", "de", "del", "desde", "donde", "dos", "el", "ella", "ellos",
    "en", "entre", "era", "eres", "es", "esa", "ese", "eso", "esta", "estan",
    "este", "esto", "estoy", "fue", "ha", "hay", "la", "las", "le", "les",
    "lo", "los", "mas", "me", "mi", "mis", "muy", "ni", "no", "nos", "o",
    "otra", "otro", "para", "pero", "por", "porque", "que", "se", "sea",
    "ser", "si", "sin", "sobre", "son", "soy", "su", "sus", "tambien", "te",
    "tu", "tus", "un", "una", "uno", "unos", "y", "ya", "yo",
}

# Umbral de similitud para la correccion de errores de tipeo.
# 0.80 tolera una o dos letras equivocadas sin confundir palabras distintas.
FUZZY_CUTOFF = 0.80

_PATRON_TOKEN = re.compile(r"[a-z0-9]+")


def quitar_tildes(texto: str) -> str:
    """Convierte 'práctica' en 'practica' descomponiendo los caracteres Unicode."""
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def normalizar(texto: str) -> str:
    """Pasa a minusculas y elimina tildes y signos de puntuacion."""
    return quitar_tildes(texto.lower())


def tokenizar(texto: str) -> list:
    """Separa la frase en palabras, descartando signos y simbolos."""
    return _PATRON_TOKEN.findall(normalizar(texto))


def clave_fonetica(palabra: str) -> str:
    """Esto es para reducir una palabra a como suena, no a como se escribe.
        EJ:
        taller / tayer        -> tayer      (yeismo: ll = y)
        constancia / constansia -> konstansia  (seseo: c ante e/i = s = z)
        analitico / analitiko -> analitiko  (c fuerte = k = qu)
        necesito / nesecito   -> nesesito
    """
    p = palabra
    p = p.replace("ll", "y")               # yeismo
    p = p.replace("ch", "\x01")            # se protege el digrafo "ch"
    p = p.replace("qu", "k")
    p = re.sub(r"c([ei])", r"s\1", p)      # seseo: ce/ci suenan como se/si
    p = p.replace("c", "k")                # c fuerte
    p = p.replace("\x01", "ch")
    p = p.replace("z", "s")
    p = p.replace("v", "b")                # b y v son el mismo sonido
    p = p.replace("w", "b")
    p = p.replace("h", "")                 # la h es muda
    p = re.sub(r"g([ei])", r"j\1", p)      # ge/gi suenan como je/ji
    p = p.replace("x", "ks")
    return re.sub(r"(.)\1+", r"\1", p)     # colapsa letras repetidas

@lru_cache(maxsize=4)
def _indice_fonetico(lexico: tuple) -> dict:
    """Mapea clave fonetica -> palabra del dominio. Se calcula una sola vez."""
    indice = {}
    for palabra in lexico:
        indice.setdefault(clave_fonetica(palabra), palabra)
    return indice

def tokens_significativos(texto: str) -> list:
    """Tokens normalizados sin stopwords, todavia SIN lematizar.
    """
    return [t for t in tokenizar(texto) if t not in STOPWORDS and len(t) > 1]

def lematizar(tokens: list) -> list:
    return [_STEMMER.stem(token) for token in tokens]

def procesar(texto: str) -> list:
    return lematizar(tokens_significativos(texto))

def bolsa_de_palabras(texto: str, vocabulario: list, lexico: list = None,
                      usar_fuzzy: bool = True):
    """Convierte una frase en el vector binario que espera la red neuronal.
    """
    indice = {raiz: i for i, raiz in enumerate(vocabulario)}
    vector = np.zeros(len(vocabulario), dtype=np.float32)
    palabras = tokens_significativos(texto)
    coincidencias = 0

    for palabra in palabras:
        raiz = _STEMMER.stem(palabra)

        if raiz in indice:
            vector[indice[raiz]] = 1.0
            coincidencias += 1
            continue

        if not usar_fuzzy or not lexico:
            continue

        fonetico = _indice_fonetico(tuple(lexico))
        clave = clave_fonetica(palabra)
        candidata = fonetico.get(clave)

        if candidata is None:
            similares = get_close_matches(
                clave, list(fonetico.keys()), n=1, cutoff=FUZZY_CUTOFF
            )
            if similares:
                candidata = fonetico[similares[0]]

        if candidata is not None:
            raiz_corregida = _STEMMER.stem(candidata)
            if raiz_corregida in indice:
                vector[indice[raiz_corregida]] = 1.0
                coincidencias += 1

    return vector, coincidencias, len(palabras)