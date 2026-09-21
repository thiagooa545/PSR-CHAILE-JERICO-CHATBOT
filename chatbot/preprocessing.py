"""Procesamiento de Lenguaje Natural (RF-01).

Pipeline de normalizacion del texto del usuario. Es el mismo modulo que usan
el entrenamiento y el servidor, garantizando que una frase se vectorice
siempre igual.

Etapas:
    1. Normalizacion  -> minusculas, sin tildes, sin signos de puntuacion.
    2. Tokenizacion   -> separacion en palabras por expresion regular.
    3. Filtrado       -> descarte de palabras vacias (stopwords) sin valor semantico.
    4. Lematizacion   -> reduccion a la raiz mediante el algoritmo Snowball en espanol
                         ("constancias", "constancia" -> "constanc").
    5. Correccion difusa -> los errores de tipeo se aproximan al termino mas
                         parecido del vocabulario aprendido (difflib).

Las etapas 1, 4 y 5 son las que permiten cumplir RF-01: entender al usuario
aunque escriba con errores ortograficos, abreviaturas o sinonimos.
"""

import re
import unicodedata
from difflib import get_close_matches
from functools import lru_cache

import numpy as np
from nltk.stem import SnowballStemmer

# Stemmer algoritmico: no requiere descargar corpus, funciona sin conexion.
_STEMMER = SnowballStemmer("spanish")

# Palabras vacias frecuentes del espanol rioplatense. No aportan informacion
# para distinguir una intencion de otra, por lo que se descartan.
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
    """Reduce una palabra a como suena, no a como se escribe.

    La mayoria de los errores ortograficos del castellano son homofonos: el
    alumno escribe lo que oye. Al colapsar los pares de letras que suenan igual,
    la palabra mal escrita y la correcta terminan en la misma clave y la
    comparacion por similitud deja de ser necesaria:

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

    Se expone aparte porque la correccion de errores de tipeo debe trabajar
    sobre la palabra completa: las raices de una palabra bien y mal escrita
    divergen demasiado ("taller" -> "tall" pero "tayer" -> "tay") y dejarian de
    parecerse entre si.
    """
    return [t for t in tokenizar(texto) if t not in STOPWORDS and len(t) > 1]


def lematizar(tokens: list) -> list:
    """Reduce cada token a su raiz mediante el algoritmo Snowball."""
    return [_STEMMER.stem(token) for token in tokens]


def procesar(texto: str) -> list:
    """Pipeline completo: texto crudo -> lista de raices significativas."""
    return lematizar(tokens_significativos(texto))


def bolsa_de_palabras(texto: str, vocabulario: list, lexico: list = None,
                      usar_fuzzy: bool = True):
    """Convierte una frase en el vector binario que espera la red neuronal.

    Args:
        texto: consulta escrita por el usuario.
        vocabulario: lista ordenada de raices aprendidas en el entrenamiento.
        lexico: palabras completas (sin lematizar) vistas en el entrenamiento.
            Es contra esta lista que se corrigen los errores de tipeo.
        usar_fuzzy: si es True, un token desconocido se asocia al termino mas
            parecido del dominio (tolerancia a errores ortograficos).

    Returns:
        (vector, coincidencias, total) donde vector es un np.ndarray de 0 y 1
        del tamano del vocabulario, coincidencias es la cantidad de palabras de
        la consulta reconocidas dentro del dominio escolar y total es la
        cantidad de palabras significativas que tenia la consulta.

        El cociente coincidencias/total es la "cobertura lexica": indica que
        porcentaje de lo que escribio el usuario pertenece realmente al dominio
        de la escuela, y se usa para decidir el fallback del RF-04.
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

        # 1) El error es homofono ("tayer" por "taller"): misma clave fonetica.
        candidata = fonetico.get(clave)

        # 2) El error es de tecleo ("alunmo" por "alumno"): se busca la clave
        #    mas parecida, ya normalizada foneticamente.
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
