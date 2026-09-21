"""Demostración de Red: testeo automatizado de la API REST.

Ejecutar (con el servidor app.py ya levantado, en otra terminal):
    python tests/test_api.py
    
Este script hace como cliente de red: abre conexiones http usando unicamente la biblioteca
estandar de python (urllib), sin que le importe nada del backend. Esto verifica
uno por uno los requerimientos del trabajo practico.

Cobertura:
    RF-01  Comprensión de lenguaje natural con errores ortográficos y sinónimos
    RF-02  Gestión de los cuatro ejes temáticos obligatorios
    RF-03  Contrato de la API REST (POST /chat con JSON)
    RF-04  Respuesta por defecto bajo el umbral de certeza
    RNF-02 Tiempo de respuesta menor a 2 segundos
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import urllib.error
import urllib.request
BASE_URL = "http://127.0.0.1:5000"
CHAT_URL = f"{BASE_URL}/chat"
HEALTH_URL = f"{BASE_URL}/health"
TIMEOUT = 10

VERDE, ROJO, AMARILLO, GRIS, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"

_resultados = {"ok": 0, "fallo": 0}


def post_chat(mensaje, content_type="application/json"):
    cuerpo = json.dumps({"message": mensaje}).encode("utf-8")
    req = urllib.request.Request(
        CHAT_URL, data=cuerpo, headers={"Content-Type": content_type}, method="POST"
    )
    inicio = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            ms = (time.perf_counter() - inicio) * 1000
            return resp.status, json.loads(resp.read().decode("utf-8")), ms
    except urllib.error.HTTPError as e:
        ms = (time.perf_counter() - inicio) * 1000
        return e.code, json.loads(e.read().decode("utf-8")), ms


def post_crudo(cuerpo_bytes, content_type="application/json"):
    req = urllib.request.Request(
        CHAT_URL, data=cuerpo_bytes, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def verificar(condicion, descripcion, detalle=""):
    if condicion:
        _resultados["ok"] += 1
        print(f"  {VERDE}[OK]{RESET}    {descripcion}")
    else:
        _resultados["fallo"] += 1
        print(f"  {ROJO}[FALLA]{RESET} {descripcion}")
    if detalle:
        print(f"          {GRIS}{detalle}{RESET}")


def titulo(texto):
    print(f"\n{AMARILLO}{'-' * 68}\n{texto}\n{'-' * 68}{RESET}")

# RF-03: contrato de la API REST
def test_api_rest():
    titulo("RF-03 | Interfaz de Programación de Aplicaciones (API REST)")

    with urllib.request.urlopen(HEALTH_URL, timeout=TIMEOUT) as resp:
        salud = json.loads(resp.read().decode("utf-8"))
    verificar(
        salud.get("status") == "ok",
        "GET /health responde que el servicio está activo",
        f"intenciones={salud.get('intenciones')} vocabulario={salud.get('vocabulario')} "
        f"umbral={salud.get('umbral_certeza')}",
    )

    status, data, _ = post_chat("Hola")
    verificar(status == 200, f"POST /chat devuelve HTTP 200 (recibido: {status})")
    verificar("response" in data, "La respuesta JSON contiene el campo 'response'")
    verificar(
        isinstance(data.get("response"), str) and len(data["response"]) > 0,
        "El campo 'response' es un texto no vacío",
        f'"{data.get("response", "")[:70]}..."',
    )

# RF-02: los cuatro ejes temáticos obligatorios
def test_ejes_tematicos():
    titulo("RF-02 | Gestión de Trámites Técnicos (ejes obligatorios)")

    casos = [
        ("Prácticas Profesionalizantes / Pasantías",
         "¿Cuándo empiezan las prácticas profesionalizantes?", "pasantias"),
        ("Seguridad y herramientas en Talleres",
         "¿Qué normas de seguridad hay en el taller?", "taller_seguridad"),
        ("Herramientas y Pañol",
         "¿Cómo retiro una herramienta del pañol?", "taller_herramientas"),
        ("Mesas de examen",
         "¿Cuándo son las mesas de examen?", "mesas_examen"),
        ("Mesas de examen - previas",
         "¿Cuántas materias previas puedo llevarme?", "previas"),
        ("Mesas de examen - equivalencias",
         "Vengo de otra escuela, ¿cómo pido equivalencias?", "equivalencias"),
        ("Emisión de documentación - constancia",
         "¿Cómo saco la constancia de alumno regular?", "alumno_regular"),
        ("Emisión de documentación - analítico",
         "Necesito el analítico para la facultad", "analitico"),
    ]

    for eje, pregunta, tag_esperado in casos:
        _, data, _ = post_chat(pregunta)
        verificar(
            data.get("tag") == tag_esperado and not data.get("fallback"),
            f"{eje}",
            f'"{pregunta}" -> tag={data.get("tag")} conf={data.get("confidence")}',
        )

# RF-01: PLN con errores ortográficos, abreviaturas y sinónimos
def test_lenguaje_natural():
    titulo("RF-01 | Procesamiento de Lenguaje Natural (robustez)")

    # Sinónimos distintos que deben activar la misma respuesta.
    grupo_constancia = [
        "constancia de alumno",
        "certificado regular",
        "papel para la obra social",
    ]
    tags = []
    for frase in grupo_constancia:
        _, data, _ = post_chat(frase)
        tags.append(data.get("tag"))
    verificar(
        len(set(tags)) == 1 and tags[0] == "alumno_regular",
        "Sinónimos distintos activan la misma intención",
        f"{grupo_constancia} -> {tags}",
    )

    # Errores ortográficos frecuentes.
    typos = [
        ("constansia de alunmo regular", "alumno_regular"),
        ("pasantais", "pasantias"),
        ("mesas de esamen", "mesas_examen"),
        ("seguridada en el tayer", "taller_seguridad"),
        ("nesecito el analitiko", "analitico"),
    ]
    for frase, tag_esperado in typos:
        _, data, _ = post_chat(frase)
        verificar(
            data.get("tag") == tag_esperado,
            f"Tolera error ortográfico: '{frase}'",
            f'-> tag={data.get("tag")} conf={data.get("confidence")}',
        )

    # Abreviaturas.
    for frase, tag_esperado in [("PPS", "pasantias"), ("EPP", "taller_seguridad")]:
        _, data, _ = post_chat(frase)
        verificar(
            data.get("tag") == tag_esperado,
            f"Entiende la abreviatura '{frase}'",
            f'-> tag={data.get("tag")}',
        )

# RF-04: respuesta por defecto (fallback)
def test_fallback():
    titulo("RF-04 | Respuesta por Defecto (umbral de certeza del 60%)")

    fuera_de_dominio = [
        "asdkjhasd qwe",
        "¿Cuál es la capital de Australia?",
        "Quiero pedir una pizza con muzzarella",
        "12345 6789",
        "!!!???",
    ]
    for frase in fuera_de_dominio:
        status, data, _ = post_chat(frase)
        verificar(
            status == 200 and data.get("fallback") is True,
            f"Deriva a Secretaría ante consulta fuera de dominio: '{frase}'",
            f'-> fallback={data.get("fallback")} conf={data.get("confidence")}',
        )

    _, data, _ = post_chat("no entiendo nada de esto xyz")
    verificar(
        "secretaria@almirantebrown36.edu.ar" in data.get("response", ""),
        "El mensaje de fallback deriva al correo institucional",
    )

# Manejo de errores: el servidor no debe caerse nunca
def test_manejo_errores():
    titulo("Manejo de Errores | Robustez del servidor ante entradas inválidas")

    status, _ = post_crudo(b'{"message": ""}')
    verificar(status == 400, f"Mensaje vacío -> HTTP 400 (recibido: {status})")

    status, _ = post_crudo(b'{"mensaje": "campo incorrecto"}')
    verificar(status == 400, f"Campo 'message' ausente -> HTTP 400 (recibido: {status})")

    status, _ = post_crudo(b'esto no es json')
    verificar(status == 400, f"Cuerpo no-JSON -> HTTP 400 (recibido: {status})")

    status, _ = post_crudo(json.dumps({"message": "a" * 600}).encode())
    verificar(status == 413, f"Mensaje de 600 caracteres -> HTTP 413 (recibido: {status})")

    status, _ = post_crudo(json.dumps({"message": "<script>alert(1)</script>"}).encode())
    verificar(status == 200, "Entrada con HTML/script no rompe el servidor")

    status, _ = post_crudo(json.dumps({"message": "emojis 🚀🔧 y ñandú"}).encode())
    verificar(status == 200, "Caracteres Unicode y emojis no rompen el servidor")

    # Método incorrecto sobre la ruta.
    try:
        with urllib.request.urlopen(CHAT_URL, timeout=TIMEOUT) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    verificar(status == 405, f"GET sobre /chat -> HTTP 405 (recibido: {status})")

    # El servidor sigue vivo después de todo lo anterior.
    status, data, _ = post_chat("Hola")
    verificar(status == 200, "El servidor sigue operativo tras las entradas inválidas")

# RNF-02: tiempo de respuesta
def test_tiempo_respuesta():
    titulo("RNF-02 | Tiempo de Respuesta (< 2 segundos)")

    consultas = [
        "¿Cómo saco la constancia de alumno regular?",
        "Requisitos para las pasantías",
        "Normas de seguridad en el taller",
        "¿Cuándo son las mesas de examen?",
        "Necesito el analítico",
        "asdasdasd",
    ]
    tiempos, internos = [], []
    for consulta in consultas:
        _, data, ms = post_chat(consulta)
        tiempos.append(ms)
        internos.append(data.get("elapsed_ms", 0.0))

    promedio, maximo = sum(tiempos) / len(tiempos), max(tiempos)
    prom_int, max_int = sum(internos) / len(internos), max(internos)

    verificar(
        max_int < 2000,
        "Procesamiento de la red neuronal por debajo de 2000 ms",
        f"promedio={prom_int:.2f} ms | máximo={max_int:.2f} ms",
    )
    verificar(
        maximo < 2000,
        "Respuesta completa de la API (ida y vuelta) por debajo de 2000 ms",
        f"promedio={promedio:.2f} ms | máximo={maximo:.2f} ms | n={len(tiempos)}",
    )


def main():
    print(f"\n{AMARILLO}{'=' * 68}")
    print(" DEMOSTRACIÓN DE RED - TESTEO DE LA API DEL CHATBOT ESCOLAR")
    print(f" Servidor bajo prueba: {BASE_URL}")
    print(f"{'=' * 68}{RESET}")

    try:
        urllib.request.urlopen(HEALTH_URL, timeout=5)
    except Exception:
        print(f"\n{ROJO}No se pudo conectar con {BASE_URL}.{RESET}")
        print("Levantá el servidor en otra terminal con:  python app.py\n")
        return 1

    test_api_rest()
    test_ejes_tematicos()
    test_lenguaje_natural()
    test_fallback()
    test_manejo_errores()
    test_tiempo_respuesta()

    total = _resultados["ok"] + _resultados["fallo"]
    color = VERDE if _resultados["fallo"] == 0 else ROJO
    print(f"\n{color}{'=' * 68}")
    print(f" RESULTADO: {_resultados['ok']}/{total} verificaciones exitosas")
    print(f"{'=' * 68}{RESET}\n")
    return 0 if _resultados["fallo"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())