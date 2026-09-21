# Chatbot Escolar con IA — Escuela Técnica N.º 36 Almirante Guillermo Brown

Asistente virtual que responde dudas sobre trámites administrativos y académicos
de la escuela. Procesa lenguaje natural con una red neuronal en Python y se
expone como API REST para integrarse en la web de la institución.

**TP PSR-TP03-C2 — Programación Sobre Redes**

---

## Arquitectura

Tres procesos independientes que se comunican por red con HTTP y JSON. La
interfaz gráfica no comparte código con el chatbot: solo conoce la dirección del
endpoint.

```
[ Frontend: página de la escuela ]     http://127.0.0.1:8080
            |  fetch() — HTTP POST con JSON
            v
[ Backend de red: Flask + CORS ]       http://127.0.0.1:5000/chat
            |  llamada en memoria
            v
[ Red neuronal Keras (MLP) ]
```

```
app.py                  Servidor de red / API REST (Flask)
train.py                Entrenamiento de la red neuronal
intents.json            Base de conocimiento (intenciones y respuestas)
requirements.txt        Dependencias

chatbot/
  config.py             Rutas e hiperparámetros
  preprocessing.py      Pipeline de PLN
  predictor.py          Motor de inferencia (no conoce HTTP)

frontend/index.html     Ventana de chat (HTML + CSS + JS)
model/                  Artefactos generados por train.py
tests/                  Suite automatizada, curl y colección Postman
docs/                   Evidencia de testeo
```

---

## Instalación

Requiere **Python 3.10 o superior** (probado con 3.12).

```bash
git clone https://github.com/thiagooa545/PSR-CHAILE-JERICO-CHATBOT.git
cd PSR-CHAILE-JERICO-CHATBOT
```

Crear y activar el entorno virtual:

```powershell
python -m venv venv           # Windows
.\venv\Scripts\Activate.ps1
```
```bash
python3 -m venv venv          # Linux / macOS
source venv/bin/activate
```

Instalar las dependencias (descarga TensorFlow, puede demorar):

```bash
pip install -r requirements.txt
```

---

## Ejecución

**1. Entrenar la red.** Solo la primera vez, o al modificar `intents.json`.

```bash
python train.py
```

**2. Levantar el servidor de red.**

```bash
python app.py
```

**3. Abrir la interfaz**, en otra terminal:

```bash
python -m http.server 8080 --directory frontend
```

Y entrar a **http://127.0.0.1:8080**.

El frontend se sirve desde otro puerto a propósito: obliga a que la comunicación
con el backend ocurra realmente por red y a través de CORS, igual que al
integrarlo en la web de la escuela.

---

## API REST

Servidor: `http://127.0.0.1:5000`

> Conviene usar `127.0.0.1` y no `localhost`: en Windows ese nombre resuelve
> primero a IPv6 y el cliente pierde hasta 2 segundos antes de reintentar.

### `POST /chat`

```json
{ "message": "¿Cómo saco la constancia de alumno regular?" }
```

```json
{
  "response": "Podés solicitar tu constancia de alumno regular en Secretaría...",
  "tag": "alumno_regular",
  "confidence": 1.0,
  "fallback": false,
  "elapsed_ms": 3.1
}
```

`tag` es la intención detectada, o `null` si no se alcanzó el umbral.
`fallback` indica si se aplicó la respuesta por defecto.

Errores: `400` mensaje vacío, ausente o cuerpo no-JSON · `405` método incorrecto
· `413` mensaje de más de 500 caracteres.

### `GET /health`

Estado del servicio y del modelo cargado.

### Ejemplo con curl

```bash
curl -X POST http://127.0.0.1:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Requisitos para las pasantías"}'
```

---

## Testeo de la API

Con el servidor levantado, en otra terminal:

```bash
python tests/test_api.py      # suite automatizada (36 verificaciones)
bash  tests/curl_tests.sh     # curl en Linux / macOS / Git Bash
.\tests\curl_tests.ps1        # curl en Windows PowerShell
```

También se puede importar `tests/postman_collection.json` en **Postman** o
**Insomnia**. La salida de la última corrida está en
[`docs/evidencia_testeo.txt`](docs/evidencia_testeo.txt).

---

## Cómo funciona

Cada consulta pasa por el mismo pipeline en el entrenamiento y en la inferencia:
normalización, tokenización, descarte de *stopwords*, lematización con el
algoritmo Snowball en español y vectorización con bolsa de palabras.

Si una palabra no está en el vocabulario, se intenta corregir antes de
descartarla. Primero por **clave fonética**: la mayoría de los errores del
castellano son homófonos (`taller`/`tayer`, `constancia`/`constansia`), así que
colapsando las letras que suenan igual la palabra mal escrita cae en la misma
clave que la correcta. Si eso no alcanza, se busca la clave más parecida con
`difflib`. La corrección se hace sobre la palabra completa y no sobre la raíz,
porque las raíces de una palabra bien y mal escrita divergen demasiado.

La red es un MLP de 128 y 64 neuronas con ReLU, dropout del 50% y salida softmax
sobre las 14 intenciones, entrenada con `categorical_crossentropy`.

Para el **fallback**, la certeza combina la probabilidad de la red con la
cobertura léxica de la consulta, y debe superar el 60%. La cobertura es
necesaria porque la red siempre reparte el 100% entre las clases que conoce: una
frase como *"quiero pedir una pizza"* comparte palabras sueltas con el dominio y
sería clasificada con más del 99% de probabilidad. Al ponderar por la fracción de
palabras reconocidas (2 de 4), la certeza cae a 0,50 y se activa el fallback.

---

## Agregar o modificar trámites

Solo hay que editar `intents.json` y volver a ejecutar `python train.py`. No hace
falta tocar el código.

```json
{
  "tag": "alumno_regular",
  "patterns": ["¿Cómo saco la constancia de alumno regular?",
               "Papel para la obra social"],
  "responses": ["Podés solicitar tu constancia en Secretaría de..."]
}
```

Cuantos más patrones tenga cada intención, mejor generaliza la red. Conviene
incluir sinónimos, abreviaturas y las formas en que los alumnos preguntan
realmente.

---

## Cumplimiento de requerimientos

| Req. | Descripción | Dónde se resuelve |
|---|---|---|
| RF-01 | PLN con errores, abreviaturas y sinónimos | `chatbot/preprocessing.py` |
| RF-02 | Los cuatro ejes temáticos de trámites | `intents.json` (14 intenciones) |
| RF-03 | API REST con endpoint `/chat` por POST | `app.py` |
| RF-04 | Respuesta por defecto bajo el umbral del 60% | `chatbot/predictor.py` |
| RNF-01 | Backend íntegramente en Python 3.x | todo el proyecto |
| RNF-02 | Respuesta en menos de 2 segundos | 3 ms de inferencia, 13 ms de ida y vuelta |
| RNF-03 | Lógica del chatbot separada del frontend | `chatbot/` no importa Flask; `app.py` no genera HTML |
