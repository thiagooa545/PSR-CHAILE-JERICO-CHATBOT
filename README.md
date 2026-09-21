# Chatbot Escolar con IA — Escuela Técnica

Asistente virtual que responde dudas recurrentes sobre trámites administrativos y
académicos de la escuela. Procesa lenguaje natural con una red neuronal en Python
y se expone como un servicio de red (API REST) para integrarse en la página web de
la institución.

**Trabajo práctico PSR-TP03-C2 — Programación Sobre Redes**

---

## Arquitectura

El sistema está dividido en tres procesos independientes que se comunican por red
mediante HTTP y JSON. La interfaz gráfica no comparte código con el chatbot: solo
conoce la dirección del endpoint.

```
[ Frontend: página de la escuela ]        http://127.0.0.1:8080
                |
                |  fetch() — HTTP POST con JSON
                v
[ Backend de red: Flask + CORS ]          http://127.0.0.1:5000/chat
                |
                |  llamada en memoria
                v
[ Procesamiento: red neuronal Keras (MLP) ]
```

### Estructura de archivos

```
.
├── app.py                      Servidor de red / API REST (Flask)
├── train.py                    Script de entrenamiento de la red neuronal
├── intents.json                Base de conocimiento (intenciones y respuestas)
├── requirements.txt            Dependencias del proyecto
│
├── chatbot/                    Paquete con la lógica desacoplada
│   ├── config.py               Rutas e hiperparámetros compartidos
│   ├── preprocessing.py        Pipeline de PLN (tokenización, lematización, fuzzy)
│   └── predictor.py            Motor de inferencia (no conoce HTTP)
│
├── frontend/
│   └── index.html              Ventana de chat (HTML + CSS + JS)
│
├── model/                      Artefactos generados por train.py
│   ├── chatbot_model.h5        Red neuronal entrenada
│   ├── words.pkl               Vocabulario (raíces lematizadas)
│   ├── classes.pkl             Etiquetas de intención
│   └── lexico.pkl              Palabras completas (corrección de errores)
│
├── tests/
│   ├── test_api.py             Suite automatizada de testeo de red
│   ├── curl_tests.ps1          Pruebas con curl (Windows)
│   ├── curl_tests.sh           Pruebas con curl (Linux / macOS / Git Bash)
│   └── postman_collection.json Colección importable en Postman o Insomnia
│
└── docs/
    └── evidencia_testeo.txt    Salida capturada de la corrida de pruebas
```

---

## Instalación

Requiere **Python 3.10 o superior** (desarrollado y probado con Python 3.12).

### 1. Clonar el repositorio

```bash
git clone <URL-DEL-REPOSITORIO>
cd PSR-CHAILE-JERICO-CHATBOT
```

### 2. Crear y activar el entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

La instalación descarga TensorFlow, por lo que puede demorar varios minutos.

---

## Ejecución

### Paso 1 — Entrenar la red neuronal

Solo es necesario la primera vez, o cada vez que se modifique `intents.json`.

```bash
python train.py
```

Genera los artefactos dentro de `model/` e imprime la precisión alcanzada:

```
[1/4] Preprocesamiento
      Intenciones (clases) : 14
      Patrones de ejemplo  : 229
      Vocabulario (raíces) : 224
      Léxico (palabras)    : 282
...
 Precisión entrenamiento : 99.48%
 Precisión validación    : 80.00%
```

### Paso 2 — Levantar el servidor de red

```bash
python app.py
```

```
Cargando modelo de red neuronal...
Modelo cargado: 14 intenciones, vocabulario de 224 términos.
Servidor escuchando en http://localhost:5000
```

### Paso 3 — Abrir la interfaz de chat

El frontend se sirve como un sitio estático aparte, desde **otro puerto**. Esto
es deliberado: obliga a que la comunicación con el backend ocurra realmente por
red y a través de CORS, tal como ocurriría al integrarlo en la web de la escuela.

En una tercera terminal:

```bash
python -m http.server 8080 --directory frontend
```

Y abrir en el navegador: **http://127.0.0.1:8080**

---

## API REST

Servidor: `http://127.0.0.1:5000`

> Se recomienda usar `127.0.0.1` en lugar de `localhost`. En Windows, `localhost`
> resuelve primero a IPv6 (`::1`) y el cliente pierde hasta 2 segundos esperando
> esa conexión antes de reintentar por IPv4.

### `POST /chat`

Petición:
```json
{ "message": "¿Cómo saco la constancia de alumno regular?" }
```

Respuesta (HTTP 200):
```json
{
  "response": "Podés solicitar tu constancia de alumno regular en Secretaría...",
  "tag": "alumno_regular",
  "confidence": 1.0,
  "fallback": false,
  "elapsed_ms": 3.1
}
```

| Campo | Descripción |
|---|---|
| `response` | Texto que muestra el chat al usuario |
| `tag` | Intención detectada, o `null` si no se alcanzó el umbral |
| `confidence` | Certeza final, entre 0 y 1 |
| `fallback` | `true` si se aplicó la respuesta por defecto |
| `elapsed_ms` | Tiempo de procesamiento de la red neuronal |

Códigos de error: `400` mensaje vacío, ausente o cuerpo no-JSON · `405` método
incorrecto · `413` mensaje de más de 500 caracteres.

### `GET /health`

Estado del servicio y del modelo cargado.

```json
{ "status": "ok", "modelo_cargado": true, "intenciones": 14,
  "vocabulario": 224, "umbral_certeza": 0.6 }
```

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
python tests/test_api.py          # suite automatizada (36 verificaciones)
bash  tests/curl_tests.sh         # pruebas con curl (Linux / macOS / Git Bash)
.\tests\curl_tests.ps1            # pruebas con curl (Windows PowerShell)
```

También se puede importar `tests/postman_collection.json` en **Postman** o
**Insomnia**: trae las 14 peticiones ya armadas.

La salida de la última corrida está guardada en
[`docs/evidencia_testeo.txt`](docs/evidencia_testeo.txt).

---

## Cómo funciona

### Procesamiento de lenguaje natural

Cada consulta atraviesa el mismo pipeline en el entrenamiento y en la inferencia,
lo que garantiza que una frase se vectorice siempre igual:

1. **Normalización** — minúsculas, sin tildes, sin signos de puntuación.
2. **Tokenización** — separación en palabras por expresión regular.
3. **Filtrado** — descarte de *stopwords* sin valor semántico.
4. **Lematización** — reducción a la raíz con el algoritmo Snowball en español
   (`constancias`, `constancia` → `constanc`).
5. **Bolsa de palabras** — vector binario del tamaño del vocabulario.

**Tolerancia a errores ortográficos.** Si una palabra no está en el vocabulario,
antes de descartarla se intenta corregirla en dos pasos:

- **Clave fonética.** La mayoría de los errores del castellano son homófonos: el
  alumno escribe lo que oye. Colapsando los pares de letras que suenan igual
  (`ll`=`y`, `b`=`v`, `c`/`z`/`s`, `qu`=`k`, `h` muda), la palabra mal escrita y la
  correcta caen en la misma clave y el error desaparece:
  `taller`/`tayer` → `tayer`, `constancia`/`constansia` → `konstansia`.
- **Similitud.** Para los errores de tecleo que no son fonéticos (`alunmo` por
  `alumno`) se busca la clave más parecida con `difflib`, con un umbral del 80%.

La corrección se aplica sobre la **palabra completa y no sobre la raíz**: las
raíces de una palabra bien y mal escrita divergen demasiado (`taller` → `tall`
pero `tayer` → `tay`) y dejarían de parecerse entre sí.

### La red neuronal

Red secuencial densa (MLP) construida con Keras:

| Capa | Salida | Parámetros |
|---|---|---|
| Dense (ReLU) | 128 | 28.800 |
| Dropout 0.5 | 128 | 0 |
| Dense (ReLU) | 64 | 8.256 |
| Dropout 0.5 | 64 | 0 |
| Dense (Softmax) | 14 | 910 |

- **ReLU** en las capas ocultas: converge rápido y evita el desvanecimiento del gradiente.
- **Dropout del 50%**: regularización necesaria porque el dataset es pequeño.
- **Softmax** en la salida: devuelve una distribución de probabilidad sobre las
  intenciones, que es lo que permite aplicar el umbral de certeza.
- **`categorical_crossentropy`**: la función de pérdida adecuada para salidas one-hot.

### Respuesta por defecto

El bot deriva a Secretaría cuando no está seguro. La decisión combina dos señales:

1. **Probabilidad** de la intención más votada por la red.
2. **Cobertura léxica**: qué porcentaje de la consulta pertenece realmente al
   dominio escolar.

La certeza final es el producto de ambas, y debe superar el **60%**.

La cobertura es necesaria porque la red siempre reparte el 100% entre las clases
que conoce. Una frase como *"quiero pedir una pizza"* comparte palabras sueltas
con el dominio (*"quiero pedir"*) y la red la clasificaría como `alumno_regular`
con más del 99% de probabilidad. Al ponderar por la fracción de palabras
reconocidas (2 de 4 → 50%), la certeza cae a 0,50 y se activa el fallback.

---

## Base de conocimiento

Para agregar o modificar trámites solo hay que editar `intents.json` y volver a
ejecutar `python train.py`. No hace falta tocar el código.

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

Intenciones cubiertas actualmente (14):

| Eje temático | Intenciones |
|---|---|
| Prácticas Profesionalizantes | `pasantias` |
| Talleres y Laboratorios | `taller_seguridad`, `taller_herramientas` |
| Mesas de examen | `mesas_examen`, `previas`, `equivalencias` |
| Emisión de documentación | `alumno_regular`, `analitico` |
| Conversación e información general | `saludo`, `despedida`, `agradecimiento`, `bot_identidad`, `horarios`, `contacto` |

---

## Cumplimiento de requerimientos

| Req. | Descripción | Dónde se resuelve |
|---|---|---|
| RF-01 | PLN con errores, abreviaturas y sinónimos | `chatbot/preprocessing.py` |
| RF-02 | Los cuatro ejes temáticos de trámites | `intents.json` |
| RF-03 | API REST con endpoint `/chat` por POST | `app.py` |
| RF-04 | Respuesta por defecto bajo el umbral del 60% | `chatbot/predictor.py` |
| RNF-01 | Backend íntegramente en Python 3.x | todo el proyecto |
| RNF-02 | Respuesta en menos de 2 segundos | 3 ms de inferencia, 13 ms de ida y vuelta |
| RNF-03 | Lógica del chatbot separada del frontend | `chatbot/` no importa Flask; `app.py` no genera HTML |
