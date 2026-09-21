#!/usr/bin/env bash
# Demostracion de Red con curl - Linux / macOS / Git Bash
#
# Uso:  bash tests/curl_tests.sh
# Requiere el servidor levantado:  python app.py

BASE="http://127.0.0.1:5000"

probar() {
    echo ""
    echo "--- $1 ---"
    echo "Enviando: $2"
    curl -s -X POST "$BASE/chat" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$2\"}"
    echo ""
}

echo "===================================================="
echo " TESTEO DE LA API DEL CHATBOT ESCOLAR (curl)"
echo "===================================================="

echo ""
echo "--- GET /health : estado del servicio ---"
curl -s "$BASE/health"
echo ""

probar "RF-02 Constancia de alumno regular" "Como saco la constancia de alumno regular?"
probar "RF-02 Pasantias"                    "Requisitos para las pasantias"
probar "RF-02 Seguridad en taller"          "Normas de seguridad en el taller"
probar "RF-02 Mesas de examen"              "Cuando son las mesas de examen?"
probar "RF-02 Analitico"                    "Necesito el analitico para la facultad"
probar "RF-01 Error ortografico"            "constansia de alunmo regular"
probar "RF-01 Sinonimo"                     "papel para la obra social"
probar "RF-01 Abreviatura"                  "PPS"
probar "RF-04 Fallback fuera de dominio"    "Cual es la capital de Australia?"

echo ""
echo "--- Manejo de errores: mensaje vacio (espera HTTP 400) ---"
curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE/chat" \
    -H "Content-Type: application/json" -d '{"message":""}'

echo ""
echo "--- Manejo de errores: cuerpo no-JSON (espera HTTP 400) ---"
curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE/chat" \
    -H "Content-Type: application/json" -d 'esto no es json'

echo ""
echo "--- Manejo de errores: GET sobre /chat (espera HTTP 405) ---"
curl -s -w "\nHTTP %{http_code}\n" "$BASE/chat"

echo ""
echo "--- RNF-02: tiempo total de respuesta ---"
curl -s -o /dev/null -w "Tiempo total: %{time_total} s\n" -X POST "$BASE/chat" \
    -H "Content-Type: application/json" -d '{"message":"Requisitos para las pasantias"}'

echo ""
echo "Testeo finalizado."
