# Demostracion de Red con curl - Windows PowerShell
#
# Uso:  .\tests\curl_tests.ps1
# Requiere el servidor levantado:  python app.py
#
# Nota: en PowerShell "curl" es un alias de Invoke-WebRequest, por eso se
# invoca curl.exe explicitamente para usar el cliente HTTP real.

$BASE = "http://127.0.0.1:5000"

function Probar($titulo, $mensaje) {
    Write-Host ""
    Write-Host "--- $titulo ---" -ForegroundColor Yellow
    Write-Host "Enviando: $mensaje" -ForegroundColor DarkGray
    $body = @{ message = $mensaje } | ConvertTo-Json -Compress
    curl.exe -s -X POST "$BASE/chat" -H "Content-Type: application/json" --data-binary $body
    Write-Host ""
}

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " TESTEO DE LA API DEL CHATBOT ESCOLAR (curl)" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "--- GET /health : estado del servicio ---" -ForegroundColor Yellow
curl.exe -s "$BASE/health"
Write-Host ""

Probar "RF-02 Constancia de alumno regular" "Como saco la constancia de alumno regular?"
Probar "RF-02 Pasantias"                    "Requisitos para las pasantias"
Probar "RF-02 Seguridad en taller"          "Normas de seguridad en el taller"
Probar "RF-02 Mesas de examen"              "Cuando son las mesas de examen?"
Probar "RF-02 Analitico"                    "Necesito el analitico para la facultad"
Probar "RF-01 Error ortografico"            "constansia de alunmo regular"
Probar "RF-01 Sinonimo"                     "papel para la obra social"
Probar "RF-01 Abreviatura"                  "PPS"
Probar "RF-04 Fallback fuera de dominio"    "Cual es la capital de Australia?"

Write-Host ""
Write-Host "--- Manejo de errores: mensaje vacio (espera HTTP 400) ---" -ForegroundColor Yellow
curl.exe -s -w "`nHTTP %{http_code}`n" -X POST "$BASE/chat" -H "Content-Type: application/json" -d '{\"message\":\"\"}'

Write-Host ""
Write-Host "--- Manejo de errores: cuerpo no-JSON (espera HTTP 400) ---" -ForegroundColor Yellow
curl.exe -s -w "`nHTTP %{http_code}`n" -X POST "$BASE/chat" -H "Content-Type: application/json" -d "esto no es json"

Write-Host ""
Write-Host "--- Manejo de errores: GET sobre /chat (espera HTTP 405) ---" -ForegroundColor Yellow
curl.exe -s -w "`nHTTP %{http_code}`n" "$BASE/chat"

Write-Host ""
Write-Host "--- RNF-02: tiempo total de respuesta ---" -ForegroundColor Yellow
curl.exe -s -o NUL -w "Tiempo total: %{time_total} s`n" -X POST "$BASE/chat" -H "Content-Type: application/json" -d '{\"message\":\"Requisitos para las pasantias\"}'

Write-Host ""
Write-Host "Testeo finalizado." -ForegroundColor Green
