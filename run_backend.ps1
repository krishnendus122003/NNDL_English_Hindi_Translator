$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
} finally {
    Pop-Location
}
