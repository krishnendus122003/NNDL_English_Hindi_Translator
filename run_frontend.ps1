$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python -m http.server 5500 --directory app
} finally {
    Pop-Location
}
