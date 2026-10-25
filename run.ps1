$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$pythonCandidates = @(
    (Join-Path $projectRoot ".venv\Scripts\python.exe"),
    (Join-Path $workspaceRoot ".venv\Scripts\python.exe")
)
$python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $python) {
    throw "Python was not found. Create a .venv in this project or activate a Python environment before launching."
}

$app = Join-Path $projectRoot "app.py"
if (-not (Test-Path -LiteralPath $app -PathType Leaf)) {
    throw "The Flask application was not found at: $app"
}

Push-Location $projectRoot
try {
    & $python $app
    $appExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($appExitCode -ne 0) {
    throw "The Flask application exited with code $appExitCode."
}
