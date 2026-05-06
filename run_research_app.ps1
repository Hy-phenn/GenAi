param(
    [string]$ArtifactsRoot = "diffusion_noise_project/diffusion_noise_project",
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 9000,
    [switch]$Share
)

$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Project virtual environment not found at $PythonExe"
}

$shareArgs = @()
if ($Share) {
    $shareArgs += "--share"
}

Push-Location $ProjectRoot
try {
    & $PythonExe -m research_extension.cli app `
        --artifacts-root $ArtifactsRoot `
        --host $ListenHost `
        --port $Port `
        @shareArgs
}
finally {
    Pop-Location
}
