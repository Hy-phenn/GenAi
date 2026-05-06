param(
    [string]$ArtifactsRoot = "diffusion_noise_project/diffusion_noise_project"
)

$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Project virtual environment not found at $PythonExe"
}

Push-Location $ProjectRoot
try {
    & $PythonExe -m research_extension.cli manifest --artifacts-root $ArtifactsRoot
    if ($LASTEXITCODE -ne 0) { throw "Manifest generation failed." }

    & $PythonExe -m research_extension.cli audit-saved-samples --artifacts-root $ArtifactsRoot
    if ($LASTEXITCODE -ne 0) { throw "Saved-sample audit failed." }

    & $PythonExe -m research_extension.cli audit-nearest-neighbors --artifacts-root $ArtifactsRoot
    if ($LASTEXITCODE -ne 0) { throw "Nearest-neighbor audit failed." }

    & $PythonExe -m research_extension.cli audit-training-dynamics --artifacts-root $ArtifactsRoot
    if ($LASTEXITCODE -ne 0) { throw "Training-dynamics audit failed." }
}
finally {
    Pop-Location
}
