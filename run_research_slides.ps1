param(
    [int]$Port = 9001,
    [string]$ListenHost = "127.0.0.1",
    [switch]$NoBrowser
)

$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
$QuartoExe = "C:\Program Files\Quarto\bin\quarto.exe"

if (-not (Test-Path $QuartoExe)) {
    throw "Quarto was not found at $QuartoExe"
}

$previewArgs = @('preview', 'docs/slides.qmd', '--port', $Port, '--host', $ListenHost)
if ($NoBrowser) {
    $previewArgs += '--no-browser'
}

Push-Location $ProjectRoot
try {
    & $QuartoExe @previewArgs
}
finally {
    Pop-Location
}
