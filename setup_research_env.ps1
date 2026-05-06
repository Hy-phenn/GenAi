$script:ResearchProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }

function Ensure-ResearchEnvironment {
    param(
        [switch]$ForceInstall
    )

    $projectRoot = $script:ResearchProjectRoot
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $venvExists = Test-Path $venvPython
    $importsOk = $false

    if ($venvExists -and -not $ForceInstall) {
        & $venvPython -c "import gradio, torch, torchvision, yaml, numpy, research_extension" 2>$null
        $importsOk = ($LASTEXITCODE -eq 0)
    }

    if (-not $venvExists) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.10 -m venv (Join-Path $projectRoot '.venv')
        }
        elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv (Join-Path $projectRoot '.venv')
        }
        else {
            throw 'Python 3.10 was not found on this machine.'
        }

        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
            throw 'Virtual environment creation failed.'
        }
    }

    if (-not $importsOk) {
        & $venvPython -m pip install --upgrade pip 'setuptools<82' wheel
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to prepare pip/setuptools/wheel in the project virtual environment.'
        }

        Push-Location $projectRoot
        try {
            & $venvPython -m pip install -e '.[all]'
        }
        finally {
            Pop-Location
        }

        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to install project Python dependencies into .venv.'
        }
    }

    $quartoExe = $null
    $quartoCommand = Get-Command quarto -ErrorAction SilentlyContinue
    if ($quartoCommand) {
        $quartoExe = $quartoCommand.Source
    }
    elseif (Test-Path 'C:\Program Files\Quarto\bin\quarto.exe') {
        $quartoExe = 'C:\Program Files\Quarto\bin\quarto.exe'
    }
    else {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw 'Quarto is missing and winget is not available to install it automatically.'
        }
        & winget install --id Posit.Quarto -e --source winget --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw 'Automatic Quarto installation failed.'
        }
        if (Test-Path 'C:\Program Files\Quarto\bin\quarto.exe') {
            $quartoExe = 'C:\Program Files\Quarto\bin\quarto.exe'
        }
        elseif (Get-Command quarto -ErrorAction SilentlyContinue) {
            $quartoExe = (Get-Command quarto -ErrorAction SilentlyContinue).Source
        }
        if (-not $quartoExe) {
            throw 'Quarto installation finished but the executable could not be located.'
        }
    }

    [pscustomobject]@{
        ProjectRoot = $projectRoot
        PythonExe = $venvPython
        QuartoExe = $quartoExe
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    Ensure-ResearchEnvironment | Format-List
}
