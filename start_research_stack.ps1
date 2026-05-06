param(
    [switch]$ForceInstall,
    [switch]$SkipRefresh,
    [switch]$NoLaunch
)

$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
. (Join-Path $ProjectRoot 'setup_research_env.ps1')
$envInfo = Ensure-ResearchEnvironment -ForceInstall:$ForceInstall

$stopScript = Join-Path $ProjectRoot 'stop_research_stack.ps1'
if (Test-Path $stopScript) {
    & powershell -ExecutionPolicy Bypass -File $stopScript | Out-Null
}

if (-not $SkipRefresh) {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot 'refresh_research_outputs.ps1')
    if ($LASTEXITCODE -ne 0) {
        throw 'Artifact refresh failed. The services were not launched.'
    }
}

$services = @(
    [pscustomobject]@{
        Name = 'Research Explorer'
        Role = 'Interactive artifact browser and metric explorer'
        Url = 'http://127.0.0.1:9000'
        Script = Join-Path $ProjectRoot 'run_research_app.ps1'
        Arguments = @('-Port', 9000)
    },
    [pscustomobject]@{
        Name = 'Slides'
        Role = 'Presentation slides built from the notebook outputs'
        Url = 'http://127.0.0.1:9001'
        Script = Join-Path $ProjectRoot 'run_research_slides.ps1'
        Arguments = @('-Port', 9001, '-NoBrowser')
    }
)

foreach ($service in $services) {
    $port = ([System.Uri]$service.Url).Port
    if (Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. Close the conflicting process before restarting the research stack."
    }
}

$outputDir = Join-Path $ProjectRoot 'research_extension_output'
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$servicesJsonPath = Join-Path $outputDir 'active_services.json'
$servicesTxtPath = Join-Path $outputDir 'active_services.txt'
$services | ConvertTo-Json -Depth 4 | Set-Content -Path $servicesJsonPath -Encoding UTF8
($services | ForEach-Object { "{0}: {1} [{2}]" -f $_.Name, $_.Url, $_.Role }) | Set-Content -Path $servicesTxtPath -Encoding UTF8

Write-Host ''
Write-Host 'Research stack ready.' -ForegroundColor Green
Write-Host "Project root: $($envInfo.ProjectRoot)"
Write-Host "Python env:   $($envInfo.PythonExe)"
Write-Host "Quarto:       $($envInfo.QuartoExe)"
Write-Host "Service map:  $servicesTxtPath"
Write-Host ''
foreach ($service in $services) {
    Write-Host ("{0}: {1}" -f $service.Name, $service.Url)
    Write-Host ("  {0}" -f $service.Role)
}
Write-Host ''

if ($NoLaunch) {
    Write-Host 'No windows launched because -NoLaunch was used.'
    return
}

foreach ($service in $services) {
    $title = $service.Name
    $script = $service.Script
    $argumentString = ($service.Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $command = "`$Host.UI.RawUI.WindowTitle = '$title'; & '$script' $argumentString"
    Start-Process powershell -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $command) | Out-Null
}
