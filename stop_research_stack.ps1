$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
$servicesJsonPath = Join-Path $ProjectRoot 'research_extension_output\active_services.json'

if (-not (Test-Path $servicesJsonPath)) {
    Write-Host 'No active service map was found.' -ForegroundColor Yellow
    exit 0
}

$services = Get-Content $servicesJsonPath -Raw | ConvertFrom-Json
foreach ($service in $services) {
    $uri = [System.Uri]$service.Url
    $port = $uri.Port
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Host ("{0}: no listener found on port {1}" -f $service.Name, $port)
        continue
    }

    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pid in $pids) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction Stop
            Write-Host ("{0}: stopped PID {1} on port {2}" -f $service.Name, $pid, $port)
        }
        catch {
            Write-Warning ("{0}: unable to stop PID {1} on port {2}" -f $service.Name, $pid, $port)
        }
    }
}
