param(
    [string]$ResourceGroup = "rg-ocr-demo",
    [string]$ContainerAppName = "pixelledger",
    [string]$StorageAccountName = "stocraccuracyks",
    [string]$BlobContainerName = "appdata",
    [string]$BlobPrefix = "auth"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI (az) is required."
}

$fqdn = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
if (-not $fqdn) {
    throw "Unable to resolve ACA FQDN."
}

$healthUrl = "https://$fqdn/_stcore/health"
Write-Host "Checking app health: $healthUrl"
$health = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 30
if ($health.StatusCode -ne 200) {
    throw "Health check failed with status $($health.StatusCode)"
}
Write-Host "Health check passed."

$expected = @("users.json", "passcode_requests.json", "login_activity.json")
foreach ($name in $expected) {
    $blobName = if ($BlobPrefix) { "$BlobPrefix/$name" } else { $name }
    $exists = az storage blob exists --account-name $StorageAccountName --auth-mode login --container-name $BlobContainerName --name $blobName --query exists -o tsv
    if ($exists -eq "true") {
        Write-Host "Found blob: $blobName"
    } else {
        Write-Warning "Missing blob: $blobName (it may appear after first app interaction)"
    }
}

Write-Host "Smoke check complete."
