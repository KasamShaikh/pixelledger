param(
    [string]$SubscriptionId = "",
    [string]$ResourceGroup = "rg-ocr-demo",
    [string]$Location = "centralindia",
    [string]$ContainerAppName = "pixelledger",
    [string]$ContainerAppEnvName = "ks-pixelledger",
    [string]$AcrName = "acrocraccuracyks",
    [string]$StorageAccountName = "stocraccuracyks",
    [string]$BlobContainerName = "appdata",
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI (az) is required."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is required for building/pushing the image."
}

if ($SubscriptionId) {
    az account set --subscription $SubscriptionId
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$imageName = "$AcrName.azurecr.io/$ContainerAppName`:$ImageTag"
$workspaceName = "log-ocr-accuracy-KS"

Write-Host "Creating resource group if needed..."
$rgExists = az group exists --name $ResourceGroup
if ($rgExists -eq "true") {
  $existingRgLocation = az group show --name $ResourceGroup --query location -o tsv
  if ($existingRgLocation -ne $Location) {
    Write-Warning "Resource group '$ResourceGroup' exists in '$existingRgLocation'. Continuing with that scope."
  }
} else {
  az group create --name $ResourceGroup --location $Location | Out-Null
}

Write-Host "Creating Log Analytics workspace if needed..."
az monitor log-analytics workspace create `
  --resource-group $ResourceGroup `
  --workspace-name $workspaceName `
  --location $Location | Out-Null

Write-Host "Creating ACR if needed..."
az acr create `
  --name $AcrName `
  --resource-group $ResourceGroup `
  --location $Location `
  --sku Basic `
  --admin-enabled true | Out-Null

Write-Host "Building and pushing Docker image..."
az acr build --registry $AcrName --image "$ContainerAppName`:$ImageTag" $repoRoot

Write-Host "Creating storage account if needed..."
az storage account create `
  --name $StorageAccountName `
  --resource-group $ResourceGroup `
  --location $Location `
  --sku Standard_LRS `
  --kind StorageV2 | Out-Null

az storage container create `
  --name $BlobContainerName `
  --account-name $StorageAccountName `
  --auth-mode login | Out-Null

Write-Host "Creating Container Apps environment if needed..."
az containerapp env create `
  --name $ContainerAppEnvName `
  --resource-group $ResourceGroup `
  --location $Location | Out-Null

Write-Host "Creating/updating Container App..."
az containerapp create `
  --name $ContainerAppName `
  --resource-group $ResourceGroup `
  --environment $ContainerAppEnvName `
  --image $imageName `
  --target-port 8501 `
  --ingress external `
  --registry-server "$AcrName.azurecr.io" `
  --cpu 1.0 `
  --memory 2.0Gi `
  --min-replicas 0 `
  --max-replicas 2 `
  --system-assigned `
  --env-vars `
      AUTH_STORAGE_BACKEND=blob `
      AUTH_BLOB_CONTAINER=$BlobContainerName `
      AUTH_BLOB_PREFIX=auth `
      AUTH_BLOB_ACCOUNT_URL="https://$StorageAccountName.blob.core.windows.net" | Out-Null

$principalId = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup --query identity.principalId -o tsv
if (-not $principalId) {
    throw "Unable to resolve managed identity principalId for container app."
}

$storageScope = az storage account show --resource-group $ResourceGroup --name $StorageAccountName --query id -o tsv
Write-Host "Assigning Storage Blob Data Contributor role to Container App managed identity..."
az role assignment create `
  --assignee-object-id $principalId `
  --assignee-principal-type ServicePrincipal `
  --role "Storage Blob Data Contributor" `
  --scope $storageScope | Out-Null

Write-Host "Container app endpoint:"
az containerapp show --name $ContainerAppName --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
