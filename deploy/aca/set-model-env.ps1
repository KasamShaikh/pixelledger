param(
    [string]$ResourceGroup = "rg-ocr-demo",
    [string]$ContainerAppName = "pixelledger",
    [string]$AzDiEndpoint,
    [string]$AzDiKey,
    [string]$AzOpenAiEndpoint,
    [string]$AzOpenAiKey,
    [string]$AzOpenAiApiVersion = "2024-12-01-preview",
    [string]$DepGpt5 = "gpt-5",
    [string]$DepGpt51 = "gpt-5.1",
    [string]$DepGpt5Mini = "gpt-5-mini",
    [string]$DepGpt54Mini = "gpt-5.4-mini",
    [string]$DepGpt4o = "gpt-4o",
    [string]$DepGpt4oMini = "gpt-4o-mini",
    [string]$DepJudge = "gpt-5"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI (az) is required."
}

az containerapp update `
  --name $ContainerAppName `
  --resource-group $ResourceGroup `
  --set-env-vars `
      AZURE_DI_ENDPOINT=$AzDiEndpoint `
      AZURE_DI_KEY=$AzDiKey `
      AZURE_OPENAI_ENDPOINT=$AzOpenAiEndpoint `
      AZURE_OPENAI_KEY=$AzOpenAiKey `
      AZURE_OPENAI_API_VERSION=$AzOpenAiApiVersion `
      AOAI_DEPLOYMENT_GPT5=$DepGpt5 `
      AOAI_DEPLOYMENT_GPT51=$DepGpt51 `
      AOAI_DEPLOYMENT_GPT5_MINI=$DepGpt5Mini `
      AOAI_DEPLOYMENT_GPT54_MINI=$DepGpt54Mini `
      AOAI_DEPLOYMENT_GPT4O=$DepGpt4o `
      AOAI_DEPLOYMENT_GPT4O_MINI=$DepGpt4oMini `
      AOAI_DEPLOYMENT_JUDGE=$DepJudge | Out-Null

Write-Host "Model environment variables updated for $ContainerAppName"
