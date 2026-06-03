# ACA Deployment (pixelledger)

This folder contains a starter deployment script for hosting this Streamlit app on Azure Container Apps.

## Locked defaults

- Resource Group: `rg-ocr-demo`
- Region: `centralindia`
- Container App: `pixelledger`
- ACA Environment: `ks-pixelledger`
- ACR: `acrocraccuracyks`
- Storage Account: `stocraccuracyks`
- Blob container for auth JSON: `appdata`

## Prerequisites

- Azure CLI logged in (`az login`)
- Docker installed
- Permissions to create resources in the subscription

## Deploy

```powershell
./deploy/aca/deploy.ps1
```

Optional parameters:

```powershell
./deploy/aca/deploy.ps1 -SubscriptionId <sub-id> -ResourceGroup rg-ocr-demo -Location centralindia
```

## Set model endpoints and keys

```powershell
./deploy/aca/set-model-env.ps1 -AzDiEndpoint <...> -AzDiKey <...> -AzOpenAiEndpoint <...> -AzOpenAiKey <...>
```

## Smoke check

```powershell
./deploy/aca/smoke-check.ps1
```

This checks ACA health endpoint and verifies expected auth JSON blobs.

## Notes

- The script sets auth storage to Blob mode using environment variables:
  - `AUTH_STORAGE_BACKEND=blob`
  - `AUTH_BLOB_CONTAINER=appdata`
  - `AUTH_BLOB_PREFIX=auth`
- `AUTH_BLOB_ACCOUNT_URL` is set directly.
- Container App uses System Assigned Managed Identity with `Storage Blob Data Contributor` RBAC on the storage account.
- Azure OpenAI and Document Intelligence env vars must be set on the container app after deployment.
