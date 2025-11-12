# Infrastructure Setup Scripts

This directory contains scripts for setting up Azure infrastructure resources.

## Prerequisites

1. **Azure CLI** installed and configured
   ```bash
   # Install Azure CLI (if not installed)
   # macOS: brew install azure-cli
   # Linux: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   
   # Login to Azure
   az login
   
   # Set your subscription (if you have multiple)
   az account set --subscription "your-subscription-id"
   ```

2. **Resource Group** created in Azure
   ```bash
   az group create --name <RESOURCE_GROUP_NAME> --location <LOCATION>
   ```

## Scripts

### setup-azure-openai.sh

Creates an Azure OpenAI service instance and deploys a GPT-4 model. Optionally deploys an embedding model for RAG (Retrieval Augmented Generation).

**Usage:**
```bash
./scripts/setup-azure-openai.sh <RESOURCE_GROUP_NAME> [OPTIONS]
```

**Options:**
- `--location LOCATION` - Azure region (default: resource group location)
- `--service-name NAME` - OpenAI service name (default: `stock-sentiment-openai`)
- `--model-name NAME` - Model name (default: `gpt-4`)
- `--model-version VERSION` - Model version (default: `turbo-2024-04-09`)
- `--deployment-name NAME` - Deployment name (default: `gpt-4`)
- `--api-version VERSION` - API version (default: `2023-05-15`)
- `--sku SKU` - SKU tier (default: `S0`)
- `--owner-email EMAIL` - Owner email for tagging
- `--deploy-embedding` - Deploy embedding model for RAG (default: enabled)
- `--no-embedding` - Skip embedding model deployment
- `--embedding-model NAME` - Embedding model name (default: `text-embedding-ada-002`)
- `--embedding-version VERSION` - Embedding model version (default: `2`)
- `--embedding-deployment NAME` - Embedding deployment name (default: `text-embedding-ada-002`)

**Example:**
```bash
# Deploy with RAG (embedding model) - recommended
./scripts/setup-azure-openai.sh my-resource-group \
  --location eastus \
  --owner-email user@example.com

# Deploy without RAG (skip embedding model)
./scripts/setup-azure-openai.sh my-resource-group \
  --location eastus \
  --no-embedding \
  --owner-email user@example.com
```

**What it does:**
1. Creates Azure OpenAI service instance
2. Deploys GPT-4 Turbo model for chat completions
3. Deploys embedding model (text-embedding-ada-002) for RAG (if enabled)
4. Retrieves API key and endpoint
5. Displays configuration information

**Output:**
The script will output:
- Service endpoint URL
- API key (save this securely!)
- Embedding deployment name (if RAG is enabled)
- Environment variables to add to `.env` file

### add-embedding-model.sh

Adds an embedding model deployment to an existing Azure OpenAI service for RAG support.

**Usage:**
```bash
./scripts/add-embedding-model.sh <RESOURCE_GROUP_NAME> <SERVICE_NAME> [OPTIONS]
```

**Options:**
- `--embedding-model NAME` - Embedding model name (default: `text-embedding-ada-002`)
- `--embedding-version VERSION` - Embedding model version (default: `2`)
- `--embedding-deployment NAME` - Embedding deployment name (default: `text-embedding-ada-002`)

**Example:**
```bash
./scripts/add-embedding-model.sh my-resource-group stock-sentiment-openai
```

**When to use:**
- You already have an Azure OpenAI service set up
- You want to add RAG support without recreating the service
- The main setup script didn't deploy the embedding model

**Output:**
The script will output the embedding deployment name to add to your `.env` file.

### setup-azure-redis.sh

Creates an Azure Cache for Redis instance.

**Usage:**
```bash
./scripts/setup-azure-redis.sh <RESOURCE_GROUP_NAME> [OPTIONS]
```

**Options:**
- `--redis-name NAME` - Redis cache name (default: `stock-sentiment-redis`)
- `--location LOCATION` - Azure region (default: resource group location)
- `--sku SKU` - SKU tier: Basic, Standard, Premium (default: `Basic`)
- `--capacity CAPACITY` - Cache size: C0, C1, C2, etc. (default: `C0`)
- `--minimum-tls VERSION` - Minimum TLS version: 1.0, 1.2 (default: `1.2`)
- `--owner-email EMAIL` - Owner email for tagging

**SKU and Capacity combinations:**
- **Basic**: C0 (250MB), C1 (1GB), C2 (2.5GB), C3 (6GB), C4 (13GB), C5 (26GB), C6 (53GB)
- **Standard**: C0-C6 (same as Basic, with replication)
- **Premium**: P1 (6GB), P2 (13GB), P3 (26GB), P4 (53GB), P5 (120GB)

**Example:**
```bash
./scripts/setup-azure-redis.sh my-resource-group \
  --location eastus \
  --sku Basic \
  --capacity C0
```

**What it does:**
1. Creates Azure Cache for Redis instance
2. Waits for Redis to be ready (may take 10-20 minutes)
3. Retrieves connection details and access keys
4. Displays configuration information

**Output:**
The script will output:
- Redis host and port
- Primary and secondary access keys (save securely!)
- Connection string
- Environment variables to add to `.env` file

## Quick Start

1. **Create a resource group** (if you don't have one):
   ```bash
   az group create --name stock-sentiment-rg --location eastus
   ```

2. **Set up Azure OpenAI (with RAG)**:
   ```bash
   # This will deploy both GPT-4 and embedding model for RAG
   ./scripts/setup-azure-openai.sh stock-sentiment-rg --location eastus
   ```

3. **Set up Azure Redis**:
   ```bash
   ./scripts/setup-azure-redis.sh stock-sentiment-rg --location eastus
   ```

4. **Copy the output** from both scripts and add to your `.env` file:
   ```bash
   cp .env.example .env
   # Then edit .env with the values from the scripts
   # Make sure to include AZURE_OPENAI_EMBEDDING_DEPLOYMENT if RAG was enabled
   ```

## Notes

- Both scripts are idempotent - they check if resources already exist before creating
- The scripts will prompt you if resources already exist
- API keys and connection strings are sensitive - never commit them to git
- Azure OpenAI service creation may take a few minutes
- Redis cache creation may take 10-20 minutes
- Make sure you have appropriate permissions in your Azure subscription

## Troubleshooting

### "Resource group not found"
- Create the resource group first: `az group create --name <NAME> --location <LOCATION>`
- Or specify `--location` explicitly in the script

### "Not logged in to Azure"
- Run `az login` first

### "Insufficient permissions"
- Make sure your Azure account has Contributor or Owner role on the subscription/resource group

### "OpenAI service not available in region"
- Check [Azure OpenAI Service regions](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/concepts/regions)
- Choose a different region that supports Azure OpenAI

### "Embedding model deployment failed"
- The embedding model (text-embedding-ada-002) may not be available in your region
- Check available models: `az cognitiveservices account list-models --name <SERVICE_NAME> --resource-group <RG_NAME>`
- You can skip embedding deployment with `--no-embedding` flag
- RAG will be disabled but the app will still work for sentiment analysis

