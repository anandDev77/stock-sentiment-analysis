#!/bin/bash

# Azure OpenAI Setup Script
# This script creates an Azure OpenAI service instance and deploys a GPT-4 model
# Usage: ./setup-azure-openai.sh <RESOURCE_GROUP_NAME> [OPTIONS]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values (can be overridden by environment variables)
SERVICE_NAME="${AZURE_OPENAI_SERVICE_NAME:-stock-sentiment-openai}"
MODEL_NAME="${AZURE_OPENAI_MODEL_NAME:-gpt-4}"
MODEL_VERSION="${AZURE_OPENAI_MODEL_VERSION:-turbo-2024-04-09}"
DEPLOYMENT_NAME="${AZURE_OPENAI_DEPLOYMENT_NAME:-gpt-4}"
API_VERSION="${AZURE_OPENAI_API_VERSION:-2023-05-15}"
SKU="${AZURE_OPENAI_SKU:-S0}"
SKU_CAPACITY="${AZURE_OPENAI_SKU_CAPACITY:-10}"
SKU_NAME="${AZURE_OPENAI_SKU_NAME:-Standard}"

# Embedding model defaults (for RAG)
EMBEDDING_MODEL_NAME="${AZURE_OPENAI_EMBEDDING_MODEL_NAME:-text-embedding-ada-002}"
EMBEDDING_MODEL_VERSION="${AZURE_OPENAI_EMBEDDING_MODEL_VERSION:-2}"
EMBEDDING_DEPLOYMENT_NAME="${AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:-text-embedding-ada-002}"
EMBEDDING_SKU_CAPACITY="${AZURE_OPENAI_EMBEDDING_SKU_CAPACITY:-10}"
EMBEDDING_SKU_NAME="${AZURE_OPENAI_EMBEDDING_SKU_NAME:-Standard}"
DEPLOY_EMBEDDING="${AZURE_OPENAI_DEPLOY_EMBEDDING:-true}"

# Infrastructure tags (can be overridden by environment variables)
TAG_PURPOSE="${INFRASTRUCTURE_TAG_PURPOSE:-stock-sentiment-analysis}"
TAG_SOLUTION="${INFRASTRUCTURE_TAG_SOLUTION:-stock-sentiment-analysis}"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    echo "Usage: $0 <RESOURCE_GROUP_NAME> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  RESOURCE_GROUP_NAME    Azure Resource Group name"
    echo ""
    echo "Options:"
    echo "  --location LOCATION           Azure region (default: resource group location)"
    echo "  --service-name NAME          OpenAI service name (default: $SERVICE_NAME)"
    echo "  --model-name NAME            Model name (default: $MODEL_NAME)"
    echo "  --model-version VERSION      Model version (default: $MODEL_VERSION)"
    echo "  --deployment-name NAME        Deployment name (default: $DEPLOYMENT_NAME)"
    echo "  --api-version VERSION        API version (default: $API_VERSION)"
    echo "  --sku SKU                    SKU tier (default: $SKU)"
    echo "  --owner-email EMAIL          Owner email for tagging"
    echo "  --deploy-embedding           Deploy embedding model for RAG (default: true)"
    echo "  --no-embedding               Skip embedding model deployment"
    echo "  --embedding-model NAME       Embedding model name (default: $EMBEDDING_MODEL_NAME)"
    echo "  --embedding-version VERSION  Embedding model version (default: $EMBEDDING_MODEL_VERSION)"
    echo "  --embedding-deployment NAME  Embedding deployment name (default: $EMBEDDING_DEPLOYMENT_NAME)"
    echo "  --help                       Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 my-resource-group --location eastus --owner-email user@example.com"
    exit 1
}

# Parse arguments
RG_NAME=""
LOCATION=""
OWNER_EMAIL=""

if [ $# -eq 0 ]; then
    usage
fi

RG_NAME=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --location)
            LOCATION="$2"
            shift 2
            ;;
        --service-name)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-version)
            MODEL_VERSION="$2"
            shift 2
            ;;
        --deployment-name)
            DEPLOYMENT_NAME="$2"
            shift 2
            ;;
        --api-version)
            API_VERSION="$2"
            shift 2
            ;;
        --sku)
            SKU="$2"
            shift 2
            ;;
        --owner-email)
            OWNER_EMAIL="$2"
            shift 2
            ;;
        --deploy-embedding)
            DEPLOY_EMBEDDING="true"
            shift
            ;;
        --no-embedding)
            DEPLOY_EMBEDDING="false"
            shift
            ;;
        --embedding-model)
            EMBEDDING_MODEL_NAME="$2"
            shift 2
            ;;
        --embedding-version)
            EMBEDDING_MODEL_VERSION="$2"
            shift 2
            ;;
        --embedding-deployment)
            EMBEDDING_DEPLOYMENT_NAME="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$RG_NAME" ]; then
    print_error "Resource Group name is required"
    usage
fi

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first."
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    print_error "Not logged in to Azure. Please run 'az login' first."
    exit 1
fi

# Get resource group location if not provided
if [ -z "$LOCATION" ]; then
    print_info "Getting resource group location..."
    LOCATION=$(az group show --name "$RG_NAME" --query location -o tsv 2>/dev/null || echo "")
    if [ -z "$LOCATION" ]; then
        print_error "Resource group '$RG_NAME' not found. Please create it first or specify --location."
        exit 1
    fi
    print_info "Using resource group location: $LOCATION"
fi

# Check if OpenAI service already exists
print_info "Checking if OpenAI service '$SERVICE_NAME' already exists..."
if az cognitiveservices account show --name "$SERVICE_NAME" --resource-group "$RG_NAME" &> /dev/null; then
    print_warning "OpenAI service '$SERVICE_NAME' already exists in resource group '$RG_NAME'"
    read -p "Do you want to continue and check deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Exiting..."
        exit 0
    fi
else
    # Create OpenAI service
    print_info "Creating Azure OpenAI service '$SERVICE_NAME'..."
    
    # Build tags from environment variables or defaults
    TAGS="purpose=${TAG_PURPOSE} solution=${TAG_SOLUTION}"
    if [ -n "$OWNER_EMAIL" ]; then
        TAG_OWNER="${INFRASTRUCTURE_TAG_OWNER:-$OWNER_EMAIL}"
        TAG_CREATED_BY="${INFRASTRUCTURE_TAG_CREATED_BY:-$OWNER_EMAIL}"
        TAGS="$TAGS owner=${TAG_OWNER} created-by=${TAG_CREATED_BY}"
    elif [ -n "${INFRASTRUCTURE_TAG_OWNER}" ]; then
        TAGS="$TAGS owner=${INFRASTRUCTURE_TAG_OWNER}"
    fi
    if [ -n "${INFRASTRUCTURE_TAG_CREATED_BY}" ] && [ -z "$OWNER_EMAIL" ]; then
        TAGS="$TAGS created-by=${INFRASTRUCTURE_TAG_CREATED_BY}"
    fi
    
    az cognitiveservices account create \
        --name "$SERVICE_NAME" \
        --resource-group "$RG_NAME" \
        --location "$LOCATION" \
        --kind OpenAI \
        --sku "$SKU" \
        --tags $TAGS
    
    if [ $? -eq 0 ]; then
        print_info "OpenAI service created successfully!"
    else
        print_error "Failed to create OpenAI service"
        exit 1
    fi
fi

# Check if deployment already exists
print_info "Checking if deployment '$DEPLOYMENT_NAME' already exists..."
if az cognitiveservices account deployment show \
    --name "$SERVICE_NAME" \
    --resource-group "$RG_NAME" \
    --deployment-name "$DEPLOYMENT_NAME" &> /dev/null; then
    print_warning "Deployment '$DEPLOYMENT_NAME' already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping deployment creation..."
    else
        print_info "Deleting existing deployment..."
        az cognitiveservices account deployment delete \
            --name "$SERVICE_NAME" \
            --resource-group "$RG_NAME" \
            --deployment-name "$DEPLOYMENT_NAME" \
            --yes
        
        # Wait a bit for deletion to complete
        sleep 5
    fi
fi

# Create deployment if it doesn't exist or was deleted
if ! az cognitiveservices account deployment show \
    --name "$SERVICE_NAME" \
    --resource-group "$RG_NAME" \
    --deployment-name "$DEPLOYMENT_NAME" &> /dev/null; then
    
    print_info "Deploying model '$MODEL_NAME' (version: $MODEL_VERSION)..."
    print_warning "This may take several minutes..."
    
    az cognitiveservices account deployment create \
        --name "$SERVICE_NAME" \
        --resource-group "$RG_NAME" \
        --deployment-name "$DEPLOYMENT_NAME" \
        --model-name "$MODEL_NAME" \
        --model-version "$MODEL_VERSION" \
        --model-format OpenAI \
        --sku-capacity "$SKU_CAPACITY" \
        --sku-name "$SKU_NAME"
    
    if [ $? -eq 0 ]; then
        print_info "Model deployment created successfully!"
    else
        print_error "Failed to create model deployment"
        exit 1
    fi
fi

# Get API key and endpoint
print_info "Retrieving API key and endpoint..."
AZURE_OPENAI_API_KEY=$(az cognitiveservices account keys list \
    --name "$SERVICE_NAME" \
    --resource-group "$RG_NAME" \
    --query "key1" --output tsv)

# Get the full service endpoint (base endpoint, not deployment-specific)
SERVICE_ENDPOINT=$(az cognitiveservices account show \
    --name "$SERVICE_NAME" \
    --resource-group "$RG_NAME" \
    --query properties.endpoint -o tsv)

# Deploy embedding model for RAG if requested
EMBEDDING_DEPLOYED="false"
if [ "$DEPLOY_EMBEDDING" = "true" ]; then
    print_info "Checking if embedding deployment '$EMBEDDING_DEPLOYMENT_NAME' already exists..."
    if az cognitiveservices account deployment show \
        --name "$SERVICE_NAME" \
        --resource-group "$RG_NAME" \
        --deployment-name "$EMBEDDING_DEPLOYMENT_NAME" &> /dev/null; then
        print_warning "Embedding deployment '$EMBEDDING_DEPLOYMENT_NAME' already exists"
        EMBEDDING_DEPLOYED="true"
    else
        print_info "Deploying embedding model '$EMBEDDING_MODEL_NAME' (version: $EMBEDDING_MODEL_VERSION)..."
        print_warning "This may take several minutes..."
        
        az cognitiveservices account deployment create \
            --name "$SERVICE_NAME" \
            --resource-group "$RG_NAME" \
            --deployment-name "$EMBEDDING_DEPLOYMENT_NAME" \
            --model-name "$EMBEDDING_MODEL_NAME" \
            --model-version "$EMBEDDING_MODEL_VERSION" \
            --model-format OpenAI \
            --sku-capacity "$EMBEDDING_SKU_CAPACITY" \
            --sku-name "$EMBEDDING_SKU_NAME"
        
        if [ $? -eq 0 ]; then
            print_info "Embedding model deployment created successfully!"
            EMBEDDING_DEPLOYED="true"
        else
            print_error "Failed to create embedding model deployment"
            print_warning "RAG will not be available, but chat completions will still work"
        fi
    fi
fi

# Display results
echo ""
print_info "=========================================="
print_info "Azure OpenAI Setup Complete!"
print_info "=========================================="
echo ""
echo "Service Name:      $SERVICE_NAME"
echo "Resource Group:    $RG_NAME"
echo "Location:          $LOCATION"
echo "Deployment Name:   $DEPLOYMENT_NAME"
echo "Model:             $MODEL_NAME ($MODEL_VERSION)"
echo "API Version:       $API_VERSION"
if [ "$EMBEDDING_DEPLOYED" = "true" ]; then
    echo "Embedding Model:   $EMBEDDING_MODEL_NAME ($EMBEDDING_MODEL_VERSION)"
    echo "Embedding Deployment: $EMBEDDING_DEPLOYMENT_NAME"
    echo "RAG Status:       ✅ Enabled"
else
    echo "RAG Status:       ⚠️  Disabled (embedding model not deployed)"
fi
echo ""
echo "Service Endpoint:  $SERVICE_ENDPOINT"
echo ""
print_warning "API Key (save this securely):"
echo "$AZURE_OPENAI_API_KEY"
echo ""
print_info "Add these to your .env file:"
echo "AZURE_OPENAI_ENDPOINT=$SERVICE_ENDPOINT"
echo "AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY"
echo "AZURE_OPENAI_DEPLOYMENT_NAME=$DEPLOYMENT_NAME"
echo "AZURE_OPENAI_API_VERSION=$API_VERSION"
if [ "$EMBEDDING_DEPLOYED" = "true" ]; then
    echo "AZURE_OPENAI_EMBEDDING_DEPLOYMENT=$EMBEDDING_DEPLOYMENT_NAME"
fi
echo ""
print_info "=========================================="

