#!/bin/bash

# Add Embedding Model to Existing Azure OpenAI Service
# This script adds an embedding model deployment to an existing Azure OpenAI service for RAG
# Usage: ./add-embedding-model.sh <RESOURCE_GROUP_NAME> <SERVICE_NAME> [OPTIONS]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
EMBEDDING_MODEL_NAME="${AZURE_OPENAI_EMBEDDING_MODEL_NAME:-text-embedding-ada-002}"
EMBEDDING_MODEL_VERSION="${AZURE_OPENAI_EMBEDDING_MODEL_VERSION:-2}"
EMBEDDING_DEPLOYMENT_NAME="${AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:-text-embedding-ada-002}"
EMBEDDING_SKU_CAPACITY="${AZURE_OPENAI_EMBEDDING_SKU_CAPACITY:-10}"
EMBEDDING_SKU_NAME="${AZURE_OPENAI_EMBEDDING_SKU_NAME:-Standard}"

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
    echo "Usage: $0 <RESOURCE_GROUP_NAME> <SERVICE_NAME> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  RESOURCE_GROUP_NAME    Azure Resource Group name"
    echo "  SERVICE_NAME          Azure OpenAI service name"
    echo ""
    echo "Options:"
    echo "  --embedding-model NAME       Embedding model name (default: $EMBEDDING_MODEL_NAME)"
    echo "  --embedding-version VERSION  Embedding model version (default: $EMBEDDING_MODEL_VERSION)"
    echo "  --embedding-deployment NAME  Embedding deployment name (default: $EMBEDDING_DEPLOYMENT_NAME)"
    echo "  --help                       Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 my-resource-group stock-sentiment-openai"
    exit 1
}

# Parse arguments
RG_NAME=""
SERVICE_NAME=""

if [ $# -lt 2 ]; then
    usage
fi

RG_NAME=$1
SERVICE_NAME=$2
shift 2

while [[ $# -gt 0 ]]; do
    case $1 in
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
if [ -z "$RG_NAME" ] || [ -z "$SERVICE_NAME" ]; then
    print_error "Resource Group name and Service name are required"
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

# Check if OpenAI service exists
print_info "Checking if OpenAI service '$SERVICE_NAME' exists..."
if ! az cognitiveservices account show --name "$SERVICE_NAME" --resource-group "$RG_NAME" &> /dev/null; then
    print_error "OpenAI service '$SERVICE_NAME' not found in resource group '$RG_NAME'"
    exit 1
fi

print_info "OpenAI service found!"

# Check if embedding deployment already exists
print_info "Checking if embedding deployment '$EMBEDDING_DEPLOYMENT_NAME' already exists..."
if az cognitiveservices account deployment show \
    --name "$SERVICE_NAME" \
    --resource-group "$RG_NAME" \
    --deployment-name "$EMBEDDING_DEPLOYMENT_NAME" &> /dev/null; then
    print_warning "Embedding deployment '$EMBEDDING_DEPLOYMENT_NAME' already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Exiting..."
        exit 0
    else
        print_info "Deleting existing deployment..."
        az cognitiveservices account deployment delete \
            --name "$SERVICE_NAME" \
            --resource-group "$RG_NAME" \
            --deployment-name "$EMBEDDING_DEPLOYMENT_NAME" \
            --yes
        
        # Wait a bit for deletion to complete
        sleep 5
    fi
fi

# Create embedding deployment
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
else
    print_error "Failed to create embedding model deployment"
    exit 1
fi

# Display results
echo ""
print_info "=========================================="
print_info "Embedding Model Deployment Complete!"
print_info "=========================================="
echo ""
echo "Service Name:           $SERVICE_NAME"
echo "Resource Group:         $RG_NAME"
echo "Embedding Model:        $EMBEDDING_MODEL_NAME ($EMBEDDING_MODEL_VERSION)"
echo "Embedding Deployment:  $EMBEDDING_DEPLOYMENT_NAME"
echo "RAG Status:            ✅ Enabled"
echo ""
print_info "Add this to your .env file:"
echo "AZURE_OPENAI_EMBEDDING_DEPLOYMENT=$EMBEDDING_DEPLOYMENT_NAME"
echo ""
print_info "=========================================="

