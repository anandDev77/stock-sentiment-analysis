#!/bin/bash

# Azure Redis Setup Script
# This script creates an Azure Cache for Redis instance in the specified resource group
# Usage: ./setup-azure-redis.sh <RESOURCE_GROUP_NAME> [OPTIONS]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values (can be overridden by environment variables)
REDIS_NAME="${AZURE_REDIS_CACHE_NAME:-stock-sentiment-redis}"
SKU="${AZURE_REDIS_SKU:-Basic}"
CAPACITY="${AZURE_REDIS_CAPACITY:-C0}"
LOCATION=""
MINIMUM_TLS_VERSION="${AZURE_REDIS_MINIMUM_TLS:-1.2}"

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
    echo "  --redis-name NAME           Redis cache name (default: $REDIS_NAME)"
    echo "  --location LOCATION         Azure region (default: resource group location)"
    echo "  --sku SKU                  SKU tier: Basic, Standard, Premium (default: $SKU)"
    echo "  --capacity CAPACITY        Cache size: C0, C1, C2, C3, C4, C5, C6 (default: $CAPACITY)"
    echo "  --minimum-tls VERSION       Minimum TLS version: 1.0, 1.2 (default: $MINIMUM_TLS_VERSION)"
    echo "  --owner-email EMAIL         Owner email for tagging"
    echo "  --help                      Show this help message"
    echo ""
    echo "SKU and Capacity combinations:"
    echo "  Basic:    C0 (250MB), C1 (1GB), C2 (2.5GB), C3 (6GB), C4 (13GB), C5 (26GB), C6 (53GB)"
    echo "  Standard: C0-C6 (same as Basic, with replication)"
    echo "  Premium:   P1 (6GB), P2 (13GB), P3 (26GB), P4 (53GB), P5 (120GB)"
    echo ""
    echo "Example:"
    echo "  $0 my-resource-group --location eastus --sku Basic --capacity C0"
    exit 1
}

# Parse arguments
RG_NAME=""
OWNER_EMAIL=""

if [ $# -eq 0 ]; then
    usage
fi

RG_NAME=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --redis-name)
            REDIS_NAME="$2"
            shift 2
            ;;
        --location)
            LOCATION="$2"
            shift 2
            ;;
        --sku)
            SKU="$2"
            shift 2
            ;;
        --capacity)
            CAPACITY="$2"
            shift 2
            ;;
        --minimum-tls)
            MINIMUM_TLS_VERSION="$2"
            shift 2
            ;;
        --owner-email)
            OWNER_EMAIL="$2"
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

# Validate SKU
if [[ ! "$SKU" =~ ^(Basic|Standard|Premium)$ ]]; then
    print_error "Invalid SKU: $SKU. Must be Basic, Standard, or Premium"
    exit 1
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

# Check if Redis cache already exists
print_info "Checking if Redis cache '$REDIS_NAME' already exists..."
if az redis show --name "$REDIS_NAME" --resource-group "$RG_NAME" &> /dev/null; then
    print_warning "Redis cache '$REDIS_NAME' already exists in resource group '$RG_NAME'"
    read -p "Do you want to continue and retrieve connection info? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Exiting..."
        exit 0
    fi
else
    # Create Redis cache
    print_info "Creating Azure Redis cache '$REDIS_NAME'..."
    print_warning "This may take 10-20 minutes..."
    
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
    
    # Create Redis cache
    # Note: --enable-non-ssl-port is not needed for Basic SKU (SSL is enabled by default)
    az redis create \
        --name "$REDIS_NAME" \
        --resource-group "$RG_NAME" \
        --location "$LOCATION" \
        --sku "$SKU" \
        --vm-size "$CAPACITY" \
        --minimum-tls-version "$MINIMUM_TLS_VERSION" \
        --tags $TAGS
    
    if [ $? -eq 0 ]; then
        print_info "Redis cache created successfully!"
    else
        print_error "Failed to create Redis cache"
        exit 1
    fi
    
    # Check provisioning state
    print_info "Checking Redis cache provisioning state..."
    PROVISIONING_STATE=$(az redis show --name "$REDIS_NAME" --resource-group "$RG_NAME" --query provisioningState -o tsv)
    
    if [ "$PROVISIONING_STATE" != "Succeeded" ]; then
        print_warning "Redis cache is still provisioning. State: $PROVISIONING_STATE"
        print_info "You can check status with: az redis show --name $REDIS_NAME --resource-group $RG_NAME"
        print_info "Continuing to retrieve connection info..."
    else
        print_info "Redis cache is ready!"
    fi
fi

# Get Redis connection details
print_info "Retrieving Redis connection information..."

REDIS_HOST=$(az redis show \
    --name "$REDIS_NAME" \
    --resource-group "$RG_NAME" \
    --query hostName -o tsv)

REDIS_PORT=$(az redis show \
    --name "$REDIS_NAME" \
    --resource-group "$RG_NAME" \
    --query port -o tsv)

REDIS_SSL_PORT=$(az redis show \
    --name "$REDIS_NAME" \
    --resource-group "$RG_NAME" \
    --query sslPort -o tsv)

# Get access keys
print_info "Retrieving Redis access keys..."
REDIS_PRIMARY_KEY=$(az redis list-keys \
    --name "$REDIS_NAME" \
    --resource-group "$RG_NAME" \
    --query primaryKey -o tsv)

REDIS_SECONDARY_KEY=$(az redis list-keys \
    --name "$REDIS_NAME" \
    --resource-group "$RG_NAME" \
    --query secondaryKey -o tsv)

# Build connection string
REDIS_CONNECTION_STRING="${REDIS_HOST}:${REDIS_SSL_PORT},password=${REDIS_PRIMARY_KEY},ssl=True,abortConnect=False"

# Display results
echo ""
print_info "=========================================="
print_info "Azure Redis Setup Complete!"
print_info "=========================================="
echo ""
echo "Cache Name:        $REDIS_NAME"
echo "Resource Group:    $RG_NAME"
echo "Location:          $LOCATION"
echo "SKU:               $SKU"
echo "Capacity:          $CAPACITY"
echo ""
echo "Host:              $REDIS_HOST"
echo "Port:              $REDIS_PORT"
echo "SSL Port:          $REDIS_SSL_PORT"
echo ""
print_warning "Primary Key (save this securely):"
echo "$REDIS_PRIMARY_KEY"
echo ""
if [ -n "$REDIS_SECONDARY_KEY" ]; then
    print_warning "Secondary Key:"
    echo "$REDIS_SECONDARY_KEY"
    echo ""
fi
print_info "Connection String:"
echo "$REDIS_CONNECTION_STRING"
echo ""
print_info "Add these to your .env file:"
echo "REDIS_HOST=$REDIS_HOST"
echo "REDIS_PORT=$REDIS_SSL_PORT"
echo "REDIS_PASSWORD=$REDIS_PRIMARY_KEY"
echo "REDIS_SSL=true"
echo "REDIS_CONNECTION_STRING=$REDIS_CONNECTION_STRING"
echo ""
print_info "=========================================="

