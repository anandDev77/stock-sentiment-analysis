#!/bin/bash

# Azure AI Search Setup Script
# This script creates an Azure AI Search service instance in the specified resource group
# Usage: ./setup-azure-ai-search.sh <RESOURCE_GROUP_NAME> [OPTIONS]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values (can be overridden by environment variables)
SEARCH_NAME="${AZURE_AI_SEARCH_SERVICE_NAME:-stock-sentiment-search}"
SKU="${AZURE_AI_SEARCH_SKU:-free}"
REPLICA_COUNT="${AZURE_AI_SEARCH_REPLICA_COUNT:-1}"
PARTITION_COUNT="${AZURE_AI_SEARCH_PARTITION_COUNT:-1}"
INDEX_NAME="${AZURE_AI_SEARCH_INDEX_NAME:-stock-articles}"
LOCATION=""

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
    echo "  --search-name NAME         Search service name (default: $SEARCH_NAME)"
    echo "                             Must be globally unique, lowercase, alphanumeric and hyphens only"
    echo "  --location LOCATION        Azure region (default: resource group location)"
    echo "  --sku SKU                  SKU tier: free, basic, standard, standard2, standard3 (default: $SKU)"
    echo "                             Note: 'free' tier has limitations (50MB, 3 indexes, 3 indexers)"
    echo "  --replica-count COUNT      Number of replicas (default: $REPLICA_COUNT)"
    echo "                             Free tier: 1 replica only"
    echo "  --partition-count COUNT    Number of partitions (default: $PARTITION_COUNT)"
    echo "                             Free tier: 1 partition only"
    echo "  --index-name NAME          Index name for vector search (default: $INDEX_NAME)"
    echo "                             Note: Index will be created automatically by the app"
    echo "  --owner-email EMAIL        Owner email for tagging"
    echo "  --help                     Show this help message"
    echo ""
    echo "SKU Details:"
    echo "  free:       50MB storage, 3 indexes, 3 indexers, 1 replica, 1 partition (suitable for dev/test)"
    echo "  basic:      2GB storage, 15 indexes, 15 indexers, 3 replicas, 1 partition"
    echo "  standard:   25GB storage, 50 indexes, 50 indexers, 12 replicas, 12 partitions"
    echo "  standard2:  100GB storage, 200 indexes, 200 indexers, 12 replicas, 12 partitions"
    echo "  standard3:  200GB storage, 200 indexes, 200 indexers, 12 replicas, 12 partitions"
    echo ""
    echo "Example:"
    echo "  $0 my-resource-group --location eastus --sku free"
    echo "  $0 my-resource-group --location eastus --sku basic --replica-count 1"
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
        --search-name)
            SEARCH_NAME="$2"
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
        --replica-count)
            REPLICA_COUNT="$2"
            shift 2
            ;;
        --partition-count)
            PARTITION_COUNT="$2"
            shift 2
            ;;
        --index-name)
            INDEX_NAME="$2"
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
if [[ ! "$SKU" =~ ^(free|basic|standard|standard2|standard3|storage_optimized_l1|storage_optimized_l2)$ ]]; then
    print_error "Invalid SKU: $SKU. Must be one of: free, basic, standard, standard2, standard3, storage_optimized_l1, storage_optimized_l2"
    exit 1
fi

# Validate search name (must be lowercase, alphanumeric and hyphens only, 2-60 chars)
if [[ ! "$SEARCH_NAME" =~ ^[a-z0-9-]{2,60}$ ]]; then
    print_error "Invalid search service name: $SEARCH_NAME"
    print_error "Must be 2-60 characters, lowercase, alphanumeric and hyphens only"
    exit 1
fi

# Free tier limitations
if [ "$SKU" = "free" ]; then
    if [ "$REPLICA_COUNT" -gt 1 ]; then
        print_warning "Free tier supports only 1 replica. Setting replica-count to 1."
        REPLICA_COUNT=1
    fi
    if [ "$PARTITION_COUNT" -gt 1 ]; then
        print_warning "Free tier supports only 1 partition. Setting partition-count to 1."
        PARTITION_COUNT=1
    fi
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

# Check if Search service already exists
print_info "Checking if Search service '$SEARCH_NAME' already exists..."
if az search service show --name "$SEARCH_NAME" --resource-group "$RG_NAME" &> /dev/null; then
    print_warning "Search service '$SEARCH_NAME' already exists in resource group '$RG_NAME'"
    read -p "Do you want to continue and retrieve connection info? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Exiting..."
        exit 0
    fi
else
    # Create Search service
    print_info "Creating Azure AI Search service '$SEARCH_NAME'..."
    print_warning "This may take 5-10 minutes..."
    
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
    
    # Create Search service
    az search service create \
        --name "$SEARCH_NAME" \
        --resource-group "$RG_NAME" \
        --sku "$SKU" \
        --location "$LOCATION" \
        --replica-count "$REPLICA_COUNT" \
        --partition-count "$PARTITION_COUNT" \
        --tags $TAGS
    
    if [ $? -eq 0 ]; then
        print_info "Search service created successfully!"
    else
        print_error "Failed to create Search service"
        exit 1
    fi
    
    # Check provisioning state
    print_info "Checking Search service provisioning state..."
    PROVISIONING_STATE=$(az search service show --name "$SEARCH_NAME" --resource-group "$RG_NAME" --query provisioningState -o tsv)
    
    if [ "$PROVISIONING_STATE" != "Succeeded" ]; then
        print_warning "Search service is still provisioning. State: $PROVISIONING_STATE"
        print_info "You can check status with: az search service show --name $SEARCH_NAME --resource-group $RG_NAME"
        print_info "Continuing to retrieve connection info..."
    else
        print_info "Search service is ready!"
    fi
fi

# Get Search service connection details
print_info "Retrieving Search service connection information..."

SEARCH_ENDPOINT=$(az search service show \
    --name "$SEARCH_NAME" \
    --resource-group "$RG_NAME" \
    --query "properties.endpoint" -o tsv)

# Get admin keys
print_info "Retrieving Search service admin keys..."
ADMIN_KEY_PRIMARY=$(az search admin-key show \
    --service-name "$SEARCH_NAME" \
    --resource-group "$RG_NAME" \
    --query "primaryKey" -o tsv)

ADMIN_KEY_SECONDARY=$(az search admin-key show \
    --service-name "$SEARCH_NAME" \
    --resource-group "$RG_NAME" \
    --query "secondaryKey" -o tsv)

# Get query keys (read-only keys)
print_info "Retrieving Search service query keys..."
QUERY_KEY_PRIMARY=$(az search query-key list \
    --service-name "$SEARCH_NAME" \
    --resource-group "$RG_NAME" \
    --query "[0].key" -o tsv 2>/dev/null || echo "")

# Get service status
SERVICE_STATUS=$(az search service show \
    --name "$SEARCH_NAME" \
    --resource-group "$RG_NAME" \
    --query "properties.status" -o tsv)

# Display results
echo ""
print_info "=========================================="
print_info "Azure AI Search Setup Complete!"
print_info "=========================================="
echo ""
echo "Service Name:      $SEARCH_NAME"
echo "Resource Group:    $RG_NAME"
echo "Location:          $LOCATION"
echo "SKU:               $SKU"
echo "Replicas:          $REPLICA_COUNT"
echo "Partitions:        $PARTITION_COUNT"
echo "Status:            $SERVICE_STATUS"
echo "Index Name:        $INDEX_NAME (will be created automatically by app)"
echo ""
echo "Endpoint:          $SEARCH_ENDPOINT"
echo ""
print_warning "Admin Key (Primary) - save this securely:"
echo "$ADMIN_KEY_PRIMARY"
echo ""
if [ -n "$ADMIN_KEY_SECONDARY" ] && [ "$ADMIN_KEY_SECONDARY" != "null" ]; then
    print_warning "Admin Key (Secondary):"
    echo "$ADMIN_KEY_SECONDARY"
    echo ""
fi
if [ -n "$QUERY_KEY_PRIMARY" ] && [ "$QUERY_KEY_PRIMARY" != "null" ]; then
    print_info "Query Key (Primary) - read-only, safer for client apps:"
    echo "$QUERY_KEY_PRIMARY"
    echo ""
fi
print_info "Add these to your .env file:"
echo "AZURE_AI_SEARCH_ENDPOINT=$SEARCH_ENDPOINT"
echo "AZURE_AI_SEARCH_API_KEY=$ADMIN_KEY_PRIMARY"
echo "AZURE_AI_SEARCH_INDEX_NAME=$INDEX_NAME"
echo ""
if [ "$SKU" = "free" ]; then
    print_warning "Note: Free tier limitations:"
    echo "  - 50MB storage"
    echo "  - 3 indexes maximum"
    echo "  - 3 indexers maximum"
    echo "  - 1 replica, 1 partition"
    echo "  - Suitable for development and testing"
    echo ""
fi
print_info "The index '$INDEX_NAME' will be created automatically"
print_info "when you first run the application with Azure AI Search configured."
echo ""
print_info "=========================================="

