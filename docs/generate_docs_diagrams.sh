#!/bin/bash

# =============================================================================
# Documentation Diagram Generator Script
# =============================================================================
# This script extracts Mermaid diagrams from index.md, generates images,
# and embeds them back into the markdown file for GitHub Pages.
#
# Requirements:
#   - mermaid-cli (npm install -g @mermaid-js/mermaid-cli)
#   - Node.js and npm
#
# Usage:
#   ./generate_docs_diagrams.sh [options]
#
# Options:
#   --format png|svg    Image format (default: png)
#   --width WIDTH       Image width in pixels (default: 1200)
#   --clean             Clean generated diagrams before running
# =============================================================================

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MD_FILE="index.md"
DIAGRAMS_DIR="diagrams"
TEMP_DIR=$(mktemp -d)
FORMAT="png"
WIDTH="1200"
CLEAN_BEFORE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BEFORE=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v mmdc &> /dev/null; then
        print_error "mermaid-cli (mmdc) not found!"
        print_info "Install it with: npm install -g @mermaid-js/mermaid-cli"
        exit 1
    fi
    
    if ! command -v node &> /dev/null; then
        print_error "Node.js not found!"
        exit 1
    fi
    
    print_success "All dependencies found"
}

# Clean old diagrams
clean_diagrams() {
    if [ "$CLEAN_BEFORE" = true ]; then
        print_info "Cleaning old diagrams..."
        rm -f "$DIAGRAMS_DIR"/architecture.png
        rm -f "$DIAGRAMS_DIR"/components.png
        rm -f "$DIAGRAMS_DIR"/dataflow.png
        rm -f "$DIAGRAMS_DIR"/rag_flow.png
        rm -f "$DIAGRAMS_DIR"/caching.png
        print_success "Cleaned old diagrams"
    fi
}

# Create diagrams directory
create_diagrams_dir() {
    if [ ! -d "$DIAGRAMS_DIR" ]; then
        mkdir -p "$DIAGRAMS_DIR"
        print_info "Created diagrams directory"
    fi
}

# Extract and generate diagrams
generate_diagrams() {
    print_info "Extracting Mermaid diagrams from $MD_FILE..."
    
    # Split diagrams into separate files and count them
    diagram_count=0
    awk -v temp_dir="$TEMP_DIR" '
    BEGIN {
        in_mermaid = 0
        diagram_num = 0
        current_file = ""
    }
    /^```mermaid/ {
        if (in_mermaid) {
            close(current_file)
        }
        in_mermaid = 1
        diagram_num++
        current_file = temp_dir "/diagram_" diagram_num ".mmd"
        next
    }
    /^```$/ {
        if (in_mermaid) {
            in_mermaid = 0
            close(current_file)
        }
        next
    }
    in_mermaid {
        print > current_file
    }
    END {
        if (in_mermaid) {
            close(current_file)
        }
        print diagram_num
    }
    ' "$MD_FILE" > "$TEMP_DIR/diagram_count.txt"
    
    # Read diagram count
    if [ -f "$TEMP_DIR/diagram_count.txt" ]; then
        diagram_count=$(cat "$TEMP_DIR/diagram_count.txt" | head -1)
    else
        diagram_count=0
    fi
    
    if [ $diagram_count -eq 0 ]; then
        print_warning "No Mermaid diagrams found in $MD_FILE"
        return
    fi
    
    print_info "Found $diagram_count diagram(s)"
    
    # Generate images for each diagram
    for i in $(seq 1 $diagram_count); do
        diagram_file="$TEMP_DIR/diagram_$i.mmd"
        
        if [ ! -f "$diagram_file" ]; then
            print_warning "Diagram file $i not found, skipping"
            continue
        fi
        
        # Determine output name based on diagram content
        diagram_content=$(cat "$diagram_file")
        output_name=""
        
        if echo "$diagram_content" | grep -qi "User Interface Layer\|Application Layer\|AI/ML Layer\|Data Layer\|External Services"; then
            output_name="architecture"
        elif echo "$diagram_content" | grep -qi "Component\|graph LR.*Frontend.*Services"; then
            output_name="components"
        elif echo "$diagram_content" | grep -qi "sequenceDiagram\|Data Flow"; then
            output_name="dataflow"
        elif echo "$diagram_content" | grep -qi "RAG\|Retrieval.*Generation\|Ingestion.*Retrieval"; then
            output_name="rag_flow"
        elif echo "$diagram_content" | grep -qi "Cache\|L1.*L2.*L3"; then
            output_name="caching"
        else
            output_name="diagram_$i"
        fi
        
        output_file="$DIAGRAMS_DIR/${output_name}.${FORMAT}"
        
        print_info "Generating diagram $i: $output_name"
        
        # Generate diagram
        if mmdc -i "$diagram_file" -o "$output_file" -w "$WIDTH" -b transparent 2>&1 | grep -v "Deprecated" | grep -v "^$"; then
            if [ -f "$output_file" ]; then
                print_success "Generated: $output_file"
            else
                print_error "Failed to generate: $output_file (file not created)"
            fi
        else
            if [ -f "$output_file" ]; then
                print_success "Generated: $output_file"
            else
                print_error "Failed to generate: $output_file"
            fi
        fi
    done
}

# Embed diagrams back into markdown
embed_diagrams() {
    print_info "Embedding diagrams into $MD_FILE..."
    
    # Create a backup
    cp "$MD_FILE" "${MD_FILE}.backup"
    
    # Replace diagram references
    # Pattern: ![Diagram Name](diagrams/diagram_X.png)
    # Replace with: ![Diagram Name](diagrams/actual_name.png)
    
    # Architecture diagram
    sed -i.tmp 's|!\[.*[Aa]rchitecture.*\](diagrams/diagram_[0-9]*\.png)|![System Architecture](diagrams/architecture.png)|g' "$MD_FILE"
    
    # Component diagram
    sed -i.tmp 's|!\[.*[Cc]omponent.*\](diagrams/diagram_[0-9]*\.png)|![Component Interaction](diagrams/components.png)|g' "$MD_FILE"
    
    # Data flow diagram
    sed -i.tmp 's|!\[.*[Dd]ata [Ff]low.*\](diagrams/diagram_[0-9]*\.png)|![Data Flow](diagrams/dataflow.png)|g' "$MD_FILE"
    
    # RAG flow diagram
    sed -i.tmp 's|!\[.*RAG.*\](diagrams/diagram_[0-9]*\.png)|![RAG Flow](diagrams/rag_flow.png)|g' "$MD_FILE"
    
    # Caching diagram
    sed -i.tmp 's|!\[.*[Cc]aching.*\](diagrams/diagram_[0-9]*\.png)|![Caching Strategy](diagrams/caching.png)|g' "$MD_FILE"
    
    # Remove temp file
    rm -f "${MD_FILE}.tmp"
    
    print_success "Diagrams embedded into $MD_FILE"
}

# Verify diagrams exist
verify_diagrams() {
    print_info "Verifying generated diagrams..."
    
    required_diagrams=("architecture" "components" "dataflow" "rag_flow" "caching")
    missing=0
    
    for diagram in "${required_diagrams[@]}"; do
        if [ -f "$DIAGRAMS_DIR/${diagram}.${FORMAT}" ]; then
            print_success "Found: ${diagram}.${FORMAT}"
        else
            print_warning "Missing: ${diagram}.${FORMAT}"
            missing=$((missing + 1))
        fi
    done
    
    if [ $missing -eq 0 ]; then
        print_success "All diagrams generated successfully"
    else
        print_warning "$missing diagram(s) missing (this is OK if not all diagrams are used)"
    fi
}

# Cleanup
cleanup() {
    print_info "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    print_success "Cleanup complete"
}

# Main execution
main() {
    print_info "Starting diagram generation for documentation..."
    
    check_dependencies
    clean_diagrams
    create_diagrams_dir
    generate_diagrams
    embed_diagrams
    verify_diagrams
    cleanup
    
    print_success "Documentation diagram generation complete!"
    print_info "Diagrams are in: $DIAGRAMS_DIR/"
    print_info "Updated file: $MD_FILE"
}

# Run main
main

