#!/bin/bash

# =============================================================================
# Mermaid Diagram Generator Script
# =============================================================================
# This script extracts Mermaid diagrams from ARCHITECTURE.md, generates images,
# and embeds them back into the markdown file for Word document conversion.
#
# Requirements:
#   - mermaid-cli (npm install -g @mermaid-js/mermaid-cli)
#   - Node.js and npm
#
# Usage:
#   ./generate_diagrams.sh [options]
#
# Options:
#   --format png|svg    Image format (default: png)
#   --width WIDTH       Image width in pixels (default: 1200)
#   --backup            Create backup of original file (default: true)
#   --clean             Clean generated diagrams before running
# =============================================================================

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MD_FILE="ARCHITECTURE.md"
DIAGRAMS_DIR="diagrams"
BACKUP_FILE="${MD_FILE}.backup"
TEMP_DIR=$(mktemp -d)
FORMAT="png"
WIDTH="1200"
CREATE_BACKUP=true
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
        --no-backup)
            CREATE_BACKUP=false
            shift
            ;;
        --clean)
            CLEAN_BEFORE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --format png|svg    Image format (default: png)"
            echo "  --width WIDTH       Image width (default: 1200)"
            echo "  --no-backup         Don't create backup file"
            echo "  --clean             Clean diagrams directory before running"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if mermaid-cli is installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v mmdc &> /dev/null; then
        log_error "mermaid-cli (mmdc) is not installed!"
        echo ""
        echo "Please install it using:"
        echo "  npm install -g @mermaid-js/mermaid-cli"
        echo ""
        echo "Or if you prefer npx (no installation needed):"
        echo "  npx -p @mermaid-js/mermaid-cli mmdc --version"
        exit 1
    fi
    
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed!"
        echo "Please install Node.js from https://nodejs.org/"
        exit 1
    fi
    
    log_success "All dependencies are installed"
}

# Clean diagrams directory
clean_diagrams() {
    if [ "$CLEAN_BEFORE" = true ]; then
        log_info "Cleaning diagrams directory..."
        if [ -d "$DIAGRAMS_DIR" ]; then
            rm -rf "$DIAGRAMS_DIR"/*
            log_success "Cleaned diagrams directory"
        fi
    fi
}

# Create directories
setup_directories() {
    log_info "Setting up directories..."
    mkdir -p "$DIAGRAMS_DIR"
    log_success "Directories created"
}

# Create backup
create_backup() {
    if [ "$CREATE_BACKUP" = true ] && [ -f "$MD_FILE" ]; then
        log_info "Creating backup: $BACKUP_FILE"
        cp "$MD_FILE" "$BACKUP_FILE"
        log_success "Backup created"
    fi
}

# Extract mermaid diagrams from markdown
extract_diagrams() {
    log_info "Extracting Mermaid diagrams from $MD_FILE..."
    
    local diagram_count=0
    local in_mermaid=false
    local diagram_content=""
    local diagram_num=0
    local output_file=""
    
    # Read the markdown file line by line
    while IFS= read -r line || [ -n "$line" ]; do
        if [[ "$line" =~ ^\`\`\`mermaid ]]; then
            in_mermaid=true
            diagram_content=""
            ((diagram_num++))
            output_file="${TEMP_DIR}/diagram_${diagram_num}.mmd"
            log_info "Found diagram #${diagram_num}"
        elif [[ "$in_mermaid" = true ]] && [[ "$line" =~ ^\`\`\` ]]; then
            in_mermaid=false
            echo "$diagram_content" > "$output_file"
            ((diagram_count++))
            log_success "Extracted diagram #${diagram_num}"
        elif [[ "$in_mermaid" = true ]]; then
            diagram_content+="$line"$'\n'
        fi
    done < "$MD_FILE"
    
    log_success "Extracted $diagram_count diagrams"
    echo "$diagram_count"
}

# Generate images from mermaid files
generate_images() {
    log_info "Generating $FORMAT images from Mermaid diagrams..."
    
    local count=0
    local diagram_files=("$TEMP_DIR"/diagram_*.mmd)
    
    for mmd_file in "${diagram_files[@]}"; do
        if [ -f "$mmd_file" ]; then
            ((count++))
            local basename=$(basename "$mmd_file" .mmd)
            local output_file="${DIAGRAMS_DIR}/${basename}.${FORMAT}"
            
            log_info "Generating image for diagram #${count}..."
            
            # Use mmdc to generate image
            if mmdc -i "$mmd_file" -o "$output_file" -w "$WIDTH" -b transparent 2>/dev/null; then
                log_success "Generated: $output_file"
            else
                log_warning "Failed to generate image for $mmd_file, trying with npx..."
                # Fallback to npx if mmdc not in PATH
                if npx -p @mermaid-js/mermaid-cli mmdc -i "$mmd_file" -o "$output_file" -w "$WIDTH" -b transparent; then
                    log_success "Generated: $output_file (via npx)"
                else
                    log_error "Failed to generate image for $mmd_file"
                    return 1
                fi
            fi
        fi
    done
    
    log_success "Generated $count images"
}

# Replace mermaid blocks with image references
embed_images() {
    log_info "Embedding images into $MD_FILE..."
    
    local temp_output="${TEMP_DIR}/output.md"
    local diagram_num=0
    local in_mermaid=false
    local diagram_start_line=0
    local line_num=0
    
    # Read original file and create new version
    while IFS= read -r line || [ -n "$line" ]; do
        ((line_num++))
        
        if [[ "$line" =~ ^\`\`\`mermaid ]]; then
            in_mermaid=true
            diagram_start_line=$line_num
            ((diagram_num++))
            local image_path="${DIAGRAMS_DIR}/diagram_${diagram_num}.${FORMAT}"
            
            # Add image reference instead of mermaid block
            echo "![Diagram ${diagram_num}](${image_path})" >> "$temp_output"
            echo "" >> "$temp_output"
        elif [[ "$in_mermaid" = true ]] && [[ "$line" =~ ^\`\`\` ]]; then
            in_mermaid=false
            # Skip the closing ```
        elif [[ "$in_mermaid" = false ]]; then
            echo "$line" >> "$temp_output"
        fi
    done < "$MD_FILE"
    
    # Replace original file
    mv "$temp_output" "$MD_FILE"
    log_success "Images embedded into $MD_FILE"
}

# Cleanup temporary files (but preserve generated diagrams)
cleanup() {
    log_info "Cleaning up temporary files..."
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
        log_success "Cleaned up temporary files"
    fi
    # DO NOT remove DIAGRAMS_DIR - those are the output files we want to keep!
}

# Main execution
main() {
    echo "============================================================================="
    echo "Mermaid Diagram Generator"
    echo "============================================================================="
    echo ""
    
    # Check if markdown file exists
    if [ ! -f "$MD_FILE" ]; then
        log_error "File not found: $MD_FILE"
        exit 1
    fi
    
    # Run steps
    check_dependencies
    clean_diagrams
    setup_directories
    create_backup
    
    local diagram_count=$(extract_diagrams)
    
    if [ "$diagram_count" -eq 0 ]; then
        log_warning "No Mermaid diagrams found in $MD_FILE"
        log_info "Checking if images already exist in $DIAGRAMS_DIR..."
        if [ -d "$DIAGRAMS_DIR" ] && [ "$(ls -A $DIAGRAMS_DIR 2>/dev/null)" ]; then
            log_success "Found existing diagrams in $DIAGRAMS_DIR"
            log_info "Skipping generation. Images are already embedded in the markdown file."
        else
            log_warning "No diagrams found and no existing images. Nothing to do."
        fi
        cleanup
        exit 0
    fi
    
    generate_images
    
    # Verify images were actually created before embedding
    local image_count=$(ls -1 "${DIAGRAMS_DIR}"/diagram_*.${FORMAT} 2>/dev/null | wc -l | tr -d ' ')
    if [ "$image_count" -eq 0 ]; then
        log_error "No images were generated! Check mermaid-cli installation and diagram syntax."
        cleanup
        exit 1
    fi
    
    log_info "Verified $image_count images were generated successfully"
    embed_images
    cleanup
    
    echo ""
    echo "============================================================================="
    log_success "All diagrams generated and embedded successfully!"
    echo "============================================================================="
    echo ""
    echo "Generated files:"
    echo "  - Diagrams: $DIAGRAMS_DIR/"
    if [ "$CREATE_BACKUP" = true ]; then
        echo "  - Backup: $BACKUP_FILE"
    fi
    echo "  - Updated: $MD_FILE"
    echo ""
    echo "You can now convert to Word using:"
    echo "  pandoc $MD_FILE -o ARCHITECTURE.docx"
    echo ""
}

# Run main function
main

