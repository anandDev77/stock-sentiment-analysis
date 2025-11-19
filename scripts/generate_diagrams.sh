#!/bin/bash
# Alternative script using Docker if mmdc is not installed locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIAGRAMS_DIR="$PROJECT_ROOT/docs/diagrams"
DIAGRAMS_SOURCE="$DIAGRAMS_DIR/mermaid_diagrams_source.md"
MERMAID_CONFIG="$DIAGRAMS_DIR/mermaid_config.json"

echo "============================================================"
echo "Mermaid Diagram Generator (Docker Version)"
echo "============================================================"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH"
    echo "   Please install Docker or use the Python script with local mmdc"
    exit 1
fi

echo "✅ Docker found"
echo ""

# Check if source file exists
if [ ! -f "$DIAGRAMS_SOURCE" ]; then
    echo "❌ Error: Source file not found: $DIAGRAMS_SOURCE"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$DIAGRAMS_DIR"

# Extract diagram count
DIAGRAM_COUNT=$(grep -c "^## Diagram" "$DIAGRAMS_SOURCE" || echo "0")
echo "📖 Found $DIAGRAM_COUNT diagrams in source file"
echo ""

# Extract and generate each diagram
echo "🎨 Generating PNG images using Docker..."
echo ""

SUCCESS=0
FAILED=0

# Extract diagrams using Python (simpler than bash regex)
python3 << 'PYTHON_SCRIPT' | while IFS= read -r line; do
import re
import sys

with open('$DIAGRAMS_SOURCE', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'## Diagram (\d+)\s*\n\n\*\*Location:\*\*.*?\n\n\*\*Context:\*\* (.*?)\n\n```mermaid\n(.*?)```'
matches = re.findall(pattern, content, re.DOTALL)

for match in matches:
    diagram_num = match[0]
    context = match[1].strip()
    mermaid_code = match[2].strip()
    
    # Sanitize filename
    filename = re.sub(r'[^\w\s-]', '', context)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename.lower()[:100]
    filename = f"{int(diagram_num):02d}_{filename}"
    
    print(f"{diagram_num}|{filename}|{context}")
PYTHON_SCRIPT

# Process each diagram
while IFS='|' read -r diagram_num filename context; do
    if [ -z "$diagram_num" ]; then
        continue
    fi
    
    echo "  Processing Diagram $diagram_num: $context"
    
    # Create temporary mermaid file
    TEMP_MMD=$(mktemp /tmp/diagram_XXXXXX.mmd)
    TEMP_PNG=$(mktemp /tmp/diagram_XXXXXX.png)
    
    # Extract mermaid code (simplified - would need full Python extraction)
    # For now, use Python to extract and save
    python3 << EXTRACT_PYTHON > "$TEMP_MMD"
import re
import sys

with open('$DIAGRAMS_SOURCE', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'## Diagram $diagram_num\s*\n\n\*\*Location:\*\*.*?\n\n\*\*Context:\*\*.*?\n\n```mermaid\n(.*?)```'
match = re.search(pattern, content, re.DOTALL)
if match:
    print(match.group(1).strip())
EXTRACT_PYTHON
    
    # Generate using Docker
    if docker run --rm \
        -v "$PROJECT_ROOT:/data" \
        -v "$MERMAID_CONFIG:/data/mermaid_config.json:ro" \
        minlag/mermaid-cli \
        -i "/data/docs/diagrams/$(basename $TEMP_MMD)" \
        -o "/data/docs/diagrams/${filename}.png" \
        -c "/data/mermaid_config.json" \
        -w 2400 \
        -H 1800 \
        -b white \
        -s 2 2>&1; then
        
        if [ -f "$DIAGRAMS_DIR/${filename}.png" ]; then
            SIZE=$(du -h "$DIAGRAMS_DIR/${filename}.png" | cut -f1)
            echo "    ✅ Generated: ${filename}.png ($SIZE)"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "    ❌ Output file not created"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "    ❌ Generation failed"
        FAILED=$((FAILED + 1))
    fi
    
    # Cleanup
    rm -f "$TEMP_MMD" "$TEMP_PNG"
done

echo ""
echo "============================================================"
echo "Generation Summary"
echo "============================================================"
echo "✅ Successful: $SUCCESS"
echo "❌ Failed: $FAILED"
echo ""
echo "📁 Output directory: $DIAGRAMS_DIR"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All diagrams generated successfully!"
    exit 0
else
    echo "⚠️  Some diagrams failed to generate"
    exit 1
fi

