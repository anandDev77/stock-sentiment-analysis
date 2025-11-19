#!/usr/bin/env python3
"""
Generate high-quality PNG images from Mermaid diagrams.

This script extracts all Mermaid diagram code blocks from mermaid_diagrams_source.md
and generates PNG images using mermaid-cli (mmdc) with proper color configuration.
"""

import re
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DIAGRAMS_SOURCE = PROJECT_ROOT / "docs" / "diagrams" / "mermaid_diagrams_source.md"
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "diagrams"
MERMAID_CONFIG = PROJECT_ROOT / "docs" / "diagrams" / "mermaid_config.json"


def extract_diagrams() -> List[Tuple[int, str, str]]:
    """
    Extract all Mermaid diagrams from the source file.
    
    Returns:
        List of tuples: (diagram_number, context, mermaid_code)
    """
    if not DIAGRAMS_SOURCE.exists():
        print(f"❌ Error: Source file not found: {DIAGRAMS_SOURCE}")
        sys.exit(1)
    
    with open(DIAGRAMS_SOURCE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all diagram blocks
    pattern = r'## Diagram (\d+)\s*\n\n\*\*Location:\*\*.*?\n\n\*\*Context:\*\* (.*?)\n\n```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    diagrams = []
    for match in matches:
        diagram_num = int(match[0])
        context = match[1].strip()
        mermaid_code = match[2].strip()
        diagrams.append((diagram_num, context, mermaid_code))
    
    print(f"✅ Found {len(diagrams)} diagrams in source file")
    return diagrams


def sanitize_filename(context: str, diagram_num: int) -> str:
    """
    Create a sanitized filename from context.
    
    Args:
        context: Diagram context/description
        diagram_num: Diagram number
        
    Returns:
        Sanitized filename
    """
    # Remove special characters and limit length
    filename = re.sub(r'[^\w\s-]', '', context)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename.lower()[:100]  # Limit length
    filename = f"{diagram_num:02d}_{filename}"
    return filename


def check_mmdc_available() -> bool:
    """Check if mermaid-cli (mmdc) is available."""
    try:
        result = subprocess.run(
            ['mmdc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def generate_diagram(diagram_num: int, context: str, mermaid_code: str) -> bool:
    """
    Generate a PNG image from Mermaid code.
    
    Args:
        diagram_num: Diagram number
        context: Diagram context
        mermaid_code: Mermaid diagram code
        
    Returns:
        True if successful, False otherwise
    """
    filename = sanitize_filename(context, diagram_num)
    output_file = DIAGRAMS_DIR / f"{filename}.png"
    
    # Create temporary mermaid file in /tmp to avoid permission issues
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as temp_file:
        temp_mmd = Path(temp_file.name)
        temp_file.write(mermaid_code)
    
    try:
        # Generate PNG using mmdc
        cmd = [
            'mmdc',
            '-i', str(temp_mmd),
            '-o', str(output_file),
            '-c', str(MERMAID_CONFIG),
            '-w', '2400',  # Width in pixels
            '-H', '1800',  # Height in pixels
            '-b', 'white',  # Background color
            '-s', '2'  # Scale factor for high quality
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            if output_file.exists():
                size_kb = output_file.stat().st_size / 1024
                print(f"  ✅ Diagram {diagram_num:2d}: {output_file.name} ({size_kb:.1f} KB)")
                return True
            else:
                print(f"  ❌ Diagram {diagram_num:2d}: Output file not created")
                if result.stdout:
                    print(f"     Output: {result.stdout[:200]}")
                return False
        else:
            print(f"  ❌ Diagram {diagram_num:2d}: mmdc error (code: {result.returncode})")
            if result.stderr:
                error_msg = result.stderr.strip()[:300]
                print(f"     Error: {error_msg}")
            if result.stdout:
                print(f"     Output: {result.stdout[:200]}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"  ❌ Diagram {diagram_num:2d}: Timeout (>60s)")
        return False
    except Exception as e:
        print(f"  ❌ Diagram {diagram_num:2d}: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        if temp_mmd.exists():
            temp_mmd.unlink()


def main():
    """Main function to generate all diagrams."""
    print("=" * 80)
    print("Mermaid Diagram Generator")
    print("=" * 80)
    print()
    
    # Check if mmdc is available
    print("🔍 Checking for mermaid-cli (mmdc)...")
    if not check_mmdc_available():
        print("❌ Error: mermaid-cli (mmdc) is not installed or not in PATH")
        print()
        print("Installation options:")
        print("  1. npm install -g @mermaid-js/mermaid-cli")
        print("  2. Or use Docker: docker pull minlag/mermaid-cli")
        print()
        print("For Docker usage, modify this script to use:")
        print("  docker run --rm -v $(pwd):/data minlag/mermaid-cli ...")
        sys.exit(1)
    
    print("✅ mermaid-cli found")
    print()
    
    # Check if config file exists
    if not MERMAID_CONFIG.exists():
        print(f"⚠️  Warning: Config file not found: {MERMAID_CONFIG}")
        print("   Will use default mermaid-cli settings")
        print()
    
    # Ensure output directory exists
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract diagrams
    print("📖 Extracting diagrams from source file...")
    diagrams = extract_diagrams()
    print()
    
    # Generate diagrams
    print("🎨 Generating PNG images...")
    print()
    
    success_count = 0
    failed_count = 0
    
    for diagram_num, context, mermaid_code in diagrams:
        if generate_diagram(diagram_num, context, mermaid_code):
            success_count += 1
        else:
            failed_count += 1
    
    print()
    print("=" * 80)
    print("Generation Summary")
    print("=" * 80)
    print(f"✅ Successful: {success_count}/{len(diagrams)}")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count}/{len(diagrams)}")
    print()
    print(f"📁 Output directory: {DIAGRAMS_DIR}")
    print()
    
    if failed_count == 0:
        print("🎉 All diagrams generated successfully!")
        return 0
    else:
        print("⚠️  Some diagrams failed to generate. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

