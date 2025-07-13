#!/usr/bin/env python3
"""
Direct test of PDF generation functionality
"""

import subprocess
import sys
from pathlib import Path

def test_pdf_generation():
    print("Testing PDF generation...")
    
    # Test simple PDF generator
    print("\n1. Testing simple PDF generator:")
    try:
        result = subprocess.run([
            sys.executable, "simple_pdf_generator.py"
        ], cwd="DATA_ANALYSIS", capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("[SUCCESS] Simple PDF generator succeeded")
            print(f"Output: {result.stdout}")
        else:
            print("[ERROR] Simple PDF generator failed")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Simple PDF generator timed out")
    except Exception as e:
        print(f"[ERROR] Error running simple PDF generator: {e}")
    
    # Check if PDF was created
    pdf_dir = Path("DATA_ANALYSIS/pdf_reports")
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"\n📁 Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            size = pdf_file.stat().st_size
            print(f"   {pdf_file.name} ({size} bytes)")
    else:
        print("\n[ERROR] PDF reports directory not found")

if __name__ == "__main__":
    test_pdf_generation()
