#!/usr/bin/env python3
"""
Test directory creation for PDF generator
"""

import sys
import os
from pathlib import Path

def test_directory_creation():
    print("Testing directory creation...")
    print(f"Current working directory: {Path.cwd()}")
    
    # Test the path resolution
    analysis_dir = "DATA_ANALYSIS"
    if not os.path.isabs(analysis_dir):
        full_path = Path.cwd() / analysis_dir
    else:
        full_path = Path(analysis_dir)
    
    print(f"Analysis directory path: {full_path}")
    
    # Create directories
    pdf_reports_dir = full_path / "pdf_reports"
    print(f"PDF reports directory: {pdf_reports_dir}")
    
    try:
        # Create directories
        full_path.mkdir(exist_ok=True)
        pdf_reports_dir.mkdir(exist_ok=True)
        
        print("✅ Directories created successfully")
        
        # Verify they exist
        if pdf_reports_dir.exists():
            print("✅ PDF reports directory exists")
        else:
            print("❌ PDF reports directory not found")
            
    except Exception as e:
        print(f"❌ Error creating directories: {e}")

if __name__ == "__main__":
    test_directory_creation()
