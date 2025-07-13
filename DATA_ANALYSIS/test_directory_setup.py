#!/usr/bin/env python3
"""
Simple test to verify directory creation and basic PDF functionality
"""

import os
import sys
from pathlib import Path

def test_directory_creation():
    print("Testing directory creation...")
    
    # Test from DATA_ANALYSIS directory (current context)
    print(f"Current working directory: {os.getcwd()}")
    
    # Create pdf_reports directory
    pdf_dir = Path("pdf_reports")
    pdf_dir.mkdir(exist_ok=True)
    
    if pdf_dir.exists():
        print(f"✅ pdf_reports directory created: {pdf_dir.absolute()}")
    else:
        print("❌ Failed to create pdf_reports directory")
        return False
    
    # Test creating a dummy file
    test_file = pdf_dir / "test.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("Test file")
        
        if test_file.exists():
            print(f"✅ Test file created successfully: {test_file}")
            # Clean up
            test_file.unlink()
            print("✅ Test file cleaned up")
        else:
            print("❌ Test file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error creating test file: {e}")
        return False
    
    return True

def test_pdf_generator_import():
    print("\nTesting PDF generator import...")
    
    try:
        from pdf_report_generator import TradePDFReportGenerator
        print("✅ PDF generator imported successfully")
        
        # Test creating instance
        generator = TradePDFReportGenerator(".")  # Use current directory
        print("✅ Generator instance created")
        
        # Check if output directory was created
        if generator.output_dir.exists():
            print(f"✅ Output directory exists: {generator.output_dir}")
        else:
            print(f"❌ Output directory not found: {generator.output_dir}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error with PDF generator: {e}")
        return False

if __name__ == "__main__":
    print("=== PDF Generator Directory Test ===")
    
    success = True
    
    if not test_directory_creation():
        success = False
    
    if not test_pdf_generator_import():
        success = False
    
    if success:
        print("\n✅ All tests passed! PDF generation should work.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
