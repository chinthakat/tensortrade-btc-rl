#!/usr/bin/env python3
"""Debug PDF Generation"""

import sys
import os
import traceback

try:
    # Import the generator
    from pdf_report_generator import TradePDFReportGenerator
    print("✅ Successfully imported TradePDFReportGenerator")
    
    # Try to create an instance
    generator = TradePDFReportGenerator(".")
    print("✅ Successfully created generator instance")
    
    # Try to generate the report
    pdf_path = generator.generate_comprehensive_report()
    print(f"✅ PDF generated successfully: {pdf_path}")
    
    # Check file details
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"✅ File exists, size: {size:,} bytes")
    else:
        print("❌ PDF file was not created")
        
except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    # Try to identify the specific issue
    print("\nDebugging information:")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')[:10]}...")
    
    # Check if analysis data exists
    if os.path.exists("combined_trades_20250713_183456.csv"):
        print("✅ Trade data file exists")
    else:
        print("❌ Trade data file not found")
