#!/usr/bin/env python3
"""Test PDF Generation"""

import sys
import os
sys.path.append('.')

try:
    from pdf_report_generator import generate_pdf_report
    print("✅ PDF generator imported successfully")
    
    # Generate report
    pdf_path = generate_pdf_report(".")
    print(f"✅ PDF generated: {pdf_path}")
    
    # Check if file exists
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"✅ File exists, size: {size:,} bytes")
    else:
        print("❌ PDF file not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
