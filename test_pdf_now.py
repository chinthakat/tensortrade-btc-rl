#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Set working directory
os.chdir(r"c:\Projects\GeminiModel\TensorTradeModel")
print(f"Working directory: {os.getcwd()}")

try:
    # Import and run
    sys.path.append("DATA_ANALYSIS")
    from simple_pdf_generator import generate_simple_pdf_report
    
    result = generate_simple_pdf_report()
    print(f"PDF generated: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
