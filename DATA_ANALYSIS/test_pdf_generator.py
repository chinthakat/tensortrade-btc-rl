#!/usr/bin/env python3
"""
Simple test for PDF report generation
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_pdf_generation():
    print("Testing PDF generation...")
    
    try:
        # Import the PDF generator
        from pdf_report_generator import TradePDFReportGenerator
        
        print("[SUCCESS] PDF generator imported successfully")
        
        # Create generator instance
        generator = TradePDFReportGenerator()
        print("[SUCCESS] Generator instance created")
        
        # Check if analysis data exists
        analysis_dir = Path(".")
        combined_files = list(analysis_dir.glob("combined_trades_*.csv"))
        
        if combined_files:
            print(f"[SUCCESS] Found {len(combined_files)} combined trade files")
            
            # Try to generate a basic report
            try:
                pdf_path = generator.generate_comprehensive_report()
                print(f"[SUCCESS] PDF report generated: {pdf_path}")
                
                # Check if file was actually created
                if os.path.exists(pdf_path):
                    file_size = os.path.getsize(pdf_path)
                    print(f"[SUCCESS] PDF file created successfully ({file_size} bytes)")
                else:
                    print("[ERROR] PDF file was not created")
                    
            except Exception as e:
                print(f"[ERROR] Error generating PDF: {str(e)}")
                print("This might be due to missing dependencies or data format issues")
        else:
            print("[WARNING]  No combined trade files found")
            print("Run the comprehensive analysis first to generate trade data")
        
    except ImportError as e:
        print(f"[ERROR] Import error: {str(e)}")
        print("Missing required packages. Install with:")
        print("pip install matplotlib seaborn reportlab pillow")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_pdf_generation()
