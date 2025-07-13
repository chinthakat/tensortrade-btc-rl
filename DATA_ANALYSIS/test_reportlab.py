#!/usr/bin/env python3
"""
Basic ReportLab test
"""

def test_reportlab():
    print("Testing ReportLab...")
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        print("[SUCCESS] ReportLab imports successful")
        
        # Create a simple PDF
        from pathlib import Path
        
        output_dir = Path("DATA_ANALYSIS/pdf_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = output_dir / "test_report.pdf"
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        
        story = [
            Paragraph("Test PDF Report", styles['Title']),
            Paragraph("This is a test PDF generated successfully!", styles['Normal'])
        ]
        
        doc.build(story)
        
        print(f"[SUCCESS] Test PDF created: {pdf_path}")
        
        # Check file size
        import os
        if os.path.exists(pdf_path):
            size = os.path.getsize(pdf_path)
            print(f"📁 File size: {size} bytes")
            return True
        else:
            print("[ERROR] PDF file not found")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    test_reportlab()
