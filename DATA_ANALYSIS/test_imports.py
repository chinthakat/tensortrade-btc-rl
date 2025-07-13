#!/usr/bin/env python3
"""
Test required packages for PDF generation
"""

def test_imports():
    print("Testing required package imports...")
    
    try:
        import matplotlib
        print(f"✅ matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ matplotlib not found")
        return False
    
    try:
        import seaborn
        print(f"✅ seaborn: {seaborn.__version__}")
    except ImportError:
        print("❌ seaborn not found")
        return False
    
    try:
        import reportlab
        print(f"✅ reportlab: {reportlab.Version}")
    except ImportError:
        print("❌ reportlab not found")
        return False
    
    try:
        from PIL import Image
        print("✅ pillow (PIL) available")
    except ImportError:
        print("❌ pillow not found")
        return False
    
    print("✅ All required packages are available!")
    return True

if __name__ == "__main__":
    test_imports()
