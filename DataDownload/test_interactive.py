#!/usr/bin/env python3
"""
Quick test script to demonstrate the new interactive features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_prompts():
    """Test the prompting functions"""
    from download_script import prompt_with_default, prompt_choice, get_available_symbols
    
    print("🧪 Testing Interactive Prompts")
    print("=" * 40)
    
    # Test default prompting
    print("\n1. Testing prompt with default:")
    print("   (Just press Enter to use default)")
    
    # Test symbol choice
    print("\n2. Testing symbol selection:")
    symbols = get_available_symbols()
    try:
        symbol = prompt_choice("📊 Available symbols:", symbols, "BTCUSDT")
        print(f"   Selected symbol: {symbol}")
    except:
        print("   Using default: BTCUSDT")

if __name__ == "__main__":
    test_prompts()
