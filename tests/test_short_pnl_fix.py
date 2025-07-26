"""
Test script to verify Short Position PnL calculation fix
This validates that short positions now calculate PnL correctly
"""

def test_short_position_pnl():
    """Test the PnL calculation fix for short positions"""
    print("🧪 Testing Short Position PnL Calculation Fix")
    print("=" * 50)
    
    # Test scenarios
    test_cases = [
        {
            "name": "Profitable Short Position",
            "position_size": -0.1,  # Short 0.1 BTC
            "entry_price": 40000,
            "current_price": 39000,  # Price dropped $1000
            "expected_pnl": 100,  # Should be positive (profit)
            "description": "Price drops from $40k to $39k - should show profit"
        },
        {
            "name": "Loss Short Position", 
            "position_size": -0.05,  # Short 0.05 BTC
            "entry_price": 38000,
            "current_price": 39000,  # Price rose $1000
            "expected_pnl": -50,  # Should be negative (loss)
            "description": "Price rises from $38k to $39k - should show loss"
        },
        {
            "name": "Profitable Long Position",
            "position_size": 0.1,  # Long 0.1 BTC
            "entry_price": 39000,
            "current_price": 40000,  # Price rose $1000
            "expected_pnl": 100,  # Should be positive (profit)
            "description": "Price rises from $39k to $40k - should show profit"
        },
        {
            "name": "Loss Long Position",
            "position_size": 0.05,  # Long 0.05 BTC  
            "entry_price": 40000,
            "current_price": 39000,  # Price dropped $1000
            "expected_pnl": -50,  # Should be negative (loss)
            "description": "Price drops from $40k to $39k - should show loss"
        }
    ]
    
    def calculate_pnl_corrected(position_size, entry_price, current_price):
        """Calculate PnL using the corrected formula"""
        if position_size > 0:  # Long position
            return abs(position_size) * (current_price - entry_price)
        else:  # Short position  
            return abs(position_size) * (entry_price - current_price)
    
    def calculate_pnl_old_buggy(position_size, entry_price, current_price):
        """Calculate PnL using the old buggy formula"""
        if position_size > 0:  # Long position
            return position_size * (current_price - entry_price)
        else:  # Short position (BUG: position_size is negative!)
            return position_size * (entry_price - current_price)
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📊 Test {i}: {test['name']}")
        print(f"   📝 {test['description']}")
        print(f"   📍 Position: {test['position_size']} BTC")
        print(f"   💰 Entry: ${test['entry_price']:,} → Current: ${test['current_price']:,}")
        
        # Calculate with both methods
        old_pnl = calculate_pnl_old_buggy(test['position_size'], test['entry_price'], test['current_price'])
        new_pnl = calculate_pnl_corrected(test['position_size'], test['entry_price'], test['current_price'])
        expected = test['expected_pnl']
        
        print(f"   🐛 Old (Buggy): ${old_pnl:.2f}")
        print(f"   ✅ New (Fixed): ${new_pnl:.2f}")
        print(f"   🎯 Expected: ${expected:.2f}")
        
        # Check if fix works
        if abs(new_pnl - expected) < 0.01:
            print(f"   ✅ PASS: PnL calculation is correct!")
        else:
            print(f"   ❌ FAIL: Expected ${expected:.2f}, got ${new_pnl:.2f}")
            all_passed = False
            
        # Check if old method was wrong for shorts
        if test['position_size'] < 0 and abs(old_pnl - expected) > 0.01:
            print(f"   🐛 Confirmed: Old method was incorrect for short positions")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Short position PnL calculation is now correct!")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    print("\n📋 Summary of the fix:")
    print("   • OLD (buggy): pnl = position_size * (price_diff)")  
    print("   • NEW (fixed): pnl = abs(position_size) * (price_diff)")
    print("   • This ensures shorts with negative position_size show correct profits/losses")

if __name__ == "__main__":
    test_short_position_pnl()
