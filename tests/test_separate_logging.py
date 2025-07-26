#!/usr/bin/env python3
"""
Test Separate Penalty Logging
Verifies that penalty errors go to file instead of terminal.
"""

import logging
import sys
import os
from io import StringIO

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_separate_penalty_logging():
    """Test that penalty errors go to file, not terminal"""
    print("📁 TESTING SEPARATE PENALTY LOGGING")
    print("=" * 50)
    
    # Remove existing log file for clean test
    log_file = 'logs/penalty_errors.log'
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Capture terminal output
    terminal_capture = StringIO()
    terminal_handler = logging.StreamHandler(terminal_capture)
    terminal_handler.setLevel(logging.ERROR)
    
    # Get root logger and add terminal capture
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    root_logger.addHandler(terminal_handler)
    
    try:
        from trading_environment import TradingEnvironment
        
        print("✅ Initializing trading environment...")
        env = TradingEnvironment()
        env.reset()
        
        print("🔍 Testing scenarios that trigger penalty errors...")
        
        # Test scenarios that should create chaos penalties
        chaos_scenarios = [
            ("Extreme leverage chaos", 200.0),
            ("Tiny position chaos", 0.001),
            ("Negative extreme chaos", -150.0),
        ]
        
        terminal_errors = 0
        file_errors = 0
        
        for scenario_name, leverage in chaos_scenarios:
            print(f"\n   Testing: {scenario_name}")
            
            # Clear terminal capture
            terminal_capture.seek(0)
            terminal_capture.truncate(0)
            
            # Execute action that should trigger chaos
            try:
                for i in range(3):  # Multiple steps to trigger chaos
                    obs, reward, done, truncated, info = env.step(leverage)
                
                # Check terminal output
                terminal_output = terminal_capture.getvalue()
                scenario_terminal_errors = terminal_output.count("PENALTY")
                terminal_errors += scenario_terminal_errors
                
                if scenario_terminal_errors > 0:
                    print(f"      ❌ {scenario_terminal_errors} penalty errors in terminal")
                else:
                    print(f"      ✅ No penalty errors in terminal")
                    
            except Exception as e:
                print(f"      ⚠️  Error in scenario: {e}")
        
        # Check if log file was created and has content
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
            file_errors = log_content.count("PENALTY")
            print(f"\n📁 Log file analysis:")
            print(f"   File created: ✅")
            print(f"   Penalty errors in file: {file_errors}")
            
            if file_errors > 0:
                print(f"   ✅ Penalty errors successfully logged to file")
            else:
                print(f"   ⚠️  No penalty errors found in file")
        else:
            print(f"\n📁 Log file analysis:")
            print(f"   File created: ❌")
            print(f"   ⚠️  Log file not created")
        
        print(f"\n🎯 SEPARATION TEST RESULTS:")
        print(f"   Terminal penalty errors: {terminal_errors}")
        print(f"   File penalty errors: {file_errors}")
        
        if terminal_errors == 0 and file_errors > 0:
            print(f"   ✅ PERFECT: Penalty errors separated to file")
            print(f"   ✅ Clean terminal during training")
            print(f"   ✅ Error details preserved in log file")
        elif terminal_errors > 0:
            print(f"   ⚠️  {terminal_errors} penalty errors still in terminal")
            print(f"   💡 Some penalty logs not separated")
        else:
            print(f"   ❓ No penalty errors triggered")
            print(f"   💡 May need more aggressive test scenarios")
            
    except ImportError as e:
        print(f"❌ Could not import trading environment: {e}")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        # Restore original handlers
        root_logger.handlers = original_handlers

def show_penalty_log_sample():
    """Show sample of penalty log file"""
    log_file = 'logs/penalty_errors.log'
    
    print(f"\n📋 PENALTY LOG SAMPLE:")
    print("=" * 50)
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        if lines:
            print(f"Total penalty errors logged: {len(lines)}")
            print(f"Last 5 entries:")
            for line in lines[-5:]:
                print(f"   {line.strip()}")
        else:
            print("Log file is empty")
    else:
        print("No penalty log file found")

if __name__ == "__main__":
    test_separate_penalty_logging()
    show_penalty_log_sample()
    
    print(f"\n💡 TO MONITOR PENALTY ERRORS:")
    print("   python penalty_monitor.py monitor    # Live monitoring")
    print("   python penalty_monitor.py tail       # Recent errors")
    print("   python penalty_monitor.py analyze    # Trend analysis")
