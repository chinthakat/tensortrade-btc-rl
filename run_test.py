import subprocess
import sys

# Run the test script and capture output
try:
    result = subprocess.run([sys.executable, 'simple_dust_test.py'], 
                          capture_output=True, text=True, 
                          encoding='utf-8', errors='replace')
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
    
except Exception as e:
    print(f"Error running test: {e}")
