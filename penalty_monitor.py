#!/usr/bin/env python3
"""
Penalty Error Monitor
Monitors the penalty_errors.log file and provides analysis of agent behavior.
"""

import os
import time
import sys
from collections import defaultdict
from datetime import datetime

def monitor_penalty_log(log_file='logs/penalty_errors.log', tail_lines=50):
    """Monitor penalty log file and provide analysis"""
    
    print("🔍 PENALTY ERROR MONITOR")
    print("=" * 60)
    print(f"Monitoring: {log_file}")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"⚠️  Log file not found: {log_file}")
        print("💡 Start training to generate penalty logs")
        return
    
    # Get initial file size
    last_size = os.path.getsize(log_file)
    penalty_counts = defaultdict(int)
    last_display_time = time.time()
    
    try:
        while True:
            current_size = os.path.getsize(log_file)
            
            # If file has grown, read new content
            if current_size > last_size:
                with open(log_file, 'r') as f:
                    f.seek(last_size)
                    new_lines = f.read().strip().split('\n')
                    
                    for line in new_lines:
                        if line.strip():
                            print(f"🚨 {line}")
                            
                            # Count penalty types
                            if "POSITION_STATE_CHAOS_PENALTY" in line:
                                penalty_counts["chaos"] += 1
                            elif "NaN position_size" in line:
                                penalty_counts["nan_position"] += 1
                            elif "NaN entry_price" in line:
                                penalty_counts["nan_entry"] += 1
                            elif "EMERGENCY_POSITION_BRAKE" in line:
                                penalty_counts["emergency_brake"] += 1
                            elif "SEVERE_SAFETY_PENALTY" in line:
                                penalty_counts["severe_safety"] += 1
                
                last_size = current_size
            
            # Display summary every 30 seconds
            if time.time() - last_display_time > 30:
                display_penalty_summary(penalty_counts)
                last_display_time = time.time()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("📊 FINAL PENALTY SUMMARY")
        display_penalty_summary(penalty_counts)
        print("Monitor stopped.")

def display_penalty_summary(penalty_counts):
    """Display penalty summary"""
    total_penalties = sum(penalty_counts.values())
    
    if total_penalties == 0:
        print(f"\n✅ No penalties in last period - agent learning discipline!")
        return
    
    print(f"\n📊 PENALTY BREAKDOWN (Total: {total_penalties}):")
    for penalty_type, count in penalty_counts.items():
        percentage = (count / total_penalties) * 100
        print(f"   {penalty_type}: {count} ({percentage:.1f}%)")
    
    # Assessment
    if penalty_counts["chaos"] > total_penalties * 0.7:
        print("🔥 HIGH CHAOS: Agent creating many invalid states")
    elif penalty_counts["emergency_brake"] > 0:
        print("🚨 EXTREME BEHAVIOR: Agent requesting dangerous positions")
    elif total_penalties < 10:
        print("🎯 IMPROVING: Penalty frequency decreasing")
    else:
        print("🔄 TRAINING: Agent still learning discipline")

def tail_penalty_log(log_file='logs/penalty_errors.log', lines=50):
    """Show last N lines of penalty log"""
    
    if not os.path.exists(log_file):
        print(f"⚠️  Log file not found: {log_file}")
        return
    
    print(f"📋 LAST {lines} PENALTY ERRORS")
    print("=" * 60)
    
    with open(log_file, 'r') as f:
        all_lines = f.readlines()
        
    if len(all_lines) == 0:
        print("✅ No penalty errors logged yet")
        return
    
    # Show last N lines
    last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
    
    for line in last_lines:
        print(line.strip())
    
    print(f"\nTotal penalty errors logged: {len(all_lines)}")

def analyze_penalty_trends(log_file='logs/penalty_errors.log'):
    """Analyze penalty trends over time"""
    
    if not os.path.exists(log_file):
        print(f"⚠️  Log file not found: {log_file}")
        return
    
    print("📈 PENALTY TREND ANALYSIS")
    print("=" * 60)
    
    penalty_times = []
    penalty_types = defaultdict(list)
    
    with open(log_file, 'r') as f:
        for line in f:
            if "PENALTY" in line:
                try:
                    # Extract timestamp
                    timestamp_str = line.split(' - ')[0]
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    penalty_times.append(timestamp)
                    
                    # Extract penalty type
                    if "CHAOS" in line:
                        penalty_types["chaos"].append(timestamp)
                    elif "NaN" in line:
                        penalty_types["nan"].append(timestamp)
                    elif "EMERGENCY" in line:
                        penalty_types["emergency"].append(timestamp)
                    
                except:
                    continue
    
    if not penalty_times:
        print("✅ No penalty data to analyze")
        return
    
    # Calculate penalty frequency
    total_time = (penalty_times[-1] - penalty_times[0]).total_seconds() / 60  # minutes
    penalty_rate = len(penalty_times) / total_time if total_time > 0 else 0
    
    print(f"Total penalties: {len(penalty_times)}")
    print(f"Time period: {total_time:.1f} minutes")
    print(f"Penalty rate: {penalty_rate:.2f} penalties/minute")
    
    # Show penalty type breakdown
    print(f"\nPenalty types:")
    for ptype, times in penalty_types.items():
        rate = len(times) / total_time if total_time > 0 else 0
        print(f"   {ptype}: {len(times)} ({rate:.2f}/min)")
    
    # Training assessment
    if penalty_rate < 1.0:
        print("\n🎯 GOOD: Low penalty rate - agent learning discipline")
    elif penalty_rate < 5.0:
        print("\n🔄 MODERATE: Medium penalty rate - agent still learning")
    else:
        print("\n🔥 HIGH: High penalty rate - agent very chaotic")

def main():
    """Main penalty monitor interface"""
    
    if len(sys.argv) < 2:
        print("🔍 PENALTY LOG UTILITIES")
        print("=" * 40)
        print("Usage:")
        print("  python penalty_monitor.py monitor    # Live monitoring")
        print("  python penalty_monitor.py tail       # Show recent errors")
        print("  python penalty_monitor.py analyze    # Trend analysis")
        return
    
    command = sys.argv[1].lower()
    
    if command == "monitor":
        monitor_penalty_log()
    elif command == "tail":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        tail_penalty_log(lines=lines)
    elif command == "analyze":
        analyze_penalty_trends()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: monitor, tail, analyze")

if __name__ == "__main__":
    main()
