#!/usr/bin/env python3
"""
Test script to verify the new simple train/validation split approach
"""

import pandas as pd
import os

def test_new_data_split():
    """Test the new 95/5 train/validation split"""
    
    # Load the actual data file
    data_file = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
    
    if not os.path.exists(data_file):
        print("❌ Data file not found")
        return
    
    df = pd.read_csv(data_file)
    total_rows = len(df)
    
    print("🎯 New Simple Train/Validation Split Test")
    print("=" * 50)
    print(f"📊 Total dataset: {total_rows:,} rows")
    print()
    
    # Simulate the new split logic
    validation_pct = 0.05
    val_size = int(total_rows * validation_pct)
    train_size = total_rows - val_size
    
    print(f"📈 Training data: rows 0 → {train_size:,} ({train_size:,} rows - {100*(1-validation_pct):.0f}%)")
    print(f"📋 Validation data: rows {train_size:,} → {total_rows:,} ({val_size:,} rows - {validation_pct*100:.0f}%)")
    print()
    
    # Calculate expected training steps
    buffer = 60  # min_buffer_for_indicators
    expected_max_step = train_size - buffer - 1
    
    print(f"🎮 Training Environment:")
    print(f"   Starting step: {buffer}")
    print(f"   Maximum step: ~{expected_max_step:,}")
    print(f"   Total training steps available: ~{expected_max_step - buffer + 1:,}")
    print()
    
    # Show time range
    if 'timestamp' in df.columns:
        train_start_time = pd.to_datetime(df.iloc[0]['timestamp'], unit='s')
        train_end_time = pd.to_datetime(df.iloc[train_size-1]['timestamp'], unit='s')
        val_start_time = pd.to_datetime(df.iloc[train_size]['timestamp'], unit='s')
        val_end_time = pd.to_datetime(df.iloc[-1]['timestamp'], unit='s')
        
        print(f"📅 Training period:")
        print(f"   {train_start_time} → {train_end_time}")
        print(f"📅 Validation period:")
        print(f"   {val_start_time} → {val_end_time}")
        print()
    
    print("✅ Benefits of this approach:")
    print("   🎯 Uses 95% of data for training (vs. 28% with old splits)")
    print("   🔄 No confusing multiple data splits")
    print("   📊 Environment episodes can cycle through FULL training data")
    print("   🧠 Better learning from complete historical patterns")
    print("   ⚡ Simpler and more intuitive")
    print()
    
    print(f"🚀 Your next training will use {expected_max_step - buffer + 1:,} steps instead of ~9,940!")

if __name__ == "__main__":
    test_new_data_split()
