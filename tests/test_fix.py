#!/usr/bin/env python3
"""
Test the complete negative price fix
"""

import sys
sys.path.append('.')

def test_negative_price_fix():
    """Test that the multi-episode training handles negative prices correctly"""
    
    print("🧪 Testing Negative Price Fix")
    print("=" * 40)
    
    try:
        from multi_episode_training import MultiEpisodeTrainer
        
        # Test with the actual data file
        data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
        base_config = {
            "training_params": {
                "initial_equity": 10000.0,
                "max_leverage": 10.0,
                "window_size": 20,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "max_risk_per_trade": 0.02,
                "maintenance_margin_rate": 0.004,
                "liquidation_fee_rate": 0.005,
                "n_envs": 1
            }
        }
        
        print("🏗️  Creating trainer (this will test data cleaning)...")
        trainer = MultiEpisodeTrainer(
            data_path, 
            base_config, 
            starting_model_path=None,
            validation_pct=0.05,
            use_simple_split=True
        )
        
        print("✅ Trainer created successfully!")
        print(f"📊 Clean data splits: {len(trainer.data_splits)}")
        
        if trainer.data_splits:
            train_data, val_data = trainer.data_splits[0]
            print(f"📈 Training data: {len(train_data):,} rows")
            print(f"📋 Validation data: {len(val_data):,} rows")
            
            # Check that all prices are positive
            price_columns = ['open', 'high', 'low', 'close']
            all_positive = True
            
            for col in price_columns:
                if col in train_data.columns:
                    min_price = train_data[col].min()
                    if min_price <= 0:
                        print(f"❌ Found non-positive {col}: {min_price}")
                        all_positive = False
                    else:
                        print(f"✅ {col} min: {min_price:.2f}")
            
            if all_positive:
                print("\n🎉 SUCCESS: All prices are positive!")
                print("🚀 Training should now complete without negative price errors")
            else:
                print("\n❌ FAILED: Still have non-positive prices")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_negative_price_fix()
