"""
Debug script to identify the exact location of the NoneType error in live trading
"""
import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from binance_integration import BinanceTrader
from trading_environment import FuturesTradingEnv
from action_space_wrapper import wrap_environment_for_algorithm
from improved_reward_configs import TREND_RIDER_CONFIG

# Set up logging to capture all errors
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_live_trading_issue():
    """Debug the exact issue in live trading environment"""
    print("🔍 Starting live trading debug session...")
    
    try:
        # Initialize trader
        trader = BinanceTrader(
            api_key="your_test_api_key",
            api_secret="your_test_api_secret",
            testnet=True,
            symbol="BTCUSDT",
            timeframe="15m",
            reward_config=TREND_RIDER_CONFIG
        )
        
        print("✅ Trader initialized")
        
        # Load the model
        await trader.load_model("models/best_model.zip")
        print("✅ Model loaded")
        
        # Get real Binance data (30 periods)
        print("📊 Fetching real Binance data...")
        market_data = await trader.fetch_startup_historical_data()
        
        if market_data is None:
            print("❌ Failed to fetch market data")
            return
            
        print(f"✅ Market data fetched: {len(market_data)} rows")
        print(f"📈 Data shape: {market_data.shape}")
        print(f"📋 Columns: {list(market_data.columns)}")
        
        # Check for any None/NaN values in the data
        print("\n🔍 Checking data quality:")
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in market_data.columns:
                null_count = market_data[col].isnull().sum()
                none_count = (market_data[col] == None).sum()
                inf_count = np.isinf(market_data[col]).sum()
                
                print(f"  {col}: nulls={null_count}, nones={none_count}, infs={inf_count}")
                
                # Show first few values
                print(f"    Sample values: {market_data[col].head().tolist()}")
                print(f"    Last few values: {market_data[col].tail().tolist()}")
        
        # Step-by-step environment creation with detailed error tracking
        print("\n🏗️ Creating trading environment...")
        
        try:
            env = FuturesTradingEnv(
                df=market_data,
                window_size=20,
                initial_equity=trader.initial_balance,
                reward_config=trader.reward_config
            )
            print("✅ Environment created successfully")
        except Exception as env_error:
            print(f"❌ Error creating environment: {env_error}")
            import traceback
            traceback.print_exc()
            return
        
        # Wrap environment
        print("🔄 Wrapping environment...")
        try:
            env = wrap_environment_for_algorithm(env, "PPO")
            print("✅ Environment wrapped successfully")
        except Exception as wrap_error:
            print(f"❌ Error wrapping environment: {wrap_error}")
            import traceback
            traceback.print_exc()
            return
        
        # Reset environment with detailed tracking
        print("🔄 Resetting environment...")
        try:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
            print("✅ Environment reset successfully")
            
            # Analyze observation
            print(f"\n📋 Observation analysis:")
            print(f"  Type: {type(obs)}")
            if isinstance(obs, dict):
                for key, value in obs.items():
                    print(f"  {key}: shape={getattr(value, 'shape', 'no shape')}, type={type(value)}")
                    if hasattr(value, 'dtype'):
                        print(f"    dtype={value.dtype}")
                    # Check for any None/NaN values in observation
                    if hasattr(value, '__iter__') and not isinstance(value, str):
                        try:
                            if np.any(pd.isnull(value)):
                                print(f"    ⚠️ Contains NaN values!")
                            if np.any(value == None):
                                print(f"    ⚠️ Contains None values!")
                        except:
                            pass
            
        except Exception as reset_error:
            print(f"❌ Error resetting environment: {reset_error}")
            import traceback
            traceback.print_exc()
            return
        
        # Make model prediction
        print("\n🤖 Making model prediction...")
        try:
            action, _ = trader.model.predict(obs, deterministic=True)
            print(f"✅ Model prediction successful: {action}")
        except Exception as model_error:
            print(f"❌ Error during model prediction: {model_error}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n🎉 Debug completed successfully! No errors found.")
        
    except Exception as e:
        print(f"❌ Unexpected error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_live_trading_issue())
