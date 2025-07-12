import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv

# Create simple test data
np.random.seed(42)
n_samples = 100
initial_price = 50000.0

timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15T')
returns = np.random.normal(0.0, 0.01, n_samples)

prices = [initial_price]
for i in range(1, n_samples):
    prices.append(prices[-1] * (1 + returns[i]))

prices = np.array(prices)

df = pd.DataFrame({
    'timestamp': [int(ts.timestamp()) for ts in timestamps],
    'open': prices,
    'high': prices * 1.005,
    'low': prices * 0.995,
    'close': prices,
    'volume': np.random.uniform(100, 1000, n_samples)
})

print("Creating environment...")
env = FuturesTradingEnv(
    df=df,
    initial_equity=10000.0,
    max_leverage=25.0,
    maintenance_margin_rate=0.004,
    liquidation_fee_rate=0.005,
    window_size=20
)

print("Resetting environment...")
obs, info = env.reset()

print(f"Initial step: {env.current_step}")
print(f"Initial price: ${env.price_data.iloc[env.current_step]['close']:.2f}")

print("Taking action...")
action = np.array([10.0])  # 10x leverage
obs, reward, terminated, truncated, info = env.step(action)

print(f"Position size: {env.position_size:.6f}")
print(f"Entry price: ${env.entry_price:.2f}")
print(f"Leverage: {env.leverage:.2f}x")
print(f"Liquidation price: ${env.liquidation_price:.2f}" if env.liquidation_price else "No liquidation price")

print("Test completed successfully!")
