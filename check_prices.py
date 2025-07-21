import pandas as pd

df = pd.read_csv('episodes/test_fix_20250721_221552/logs/trades_test_fix_20250721_221552.csv')

zero_entries = df[df['entry_price'] == 0.0]
zero_closes = df[df['close_price'] == 0.0]

print(f'Zero entry prices: {len(zero_entries)}')
print(f'Zero close prices: {len(zero_closes)}')

print(f'Min entry price: {df["entry_price"].min()}')
print(f'Max entry price: {df["entry_price"].max()}')
print(f'Mean entry price: {df["entry_price"].mean():.2f}')

if len(zero_entries) > 0:
    print('\nSample zero entry price trades:')
    print(zero_entries[['trade_id', 'entry_datetime', 'entry_price', 'side', 'status']].head(3))
