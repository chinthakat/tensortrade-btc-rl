# Current Observation Space & Technical Indicators

## **📊 Observation Space Structure**

The trading environment uses a **Dictionary observation space** with two main components:

### **🎯 Space Definition:**
```python
observation_space = spaces.Dict({
    'market_features': spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(60, 27),  # 60 time steps × 27 features
        dtype=np.float32
    ),
    'portfolio_features': spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(12,),  # 12 portfolio state features
        dtype=np.float32
    )
})
```

### **📈 Total Observation Dimensions:**
- **Market Features**: `60 × 27 = 1,620 values`
- **Portfolio Features**: `12 values`  
- **Total**: `1,632 observation values per step`

---

## **🔍 Market Features (60 × 27 Matrix)**

### **Time Window Configuration:**
- **Window Size**: 60 time steps (15-minute intervals = 15 hours of market data)
- **Lookback**: Rolling 60-period historical context
- **Scaling**: StandardScaler normalization applied

### **📊 27 Technical Indicators:**

#### **1. Basic Price Features (4 indicators)**
1. **`returns`** - Price percentage change
2. **`log_returns`** - Logarithmic returns  
3. **`high_low_pct`** - (High - Low) / Close ratio
4. **`close_open_pct`** - (Close - Open) / Open ratio

#### **2. Moving Averages (4 indicators)**
5. **`sma_10`** - 10-period Simple Moving Average
6. **`sma_20`** - 20-period Simple Moving Average
7. **`ema_10`** - 10-period Exponential Moving Average
8. **`ema_20`** - 20-period Exponential Moving Average

#### **3. Momentum Indicators (2 indicators)**
9. **`rsi`** - 14-period Relative Strength Index
10. **`stoch_k`** - Stochastic Oscillator %K

#### **4. Volatility Indicators (2 indicators)**
11. **`bb_width`** - Bollinger Bands width
12. **`atr`** - 14-period Average True Range

#### **5. MACD System (3 indicators)**
13. **`macd`** - MACD Line (12,26,9)
14. **`macd_signal`** - MACD Signal Line
15. **`macd_histogram`** - MACD Histogram

#### **6. Volume & Position (2 indicators)**
16. **`volume_ratio`** - Volume / 20-period SMA
17. **`price_position`** - (Close - SMA20) / SMA20

#### **7. ADX Directional System (4 indicators)**
18. **`adx`** - Average Directional Index (trend strength)
19. **`di_plus`** - Positive Directional Indicator
20. **`di_minus`** - Negative Directional Indicator  
21. **`directional_bias`** - +1 (bullish), -1 (bearish), 0 (neutral)

#### **8. Advanced Momentum (4 indicators)**
22. **`psar_trend`** - Parabolic SAR trend direction
23. **`williams_r`** - Williams %R oscillator
24. **`williams_signal`** - Williams %R buy/sell signals
25. **`cci`** - Commodity Channel Index
26. **`cci_signal`** - CCI buy/sell signals

#### **9. Cross Signals (2 indicators)**
27. **`ma_cross_signal`** - EMA 10/20 crossover signals
28. **`macd_cross_signal`** - MACD line/signal crossover

#### **10. Multi-Timeframe Trends (4 indicators)**
29. **`trend_5`** - 5-period trend direction
30. **`trend_10`** - 10-period trend direction  
31. **`trend_20`** - 20-period trend direction
32. **`composite_direction`** - Combined directional score (-5 to +5)

**Note**: Only 27 of the 32 defined indicators are currently active in the observation space.

---

## **💰 Portfolio Features (12 Values)**

### **Core Metrics (3 features)**
1. **`equity_ratio`** - Current equity / Initial equity
2. **`balance_ratio`** - Current balance / Initial equity
3. **`normalized_leverage`** - Current leverage / Max leverage

### **P&L Information (3 features)**
4. **`unrealized_pnl_ratio`** - Unrealized P&L / Initial equity
5. **`realized_pnl_ratio`** - Total realized P&L / Initial equity
6. **`margin_ratio`** - Margin used / Initial equity

### **Risk Metrics (3 features)**
7. **`drawdown`** - Current drawdown from equity peak
8. **`balance_trend_slope`** - Balance trend (normalized slope)
9. **`balance_trend`** - Recent balance change

### **Trading Behavior (3 features)**
10. **`consecutive_losses_norm`** - Consecutive losses / 10 (capped at 1.0)
11. **`consecutive_wins_norm`** - Consecutive wins / 10 (capped at 1.0)
12. **`penalty_multiplier_norm`** - Loss penalty multiplier / 3 (capped at 1.0)

---

## **⚙️ Action Space Configuration**

### **Enhanced Action Space (Current)**
```python
action_space = spaces.Dict({
    'action_type': spaces.Discrete(4),  # 0=HOLD, 1=BUY, 2=SELL, 3=CANCEL
    'leverage': spaces.Box(low=0.1, high=max_leverage, shape=(1,)),
    'risk_percentage': spaces.Box(low=0.01, high=1.0, shape=(1,))
})
```

### **Action Types:**
- **0 = HOLD**: Maintain current position
- **1 = BUY**: Open/increase long position
- **2 = SELL**: Open/increase short position  
- **3 = CANCEL**: Close current position

---

## **🎯 Key Characteristics**

### **Strengths:**
✅ **Rich Feature Set**: 27 technical indicators covering all major categories  
✅ **Multi-Timeframe**: 5, 10, 20-period trend analysis  
✅ **Directional Intelligence**: ADX system + composite scoring  
✅ **Portfolio Awareness**: 12 comprehensive portfolio metrics  
✅ **Proper Scaling**: StandardScaler normalization prevents bias  
✅ **Time Series Context**: 60-step historical window  

### **Potential Issues:**
⚠️ **High Dimensionality**: 1,632 total features may cause overfitting  
⚠️ **Feature Redundancy**: Multiple similar indicators (SMA/EMA overlap)  
⚠️ **Memory Requirements**: Large observation space for RL training  
⚠️ **Curse of Dimensionality**: Complex feature space may slow learning  

### **Missing Indicators (Not in Active Set):**
- Parabolic SAR price values
- Multi-timeframe trends (trend_5, trend_10, trend_20)  
- Composite direction score
- Additional momentum indicators (only 27/32 active)

---

## **📋 Summary Statistics**

| Component | Count | Dimension | Total Values |
|-----------|-------|-----------|--------------|
| **Basic Price** | 4 | 60 × 4 | 240 |
| **Moving Averages** | 4 | 60 × 4 | 240 |
| **Momentum** | 2 | 60 × 2 | 120 |
| **Volatility** | 2 | 60 × 2 | 120 |
| **MACD** | 3 | 60 × 3 | 180 |
| **Volume/Position** | 2 | 60 × 2 | 120 |
| **ADX System** | 4 | 60 × 4 | 240 |
| **Advanced Momentum** | 4 | 60 × 4 | 240 |
| **Cross Signals** | 2 | 60 × 2 | 120 |
| **Portfolio State** | 12 | 12 × 1 | 12 |
| **TOTAL** | **39** | | **1,632** |

This is a **sophisticated observation space** with comprehensive market intelligence and portfolio awareness! 🎯
