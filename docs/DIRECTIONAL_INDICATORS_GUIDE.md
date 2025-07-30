# Trade Directional Indicators Guide

## Overview
This guide documents all directional indicators available in the TensorTradeModel for enhanced trading decision-making. These indicators help the model identify market direction, trend strength, and optimal entry/exit points.

## Current Directional Indicators

### **🔄 Trend Following Indicators**

#### 1. **Moving Averages**
- **SMA 10 & 20**: Simple moving averages for trend identification
- **EMA 10 & 20**: Exponential moving averages (more responsive)
- **Usage**: Price above MA = bullish bias, below = bearish bias

#### 2. **Moving Average Cross Signal** (NEW ✨)
- **Signal**: `ma_cross_signal` 
- **Values**: +1 (bullish), -1 (bearish), 0 (neutral)
- **Logic**: EMA 10 above EMA 20 = bullish signal

#### 3. **MACD System**
- **MACD Line**: Momentum indicator
- **Signal Line**: MACD smoothed
- **Histogram**: MACD - Signal difference
- **Cross Signal** (NEW ✨): `macd_cross_signal` for directional bias

#### 4. **Parabolic SAR** (NEW ✨)
- **Indicator**: `psar_trend`
- **Values**: +1 (price above PSAR = bullish), -1 (price below PSAR = bearish)
- **Usage**: Trend direction and trailing stop-loss levels

### **📈 Momentum Indicators**

#### 5. **RSI (Relative Strength Index)**
- **Range**: 0-100
- **Directional Bias**: >70 overbought (bearish reversal), <30 oversold (bullish reversal)

#### 6. **Stochastic Oscillator**
- **Range**: 0-100
- **Usage**: Momentum and reversal point identification

#### 7. **Williams %R** (NEW ✨)
- **Indicator**: `williams_r` & `williams_signal`
- **Signals**: >-20 overbought (bearish), <-80 oversold (bullish)
- **Values**: +1 (bullish signal), -1 (bearish signal), 0 (neutral)

#### 8. **CCI (Commodity Channel Index)** (NEW ✨)
- **Indicator**: `cci` & `cci_signal`  
- **Signals**: >100 bullish momentum, <-100 bearish momentum
- **Values**: +1 (bullish), -1 (bearish), 0 (neutral)

### **🎯 Directional Strength Indicators**

#### 9. **ADX System** (NEW ✨)
- **ADX**: Trend strength (>25 = strong trend, <20 = weak trend)
- **DI+**: Positive directional movement
- **DI-**: Negative directional movement
- **Directional Bias**: `directional_bias` (+1/-1/0 based on DI+/DI- comparison)

#### 10. **Multi-Timeframe Trends** (NEW ✨)
- **trend_5**: 5-period price direction
- **trend_10**: 10-period price direction  
- **trend_20**: 20-period price direction
- **Values**: +1 (uptrend), -1 (downtrend)

### **🧮 Composite Indicators**

#### 11. **Composite Direction Score** (NEW ✨)
- **Indicator**: `composite_direction`
- **Range**: -5 to +5
- **Components**: 
  - Directional bias (ADX-based)
  - PSAR trend
  - MA cross signal
  - MACD cross signal
  - RSI overbought/oversold
- **Usage**: Higher positive values = stronger bullish bias, negative = bearish bias

## Practical Trading Signals

### **Strong Bullish Signals**
```python
# Example bullish conditions
composite_direction >= 2 and
adx > 25 and  # Strong trend
psar_trend == 1 and  # Price above PSAR
ma_cross_signal == 1 and  # EMA 10 > EMA 20
rsi < 70  # Not overbought
```

### **Strong Bearish Signals**  
```python
# Example bearish conditions
composite_direction <= -2 and
adx > 25 and  # Strong trend
psar_trend == -1 and  # Price below PSAR
ma_cross_signal == -1 and  # EMA 10 < EMA 20
rsi > 30  # Not oversold
```

### **Trend Strength Confirmation**
```python
# Strong trend conditions
adx > 25 and
abs(composite_direction) >= 2 and
(trend_5 == trend_10 == trend_20)  # All timeframes aligned
```

### **Reversal Signals**
```python
# Potential reversal conditions
williams_signal != 0 or  # Williams %R signal
cci_signal != 0 or       # CCI signal  
(rsi > 70 and composite_direction > 0) or  # Overbought in uptrend
(rsi < 30 and composite_direction < 0)     # Oversold in downtrend
```

## Feature Engineering Benefits

### **Enhanced Model Input**
- **Before**: 17 features
- **After**: 30+ features including 13 new directional indicators
- **Improvement**: 76% more directional information for the model

### **Directional Clarity**
- **Trend Identification**: Multiple timeframe trend analysis
- **Momentum Confirmation**: Cross-validation through multiple momentum indicators
- **Strength Measurement**: ADX for trend strength quantification
- **Composite Scoring**: Single unified directional bias score

### **Risk Management**
- **PSAR Integration**: Natural stop-loss levels
- **Divergence Detection**: Multiple indicator confirmation required
- **Overbought/Oversold**: Williams %R and CCI for reversal points

## Implementation Notes

### **Data Quality**
- All indicators include proper NaN handling
- Consistent scaling and normalization
- No look-ahead bias in calculations

### **Performance Considerations**
- pandas_ta library for efficient calculations
- Vectorized operations for speed
- Memory-efficient feature storage

### **Model Training Impact**
- Enhanced state representation for RL agent
- Better pattern recognition capabilities
- Improved directional bias detection
- More informed trading decisions

## Usage Example

```python
# Access directional indicators in the environment
def get_market_direction(self):
    current_step = self.current_step
    
    # Get current feature values
    composite = self.feature_columns_scaled.iloc[current_step]['composite_direction']
    adx = self.feature_columns_scaled.iloc[current_step]['adx'] 
    psar_trend = self.feature_columns_scaled.iloc[current_step]['psar_trend']
    
    # Determine market direction
    if composite >= 2 and adx > 0.5:  # Strong bullish
        return "STRONG_BULLISH"
    elif composite <= -2 and adx > 0.5:  # Strong bearish  
        return "STRONG_BEARISH"
    elif abs(composite) >= 1:  # Weak trend
        return "WEAK_BULLISH" if composite > 0 else "WEAK_BEARISH"
    else:
        return "NEUTRAL"
```

## Next Steps

1. **Test New Indicators**: Run training episodes to validate indicator effectiveness
2. **Hyperparameter Tuning**: Optimize indicator periods for your trading timeframe
3. **Feature Selection**: Use feature importance analysis to identify most valuable indicators
4. **Custom Indicators**: Add domain-specific indicators based on your trading strategy

---

*Updated: January 2025*  
*Model Version: Enhanced with 13 new directional indicators*
