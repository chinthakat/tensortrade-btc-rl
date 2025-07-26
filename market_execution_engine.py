"""
Market Execution Engine - Realistic Price Execution Model

This module implements realistic market execution with bid-ask spreads, slippage,
and market impact modeling to eliminate trade anomalies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging
import os

class MarketExecutionEngine:
    """
    Realistic market execution engine that models real-world trading conditions
    """
    
    def __init__(self, 
                 base_spread_bps: float = 2.0,  # 0.02% base spread
                 base_slippage_bps: float = 1.0,  # 0.01% base slippage
                 volatility_multiplier: float = 1.5,  # Spread increases with volatility
                 size_impact_factor: float = 0.0001):  # Market impact per $1000
        """
        Initialize market execution engine
        
        Args:
            base_spread_bps: Base bid-ask spread in basis points
            base_slippage_bps: Base slippage in basis points
            volatility_multiplier: Factor to increase spread during high volatility
            size_impact_factor: Market impact factor per $1000 position size
        """
        self.base_spread_bps = base_spread_bps
        self.base_slippage_bps = base_slippage_bps
        self.volatility_multiplier = volatility_multiplier
        self.size_impact_factor = size_impact_factor
        
    def calculate_volatility_adjustment(self, 
                                      current_price: float,
                                      high_price: float, 
                                      low_price: float) -> float:
        """
        Calculate volatility adjustment for spreads and slippage
        
        Args:
            current_price: Current market price
            high_price: High price of current period
            low_price: Low price of current period
            
        Returns:
            Volatility multiplier (1.0 = normal, >1.0 = high volatility)
        """
        if current_price <= 0:
            return 1.0
            
        # Calculate intrabar volatility
        price_range = (high_price - low_price) / current_price
        
        # Normalize volatility (0.5% range = 1.0x, 2% range = 2.0x, etc.)
        volatility_factor = max(1.0, price_range / 0.005)
        
        return min(volatility_factor, 3.0)  # Cap at 3x normal
    
    def calculate_market_impact(self, 
                              position_value: float, 
                              current_price: float) -> float:
        """
        Calculate market impact based on position size
        
        Args:
            position_value: Dollar value of the position
            current_price: Current market price
            
        Returns:
            Market impact as percentage of price
        """
        # Market impact increases non-linearly with position size
        size_factor = abs(position_value) / 1000.0  # Per $1000
        
        # Use square root scaling for realistic impact
        # Small orders: minimal impact, large orders: significant impact
        impact_pct = self.size_impact_factor * np.sqrt(size_factor)
        
        # Cap maximum impact at 0.5%
        return min(impact_pct, 0.005)
    
    def get_execution_price(self,
                           market_price: float,
                           high_price: float,
                           low_price: float,
                           side: str,  # 'BUY' or 'SELL'
                           position_value: float) -> Tuple[float, Dict]:
        """
        Calculate realistic execution price with spreads, slippage, and market impact
        
        Args:
            market_price: Base market price (close price)
            high_price: High price of current period
            low_price: Low price of current period  
            side: Trade side ('BUY' or 'SELL')
            position_value: Dollar value of position
            
        Returns:
            Tuple of (execution_price, execution_details)
        """
        if market_price <= 0:
            return market_price, {'error': 'Invalid market price'}
        
        # Calculate components
        volatility_adj = self.calculate_volatility_adjustment(market_price, high_price, low_price)
        market_impact = self.calculate_market_impact(position_value, market_price)
        
        # Adjust spread and slippage for volatility
        adjusted_spread_bps = self.base_spread_bps * volatility_adj
        adjusted_slippage_bps = self.base_slippage_bps * volatility_adj
        
        # Convert basis points to percentages
        spread_pct = adjusted_spread_bps / 10000.0
        slippage_pct = adjusted_slippage_bps / 10000.0
        
        # Calculate bid-ask prices
        half_spread = market_price * spread_pct / 2.0
        bid_price = market_price - half_spread
        ask_price = market_price + half_spread
        
        # Determine base execution price (before slippage and impact)
        if side.upper() == 'BUY':
            base_execution_price = ask_price  # Buy at ask
            # Additional slippage and impact work against buyer
            total_adverse_impact = slippage_pct + market_impact
            execution_price = base_execution_price * (1 + total_adverse_impact)
        else:  # SELL
            base_execution_price = bid_price  # Sell at bid
            # Additional slippage and impact work against seller
            total_adverse_impact = slippage_pct + market_impact
            execution_price = base_execution_price * (1 - total_adverse_impact)
        
        # Ensure execution price stays within reasonable bounds (high/low of period)
        # This prevents unrealistic executions outside the trading range
        if execution_price > high_price:
            execution_price = high_price
        elif execution_price < low_price:
            execution_price = low_price
        
        # Execution details for analysis
        execution_details = {
            'market_price': market_price,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'base_execution_price': base_execution_price,
            'final_execution_price': execution_price,
            'spread_bps': adjusted_spread_bps,
            'slippage_bps': adjusted_slippage_bps,
            'market_impact_pct': market_impact * 100,
            'volatility_multiplier': volatility_adj,
            'total_cost_bps': abs(execution_price - market_price) / market_price * 10000,
            'side': side,
            'position_value': position_value
        }
        
        return execution_price, execution_details

class TradeExecutionLogger:
    """
    Enhanced logging for trade executions with market impact analysis
    """
    
    def __init__(self, log_file: str = 'logs/execution_analysis.log'):
        """Initialize execution logger"""
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create execution logger
        self.logger = logging.getLogger('execution')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # File handler for execution logs
        handler = logging.FileHandler(log_file, mode='a')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - EXECUTION - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_execution(self, 
                     trade_id: str,
                     step: int,
                     execution_price: float,
                     execution_details: Dict):
        """
        Log detailed execution information
        
        Args:
            trade_id: Trade identifier
            step: Current step
            execution_price: Final execution price
            execution_details: Detailed execution breakdown
        """
        log_data = {
            'trade_id': trade_id,
            'step': step,
            'market_price': execution_details['market_price'],
            'execution_price': execution_price,
            'spread_bps': execution_details['spread_bps'],
            'slippage_bps': execution_details['slippage_bps'],
            'market_impact_pct': execution_details['market_impact_pct'],
            'total_cost_bps': execution_details['total_cost_bps'],
            'side': execution_details['side'],
            'volatility_mult': execution_details['volatility_multiplier']
        }
        
        self.logger.info(f"EXECUTION: {log_data}")

# Global execution engine instance
_execution_engine = None
_execution_logger = None

def get_execution_engine() -> MarketExecutionEngine:
    """Get global execution engine instance"""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = MarketExecutionEngine()
    return _execution_engine

def get_execution_logger() -> TradeExecutionLogger:
    """Get global execution logger instance"""
    global _execution_logger
    if _execution_logger is None:
        _execution_logger = TradeExecutionLogger()
    return _execution_logger

def calculate_realistic_execution_price(market_price: float,
                                      high_price: float,
                                      low_price: float,
                                      side: str,
                                      position_value: float) -> float:
    """
    Convenience function to calculate realistic execution price
    
    Returns:
        Realistic execution price accounting for spreads, slippage, and market impact
    """
    engine = get_execution_engine()
    execution_price, _ = engine.get_execution_price(
        market_price, high_price, low_price, side, position_value
    )
    return execution_price
