"""
Live Trading Interface for Binance Futures
Real-time trading with trained RL models
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from binance.client import Client
from binance.exceptions import BinanceAPIException
from stable_baselines3 import PPO, A2C, SAC

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt, Confirm, FloatPrompt
from rich import print as rprint

from trading_environment import FuturesTradingEnv, TradeLogger

console = Console()

class BinanceFuturesLiveTrader:
    """Live trading interface for Binance Futures"""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        model_path: str,
        symbol: str = "BTCUSDT",
        testnet: bool = True,
        max_position_size: float = 1000.0,  # Maximum position size in USDT
        risk_per_trade: float = 0.02  # 2% risk per trade
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.model_path = model_path
        self.symbol = symbol
        self.testnet = testnet
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        
        # Initialize Binance client
        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            console.print("⚠️  [yellow]Connected to Binance Testnet[/yellow]")
        else:
            self.client = Client(api_key, api_secret)
            console.print("🔴 [red]Connected to Binance Live Trading[/red]")
        
        # Load trained model
        self.model = self._load_model()
        
        # Trading state
        self.is_trading = False
        self.current_position = None
        self.trade_logger = None
        self.historical_data = pd.DataFrame()
        
        # Risk management
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_trades_per_day = 10
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Initialize trade logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"live_trading_logs/live_trades_{timestamp}.csv"
        os.makedirs("live_trading_logs", exist_ok=True)
        self.trade_logger = TradeLogger(log_file)
    
    def _load_model(self):
        """Load the trained RL model"""
        try:
            # Determine model type from filename
            if "ppo" in self.model_path.lower():
                model = PPO.load(self.model_path)
            elif "a2c" in self.model_path.lower():
                model = A2C.load(self.model_path)
            elif "sac" in self.model_path.lower():
                model = SAC.load(self.model_path)
            else:
                model = PPO.load(self.model_path)  # Default
            
            console.print(f"✅ Model loaded from: [green]{self.model_path}[/green]")
            return model
            
        except Exception as e:
            console.print(f"[red]❌ Error loading model: {str(e)}[/red]")
            return None
    
    def get_account_info(self) -> Dict:
        """Get futures account information"""
        try:
            account = self.client.futures_account()
            balance = float(account['totalWalletBalance'])
            available_balance = float(account['availableBalance'])
            total_unrealized_pnl = float(account['totalUnrealizedProfit'])
            
            return {
                'balance': balance,
                'available_balance': available_balance,
                'unrealized_pnl': total_unrealized_pnl,
                'margin_ratio': float(account['totalMaintMargin']) / balance if balance > 0 else 0
            }
        except BinanceAPIException as e:
            console.print(f"[red]❌ Error getting account info: {str(e)}[/red]")
            return {}
    
    def get_current_position(self) -> Optional[Dict]:
        """Get current position for the symbol"""
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for position in positions:
                if float(position['positionAmt']) != 0:
                    return {
                        'symbol': position['symbol'],
                        'size': float(position['positionAmt']),
                        'entry_price': float(position['entryPrice']),
                        'mark_price': float(position['markPrice']),
                        'unrealized_pnl': float(position['unRealizedProfit']),
                        'percentage': float(position['percentage'])
                    }
            return None
        except BinanceAPIException as e:
            console.print(f"[red]❌ Error getting position: {str(e)}[/red]")
            return None
    
    def get_historical_data(self, interval: str = "15m", limit: int = 200) -> pd.DataFrame:
        """Get historical kline data"""
        try:
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to appropriate data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['timestamp'] = df['timestamp'].astype(int) // 1000  # Convert to seconds
            
            # Keep only required columns
            df = df[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            
            return df
            
        except BinanceAPIException as e:
            console.print(f"[red]❌ Error getting historical data: {str(e)}[/red]")
            return pd.DataFrame()
    
    def calculate_position_size(self, account_balance: float, entry_price: float, stop_loss_price: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = account_balance * self.risk_per_trade
        price_difference = abs(entry_price - stop_loss_price)
        
        if price_difference == 0:
            return 0
        
        position_size_risk = risk_amount / price_difference
        position_size_max = min(self.max_position_size / entry_price, account_balance * 0.1 / entry_price)
        
        return min(position_size_risk, position_size_max)
    
    def place_market_order(self, side: str, quantity: float) -> Optional[Dict]:
        """Place a market order"""
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=quantity
            )
            
            console.print(f"✅ Order placed: {side} {quantity} {self.symbol}")
            return order
            
        except BinanceAPIException as e:
            console.print(f"[red]❌ Error placing order: {str(e)}[/red]")
            return None
    
    def close_position(self) -> bool:
        """Close current position"""
        position = self.get_current_position()
        if not position:
            return True
        
        size = abs(position['size'])
        side = "SELL" if position['size'] > 0 else "BUY"
        
        order = self.place_market_order(side, size)
        if order:
            # Log the trade
            if self.trade_logger:
                self._log_trade_close(position, order)
            return True
        
        return False
    
    def _log_trade_close(self, position: Dict, close_order: Dict):
        """Log a closed trade"""
        pnl = position['unrealized_pnl']
        
        trade_data = {
            'trade_id': f"LIVE_{self.total_trades:05d}",
            'training_step': 0,
            'training_iteration': 0,
            'entry_datetime': datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M'),
            'close_datetime': datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M'),
            'side': 'LONG' if position['size'] > 0 else 'SHORT',
            'entry_action': 'BUY' if position['size'] > 0 else 'SELL',
            'entry_price': position['entry_price'],
            'close_price': position['mark_price'],
            'net_pnl': pnl,
            'close_reward': 0.0,
            'entry_net_worth': 0.0,
            'close_net_worth': 0.0,
            'trade_duration_hours': 0.0,
            'status': 'CLOSED',
            'win_loss': 'WIN' if pnl > 0 else 'LOSS',
            'position_size': abs(position['size']),
            'fees_paid': 0.0,
            'stop_loss_price': 0.0,
            'take_profit_price': 0.0,
            'close_reason': 'MANUAL'
        }
        
        self.trade_logger.log_trade(trade_data)
        
        # Update statistics
        self.total_trades += 1
        self.daily_trades += 1
        self.total_pnl += pnl
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
    
    def create_trading_dashboard(self) -> Layout:
        """Create a live trading dashboard"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Header
        header_text = "[bold blue]🤖 Live Trading Dashboard[/bold blue]"
        if self.testnet:
            header_text += " [yellow](TESTNET)[/yellow]"
        else:
            header_text += " [red](LIVE)[/red]"
        
        layout["header"].update(Panel(header_text, style="blue"))
        
        # Account info
        account_info = self.get_account_info()
        account_table = Table(title="Account Information")
        account_table.add_column("Metric", style="cyan")
        account_table.add_column("Value", style="green")
        
        if account_info:
            account_table.add_row("Balance", f"${account_info['balance']:.2f}")
            account_table.add_row("Available", f"${account_info['available_balance']:.2f}")
            account_table.add_row("Unrealized PnL", f"${account_info['unrealized_pnl']:.2f}")
            account_table.add_row("Margin Ratio", f"{account_info['margin_ratio']*100:.1f}%")
        
        layout["left"].update(account_table)
        
        # Position info
        position = self.get_current_position()
        position_table = Table(title="Current Position")
        position_table.add_column("Metric", style="cyan")
        position_table.add_column("Value", style="green")
        
        if position:
            position_table.add_row("Symbol", position['symbol'])
            position_table.add_row("Size", f"{position['size']:.4f}")
            position_table.add_row("Entry Price", f"${position['entry_price']:.2f}")
            position_table.add_row("Mark Price", f"${position['mark_price']:.2f}")
            position_table.add_row("Unrealized PnL", f"${position['unrealized_pnl']:.2f}")
            position_table.add_row("Percentage", f"{position['percentage']:.2f}%")
        else:
            position_table.add_row("Status", "No Position")
        
        layout["right"].update(position_table)
        
        # Footer with trading stats
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        footer_text = f"Total Trades: {self.total_trades} | Win Rate: {win_rate:.1f}% | Total PnL: ${self.total_pnl:.2f} | Daily PnL: ${self.daily_pnl:.2f}"
        layout["footer"].update(Panel(footer_text, style="green"))
        
        return layout
    
    async def trading_loop(self):
        """Main trading loop"""
        console.print("[bold]🚀 Starting live trading loop...[/bold]")
        
        with Live(self.create_trading_dashboard(), refresh_per_second=1, console=console) as live:
            while self.is_trading:
                try:
                    # Check daily limits
                    if self.daily_pnl < -self.get_account_info()['balance'] * self.daily_loss_limit:
                        console.print("[red]❌ Daily loss limit reached. Stopping trading.[/red]")
                        break
                    
                    if self.daily_trades >= self.max_trades_per_day:
                        console.print("[yellow]⚠️  Daily trade limit reached. Stopping trading.[/yellow]")
                        break
                    
                    # Get market data
                    market_data = self.get_historical_data()
                    if market_data.empty:
                        await asyncio.sleep(5)
                        continue
                    
                    # Create environment observation
                    # This is a simplified version - in a real implementation,
                    # you would need to properly format the data for your model
                    
                    # Update dashboard
                    live.update(self.create_trading_dashboard())
                    
                    # Wait before next iteration
                    await asyncio.sleep(10)  # Check every 10 seconds
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]⚠️  Trading interrupted by user[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]❌ Error in trading loop: {str(e)}[/red]")
                    await asyncio.sleep(5)
        
        # Close any open positions when stopping
        if self.get_current_position():
            if Confirm.ask("Close current position before stopping?"):
                self.close_position()
        
        console.print("[bold green]✅ Trading stopped[/bold green]")
    
    def start_trading(self):
        """Start the live trading system"""
        if not self.model:
            console.print("[red]❌ No model loaded. Cannot start trading.[/red]")
            return
        
        # Display risk warning
        risk_warning = """
        ⚠️  RISK WARNING ⚠️
        
        Live trading involves significant financial risk.
        - You can lose all your capital
        - Past performance does not guarantee future results
        - This is experimental software
        - Start with small amounts
        - Use testnet first
        
        Make sure you understand the risks before proceeding.
        """
        
        console.print(Panel(risk_warning, title="Risk Warning", border_style="red"))
        
        if not Confirm.ask("Do you understand and accept the risks?"):
            console.print("[yellow]Trading cancelled by user[/yellow]")
            return
        
        # Final confirmation for live trading
        if not self.testnet:
            if not Confirm.ask("⚠️  This is LIVE TRADING with real money. Are you sure?"):
                console.print("[yellow]Trading cancelled by user[/yellow]")
                return
        
        self.is_trading = True
        
        # Start trading loop
        try:
            asyncio.run(self.trading_loop())
        except KeyboardInterrupt:
            console.print("\n[yellow]Trading interrupted[/yellow]")
        finally:
            self.is_trading = False

def setup_live_trading():
    """Setup live trading interface"""
    console.print("[bold]🔧 Live Trading Setup[/bold]")
    
    # Get API credentials
    console.print("\n[yellow]📋 API Configuration:[/yellow]")
    console.print("You need Binance Futures API credentials.")
    console.print("Create them at: https://www.binance.com/en/my/settings/api-management")
    
    api_key = Prompt.ask("🔑 Enter your Binance API Key")
    api_secret = Prompt.ask("🔐 Enter your Binance API Secret", password=True)
    
    # Select model
    from pathlib import Path
    models_dir = Path("models")
    if not models_dir.exists():
        console.print("[red]❌ No models directory found![/red]")
        return
    
    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        console.print("[red]❌ No trained models found![/red]")
        return
    
    console.print("\n[bold]🤖 Available Models:[/bold]")
    table = Table()
    table.add_column("Index", style="cyan")
    table.add_column("Model", style="green")
    
    for i, model_file in enumerate(model_files):
        table.add_row(str(i+1), model_file.name)
    
    console.print(table)
    
    from rich.prompt import IntPrompt
    choice = IntPrompt.ask("Select model", default=1)
    
    if 1 <= choice <= len(model_files):
        selected_model = model_files[choice-1]
    else:
        console.print("[red]Invalid selection[/red]")
        return
    
    # Trading parameters
    testnet = Confirm.ask("Use testnet (recommended for first time)?", default=True)
    symbol = Prompt.ask("Trading symbol", default="BTCUSDT")
    max_position = FloatPrompt.ask("Maximum position size (USDT)", default=1000.0)
    risk_per_trade = FloatPrompt.ask("Risk per trade (0.02 = 2%)", default=0.02)
    
    # Create trader instance
    trader = BinanceFuturesLiveTrader(
        api_key=api_key,
        api_secret=api_secret,
        model_path=str(selected_model),
        symbol=symbol,
        testnet=testnet,
        max_position_size=max_position,
        risk_per_trade=risk_per_trade
    )
    
    # Test connection
    account_info = trader.get_account_info()
    if account_info:
        console.print(f"✅ [green]Connected successfully![/green]")
        console.print(f"💰 Account Balance: ${account_info['balance']:.2f}")
        
        # Start trading
        trader.start_trading()
    else:
        console.print("[red]❌ Failed to connect to Binance[/red]")

if __name__ == "__main__":
    setup_live_trading()
