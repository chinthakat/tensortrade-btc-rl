#!/usr/bin/env python3
"""
Live Position Sync Fix - Apply position synchronization immediately to running system
"""
import asyncio
import sys
import os
import time
from datetime import datetime
from rich.console import Console

console = Console()

async def force_position_sync():
    """Force immediate position synchronization for running system"""
    console.print("[bold red]🔄 FORCE POSITION SYNCHRONIZATION[/bold red]")
    console.print("[yellow]This will sync the running system's position tracking with actual Binance positions[/yellow]")
    
    try:
        # Import and initialize
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from binance_integration import LiveBTCUSDTTradingSystem
        
        model_path = "models/best_model.zip"
        trading_system = LiveBTCUSDTTradingSystem(model_path)
        
        # Initialize account to get client connection
        account_params = await trading_system.connector.initialize_account_parameters()
        
        if not account_params.get('trading_enabled', False):
            console.print("[red]❌ Trading not enabled - check connection[/red]")
            return
        
        console.print("[green]✅ Connected to Binance testnet[/green]")
        
        # Get actual positions
        console.print("[cyan]🔍 Checking actual positions on Binance...[/cyan]")
        positions = trading_system.connector.client.futures_position_information(symbol="BTCUSDT")
        
        actual_positions = []
        for position in positions:
            pos_amt = float(position['positionAmt'])
            if pos_amt != 0:
                actual_positions.append({
                    'symbol': position['symbol'],
                    'amount': pos_amt,
                    'side': 'LONG' if pos_amt > 0 else 'SHORT',
                    'size': abs(pos_amt),
                    'entry_price': float(position['entryPrice']),
                    'unrealized_pnl': float(position['unrealizedProfit']),
                    'notional': float(position['notional'])
                })
        
        console.print(f"[blue]📊 Found {len(actual_positions)} actual positions on Binance:[/blue]")
        for pos in actual_positions:
            console.print(f"  • {pos['symbol']}: {pos['side']} {pos['size']:.6f} BTC")
            console.print(f"    Entry: ${pos['entry_price']:.2f}, PnL: ${pos['unrealized_pnl']:.2f}")
            console.print(f"    Notional: ${abs(pos['notional']):.2f}")
        
        # Test the sync method
        console.print("\n[cyan]🔄 Testing position synchronization method...[/cyan]")
        synced_count = await trading_system.sync_positions_with_binance()
        console.print(f"[green]✅ Position sync returned: {synced_count} positions[/green]")
        
        # Check active_trades after sync
        console.print(f"[blue]📋 Active trades after sync: {len(trading_system.active_trades)}[/blue]")
        for trade_id, trade in trading_system.active_trades.items():
            console.print(f"  • {trade_id}: {trade.side} {trade.quantity:.6f} {trade.symbol}")
            console.print(f"    Status: {trade.status}, Entry: ${trade.entry_price:.2f}")
        
        # Show recommendations
        console.print(f"\n[bold yellow]📋 POSITION TRACKING ANALYSIS:[/bold yellow]")
        console.print(f"• Actual Binance positions: {len(actual_positions)}")
        console.print(f"• Tracked positions after sync: {len(trading_system.active_trades)}")
        console.print(f"• Sync method result: {synced_count}")
        
        if len(actual_positions) != len(trading_system.active_trades):
            console.print(f"[red]⚠️ MISMATCH DETECTED: Position tracking still not synchronized[/red]")
            console.print(f"[yellow]The position sync method needs to be called more frequently in the live system[/yellow]")
        else:
            console.print(f"[green]✅ Position tracking is now synchronized![/green]")
        
        console.print(f"\n[cyan]💡 SOLUTION: The system needs periodic position sync calls in the trading loop[/cyan]")
        
    except Exception as e:
        console.print(f"[red]❌ Sync test failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(force_position_sync())
