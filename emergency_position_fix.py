#!/usr/bin/env python3
"""
Emergency Position Management and System Restart
Fixes position tracking issues and closes existing positions
"""
import asyncio
import sys
import os
import time
from datetime import datetime
from rich.console import Console

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from binance_integration import LiveBTCUSDTTradingSystem

console = Console()

async def emergency_position_management():
    """Emergency position management and restart"""
    console.print("[bold red]🚨 EMERGENCY POSITION MANAGEMENT SYSTEM[/bold red]")
    console.print("[yellow]This will:[/yellow]")
    console.print("  1. Connect to Binance testnet")
    console.print("  2. Check actual positions vs tracked positions")
    console.print("  3. Close all existing positions")
    console.print("  4. Clear active trades tracking")
    console.print("  5. Restart with fixed position synchronization")
    
    # Initialize the trading system
    console.print("\n[cyan]🔧 Initializing trading system...[/cyan]")
    model_path = "models/best_model.zip"
    trading_system = LiveBTCUSDTTradingSystem(model_path)
    
    try:
        # Initialize account parameters
        account_params = await trading_system.connector.initialize_account_parameters()
        
        if not account_params.get('trading_enabled', False):
            console.print("[red]❌ Trading not enabled - check API credentials[/red]")
            return
        
        console.print(f"[green]✅ Connected to Binance testnet[/green]")
        console.print(f"  • Balance: ${account_params.get('balance', 0):,.2f} USDT")
        console.print(f"  • Leverage: {account_params.get('leverage', 1)}x")
        
        # Check position discrepancy
        console.print("\n[cyan]🔍 Analyzing position tracking discrepancy...[/cyan]")
        
        # Get actual positions from Binance
        positions = trading_system.connector.client.futures_position_information(symbol="BTCUSDT")
        actual_positions = 0
        total_position_size = 0
        
        for position in positions:
            pos_amt = float(position['positionAmt'])
            if pos_amt != 0:
                actual_positions += 1
                total_position_size += abs(pos_amt)
                side = "LONG" if pos_amt > 0 else "SHORT"
                console.print(f"  • Actual position: {side} {abs(pos_amt):.6f} BTC")
                console.print(f"    Entry Price: ${float(position['entryPrice']):.2f}")
                console.print(f"    Unrealized PnL: ${float(position['unrealizedProfit']):.2f}")
        
        # Check tracked positions
        tracked_positions = len(trading_system.active_trades)
        console.print(f"\n[blue]📊 Position Analysis:[/blue]")
        console.print(f"  • Actual positions on Binance: {actual_positions}")
        console.print(f"  • Tracked positions in system: {tracked_positions}")
        console.print(f"  • Total position size: {total_position_size:.6f} BTC")
        
        if actual_positions != tracked_positions:
            console.print(f"[red]⚠️ POSITION TRACKING MISMATCH DETECTED![/red]")
            console.print(f"   System lost track of {actual_positions - tracked_positions} positions")
        
        # Emergency position closure
        if actual_positions > 0:
            console.print("\n[yellow]🚨 Closing all existing positions...[/yellow]")
            closed_count = await trading_system.close_all_positions_emergency()
            console.print(f"[green]✅ Closed {closed_count} positions[/green]")
        else:
            console.print("\n[green]✅ No positions to close[/green]")
        
        # Clear active trades tracking
        console.print("\n[cyan]🧹 Clearing active trades tracking...[/cyan]")
        trading_system.active_trades.clear()
        console.print(f"[green]✅ Cleared {tracked_positions} tracked trades[/green]")
        
        # Verify positions are closed
        console.print("\n[cyan]🔍 Verifying all positions are closed...[/cyan]")
        await asyncio.sleep(2)  # Wait for orders to process
        
        final_positions = trading_system.connector.client.futures_position_information(symbol="BTCUSDT")
        remaining_positions = sum(1 for pos in final_positions if float(pos['positionAmt']) != 0)
        
        if remaining_positions == 0:
            console.print("[green]✅ All positions successfully closed[/green]")
        else:
            console.print(f"[red]❌ {remaining_positions} positions still open - manual intervention required[/red]")
            return
        
        # Test new position synchronization
        console.print("\n[cyan]🧪 Testing new position synchronization...[/cyan]")
        synced_count = await trading_system.sync_positions_with_binance()
        console.print(f"[green]✅ Position sync working: {synced_count} positions detected[/green]")
        
        console.print("\n[bold green]🎯 EMERGENCY FIX COMPLETE![/bold green]")
        console.print("[green]The system now includes:[/green]")
        console.print("  ✅ Position synchronization with Binance every 30 seconds")
        console.print("  ✅ Emergency position closure capability")
        console.print("  ✅ Phantom trade creation for untracked positions")
        console.print("  ✅ Real-time position count in action logs")
        
        # Ask if user wants to restart trading
        console.print("\n[cyan]Would you like to restart live trading with fixed position tracking? (y/n)[/cyan]")
        restart = input().lower().strip()
        
        if restart == 'y':
            console.print("\n[bold green]🚀 Restarting live trading with fixed position tracking...[/bold green]")
            # Start the trading system with all fixes
            await trading_system.start()
        else:
            console.print("\n[yellow]⏸️ Emergency fix complete - trading not restarted[/yellow]")
            console.print("[cyan]Run python launch_live_trading.py when ready to trade[/cyan]")
    
    except Exception as e:
        console.print(f"[red]❌ Emergency management failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(emergency_position_management())
