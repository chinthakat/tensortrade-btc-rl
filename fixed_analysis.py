"""
Fixed episode analysis function with sub-episode support
"""

def generate_episode_report_fixed(episode_info: dict, tracker) -> str:
    """Generate comprehensive markdown report for an episode with sub-episode analysis"""
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        episode_name = episode_info['name']
        logs_dir = episode_info['logs_dir']
        models_dir = episode_info['models_dir']
        
        # Start building the report
        report = f"# Episode Performance Analysis: {episode_name}\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 📊 Episode Overview\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        report += f"| Episode Name | {episode_name} |\n"
        report += f"| Log Files Found | {episode_info['log_count']} |\n"
        report += f"| Model Files Found | {episode_info['model_count']} |\n"
        report += f"| Analysis Date | {datetime.now().strftime('%Y-%m-%d')} |\n\n"
        
        # Analyze trading logs if available
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.csv"))
            
            if log_files:
                report += "## 📈 Trading Performance Analysis\n\n"
                
                # Process each log file
                for log_file in log_files:
                    try:
                        df = pd.read_csv(log_file)
                        
                        if len(df) > 0:
                            # Extract sub-episode information from trade_id
                            df['sub_episode'] = df['trade_id'].str.extract(r'(EP\d+)')[0]
                            
                            # Calculate basic metrics using correct column names
                            total_trades = len(df)
                            
                            # Use 'net_pnl' instead of 'pnl' and filter for completed trades only
                            completed_trades = df[df['status'] == 'CLOSED']
                            
                            if len(completed_trades) > 0:
                                # Overall metrics
                                profitable_trades = len(completed_trades[completed_trades['net_pnl'] > 0])
                                losing_trades = len(completed_trades[completed_trades['net_pnl'] < 0])
                                win_rate = (profitable_trades / len(completed_trades) * 100) if len(completed_trades) > 0 else 0
                                
                                # Calculate financial metrics using completed trades
                                total_pnl = completed_trades['net_pnl'].sum()
                                avg_profit = completed_trades[completed_trades['net_pnl'] > 0]['net_pnl'].mean() if profitable_trades > 0 else 0
                                avg_loss = completed_trades[completed_trades['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
                                max_profit = completed_trades['net_pnl'].max()
                                max_loss = completed_trades['net_pnl'].min()
                                
                                # Calculate position size metrics
                                position_sizes = completed_trades['position_size'].abs()
                                avg_position_size = position_sizes.mean() if len(position_sizes) > 0 else 0
                                max_position_size = position_sizes.max() if len(position_sizes) > 0 else 0
                                min_position_size = position_sizes.min() if len(position_sizes) > 0 else 0
                                
                                # Calculate cumulative returns and drawdown
                                completed_trades_sorted = completed_trades.sort_values('close_datetime')
                                completed_trades_sorted['cumulative_pnl'] = completed_trades_sorted['net_pnl'].cumsum()
                                max_drawdown = 0
                                peak = 0
                                for value in completed_trades_sorted['cumulative_pnl']:
                                    if value > peak:
                                        peak = value
                                    drawdown = (peak - value) / abs(peak) if peak != 0 else 0
                                    max_drawdown = max(max_drawdown, drawdown)
                                
                                # Sub-episode analysis
                                sub_episode_stats = {}
                                unique_sub_episodes = completed_trades['sub_episode'].dropna().unique()
                                
                                for sub_ep in sorted(unique_sub_episodes):
                                    sub_trades = completed_trades[completed_trades['sub_episode'] == sub_ep]
                                    if len(sub_trades) > 0:
                                        sub_profitable = len(sub_trades[sub_trades['net_pnl'] > 0])
                                        sub_losing = len(sub_trades[sub_trades['net_pnl'] < 0])
                                        sub_win_rate = (sub_profitable / len(sub_trades) * 100) if len(sub_trades) > 0 else 0
                                        sub_total_pnl = sub_trades['net_pnl'].sum()
                                        sub_avg_position = sub_trades['position_size'].abs().mean()
                                        
                                        # Calculate fees and adjusted P&L
                                        sub_total_fees = sub_trades['fees_paid'].sum() if 'fees_paid' in sub_trades.columns else 0
                                        sub_adjusted_pnl = sub_total_pnl - sub_total_fees  # P&L after fees
                                        
                                        sub_episode_stats[sub_ep] = {
                                            'total_trades': len(sub_trades),
                                            'profitable': sub_profitable,
                                            'losing': sub_losing,
                                            'win_rate': sub_win_rate,
                                            'total_pnl': sub_total_pnl,
                                            'total_fees': sub_total_fees,
                                            'adjusted_pnl': sub_adjusted_pnl,
                                            'avg_position_size': sub_avg_position,
                                            'start_net_worth': sub_trades['entry_net_worth'].iloc[0] if len(sub_trades) > 0 else 0,
                                            'end_net_worth': sub_trades['close_net_worth'].iloc[-1] if len(sub_trades) > 0 else 0
                                        }
                            else:
                                profitable_trades = losing_trades = 0
                                win_rate = total_pnl = avg_profit = avg_loss = max_profit = max_loss = max_drawdown = 0
                                avg_position_size = max_position_size = min_position_size = 0
                                sub_episode_stats = {}
                            
                            # Build report sections
                            report += f"### Log File: {log_file.name}\n\n"
                            
                            # Overall Trade Summary
                            report += "#### Overall Trade Summary\n"
                            report += "| Metric | Value |\n"
                            report += "|--------|--------|\n"
                            report += f"| Total Trades | {total_trades:,} |\n"
                            report += f"| Completed Trades | {len(completed_trades):,} |\n"
                            report += f"| Profitable Trades | {profitable_trades:,} ({win_rate:.1f}%) |\n"
                            report += f"| Losing Trades | {losing_trades:,} ({100-win_rate:.1f}%) |\n"
                            report += f"| Win Rate | {win_rate:.1f}% |\n\n"
                            
                            # Financial Performance
                            report += "#### Financial Performance\n"
                            report += "| Metric | Value |\n"
                            report += "|--------|--------|\n"
                            report += f"| Total P&L | ${total_pnl:.2f} |\n"
                            report += f"| Average Profit | ${avg_profit:.2f} |\n"
                            report += f"| Average Loss | ${avg_loss:.2f} |\n"
                            report += f"| Max Single Profit | ${max_profit:.2f} |\n"
                            report += f"| Max Single Loss | ${max_loss:.2f} |\n"
                            report += f"| Max Drawdown | {max_drawdown*100:.2f}% |\n\n"
                            
                            # Position Size Analysis
                            report += "#### Position Size Analysis\n"
                            report += "| Metric | Value |\n"
                            report += "|--------|--------|\n"
                            report += f"| Average Position Size | ${avg_position_size:.4f} |\n"
                            report += f"| Maximum Position Size | ${max_position_size:.4f} |\n"
                            report += f"| Minimum Position Size | ${min_position_size:.4f} |\n\n"
                            
                            # Add sub-episode analysis if available
                            if sub_episode_stats:
                                report += "#### Sub-Episode Performance Analysis\n\n"
                                
                                # Summary table with fees breakdown
                                report += "| Sub-Episode | Total Trades | Win Rate | Raw P&L | Fees Paid | Adj P&L | Net Worth Change | Discrepancy |\n"
                                report += "|-------------|--------------|----------|---------|-----------|---------|------------------|-------------|\n"
                                
                                for sub_ep, stats in sub_episode_stats.items():
                                    net_worth_change = stats['end_net_worth'] - stats['start_net_worth']
                                    discrepancy = net_worth_change - stats['adjusted_pnl']
                                    report += f"| {sub_ep} | {stats['total_trades']:,} | {stats['win_rate']:.1f}% | ${stats['total_pnl']:.2f} | ${stats['total_fees']:.2f} | ${stats['adjusted_pnl']:.2f} | ${net_worth_change:.2f} | ${discrepancy:.2f} |\n"
                                
                                report += "\n##### Detailed Sub-Episode Breakdown\n\n"
                                
                                # Detailed breakdown for each sub-episode
                                for sub_ep, stats in sub_episode_stats.items():
                                    net_worth_change = stats['end_net_worth'] - stats['start_net_worth']
                                    discrepancy = net_worth_change - stats['adjusted_pnl']
                                    roi = (net_worth_change / stats['start_net_worth'] * 100) if stats['start_net_worth'] > 0 else 0
                                    
                                    report += f"**{sub_ep}:**\n"
                                    report += f"- **Trades:** {stats['total_trades']:,} (Win: {stats['profitable']}, Loss: {stats['losing']})\n"
                                    report += f"- **Win Rate:** {stats['win_rate']:.1f}%\n"
                                    report += f"- **Raw P&L:** ${stats['total_pnl']:.2f}\n"
                                    report += f"- **Fees Paid:** ${stats['total_fees']:.2f}\n"
                                    report += f"- **Adjusted P&L:** ${stats['adjusted_pnl']:.2f} (after fees)\n"
                                    report += f"- **Net Worth:** ${stats['start_net_worth']:.2f} → ${stats['end_net_worth']:.2f} ({net_worth_change:+.2f})\n"
                                    report += f"- **Discrepancy:** ${discrepancy:.2f} (Net Worth Change vs Adjusted P&L)\n"
                                    report += f"- **ROI:** {roi:+.2f}%\n"
                                    report += f"- **Avg Position:** ${stats['avg_position_size']:.4f}\n\n"
                                
                                # Add explanation section
                                report += "##### Understanding P&L vs Net Worth Discrepancy\n\n"
                                report += "**Why Net Worth Change ≠ P&L:**\n"
                                report += "- **Raw P&L**: Profit/Loss from price movements only\n"
                                report += "- **Fees**: Trading fees are deducted separately from net worth\n"
                                report += "- **Adjusted P&L**: Raw P&L minus fees (should be closer to net worth change)\n"
                                report += "- **Discrepancy**: Any remaining difference may indicate:\n"
                                report += "  - Slippage or spread costs\n"
                                report += "  - Position management overhead\n"
                                report += "  - Other trading environment factors\n"
                                report += "  - Calculation timing differences between trades\n\n"
                            
                        else:
                            report += f"### Log File: {log_file.name}\n**No trading data found**\n\n"
                            
                    except Exception as e:
                        report += f"### Log File: {log_file.name}\n**Error reading file:** {str(e)}\n\n"
            else:
                report += "**No log files found for analysis**\n\n"
        
        # Model analysis
        if models_dir.exists():
            model_files = list(models_dir.glob("*.zip"))
            
            if model_files:
                report += "## 🤖 Model Information\n\n"
                report += "| Model File | Size | Modified |\n"
                report += "|------------|------|----------|\n"
                
                for model_file in model_files:
                    file_size = model_file.stat().st_size / (1024 * 1024)  # MB
                    mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                    report += f"| {model_file.name} | {file_size:.1f} MB | {mod_time.strftime('%Y-%m-%d %H:%M')} |\n"
                report += "\n"
        
        # Analysis insights
        report += "## 💡 Analysis Insights\n\n"
        report += "### Key Observations\n"
        report += "- Review the win rate and average profit/loss ratios\n"
        report += "- Monitor position sizing consistency\n"
        report += "- Check for any unusual action distributions\n"
        report += "- Evaluate drawdown periods for risk management\n"
        report += "- Compare performance across sub-episodes\n\n"
        
        report += "### Recommendations\n"
        report += "1. **If Win Rate < 50%**: Focus on entry signal improvement\n"
        report += "2. **If Average Loss > Average Profit**: Implement better stop-loss strategies\n"
        report += "3. **If High Drawdown**: Consider position sizing adjustments\n"
        report += "4. **If Low Trade Count**: Evaluate signal frequency and market conditions\n"
        report += "5. **Sub-Episode Analysis**: Look for patterns in performance across different market periods\n\n"
        
        report += "---\n\n"
        report += "*This report was automatically generated by the TensorTrade Model Analysis System*\n"
        
        return report
        
    except Exception as e:
        return f"Error generating report: {str(e)}"
