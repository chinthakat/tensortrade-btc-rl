#!/usr/bin/env python3
"""
PDF Report Generator for Trade Analysis
======================================

This module generates comprehensive PDF reports from trade analysis data,
including charts, tables, and statistical summaries.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

# Import clean trade analyzer
try:
    from clean_trade_analyzer import CleanTradeStatsAnalyzer
except ImportError:
    CleanTradeStatsAnalyzer = None

# Set matplotlib style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TradePDFReportGenerator:
    """Generate comprehensive PDF reports from trade analysis data"""
    
    def __init__(self, analysis_dir: str = "DATA_ANALYSIS"):
        # Handle relative paths properly
        if not os.path.isabs(analysis_dir):
            # If we're already in the DATA_ANALYSIS directory, use current directory
            if os.path.basename(os.getcwd()) == "DATA_ANALYSIS":
                self.analysis_dir = Path.cwd()
            else:
                # Otherwise, make it relative to current working directory
                self.analysis_dir = Path.cwd() / analysis_dir
        else:
            self.analysis_dir = Path(analysis_dir)
        
        self.reports_dir = self.analysis_dir / "reports"
        self.output_dir = self.analysis_dir / "pdf_reports"
        
        # Create all necessary directories
        self.analysis_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # ReportLab styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            spaceAfter=30
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.darkgreen,
            spaceAfter=12
        )
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.darkred,
            spaceAfter=8
        )
    
    def generate_comprehensive_report(self, episode_name: str = None) -> str:
        """Generate a comprehensive PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if episode_name:
            pdf_filename = f"trade_analysis_report_{episode_name}_{timestamp}.pdf"
        else:
            pdf_filename = f"comprehensive_trade_analysis_{timestamp}.pdf"
        
        pdf_path = self.output_dir / pdf_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title page
        story.extend(self._create_title_page())
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary())
        story.append(PageBreak())
        
        # Comprehensive Action Summary Table
        story.extend(self._create_comprehensive_summary_table())
        story.append(PageBreak())
        
        # Performance Overview
        story.extend(self._create_performance_overview())
        story.append(PageBreak())
        
        # Action Analysis
        story.extend(self._create_action_analysis())
        story.append(PageBreak())
        
        # Reward Analysis
        story.extend(self._create_reward_analysis())
        story.append(PageBreak())
        
        # Top Trades Analysis
        story.extend(self._create_top_trades_analysis())
        story.append(PageBreak())
        
        # Anomaly Detection Results
        story.extend(self._create_anomaly_analysis())
        story.append(PageBreak())
        
        # Statistical Summary
        story.extend(self._create_statistical_summary())
        
        # Build PDF
        doc.build(story)
        
        return str(pdf_path)
    
    def _create_title_page(self) -> List:
        """Create the title page"""
        story = []
        
        # Main title
        story.append(Paragraph("Trade Analysis Report", self.title_style))
        story.append(Spacer(1, 20))
        
        # Subtitle
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Comprehensive Trading Performance Analysis", subtitle_style))
        story.append(Spacer(1, 40))
        
        # Report details
        details = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: {self._get_analysis_period()}",
            f"Total Episodes Analyzed: {self._get_episode_count()}",
            f"Total Trades: {self._get_total_trades()}"
        ]
        
        for detail in details:
            story.append(Paragraph(detail, self.styles['Normal']))
            story.append(Spacer(1, 12))
        
        return story
    
    def _create_executive_summary(self) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.heading_style))
        
        # Load summary data
        summary_data = self._load_summary_data()
        
        if summary_data:
            # Key metrics table
            metrics_data = [
                ['Metric', 'Value'],
                ['Total Profit/Loss', f"${summary_data.get('total_pnl', 0):,.2f}"],
                ['Win Rate', f"{summary_data.get('win_rate', 0)*100:.1f}%"],
                ['Average Trade Duration', f"{summary_data.get('avg_duration', 0):.1f} hours"],
                ['Best Trade', f"${summary_data.get('best_trade', 0):,.2f}"],
                ['Worst Trade', f"${summary_data.get('worst_trade', 0):,.2f}"],
                ['Total Trades', f"{summary_data.get('total_trades', 0):,}"]
            ]
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 20))
        
        # Key insights
        insights = self._generate_key_insights()
        story.append(Paragraph("Key Insights:", self.subheading_style))
        for insight in insights:
            story.append(Paragraph(f"• {insight}", self.styles['Normal']))
            story.append(Spacer(1, 6))
        
        return story
    
    def _create_performance_overview(self) -> List:
        """Create performance overview with charts"""
        story = []
        
        story.append(Paragraph("Performance Overview", self.heading_style))
        
        # Generate performance charts
        chart_path = self._generate_performance_charts()
        if chart_path and os.path.exists(chart_path):
            img = Image(chart_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 20))
        
        return story
    
    def _create_action_analysis(self) -> List:
        """Create action type analysis"""
        story = []
        
        story.append(Paragraph("Action Type Analysis", self.heading_style))
        
        # Load clean action data
        action_data = self._load_action_summaries()
        
        if action_data:
            # Enhanced action statistics table
            table_data = [['Action', 'Count', '%', 'Win Rate', 'Avg P&L', 'Min P&L', 'Max P&L', 'Avg Reward', 'P95 Reward']]
            
            for action, stats in action_data.items():
                # Skip debug entries
                if any(skip in str(action) for skip in ['BUY:', 'ACTION_', 'DEBUG']):
                    continue
                
                table_data.append([
                    action,
                    f"{stats.get('count', 0):,}",
                    f"{stats.get('percentage', 0):.1f}%",
                    f"{stats.get('win_rate', 0):.1f}%" if 'win_rate' in stats else "N/A",
                    f"${stats.get('avg_pnl', 0):.2f}",
                    f"${stats.get('min_pnl', 0):.2f}" if 'min_pnl' in stats else "N/A",
                    f"${stats.get('max_pnl', 0):.2f}" if 'max_pnl' in stats else "N/A",
                    f"{stats.get('avg_reward', 0):.6f}",
                    f"{stats.get('p95_reward', 0):.6f}" if 'p95_reward' in stats else "N/A"
                ])
            
            if len(table_data) > 1:  # More than just header
                action_table = Table(table_data, colWidths=[0.8*inch, 0.7*inch, 0.5*inch, 0.7*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
                action_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(action_table)
                story.append(Spacer(1, 20))
            else:
                story.append(Paragraph("No clean action data available after filtering debug entries.", self.styles['Normal']))
        else:
            story.append(Paragraph("No action analysis data available.", self.styles['Normal']))
        
        return story
    
    def _create_reward_analysis(self) -> List:
        """Create reward analysis section"""
        story = []
        
        story.append(Paragraph("Reward Analysis", self.heading_style))
        
        # Generate reward distribution chart
        reward_chart_path = self._generate_reward_charts()
        if reward_chart_path and os.path.exists(reward_chart_path):
            img = Image(reward_chart_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 20))
        
        return story
    
    def _create_top_trades_analysis(self) -> List:
        """Create top trades analysis"""
        story = []
        
        story.append(Paragraph("Top Trades Analysis", self.heading_style))
        
        # Load top trades data
        top_trades = self._load_top_trades()
        
        if top_trades:
            story.append(Paragraph("Best Performing Trades:", self.subheading_style))
            
            # Best trades table
            best_trades_data = [['Date', 'Action', 'PnL', 'Reward', 'Duration']]
            
            for trade in top_trades.get('best', [])[:10]:
                best_trades_data.append([
                    trade.get('close_datetime', '')[:10],  # Date only
                    trade.get('entry_action', ''),
                    f"${trade.get('net_pnl', 0):.2f}",
                    f"{trade.get('close_reward', 0):.4f}",
                    f"{trade.get('trade_duration_hours', 0):.1f}h"
                ])
            
            best_table = Table(best_trades_data, colWidths=[1.2*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
            best_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.green),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(best_table)
            story.append(Spacer(1, 20))
        
        return story
    
    def _create_anomaly_analysis(self) -> List:
        """Create anomaly detection analysis"""
        story = []
        
        story.append(Paragraph("Anomaly Detection Results", self.heading_style))
        
        # Load anomaly data
        anomalies = self._load_anomalies()
        
        if anomalies:
            story.append(Paragraph(f"Total anomalies detected: {len(anomalies)}", self.styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Sample anomalies table
            if len(anomalies) > 0:
                anomaly_data = [['Date', 'Action', 'PnL', 'Reward', 'Anomaly Type']]
                
                for anomaly in anomalies[:10]:  # Show first 10
                    anomaly_data.append([
                        anomaly.get('close_datetime', '')[:10],
                        anomaly.get('entry_action', ''),
                        f"${anomaly.get('net_pnl', 0):.2f}",
                        f"{anomaly.get('close_reward', 0):.4f}",
                        anomaly.get('anomaly_type', 'Unknown')
                    ])
                
                anomaly_table = Table(anomaly_data, colWidths=[1.2*inch, 0.8*inch, 1*inch, 1*inch, 1.2*inch])
                anomaly_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightpink),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(anomaly_table)
        else:
            story.append(Paragraph("No anomalies detected in the analyzed data.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_statistical_summary(self) -> List:
        """Create detailed statistical summary"""
        story = []
        
        story.append(Paragraph("Statistical Summary", self.heading_style))
        
        # Load all trade data for statistics
        stats = self._calculate_detailed_statistics()
        
        if stats:
            # Risk metrics
            story.append(Paragraph("Risk Metrics:", self.subheading_style))
            risk_data = [
                ['Metric', 'Value'],
                ['Sharpe Ratio', f"{stats.get('sharpe_ratio', 0):.3f}"],
                ['Max Drawdown', f"{stats.get('max_drawdown', 0)*100:.1f}%"],
                ['Volatility', f"{stats.get('volatility', 0)*100:.1f}%"],
                ['Value at Risk (95%)', f"${stats.get('var_95', 0):.2f}"],
                ['Average Loss', f"${stats.get('avg_loss', 0):.2f}"],
                ['Average Win', f"${stats.get('avg_win', 0):.2f}"]
            ]
            
            risk_table = Table(risk_data, colWidths=[2*inch, 2*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(risk_table)
        
        return story
    
    def _generate_performance_charts(self) -> Optional[str]:
        """Generate performance charts and save as image"""
        try:
            # Load trade data
            combined_trades = self._load_combined_trades()
            if combined_trades is None or combined_trades.empty:
                return None
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Trading Performance Overview', fontsize=16, fontweight='bold')
            
            # 1. Cumulative PnL over time
            combined_trades['cumulative_pnl'] = combined_trades['net_pnl'].cumsum()
            ax1.plot(range(len(combined_trades)), combined_trades['cumulative_pnl'], 
                    color='blue', linewidth=2)
            ax1.set_title('Cumulative P&L Over Time')
            ax1.set_xlabel('Trade Number')
            ax1.set_ylabel('Cumulative P&L ($)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Win/Loss distribution
            win_loss_counts = combined_trades['win_loss'].value_counts()
            colors_pie = ['green', 'red']
            ax2.pie(win_loss_counts.values, labels=win_loss_counts.index, 
                   autopct='%1.1f%%', colors=colors_pie)
            ax2.set_title('Win/Loss Distribution')
            
            # 3. Trade duration histogram
            ax3.hist(combined_trades['trade_duration_hours'], bins=30, 
                    color='skyblue', alpha=0.7, edgecolor='black')
            ax3.set_title('Trade Duration Distribution')
            ax3.set_xlabel('Duration (hours)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
            # 4. Reward vs PnL scatter
            ax4.scatter(combined_trades['close_reward'], combined_trades['net_pnl'], 
                       alpha=0.6, c=combined_trades['net_pnl'], cmap='RdYlGn')
            ax4.set_title('Reward vs P&L Correlation')
            ax4.set_xlabel('Close Reward')
            ax4.set_ylabel('Net P&L ($)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / "performance_charts.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error generating performance charts: {e}")
            return None
    
    def _generate_reward_charts(self) -> Optional[str]:
        """Generate reward analysis charts"""
        try:
            combined_trades = self._load_combined_trades()
            if combined_trades is None or combined_trades.empty:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Reward Analysis', fontsize=16, fontweight='bold')
            
            # 1. Reward distribution
            ax1.hist(combined_trades['close_reward'], bins=50, 
                    color='purple', alpha=0.7, edgecolor='black')
            ax1.set_title('Reward Distribution')
            ax1.set_xlabel('Close Reward')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # 2. Reward by action type
            action_rewards = combined_trades.groupby('entry_action')['close_reward'].mean()
            ax2.bar(action_rewards.index, action_rewards.values, 
                   color=['blue', 'red', 'green'][:len(action_rewards)])
            ax2.set_title('Average Reward by Action Type')
            ax2.set_xlabel('Action Type')
            ax2.set_ylabel('Average Reward')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / "reward_charts.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"Error generating reward charts: {e}")
            return None
    
    def _load_combined_trades(self) -> Optional[pd.DataFrame]:
        """Load combined trades data"""
        try:
            # Look for the most recent combined trades file
            pattern = "combined_trades_*.csv"
            files = list(self.analysis_dir.glob(pattern))
            
            if not files:
                return None
            
            # Get the most recent file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            return pd.read_csv(latest_file)
            
        except Exception as e:
            print(f"Error loading combined trades: {e}")
            return None
    
    def _load_summary_data(self) -> Optional[Dict]:
        """Load summary data from reports"""
        try:
            # Look for summary reports
            report_files = list(self.reports_dir.glob("*.json"))
            
            if not report_files:
                return None
            
            # Load the most recent report
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_report, 'r') as f:
                data = json.load(f)
            
            return data.get('summary', {})
            
        except Exception as e:
            print(f"Error loading summary data: {e}")
            return None
    
    def _load_action_summaries(self) -> Optional[Dict]:
        """Load clean action type summaries"""
        try:
            # Try to load clean stats first
            clean_stats_dir = self.analysis_dir / "clean_stats"
            if clean_stats_dir.exists():
                stats_files = list(clean_stats_dir.glob("clean_action_stats_*.json"))
                if stats_files:
                    latest_file = max(stats_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        return json.load(f)
            
            # Fallback to generating clean stats on-the-fly
            from clean_trade_analyzer import CleanTradeStatsAnalyzer
            analyzer = CleanTradeStatsAnalyzer(str(self.analysis_dir))
            results = analyzer.analyze_combined_trades()
            
            if results and 'stats' in results:
                return results['stats']
            
            # Final fallback to old method
            summary_files = list(self.summaries_dir.glob("action_summary_*.csv"))
            
            if not summary_files:
                return None
            
            latest_file = max(summary_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Convert to dictionary format
            result = {}
            for _, row in df.iterrows():
                action_type = row['action_type']
                # Skip debug entries
                if 'BUY:' in str(action_type) or 'ACTION_' in str(action_type):
                    continue
                
                result[action_type] = {
                    'count': row['count'],
                    'percentage': row['percentage'],
                    'avg_reward': row['avg_reward'],
                    'avg_pnl': row['avg_pnl']
                }
            
            return result
            
        except Exception as e:
            print(f"Error loading action summaries: {e}")
            return None
    
    def _load_top_trades(self) -> Optional[Dict]:
        """Load top trades data"""
        try:
            extract_files = list(self.extracts_dir.glob("*.csv"))
            
            if not extract_files:
                return None
            
            result = {'best': [], 'worst': []}
            
            for file_path in extract_files:
                if 'profit' in file_path.name.lower():
                    df = pd.read_csv(file_path)
                    result['best'] = df.to_dict('records')
                elif 'loss' in file_path.name.lower():
                    df = pd.read_csv(file_path)
                    result['worst'] = df.to_dict('records')
            
            return result
            
        except Exception as e:
            print(f"Error loading top trades: {e}")
            return None
    
    def _load_anomalies(self) -> Optional[List[Dict]]:
        """Load anomaly data"""
        try:
            anomaly_files = list(self.anomalies_dir.glob("*.csv"))
            
            if not anomaly_files:
                return None
            
            all_anomalies = []
            for file_path in anomaly_files:
                df = pd.read_csv(file_path)
                all_anomalies.extend(df.to_dict('records'))
            
            return all_anomalies
            
        except Exception as e:
            print(f"Error loading anomalies: {e}")
            return None
    
    def _calculate_detailed_statistics(self) -> Optional[Dict]:
        """Calculate detailed trading statistics"""
        try:
            combined_trades = self._load_combined_trades()
            if combined_trades is None or combined_trades.empty:
                return None
            
            # Calculate various metrics
            returns = combined_trades['net_pnl']
            win_trades = returns[returns > 0]
            loss_trades = returns[returns < 0]
            
            stats = {
                'total_trades': len(combined_trades),
                'win_rate': len(win_trades) / len(combined_trades) if len(combined_trades) > 0 else 0,
                'avg_win': win_trades.mean() if len(win_trades) > 0 else 0,
                'avg_loss': loss_trades.mean() if len(loss_trades) > 0 else 0,
                'total_pnl': returns.sum(),
                'best_trade': returns.max() if len(returns) > 0 else 0,
                'worst_trade': returns.min() if len(returns) > 0 else 0,
                'volatility': returns.std() if len(returns) > 1 else 0,
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'var_95': np.percentile(returns, 5) if len(returns) > 0 else 0,
                'avg_duration': combined_trades['trade_duration_hours'].mean()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return None
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = returns.cumsum()
            rolling_max = cumulative.expanding().max()
            drawdown = cumulative - rolling_max
            return abs(drawdown.min()) / rolling_max.max() if rolling_max.max() > 0 else 0
        except:
            return 0
    
    def _generate_key_insights(self) -> List[str]:
        """Generate key insights for the report"""
        insights = []
        
        try:
            stats = self._calculate_detailed_statistics()
            if stats:
                # Win rate insight
                if stats['win_rate'] > 0.6:
                    insights.append(f"Strong win rate of {stats['win_rate']*100:.1f}% indicates good trade selection")
                elif stats['win_rate'] < 0.4:
                    insights.append(f"Low win rate of {stats['win_rate']*100:.1f}% suggests need for strategy improvement")
                
                # Profit factor insight
                total_wins = abs(stats['avg_win'] * stats['total_trades'] * stats['win_rate'])
                total_losses = abs(stats['avg_loss'] * stats['total_trades'] * (1 - stats['win_rate']))
                if total_losses > 0:
                    profit_factor = total_wins / total_losses
                    if profit_factor > 1.5:
                        insights.append(f"Excellent profit factor of {profit_factor:.2f} shows strong profitability")
                    elif profit_factor < 1.0:
                        insights.append(f"Profit factor of {profit_factor:.2f} indicates losses exceed gains")
                
                # Sharpe ratio insight
                if stats['sharpe_ratio'] > 1.0:
                    insights.append(f"Good risk-adjusted returns with Sharpe ratio of {stats['sharpe_ratio']:.2f}")
                elif stats['sharpe_ratio'] < 0:
                    insights.append(f"Negative Sharpe ratio of {stats['sharpe_ratio']:.2f} indicates poor risk management")
        
        except Exception as e:
            insights.append("Unable to generate detailed insights due to data limitations")
        
        if not insights:
            insights.append("Analysis completed successfully with comprehensive trade data")
        
        return insights
    
    def _get_analysis_period(self) -> str:
        """Get the analysis period from data"""
        try:
            combined_trades = self._load_combined_trades()
            if combined_trades is not None and not combined_trades.empty:
                start_date = pd.to_datetime(combined_trades['entry_datetime']).min()
                end_date = pd.to_datetime(combined_trades['close_datetime']).max()
                return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        except:
            pass
        return "Unknown period"
    
    def _get_episode_count(self) -> int:
        """Get total number of episodes analyzed"""
        try:
            report_files = list(self.reports_dir.glob("*.json"))
            return len(report_files)
        except:
            return 0
    
    def _get_total_trades(self) -> int:
        """Get total number of trades"""
        try:
            combined_trades = self._load_combined_trades()
            if combined_trades is not None:
                return len(combined_trades)
        except:
            pass
        return 0
    
    def _create_comprehensive_summary_table(self) -> List:
        """Create comprehensive action-level summary table"""
        story = []
        
        story.append(Paragraph("COMPREHENSIVE ACTION SUMMARY", self.heading_style))
        story.append(Spacer(1, 12))
        
        # Use clean trade analyzer if available
        if CleanTradeStatsAnalyzer:
            try:
                # Initialize clean analyzer
                clean_analyzer = CleanTradeStatsAnalyzer(self.analysis_dir)
                
                # Analyze combined trades
                clean_stats = clean_analyzer.analyze_combined_trades()
                
                if clean_stats and 'action_stats' in clean_stats:
                    # Generate summary table
                    summary_table = clean_analyzer.generate_clean_summary_table(clean_stats['action_stats'])
                    
                    if not summary_table.empty:
                        # Create header with separators
                        separator_line = "=" * 120
                        story.append(Paragraph(separator_line, self.styles['Code']))
                        story.append(Paragraph("SUMMARY TABLE", ParagraphStyle(
                            'TableTitle',
                            parent=self.styles['Normal'],
                            fontSize=14,
                            textColor=colors.darkblue,
                            alignment=TA_CENTER,
                            fontName='Helvetica-Bold'
                        )))
                        story.append(Paragraph(separator_line, self.styles['Code']))
                        story.append(Spacer(1, 10))
                        
                        # Convert dataframe to table data
                        table_data = [summary_table.columns.tolist()]  # Headers
                        for _, row in summary_table.iterrows():
                            table_data.append(row.tolist())
                        
                        # Create ReportLab table with proper styling
                        col_widths = [0.8*inch, 0.7*inch, 0.8*inch, 0.9*inch, 0.7*inch, 
                                    0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.8*inch, 
                                    0.8*inch, 0.8*inch, 0.8*inch, 0.9*inch]
                        
                        summary_rltable = Table(table_data, colWidths=col_widths)
                        summary_rltable.setStyle(TableStyle([
                            # Header row styling
                            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 8),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                            
                            # Data rows styling
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 1), (-1, -1), 7),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                            
                            # Alternate row colors
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                            
                            # Right-align numeric columns
                            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),  # Count
                            ('ALIGN', (2, 1), (2, -1), 'RIGHT'),  # Percentage  
                            ('ALIGN', (3, 1), (3, -1), 'RIGHT'),  # Total P&L
                            ('ALIGN', (4, 1), (4, -1), 'RIGHT'),  # Avg P&L
                            ('ALIGN', (5, 1), (5, -1), 'RIGHT'),  # Min P&L
                            ('ALIGN', (6, 1), (6, -1), 'RIGHT'),  # Max P&L
                            ('ALIGN', (7, 1), (7, -1), 'RIGHT'),  # P95 P&L
                            ('ALIGN', (8, 1), (8, -1), 'RIGHT'),  # Win Rate
                            ('ALIGN', (9, 1), (9, -1), 'RIGHT'),  # Avg Reward
                            ('ALIGN', (10, 1), (10, -1), 'RIGHT'), # Min Reward
                            ('ALIGN', (11, 1), (11, -1), 'RIGHT'), # Max Reward
                            ('ALIGN', (12, 1), (12, -1), 'RIGHT'), # P95 Reward
                            ('ALIGN', (13, 1), (13, -1), 'RIGHT'), # Profit Factor
                        ]))
                        
                        story.append(summary_rltable)
                        story.append(Spacer(1, 15))
                        
                        # Add footer separator
                        story.append(Paragraph("=" * 120, self.styles['Code']))
                        story.append(Spacer(1, 10))
                        
                        # Add summary insights
                        story.append(Paragraph("Key Insights:", self.subheading_style))
                        total_trades = sum(stats.get('count', 0) for stats in clean_stats['action_stats'].values())
                        total_pnl = sum(stats.get('total_pnl', 0) for stats in clean_stats['action_stats'].values())
                        
                        insights = [
                            f"Total Actions Analyzed: {total_trades:,}",
                            f"Overall P&L: ${total_pnl:,.2f}",
                            f"Most Frequent Action: {max(clean_stats['action_stats'].keys(), key=lambda x: clean_stats['action_stats'][x]['count'])}",
                            f"Most Profitable Action: {max(clean_stats['action_stats'].keys(), key=lambda x: clean_stats['action_stats'][x]['total_pnl'])}"
                        ]
                        
                        for insight in insights:
                            story.append(Paragraph(f"• {insight}", self.styles['Normal']))
                            story.append(Spacer(1, 4))
                        
                    else:
                        story.append(Paragraph("No clean trade data available for summary table.", self.styles['Normal']))
                else:
                    story.append(Paragraph("No action statistics available.", self.styles['Normal']))
            
            except Exception as e:
                story.append(Paragraph(f"Error generating clean summary: {str(e)}", self.styles['Normal']))
        else:
            story.append(Paragraph("Clean Trade Analyzer not available.", self.styles['Normal']))
        
        return story
def generate_pdf_report(analysis_dir: str = "DATA_ANALYSIS", episode_name: str = None) -> str:
    """Convenience function to generate PDF report"""
    # Handle relative paths properly when called from different locations
    if not os.path.isabs(analysis_dir):
        analysis_dir = str(Path.cwd() / analysis_dir)
    
    generator = TradePDFReportGenerator(analysis_dir)
    return generator.generate_comprehensive_report(episode_name)


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Generate comprehensive PDF trade analysis report')
    parser.add_argument('--episode', type=str, help='Specific episode name to analyze')
    parser.add_argument('--analysis-dir', type=str, default='DATA_ANALYSIS', 
                       help='Analysis directory path')
    
    args = parser.parse_args()
    
    try:
        # Generate report
        pdf_path = generate_pdf_report(args.analysis_dir, args.episode)
        print(f"[SUCCESS] PDF report generated successfully: {pdf_path}")
        
        # Check file size
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"[FILE] File size: {file_size:,} bytes")
        
    except Exception as e:
        print(f"[ERROR] Error generating PDF report: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure analysis data exists (run comprehensive analysis first)")
        print("2. Check that required packages are installed:")
        print("   pip install matplotlib seaborn reportlab pillow")
        print("3. Verify DATA_ANALYSIS directory contains trade data")
        sys.exit(1)
