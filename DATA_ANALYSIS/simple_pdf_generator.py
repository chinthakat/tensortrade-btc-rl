#!/usr/bin/env python3
"""
Simple PDF Report Generator (Minimal Dependencies)
=================================================

This is a simplified version that creates basic PDF reports without requiring
all the heavy dependencies like matplotlib and seaborn.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class SimplePDFReportGenerator:
    """Generate basic PDF reports from trade analysis data"""
    
    def __init__(self, analysis_dir: str = "DATA_ANALYSIS"):
        # Handle relative paths properly
        if not os.path.isabs(analysis_dir):
            self.analysis_dir = Path.cwd() / analysis_dir
        else:
            self.analysis_dir = Path(analysis_dir)
        
        self.reports_dir = self.analysis_dir / "reports"
        self.output_dir = self.analysis_dir / "pdf_reports"
        
        # Create all necessary directories
        self.analysis_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
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
    
    def generate_simple_report(self, episode_name: str = None) -> str:
        """Generate a simple PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if episode_name:
            pdf_filename = f"trade_analysis_report_{episode_name}_{timestamp}.pdf"
        else:
            pdf_filename = f"trade_analysis_report_{timestamp}.pdf"
        
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
        
        # Summary data
        story.extend(self._create_summary_section())
        story.append(PageBreak())
        
        # Trade statistics
        story.extend(self._create_statistics_section())
        
        # Build PDF
        try:
            doc.build(story)
            return str(pdf_path)
        except Exception as e:
            raise Exception(f"Failed to build PDF: {str(e)}")
    
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
        story.append(Paragraph("Trading Performance Analysis", subtitle_style))
        story.append(Spacer(1, 40))
        
        # Report details
        details = [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Directory: {self.analysis_dir}",
            f"Total Episodes: {self._get_episode_count()}",
            f"Total Trades: {self._get_total_trades()}"
        ]
        
        for detail in details:
            story.append(Paragraph(detail, self.styles['Normal']))
            story.append(Spacer(1, 12))
        
        return story
    
    def _create_summary_section(self) -> List:
        """Create summary section"""
        story = []
        
        story.append(Paragraph("Analysis Summary", self.heading_style))
        
        # Load trade data
        combined_trades = self._load_combined_trades()
        
        if combined_trades is not None and not combined_trades.empty:
            # Create summary table
            total_trades = len(combined_trades)
            total_pnl = combined_trades['net_pnl'].sum()
            wins = len(combined_trades[combined_trades['net_pnl'] > 0])
            losses = len(combined_trades[combined_trades['net_pnl'] <= 0])
            win_rate = wins / total_trades * 100 if total_trades > 0 else 0
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Trades', f"{total_trades:,}"],
                ['Total P&L', f"${total_pnl:,.2f}"],
                ['Wins', f"{wins:,}"],
                ['Losses', f"{losses:,}"],
                ['Win Rate', f"{win_rate:.1f}%"],
                ['Average P&L', f"${total_pnl/total_trades:.2f}" if total_trades > 0 else "$0.00"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
        else:
            story.append(Paragraph("No trade data found for analysis.", self.styles['Normal']))
        
        return story
    
    def _create_statistics_section(self) -> List:
        """Create statistics section"""
        story = []
        
        story.append(Paragraph("Detailed Statistics", self.heading_style))
        
        # Load trade data
        combined_trades = self._load_combined_trades()
        
        if combined_trades is not None and not combined_trades.empty:
            # Action type analysis
            if 'entry_action' in combined_trades.columns:
                action_stats = combined_trades.groupby('entry_action').agg({
                    'net_pnl': ['count', 'sum', 'mean'],
                    'close_reward': 'mean'
                }).round(4)
                
                # Flatten column names
                action_stats.columns = ['Count', 'Total_PnL', 'Avg_PnL', 'Avg_Reward']
                action_stats = action_stats.reset_index()
                
                # Create action analysis table
                action_data = [['Action Type', 'Count', 'Total P&L', 'Avg P&L', 'Avg Reward']]
                
                for _, row in action_stats.iterrows():
                    action_data.append([
                        row['entry_action'],
                        f"{int(row['Count']):,}",
                        f"${row['Total_PnL']:,.2f}",
                        f"${row['Avg_PnL']:.2f}",
                        f"{row['Avg_Reward']:.4f}"
                    ])
                
                action_table = Table(action_data, colWidths=[1*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
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
                
                story.append(Paragraph("Action Type Analysis:", self.styles['Heading3']))
                story.append(action_table)
                story.append(Spacer(1, 20))
        
        return story
    
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
    
    def _get_episode_count(self) -> int:
        """Get total number of episodes"""
        try:
            episodes_dir = Path("episodes")
            if episodes_dir.exists():
                return len([d for d in episodes_dir.iterdir() if d.is_dir()])
        except:
            pass
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


def generate_simple_pdf_report(analysis_dir: str = "DATA_ANALYSIS", episode_name: str = None) -> str:
    """Convenience function to generate simple PDF report"""
    # Handle relative paths properly
    if not os.path.isabs(analysis_dir):
        analysis_dir = str(Path.cwd() / analysis_dir)
    
    generator = SimplePDFReportGenerator(analysis_dir)
    return generator.generate_simple_report(episode_name)


if __name__ == "__main__":
    print("Starting PDF generation...")
    
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Generate simple PDF trade analysis report')
    parser.add_argument('--episode', type=str, help='Specific episode name to analyze')
    parser.add_argument('--analysis-dir', type=str, default='.', 
                       help='Analysis directory path')
    
    args = parser.parse_args()
    
    try:
        print(f"Analysis directory: {args.analysis_dir}")
        print(f"Episode: {args.episode}")
        
        # Generate report
        pdf_path = generate_simple_pdf_report(args.analysis_dir, args.episode)
        print(f"[SUCCESS] PDF report generated successfully: {pdf_path}")
        
        # Check file size
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"[INFO] File size: {file_size:,} bytes")
        else:
            print(f"[ERROR] PDF file not found at: {pdf_path}")
        
    except Exception as e:
        print(f"[ERROR] Error generating PDF report: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure analysis data exists (run comprehensive analysis first)")
        print("2. Install required package: pip install reportlab")
        print("3. Verify DATA_ANALYSIS directory contains trade data")
        sys.exit(1)
