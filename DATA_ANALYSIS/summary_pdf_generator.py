#!/usr/bin/env python3
"""
Summary-Only PDF Generator
Creates a focused PDF with just the comprehensive action summary table
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.utils import ImageReader

# Import our analyzer
from clean_trade_analyzer import CleanTradeStatsAnalyzer

class SummaryOnlyPDFGenerator:
    def __init__(self, analysis_dir: str = "."):
        self.analysis_dir = Path(analysis_dir)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.darkgreen,
            spaceBefore=20,
            spaceAfter=12
        )
        
        # Subheading style
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceBefore=15,
            spaceAfter=8
        )

    def generate_summary_report(self) -> str:
        """Generate a summary-only PDF report"""
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.analysis_dir / "pdf_reports"
        output_dir.mkdir(exist_ok=True)
        
        pdf_filename = f"action_summary_report_{timestamp}.pdf"
        pdf_path = output_dir / pdf_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph("TRADING ACTION SUMMARY REPORT", self.title_style))
        story.append(Spacer(1, 20))
        
        # Timestamp
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Split Comprehensive Action Summary into Two Tables
        story.extend(self._create_split_summary_tables())
        
        # Build PDF
        doc.build(story)
        
        return str(pdf_path)
    
    def _create_comprehensive_summary_table(self):
        """Create comprehensive action-level summary table"""
        story = []
        
        story.append(Paragraph("COMPREHENSIVE ACTION SUMMARY", self.heading_style))
        story.append(Spacer(1, 12))
        
        try:
            # Initialize clean analyzer
            clean_analyzer = CleanTradeStatsAnalyzer(str(self.analysis_dir))
            
            # Analyze combined trades
            clean_stats = clean_analyzer.analyze_combined_trades()
            
            if clean_stats and 'stats' in clean_stats:
                # Generate summary table
                summary_table = clean_analyzer.generate_clean_summary_table(clean_stats['stats'])
                
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
                    total_trades = sum(stats.get('count', 0) for stats in clean_stats['stats'].values())
                    total_pnl = sum(stats.get('total_pnl', 0) for stats in clean_stats['stats'].values())
                    
                    insights = [
                        f"Total Actions Analyzed: {total_trades:,}",
                        f"Overall P&L: ${total_pnl:,.2f}",
                        f"Most Frequent Action: {max(clean_stats['stats'].keys(), key=lambda x: clean_stats['stats'][x]['count'])}",
                        f"Most Profitable Action: {max(clean_stats['stats'].keys(), key=lambda x: clean_stats['stats'][x]['total_pnl'])}"
                    ]
                    
                    for insight in insights:
                        story.append(Paragraph(f"• {insight}", self.styles['Normal']))
                        story.append(Spacer(1, 4))
                    
                    # Add action breakdown
                    story.append(Spacer(1, 20))
                    story.append(Paragraph("Action Breakdown:", self.subheading_style))
                    
                    for action_type, stats in clean_stats['stats'].items():
                        pnl = stats.get('total_pnl', 0)
                        count = stats.get('count', 0)
                        win_rate = stats.get('win_rate', 0) * 100
                        
                        breakdown_text = f"{action_type}: {count} trades, ${pnl:,.2f} P&L, {win_rate:.1f}% win rate"
                        story.append(Paragraph(f"• {breakdown_text}", self.styles['Normal']))
                        story.append(Spacer(1, 3))
                        
                else:
                    story.append(Paragraph("No clean trade data available for summary table.", self.styles['Normal']))
            else:
                story.append(Paragraph("No action statistics available.", self.styles['Normal']))
        
        except Exception as e:
            story.append(Paragraph(f"Error generating clean summary: {str(e)}", self.styles['Normal']))
            import traceback
            story.append(Paragraph(f"Details: {traceback.format_exc()}", self.styles['Code']))
        
        return story

    def _create_split_summary_tables(self):
        """Create two separate tables to avoid column overflow"""
        story = []
        
        story.append(Paragraph("ACTION SUMMARY ANALYSIS", self.heading_style))
        story.append(Spacer(1, 12))
        
        try:
            # Initialize clean analyzer
            clean_analyzer = CleanTradeStatsAnalyzer(str(self.analysis_dir))
            
            # Analyze combined trades
            clean_stats = clean_analyzer.analyze_combined_trades()
            
            if clean_stats and 'stats' in clean_stats:
                # === TABLE 1: Basic Metrics ===
                story.append(Paragraph("Table 1: Basic Action Metrics", self.subheading_style))
                story.append(Spacer(1, 8))
                
                # Create basic metrics table
                basic_headers = ['Action Type', 'Count', 'Percentage', 'Total P&L', 'Avg P&L', 'Win Rate', 'Profit Factor']
                basic_data = [basic_headers]
                
                for action_type, stats in clean_stats['stats'].items():
                    row = [
                        action_type,
                        f"{stats.get('count', 0):,}",
                        f"{stats.get('percentage', 0):.1f}%",
                        f"${stats.get('total_pnl', 0):,.2f}",
                        f"${stats.get('avg_pnl', 0):.2f}",
                        f"{stats.get('win_rate', 0)*100:.1f}%",
                        f"{stats.get('profit_factor', 0):.2f}" if stats.get('profit_factor', 0) != float('inf') else "∞"
                    ]
                    basic_data.append(row)
                
                # Create basic table
                basic_col_widths = [1.2*inch, 0.8*inch, 0.9*inch, 1.0*inch, 0.9*inch, 0.8*inch, 0.9*inch]
                basic_table = Table(basic_data, colWidths=basic_col_widths)
                basic_table.setStyle(TableStyle([
                    # Header styling
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    
                    # Data styling
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    
                    # Right-align numeric columns
                    ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),  # All numeric columns
                    ('ALIGN', (0, 1), (0, -1), 'LEFT'),   # Action type column left-aligned
                ]))
                
                story.append(basic_table)
                story.append(Spacer(1, 20))
                
                # === TABLE 2: Advanced Metrics ===
                story.append(Paragraph("Table 2: Advanced P&L and Reward Metrics", self.subheading_style))
                story.append(Spacer(1, 8))
                
                # Create advanced metrics table
                advanced_headers = ['Action Type', 'Min P&L', 'Max P&L', 'P95 P&L', 'Avg Reward', 'Min Reward', 'Max Reward', 'P95 Reward']
                advanced_data = [advanced_headers]
                
                for action_type, stats in clean_stats['stats'].items():
                    row = [
                        action_type,
                        f"${stats.get('min_pnl', 0):.2f}",
                        f"${stats.get('max_pnl', 0):.2f}",
                        f"${stats.get('p95_pnl', 0):.2f}",
                        f"{stats.get('avg_reward', 0):.6f}",
                        f"{stats.get('min_reward', 0):.6f}",
                        f"{stats.get('max_reward', 0):.6f}",
                        f"{stats.get('p95_reward', 0):.6f}"
                    ]
                    advanced_data.append(row)
                
                # Create advanced table
                advanced_col_widths = [1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch]
                advanced_table = Table(advanced_data, colWidths=advanced_col_widths)
                advanced_table.setStyle(TableStyle([
                    # Header styling
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    
                    # Data styling
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    
                    # Right-align numeric columns
                    ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),  # All numeric columns
                    ('ALIGN', (0, 1), (0, -1), 'LEFT'),   # Action type column left-aligned
                ]))
                
                story.append(advanced_table)
                story.append(Spacer(1, 20))
                
                # === SUMMARY INSIGHTS ===
                story.append(Paragraph("Summary Insights", self.subheading_style))
                story.append(Spacer(1, 8))
                
                total_trades = sum(stats.get('count', 0) for stats in clean_stats['stats'].values())
                total_pnl = sum(stats.get('total_pnl', 0) for stats in clean_stats['stats'].values())
                
                # Find best performing action
                best_action = max(clean_stats['stats'].keys(), key=lambda x: clean_stats['stats'][x]['total_pnl'])
                most_frequent = max(clean_stats['stats'].keys(), key=lambda x: clean_stats['stats'][x]['count'])
                
                insights = [
                    f"📊 Total Actions Analyzed: {total_trades:,}",
                    f"💰 Overall P&L: ${total_pnl:,.2f}",
                    f"🔥 Most Frequent Action: {most_frequent} ({clean_stats['stats'][most_frequent]['count']:,} times)",
                    f"⭐ Most Profitable Action: {best_action} (${clean_stats['stats'][best_action]['total_pnl']:,.2f})",
                    f"📈 Best Win Rate: {max(stats['win_rate']*100 for stats in clean_stats['stats'].values()):.1f}%"
                ]
                
                for insight in insights:
                    # Replace emojis with text for compatibility
                    insight_clean = insight.replace('📊', '[STATS]').replace('💰', '[MONEY]').replace('🔥', '[FREQ]').replace('⭐', '[STAR]').replace('📈', '[UP]')
                    story.append(Paragraph(f"• {insight_clean}", self.styles['Normal']))
                    story.append(Spacer(1, 4))
                    
            else:
                story.append(Paragraph("No action statistics available for analysis.", self.styles['Normal']))
        
        except Exception as e:
            story.append(Paragraph(f"Error generating split summary tables: {str(e)}", self.styles['Normal']))
            import traceback
            story.append(Paragraph(f"Details: {traceback.format_exc()}", self.styles['Code']))
        
        return story

def generate_summary_pdf(analysis_dir: str = ".") -> str:
    """Convenience function to generate summary PDF report"""
    generator = SummaryOnlyPDFGenerator(analysis_dir)
    return generator.generate_summary_report()

if __name__ == "__main__":
    try:
        pdf_path = generate_summary_pdf()
        print(f"[SUCCESS] Summary PDF generated: {pdf_path}")
        
        # Check file size
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"[FILE] File size: {file_size:,} bytes")
        
    except Exception as e:
        print(f"[ERROR] Error generating summary PDF: {str(e)}")
        import traceback
        traceback.print_exc()
