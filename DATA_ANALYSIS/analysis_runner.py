#!/usr/bin/env python3
"""
Batch Analysis Runner
====================

This script provides a convenient way to run comprehensive trade analysis
with various options and configurations.
"""

import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm, Prompt, IntPrompt
from rich.panel import Panel

console = Console()

def run_comprehensive_analysis():
    """Run the full episode trade analyzer"""
    console.print("[bold]🚀 Running Comprehensive Episode Analysis[/bold]")
    
    try:
        result = subprocess.run([sys.executable, "episode_trade_analyzer.py"], 
                              cwd="DATA_ANALYSIS", 
                              capture_output=False)
        
        if result.returncode == 0:
            console.print("[green][SUCCESS] Comprehensive analysis completed successfully![/green]")
        else:
            console.print("[red][ERROR] Analysis failed with errors[/red]")
            
    except Exception as e:
        console.print(f"[red][ERROR] Error running analysis: {str(e)}[/red]")

def run_quick_analysis():
    """Run the quick analyzer tool"""
    console.print("[bold][QUICK] Running Quick Analysis[/bold]")
    
    try:
        result = subprocess.run([sys.executable, "quick_analyzer.py", "--interactive"], 
                              cwd="DATA_ANALYSIS", 
                              capture_output=False)
        
        if result.returncode == 0:
            console.print("[green][SUCCESS] Quick analysis completed![/green]")
        else:
            console.print("[red][ERROR] Quick analysis failed[/red]")
            
    except Exception as e:
        console.print(f"[red][ERROR] Error running quick analysis: {str(e)}[/red]")

def run_clean_analysis():
    """Run clean trade statistics analyzer"""
    console.print("[bold]🧹 Running Clean Trade Analysis[/bold]")
    
    try:
        result = subprocess.run([sys.executable, "clean_trade_analyzer.py"], 
                              cwd="DATA_ANALYSIS", 
                              capture_output=False)
        
        if result.returncode == 0:
            console.print("[green][SUCCESS] Clean analysis completed![/green]")
        else:
            console.print("[red][ERROR] Clean analysis failed[/red]")
            
    except Exception as e:
        console.print(f"[red][ERROR] Error running clean analysis: {str(e)}[/red]")

def generate_pdf_report():
    """Generate comprehensive PDF report"""
    console.print("[bold][INFO] Generating PDF Report[/bold]")
    
    # Ensure directories exist first
    pdf_dir = Path("pdf_reports")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try the full PDF generator first
        result = subprocess.run([sys.executable, "pdf_report_generator.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green][SUCCESS] PDF report generated successfully![/green]")
            
            # Show where the report was saved
            pdf_dir = Path("pdf_reports")
            if pdf_dir.exists():
                pdf_files = list(pdf_dir.glob("*.pdf"))
                if pdf_files:
                    latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
                    console.print(f"[cyan][INFO] Report saved: {latest_pdf}[/cyan]")
        else:
            # If full generator fails, try simple generator
            console.print("[yellow][WARNING] Full PDF generator failed, trying simple version...[/yellow]")
            console.print(f"[red]Error details: {result.stderr}[/red]")
            console.print(f"[red]Return code: {result.returncode}[/red]")
            
            result = subprocess.run([sys.executable, "simple_pdf_generator.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("[green][SUCCESS] Simple PDF report generated successfully![/green]")
                
                # Show where the report was saved
                pdf_dir = Path("pdf_reports")
                if pdf_dir.exists():
                    pdf_files = list(pdf_dir.glob("*.pdf"))
                    if pdf_files:
                        latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
                        console.print(f"[cyan][INFO] Report saved: {latest_pdf}[/cyan]")
            else:
                console.print("[red][ERROR] Both PDF generators failed[/red]")
                console.print(f"[red][ERROR] Error: {result.stderr}[/red]")
            
    except Exception as e:
        console.print(f"[red][ERROR] Error generating PDF: {str(e)}[/red]")

def generate_custom_pdf_report():
    """Generate PDF report with custom options"""
    console.print("[bold]📄 Custom PDF Report Generator[/bold]")
    
    # Ask for report type
    report_type = Prompt.ask(
        "Select report type",
        choices=["summary", "full"],
        default="summary"
    )
    
    if report_type == "summary":
        # Generate summary-only report
        try:
            result = subprocess.run([sys.executable, "summary_pdf_generator.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("[green][SUCCESS] Summary PDF report generated successfully![/green]")
                
                # Show where the report was saved
                pdf_dir = Path("pdf_reports")
                if pdf_dir.exists():
                    pdf_files = list(pdf_dir.glob("*summary*.pdf"))
                    if pdf_files:
                        latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
                        console.print(f"[cyan][INFO] Report saved: {latest_pdf}[/cyan]")
            else:
                console.print("[red][ERROR] Summary PDF generation failed[/red]")
                console.print(f"[red]Error: {result.stderr}[/red]")
        except Exception as e:
            console.print(f"[red][ERROR] Error: {str(e)}[/red]")
        return
    
    # Full report generation logic
    # Ask for episode selection
    episode_choice = Prompt.ask(
        "Select report scope",
        choices=["all", "specific", "latest"],
        default="all"
    )
    
    episode_name = None
    if episode_choice == "specific":
        # List available episodes
        episodes_dir = Path("episodes")
        if episodes_dir.exists():
            episode_dirs = [d.name for d in episodes_dir.iterdir() if d.is_dir()]
            if episode_dirs:
                console.print("\n[bold]Available episodes:[/bold]")
                for i, ep in enumerate(episode_dirs, 1):
                    console.print(f"  {i}. {ep}")
                
                try:
                    choice_idx = IntPrompt.ask(
                        "Select episode number",
                        default=1,
                        show_default=True
                    ) - 1
                    
                    if 0 <= choice_idx < len(episode_dirs):
                        episode_name = episode_dirs[choice_idx]
                    else:
                        console.print("[yellow]Invalid selection, using all episodes[/yellow]")
                except:
                    console.print("[yellow]Invalid input, using all episodes[/yellow]")
    
    elif episode_choice == "latest":
        # Find the latest episode
        episodes_dir = Path("episodes")
        if episodes_dir.exists():
            episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
            if episode_dirs:
                latest_episode = max(episode_dirs, key=lambda x: x.stat().st_mtime)
                episode_name = latest_episode.name
                console.print(f"[cyan]Using latest episode: {episode_name}[/cyan]")
    
    # Generate report
    try:
        if episode_name:
            result = subprocess.run([
                sys.executable, "pdf_report_generator.py", 
                "--episode", episode_name
            ], cwd="DATA_ANALYSIS", capture_output=False)
        else:
            result = subprocess.run([sys.executable, "pdf_report_generator.py"], 
                                  cwd="DATA_ANALYSIS", capture_output=False)
        
        if result.returncode == 0:
            console.print("[green][SUCCESS] Custom PDF report generated successfully![/green]")
        else:
            console.print("[red][ERROR] PDF generation failed[/red]")
            
    except Exception as e:
        console.print(f"[red][ERROR] Error generating custom PDF: {str(e)}[/red]")

def view_analysis_results():
    """View existing analysis results"""
    analysis_dir = Path("DATA_ANALYSIS")
    
    if not analysis_dir.exists():
        console.print("[yellow]No analysis directory found. Run analysis first.[/yellow]")
        return
    
    console.print("[bold]📁 Analysis Results Directory Structure:[/bold]")
    
    subdirs = {
        "reports": "📊 Detailed JSON analysis reports",
        "extracts": "🏆 Top profit/loss trade extracts", 
        "anomalies": "[WARNING]  Detected reward-PnL anomalies",
        "summaries": "📈 CSV summaries by action type",
        "pdf_reports": "📄 Generated PDF reports"
    }
    
    for subdir, description in subdirs.items():
        subdir_path = analysis_dir / subdir
        if subdir_path.exists():
            file_count = len(list(subdir_path.glob("*")))
            console.print(f"   {description}: {file_count} files")
        else:
            console.print(f"   {description}: [dim]No files[/dim]")
    
    # Show recent files
    recent_files = []
    for pattern in ["*.json", "*.csv", "*.pdf"]:
        recent_files.extend(analysis_dir.rglob(pattern))
    
    if recent_files:
        # Sort by modification time
        recent_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        console.print(f"\n[bold]📅 Recent Analysis Files (Top 10):[/bold]")
        for i, file_path in enumerate(recent_files[:10]):
            rel_path = file_path.relative_to(analysis_dir)
            console.print(f"   {i+1}. {rel_path}")

def clean_analysis_results():
    """Clean old analysis results"""
    analysis_dir = Path("DATA_ANALYSIS")
    
    if not analysis_dir.exists():
        console.print("[yellow]No analysis directory found.[/yellow]")
        return
    
    # Count files to be cleaned
    file_patterns = ["*.json", "*.csv"]
    files_to_clean = []
    
    for pattern in file_patterns:
        files_to_clean.extend(analysis_dir.rglob(pattern))
    
    if not files_to_clean:
        console.print("[green]No analysis files to clean.[/green]")
        return
    
    console.print(f"[yellow]Found {len(files_to_clean)} analysis files to clean:[/yellow]")
    
    # Group by subdirectory
    by_subdir = {}
    for file_path in files_to_clean:
        subdir = file_path.parent.name
        if subdir not in by_subdir:
            by_subdir[subdir] = []
        by_subdir[subdir].append(file_path)
    
    for subdir, files in by_subdir.items():
        console.print(f"   {subdir}: {len(files)} files")
    
    if Confirm.ask("Delete all analysis files?"):
        deleted_count = 0
        for file_path in files_to_clean:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                console.print(f"[red]Error deleting {file_path}: {str(e)}[/red]")
        
        console.print(f"[green][SUCCESS] Deleted {deleted_count} files[/green]")
    else:
        console.print("[yellow]Cleanup cancelled[/yellow]")

def setup_analysis_environment():
    """Setup analysis environment and dependencies"""
    console.print("[bold]🔧 Setting up Analysis Environment[/bold]")
    
    # Check if DATA_ANALYSIS directory exists
    analysis_dir = Path("DATA_ANALYSIS")
    if not analysis_dir.exists():
        console.print("[yellow]Creating DATA_ANALYSIS directory...[/yellow]")
        analysis_dir.mkdir()
    
    # Create subdirectories
    subdirs = ["reports", "extracts", "anomalies", "summaries", "quick_analysis", "pdf_reports"]
    for subdir in subdirs:
        subdir_path = analysis_dir / subdir
        if not subdir_path.exists():
            console.print(f"[yellow]Creating {subdir} directory...[/yellow]")
            subdir_path.mkdir()
    
    console.print("[green][SUCCESS] Analysis environment ready![/green]")

def install_pdf_dependencies():
    """Install required dependencies for PDF generation"""
    console.print("[bold]📦 Installing PDF Generation Dependencies[/bold]")
    
    required_packages = [
        "matplotlib",
        "seaborn", 
        "reportlab",
        "pillow"
    ]
    
    console.print("[yellow]Installing required packages...[/yellow]")
    
    try:
        for package in required_packages:
            console.print(f"Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[green][SUCCESS] {package} installed successfully[/green]")
            else:
                console.print(f"[red][ERROR] Failed to install {package}: {result.stderr}[/red]")
    
    except Exception as e:
        console.print(f"[red][ERROR] Error installing dependencies: {str(e)}[/red]")

def run_enhanced_trade_analysis():
    """Run the enhanced trade analyzer on a specific file"""
    console.print("[bold]🔬 Enhanced Trade Analysis[/bold]")
    
    # Find available trade files
    episodes_dir = Path("../episodes")
    if not episodes_dir.exists():
        console.print("[red][ERROR] Episodes directory not found[/red]")
        return
    
    trade_files = []
    for episode_dir in episodes_dir.iterdir():
        if episode_dir.is_dir():
            logs_dir = episode_dir / "logs"
            if logs_dir.exists():
                for file in logs_dir.glob("trades_*.csv"):
                    trade_files.append(file)
    
    if not trade_files:
        console.print("[red][ERROR] No trade files found in episodes[/red]")
        return
    
    # Display available files
    console.print("\n[cyan]Available trade files:[/cyan]")
    for i, file in enumerate(trade_files):
        file_size = file.stat().st_size / 1024  # KB
        console.print(f"  {i+1}. {file.name} ({file_size:.1f} KB)")
    
    try:
        choice = IntPrompt.ask("Select file to analyze", default=1)
        if 1 <= choice <= len(trade_files):
            selected_file = trade_files[choice-1]
            
            console.print(f"[cyan]Analyzing: {selected_file.name}[/cyan]")
            
            # Run enhanced analyzer
            result = subprocess.run([
                sys.executable, 
                "enhanced_trade_analyzer.py", 
                str(selected_file)
            ], cwd="DATA_ANALYSIS", capture_output=False)
            
            if result.returncode == 0:
                console.print("[green][SUCCESS] Enhanced analysis completed successfully![/green]")
            else:
                console.print("[red][ERROR] Enhanced analysis failed[/red]")
        else:
            console.print("[red][ERROR] Invalid selection[/red]")
            
    except Exception as e:
        console.print(f"[red][ERROR] Error running enhanced analysis: {str(e)}[/red]")

def main():
    """Main menu interface"""
    
    while True:
        console.clear()
        
        # Display title
        title_panel = Panel.fit(
            "[bold]📊 Trade Analysis Tool Suite[/bold]\n"
            "Comprehensive analysis of episode trade logs",
            border_style="blue"
        )
        console.print(title_panel)
        
        # Menu options
        menu_options = [
            "1. 🚀 Run Comprehensive Analysis (All Episodes)",
            "2. [QUICK] Run Quick Analysis (Single File)",
            "3. 🧹 Run Clean Trade Analysis (Filtered)",
            "4. 📄 Generate PDF Report (Standard)",
            "5. 📋 Generate Custom PDF Report",
            "6. 📁 View Analysis Results",
            "7. 🗑️  Clean Analysis Results", 
            "8. 🔧 Setup Analysis Environment",
            "9. 📦 Install PDF Dependencies",
            "10. 🔬 Run Enhanced Trade Analysis",
            "11. ❌ Exit"
        ]
        
        console.print("\n[bold]Select an option:[/bold]")
        for option in menu_options:
            console.print(f"   {option}")
        
        choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"], default="1")
        
        if choice == "1":
            run_comprehensive_analysis()
        elif choice == "2":
            run_quick_analysis()
        elif choice == "3":
            run_clean_analysis()
        elif choice == "4":
            generate_pdf_report()
        elif choice == "5":
            generate_custom_pdf_report()
        elif choice == "6":
            view_analysis_results()
        elif choice == "7":
            clean_analysis_results()
        elif choice == "8":
            setup_analysis_environment()
        elif choice == "9":
            install_pdf_dependencies()
        elif choice == "10":
            run_enhanced_trade_analysis()
        elif choice == "11":
            console.print("[yellow]Goodbye! 👋[/yellow]")
            break
        
        # Wait for user input before continuing
        if choice != "11":
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()

if __name__ == "__main__":
    main()
