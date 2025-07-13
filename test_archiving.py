"""
Test Script for Log Archiving Functionality
Run this to test the archiving system without starting training
"""

from rich.console import Console
from log_archiver import archive_startup_logs, LogArchiver

console = Console()

def test_archiving():
    """Test the archiving functionality"""
    console.print("[bold blue]🧪 Testing Log Archiving System[/bold blue]")
    
    # Test with very short age to archive existing files
    console.print("\n[bold]Testing with aggressive archiving (1 hour old files)...[/bold]")
    
    success = archive_startup_logs(
        base_dir=".",
        log_age_days=0.04,      # ~1 hour in days (1/24)
        model_age_days=0.04,
        tensorboard_age_days=0.04
    )
    
    if success:
        console.print("[green]✅ Archiving test completed successfully![/green]")
    else:
        console.print("[red]❌ Archiving test failed![/red]")
    
    # Show archive directory contents
    console.print("\n[bold]Archive directory contents:[/bold]")
    try:
        from pathlib import Path
        archive_dir = Path("archive")
        if archive_dir.exists():
            archive_files = list(archive_dir.glob("*.zip"))
            if archive_files:
                for archive_file in archive_files:
                    file_size = archive_file.stat().st_size / 1024  # KB
                    console.print(f"  📦 {archive_file.name} ({file_size:.1f} KB)")
            else:
                console.print("  (No archive files found)")
        else:
            console.print("  (Archive directory doesn't exist yet)")
    except Exception as e:
        console.print(f"  [red]Error reading archive directory: {e}[/red]")

if __name__ == "__main__":
    test_archiving()
