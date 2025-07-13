"""
Log Archiving Utility for TensorTradeModel
Automatically archives old logs when training starts
"""

import os
import shutil
import zipfile
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Rich library for beautiful console output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def load_archive_config(config_path: str = "archive_config.json") -> Dict[str, Any]:
    """Load archiving configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default config if file doesn't exist or is invalid
        return {
            "archiving": {
                "enabled": True,
                "log_age_days": 3,
                "model_age_days": 14,
                "tensorboard_age_days": 7,
                "max_archives": 15,
                "keep_latest_logs": 3,
                "keep_latest_models": 5,
                "keep_latest_tensorboard": 2,
                "exclude_models": ["best_model.zip"]
            },
            "startup_settings": {
                "archive_on_main_start": True,
                "archive_on_training_start": True,
                "show_archive_progress": True
            }
        }

class LogArchiver:
    """Utility class for archiving logs and models"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.archive_dir = self.base_dir / "archive"
        self.logs_dir = self.base_dir / "logs"
        self.models_dir = self.base_dir / "models"
        self.tensorboard_dir = self.base_dir / "tensorboard_logs"
        
        # Create archive directory if it doesn't exist
        self.archive_dir.mkdir(exist_ok=True)
        
    def archive_logs(self, max_age_days: int = 7, keep_latest: int = 3) -> bool:
        """
        Archive old log files
        
        Args:
            max_age_days: Archive files older than this many days
            keep_latest: Always keep this many latest files
            
        Returns:
            bool: True if archiving was successful
        """
        try:
            console.print("\n[bold]📦 Archiving Old Logs...[/bold]")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"logs_archive_{timestamp}.zip"
            archive_path = self.archive_dir / archive_name
            
            # Get all log files
            log_files = self._get_files_to_archive(
                self.logs_dir, 
                patterns=["*.csv", "*.log", "*.npz"],
                max_age_days=max_age_days,
                keep_latest=keep_latest
            )
            
            if not log_files:
                console.print("[green]✅ No old logs to archive[/green]")
                return True
                
            # Create archive
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Creating log archive...", total=len(log_files))
                
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in log_files:
                        # Calculate relative path for archive
                        rel_path = file_path.relative_to(self.base_dir)
                        zipf.write(file_path, rel_path)
                        progress.update(task, advance=1)
                        
                        # Remove original file after archiving
                        file_path.unlink()
                        
            console.print(f"[green]✅ Archived {len(log_files)} log files to:[/green] [blue]{archive_path}[/blue]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error archiving logs: {str(e)}[/red]")
            return False
    
    def archive_old_models(self, max_age_days: int = 30, keep_latest: int = 5) -> bool:
        """
        Archive old model files
        
        Args:
            max_age_days: Archive files older than this many days
            keep_latest: Always keep this many latest files
            
        Returns:
            bool: True if archiving was successful
        """
        try:
            console.print("\n[bold]📦 Archiving Old Models...[/bold]")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"models_archive_{timestamp}.zip"
            archive_path = self.archive_dir / archive_name
            
            # Get all model files (excluding best_model.zip which should always be kept)
            model_files = self._get_files_to_archive(
                self.models_dir,
                patterns=["*.zip", "*.pkl"],
                max_age_days=max_age_days,
                keep_latest=keep_latest,
                exclude_patterns=["best_model.zip"]
            )
            
            if not model_files:
                console.print("[green]✅ No old models to archive[/green]")
                return True
                
            # Create archive
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Creating model archive...", total=len(model_files))
                
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in model_files:
                        # Calculate relative path for archive
                        rel_path = file_path.relative_to(self.base_dir)
                        zipf.write(file_path, rel_path)
                        progress.update(task, advance=1)
                        
                        # Remove original file after archiving
                        file_path.unlink()
                        
            console.print(f"[green]✅ Archived {len(model_files)} model files to:[/green] [blue]{archive_path}[/blue]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error archiving models: {str(e)}[/red]")
            return False
    
    def archive_tensorboard_logs(self, max_age_days: int = 14, keep_latest: int = 2) -> bool:
        """
        Archive old TensorBoard logs
        
        Args:
            max_age_days: Archive directories older than this many days
            keep_latest: Always keep this many latest directories
            
        Returns:
            bool: True if archiving was successful
        """
        try:
            console.print("\n[bold]📦 Archiving Old TensorBoard Logs...[/bold]")
            
            if not self.tensorboard_dir.exists():
                console.print("[green]✅ No TensorBoard logs to archive[/green]")
                return True
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"tensorboard_archive_{timestamp}.zip"
            archive_path = self.archive_dir / archive_name
            
            # Get directories to archive
            dirs_to_archive = self._get_dirs_to_archive(
                self.tensorboard_dir,
                max_age_days=max_age_days,
                keep_latest=keep_latest
            )
            
            if not dirs_to_archive:
                console.print("[green]✅ No old TensorBoard logs to archive[/green]")
                return True
                
            # Create archive
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Creating TensorBoard archive...", total=len(dirs_to_archive))
                
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for dir_path in dirs_to_archive:
                        # Add all files in the directory
                        for file_path in dir_path.rglob("*"):
                            if file_path.is_file():
                                rel_path = file_path.relative_to(self.base_dir)
                                zipf.write(file_path, rel_path)
                        
                        progress.update(task, advance=1)
                        
                        # Remove original directory after archiving
                        shutil.rmtree(dir_path)
                        
            console.print(f"[green]✅ Archived {len(dirs_to_archive)} TensorBoard directories to:[/green] [blue]{archive_path}[/blue]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error archiving TensorBoard logs: {str(e)}[/red]")
            return False
    
    def archive_all(self, 
                   log_age_days: int = 7, 
                   model_age_days: int = 30, 
                   tensorboard_age_days: int = 14) -> bool:
        """
        Archive all old files
        
        Args:
            log_age_days: Archive log files older than this many days
            model_age_days: Archive model files older than this many days
            tensorboard_age_days: Archive TensorBoard logs older than this many days
            
        Returns:
            bool: True if all archiving was successful
        """
        results = []
        
        # Archive logs
        results.append(self.archive_logs(max_age_days=log_age_days))
        
        # Archive models
        results.append(self.archive_old_models(max_age_days=model_age_days))
        
        # Archive TensorBoard logs
        results.append(self.archive_tensorboard_logs(max_age_days=tensorboard_age_days))
        
        success = all(results)
        if success:
            console.print("\n[bold green]🎉 All archiving completed successfully![/bold green]")
        else:
            console.print("\n[bold yellow]⚠️ Some archiving operations had issues[/bold yellow]")
            
        return success
    
    def archive_everything_now(self) -> bool:
        """
        Archive ALL files immediately, ignoring age and count restrictions
        
        Returns:
            bool: True if archiving was successful
        """
        results = []
        
        # Archive ALL logs (0 age, 0 keep latest)
        results.append(self.archive_logs(max_age_days=0, keep_latest=0))
        
        # Archive ALL models (0 age, 0 keep latest) 
        results.append(self.archive_old_models(max_age_days=0, keep_latest=0))
        
        # Archive ALL TensorBoard logs (0 age, 0 keep latest)
        results.append(self.archive_tensorboard_logs(max_age_days=0, keep_latest=0))
        
        success = all(results)
        if success:
            console.print("\n[bold green]🎉 Complete archiving finished - ALL files archived![/bold green]")
        else:
            console.print("\n[bold yellow]⚠️ Some archiving operations had issues[/bold yellow]")
            
        return success

    def _get_files_to_archive(self, 
                             directory: Path, 
                             patterns: List[str],
                             max_age_days: int,
                             keep_latest: int,
                             exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Get files that should be archived based on age and count criteria"""
        if not directory.exists():
            return []
            
        exclude_patterns = exclude_patterns or []
        files = []
        
        # Collect all matching files
        for pattern in patterns:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    # Check if file should be excluded
                    excluded = any(file_path.match(exclude) for exclude in exclude_patterns)
                    if not excluded:
                        files.append(file_path)
        
        if not files:
            return []
            
        # Sort by modification time (newest first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Always keep the latest N files
        files_to_check = files[keep_latest:]
          # Filter by age
        files_to_archive = []
        
        # Special case: max_age_days=0 means archive everything (regardless of age)
        if max_age_days == 0:
            files_to_archive = files_to_check
        else:
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            for file_path in files_to_check:
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    files_to_archive.append(file_path)
                
        return files_to_archive
    
    def _get_dirs_to_archive(self, 
                            directory: Path, 
                            max_age_days: int,
                            keep_latest: int) -> List[Path]:
        """Get directories that should be archived based on age and count criteria"""
        if not directory.exists():
            return []
            
        # Get all subdirectories
        dirs = [d for d in directory.iterdir() if d.is_dir()]
        
        if not dirs:
            return []
            
        # Sort by modification time (newest first)
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        
        # Always keep the latest N directories
        dirs_to_check = dirs[keep_latest:]
          # Filter by age
        dirs_to_archive = []
        
        # Special case: max_age_days=0 means archive everything (regardless of age)
        if max_age_days == 0:
            dirs_to_archive = dirs_to_check
        else:
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            for dir_path in dirs_to_check:
                dir_age = current_time - dir_path.stat().st_mtime
                if dir_age > max_age_seconds:
                    dirs_to_archive.append(dir_path)
                
        return dirs_to_archive
    
    def cleanup_old_archives(self, max_archives: int = 10) -> bool:
        """
        Keep only the latest N archive files to prevent archive directory from growing too large
        
        Args:
            max_archives: Maximum number of archive files to keep
            
        Returns:
            bool: True if cleanup was successful
        """
        try:
            archive_files = list(self.archive_dir.glob("*.zip"))
            
            if len(archive_files) <= max_archives:
                return True
                
            # Sort by modification time (newest first)
            archive_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove old archives
            files_to_remove = archive_files[max_archives:]
            for file_path in files_to_remove:
                file_path.unlink()
                console.print(f"[yellow]🗑️ Removed old archive:[/yellow] [blue]{file_path.name}[/blue]")
                
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error cleaning up old archives: {str(e)}[/red]")
            return False

def archive_startup_logs(base_dir: str = ".", 
                        log_age_days: Optional[int] = None, 
                        model_age_days: Optional[int] = None, 
                        tensorboard_age_days: Optional[int] = None,
                        keep_latest_logs: Optional[int] = None,
                        keep_latest_models: Optional[int] = None,
                        keep_latest_tensorboard: Optional[int] = None,
                        config_path: str = "archive_config.json") -> bool:
    """
    Convenience function to archive logs at startup
    
    Args:
        base_dir: Base directory of the project
        log_age_days: Archive log files older than this many days (overrides config)
        model_age_days: Archive model files older than this many days (overrides config)
        tensorboard_age_days: Archive TensorBoard logs older than this many days (overrides config)
        keep_latest_logs: Number of latest log files to keep (overrides config)
        keep_latest_models: Number of latest model files to keep (overrides config)
        keep_latest_tensorboard: Number of latest tensorboard dirs to keep (overrides config)
        config_path: Path to archive configuration file
        
    Returns:
        bool: True if archiving was successful
    """
    # Load configuration
    config = load_archive_config(config_path)
    archiving_config = config.get("archiving", {})
    startup_config = config.get("startup_settings", {})
    
    # Check if archiving is enabled
    if not archiving_config.get("enabled", True):
        console.print("[yellow]ℹ️  Log archiving is disabled in configuration[/yellow]")
        return True
    
    # Use provided values or fall back to config
    log_age = log_age_days if log_age_days is not None else archiving_config.get("log_age_days", 3)
    model_age = model_age_days if model_age_days is not None else archiving_config.get("model_age_days", 14)
    tensorboard_age = tensorboard_age_days if tensorboard_age_days is not None else archiving_config.get("tensorboard_age_days", 7)
    
    archiver = LogArchiver(base_dir)
    
    # Archive old files
    success = archiver.archive_all(
        log_age_days=log_age,
        model_age_days=model_age, 
        tensorboard_age_days=tensorboard_age
    )
    
    # Cleanup old archives
    max_archives = archiving_config.get("max_archives", 15)
    archiver.cleanup_old_archives(max_archives=max_archives)
    
    return success

if __name__ == "__main__":
    # Test the archiver
    archive_startup_logs()
