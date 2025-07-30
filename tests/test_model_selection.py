#!/usr/bin/env python3
"""
Test the improved model selection display
"""

import sys
import os
from pathlib import Path

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_selection():
    """Test the improved model selection display"""
    try:
        from multi_episode_training import get_existing_models
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        
        console.print("🧪 Testing Improved Model Selection...")
        
        # Get models using the new filtering logic
        existing_models = get_existing_models()
        
        console.print(f"\n📊 Found {len(existing_models)} important models (filtered from all available)")
        
        if existing_models:
            # Display the filtered models
            model_table = Table(title="Important Available Models (Test)")
            model_table.add_column("Index", style="cyan", no_wrap=True)
            model_table.add_column("Model Name", style="green")
            model_table.add_column("Type", style="yellow")
            model_table.add_column("Episode/Location", style="blue")
            model_table.add_column("Size (MB)", style="magenta")
            model_table.add_column("Modified", style="white")
            
            # Sort models by priority (lower is better) and modification time (newer first)
            model_list = list(existing_models.items())
            model_list.sort(key=lambda x: (x[1]['priority'], -x[1]['modified'].timestamp()))
            
            for i, (rel_path, info) in enumerate(model_list):
                # Determine model type for display
                if "best_" in info['name'] or info['episode'] == 'models':
                    model_type = "🏆 Best"
                elif "final_" in info['name']:
                    model_type = "✅ Final"
                elif "interrupted_" in info['name']:
                    model_type = "⚠️ Interrupted"
                elif "checkpoint_" in info['name']:
                    model_type = "📝 Checkpoint"
                else:
                    model_type = "📦 Model"
                
                # Truncate long model names for better display
                display_name = info['name']
                if len(display_name) > 45:
                    display_name = display_name[:42] + "..."
                
                model_table.add_row(
                    str(i+1),
                    display_name,
                    model_type,
                    info['episode'],
                    f"{info['size_mb']:.1f}",
                    info['modified'].strftime("%m-%d %H:%M")
                )
            
            console.print(model_table)
            
            # Show helpful info about model types
            console.print("\n[dim]Model Types:[/dim]")
            console.print("[dim]🏆 Best = Highest performing models[/dim]")
            console.print("[dim]✅ Final = Completed episode models[/dim]") 
            console.print("[dim]⚠️ Interrupted = Partially trained models[/dim]")
            console.print("[dim]📝 Checkpoint = Latest checkpoint per episode[/dim]")
            
        else:
            console.print("No models found.")
        
        console.print("\n✅ Model selection filtering test completed!")
        return True
        
    except Exception as e:
        console.print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_selection()
    sys.exit(0 if success else 1)
