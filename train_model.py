"""
Interactive Training Script for Binance Futures Trading Bot
Supports multiple model architectures and configurable parameters
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Rich library for beautiful console output
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# ML libraries
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Local imports
from trading_environment import FuturesTradingEnv
from model_architectures import CNNLSTMFeatureExtractor, AttentionCNNLSTMExtractor, ResNetLSTMExtractor

console = Console()

class TradingProgressCallback(BaseCallback):
    """Custom callback for training progress with status updates"""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.progress = None
        self.task_id = None
        
    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            "Training RL Agent...", 
            total=self.locals.get('total_timesteps', 1000000)
        )
    
    def _on_step(self) -> bool:
        """Called after each step"""
        if self.n_calls % self.check_freq == 0:
            if self.progress and self.task_id is not None:
                self.progress.update(self.task_id, completed=self.num_timesteps)
                
                # Log current performance
                if hasattr(self.training_env, 'get_attr'):
                    try:
                        env_stats = self.training_env.get_attr('_get_info')[0]
                        equity = env_stats.get('equity', 0)
                        trades = env_stats.get('episode_trades', 0)
                        
                        self.progress.update(
                            self.task_id, 
                            description=f"Training - Equity: ${equity:.2f}, Trades: {trades}"
                        )
                    except:
                        pass
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training"""
        if self.progress:
            self.progress.stop()

class ModelConfig:
    """Configuration class for model parameters"""
    
    AVAILABLE_MODELS = {
        "cnn_lstm": {
            "name": "CNN-LSTM",
            "class": CNNLSTMFeatureExtractor,
            "description": "Hybrid CNN-LSTM architecture for temporal feature extraction"
        },
        "attention_cnn_lstm": {
            "name": "Attention CNN-LSTM", 
            "class": AttentionCNNLSTMExtractor,
            "description": "CNN-LSTM with multi-head attention mechanism"
        },
        "resnet_lstm": {
            "name": "ResNet-LSTM",
            "class": ResNetLSTMExtractor,
            "description": "ResNet-style CNN with LSTM for robust feature learning"
        }
    }
    
    AVAILABLE_ALGORITHMS = {
        "ppo": {"name": "PPO", "class": PPO, "description": "Proximal Policy Optimization"},
        "a2c": {"name": "A2C", "class": A2C, "description": "Advantage Actor-Critic"},
        "sac": {"name": "SAC", "class": SAC, "description": "Soft Actor-Critic (continuous actions)"}
    }

def display_welcome():
    """Display welcome message and system info"""
    welcome_text = Text()
    welcome_text.append("🚀 Binance Futures Trading Bot Trainer 🚀\n", style="bold blue")
    welcome_text.append("Advanced Reinforcement Learning for Cryptocurrency Trading\n", style="cyan")
    welcome_text.append("Powered by TensorTrade, Stable-Baselines3, and PyTorch", style="green")
    
    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))
    
    # System info
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    console.print(f"🖥️  Computing Device: [bold]{device}[/bold]")
    if torch.cuda.is_available():
        console.print(f"🔥 GPU: {torch.cuda.get_device_name()}")
    console.print()

def get_data_files() -> Dict[str, str]:
    """Get available data files from the data directory"""
    data_dir = Path("data")
    csv_files = {}
    
    if data_dir.exists():
        for file in data_dir.rglob("*.csv"):
            relative_path = str(file.relative_to(data_dir))
            csv_files[relative_path] = str(file)
    
    return csv_files

def select_data_file() -> str:
    """Interactive data file selection"""
    console.print("[bold]📊 Available Data Files:[/bold]")
    
    data_files = get_data_files()
    
    if not data_files:
        console.print("[red]❌ No CSV files found in data directory![/red]")
        return None
    
    # Display available files
    table = Table(title="Data Files")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("File", style="green")
    table.add_column("Path", style="yellow")
    
    file_list = list(data_files.items())
    for i, (name, path) in enumerate(file_list):
        table.add_row(str(i+1), name, path)
    
    console.print(table)
    
    while True:
        try:
            choice = IntPrompt.ask(
                "\n🎯 Select data file (enter number)",
                default=1,
                show_default=True
            )
            if 1 <= choice <= len(file_list):
                selected_file = file_list[choice-1][1]
                console.print(f"✅ Selected: [green]{selected_file}[/green]")
                return selected_file
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
        except KeyboardInterrupt:
            console.print("\n[red]Cancelled by user[/red]")
            return None

def select_model_architecture() -> Dict[str, Any]:
    """Interactive model architecture selection"""
    console.print("\n[bold]🧠 Available Model Architectures:[/bold]")
    
    table = Table(title="Model Architectures")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Description", style="yellow")
    
    models = list(ModelConfig.AVAILABLE_MODELS.items())
    for i, (key, model_info) in enumerate(models):
        table.add_row(str(i+1), model_info["name"], model_info["description"])
    
    console.print(table)
    
    while True:
        try:
            choice = IntPrompt.ask(
                "\n🎯 Select model architecture (enter number)",
                default=1,
                show_default=True
            )
            if 1 <= choice <= len(models):
                selected_model = models[choice-1]
                console.print(f"✅ Selected: [green]{selected_model[1]['name']}[/green]")
                return selected_model
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
        except KeyboardInterrupt:
            console.print("\n[red]Cancelled by user[/red]")
            return None

def select_algorithm() -> Dict[str, Any]:
    """Interactive algorithm selection"""
    console.print("\n[bold]⚡ Available RL Algorithms:[/bold]")
    
    table = Table(title="RL Algorithms")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Description", style="yellow")
    
    algorithms = list(ModelConfig.AVAILABLE_ALGORITHMS.items())
    for i, (key, algo_info) in enumerate(algorithms):
        table.add_row(str(i+1), algo_info["name"], algo_info["description"])
    
    console.print(table)
    
    while True:
        try:
            choice = IntPrompt.ask(
                "\n🎯 Select RL algorithm (enter number)",
                default=1,
                show_default=True
            )
            if 1 <= choice <= len(algorithms):
                selected_algo = algorithms[choice-1]
                console.print(f"✅ Selected: [green]{selected_algo[1]['name']}[/green]")
                return selected_algo
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
        except KeyboardInterrupt:
            console.print("\n[red]Cancelled by user[/red]")
            return None

def get_training_parameters() -> Dict[str, Any]:
    """Get training parameters from user"""
    console.print("\n[bold]⚙️  Training Parameters:[/bold]")
    
    params = {}
    
    # Training steps
    params['total_timesteps'] = IntPrompt.ask(
        "🎯 Total training timesteps",
        default=1000000,
        show_default=True
    )
    
    # Leverage
    params['max_leverage'] = FloatPrompt.ask(
        "💪 Maximum leverage",
        default=25.0,
        show_default=True
    )
    
    # Initial equity
    params['initial_equity'] = FloatPrompt.ask(
        "💰 Initial equity (USDT)",
        default=10000.0,
        show_default=True
    )
    
    # Window size
    params['window_size'] = IntPrompt.ask(
        "📊 Lookback window size",
        default=60,
        show_default=True
    )
    
    # Stop loss percentage
    params['stop_loss_pct'] = FloatPrompt.ask(
        "🛑 Stop loss percentage (0.02 = 2%)",
        default=0.02,
        show_default=True
    )
    
    # Take profit percentage
    params['take_profit_pct'] = FloatPrompt.ask(
        "🎯 Take profit percentage (0.04 = 4%)",
        default=0.04,
        show_default=True
    )
    
    # Number of parallel environments
    params['n_envs'] = IntPrompt.ask(
        "🔄 Number of parallel environments",
        default=4,
        show_default=True
    )
    
    return params

def check_existing_models() -> Optional[str]:
    """Check for existing trained models"""
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        return None
    
    console.print("\n[bold]🎯 Existing Models Found:[/bold]")
    
    table = Table(title="Trained Models")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Modified", style="yellow")
    
    for i, model_file in enumerate(model_files):
        mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
        table.add_row(str(i+1), model_file.name, mod_time.strftime("%Y-%m-%d %H:%M"))
    
    console.print(table)
    
    # Ask if user wants to continue training existing model
    if Confirm.ask("\n🔄 Continue training an existing model?", default=False):
        while True:
            try:
                choice = IntPrompt.ask(
                    "🎯 Select model to continue (enter number)",
                    default=1,
                    show_default=True
                )
                if 1 <= choice <= len(model_files):
                    selected_model = model_files[choice-1]
                    console.print(f"✅ Selected: [green]{selected_model.name}[/green]")
                    return str(selected_model)
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
            except KeyboardInterrupt:
                console.print("\n[red]Cancelled by user[/red]")
                break
    
    return None

def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate data file"""
    console.print(f"\n[bold]📈 Loading data from: [green]{file_path}[/green][/bold]")
    
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            console.print(f"[red]❌ Missing required columns: {missing_columns}[/red]")
            return None
        
        # Convert timestamp to datetime for display
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        console.print(f"✅ Loaded {len(df)} rows of data")
        console.print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        console.print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        console.print(f"[red]❌ Error loading data: {str(e)}[/red]")
        return None

def create_environment(df: pd.DataFrame, params: Dict[str, Any], log_file: str = None, training_iteration: int = 0):
    """Create trading environment with specified parameters"""
    return FuturesTradingEnv(
        df=df,
        initial_equity=params['initial_equity'],
        max_leverage=params['max_leverage'],
        window_size=params['window_size'],
        stop_loss_pct=params['stop_loss_pct'],
        take_profit_pct=params['take_profit_pct'],
        log_file=log_file,
        training_iteration=training_iteration
    )

def save_config(config: Dict[str, Any], filename: str):
    """Save training configuration"""
    os.makedirs("configs", exist_ok=True)
    config_path = f"configs/{filename}"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    console.print(f"💾 Configuration saved to: [green]{config_path}[/green]")

def train_model(
    model_class,
    env_fn,
    model_config: Dict[str, Any],
    training_params: Dict[str, Any],
    existing_model_path: Optional[str] = None
):
    """Train the RL model"""
    console.print("\n[bold]🚀 Starting Training Process...[/bold]")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Setup vectorized environment
    with console.status("[bold green]Setting up training environment..."):
        vec_env = make_vec_env(env_fn, n_envs=training_params['n_envs'])
    
    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": model_config[1]["class"],
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 128]  # Additional network layers
    }
    
    # Initialize or load model
    if existing_model_path and os.path.exists(existing_model_path):
        console.print(f"🔄 Loading existing model: [green]{existing_model_path}[/green]")
        model = model_class[1]["class"].load(existing_model_path, env=vec_env)
        # Update tensorboard log
        model.tensorboard_log = "./tensorboard_logs/"
    else:
        console.print("🆕 Creating new model...")
        model = model_class[1]["class"](
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048 if model_class[0] == "ppo" else 256,
            device="auto"
        )
    
    # Setup callbacks
    progress_callback = TradingProgressCallback(check_freq=1000)
    
    # Create evaluation environment
    eval_env = env_fn()
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Start training
    try:
        model.learn(
            total_timesteps=training_params['total_timesteps'],
            callback=[progress_callback, eval_callback],
            progress_bar=False  # We use our custom progress bar
        )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"trading_bot_{model_config[0]}_{timestamp}"
        model_path = f"models/{model_name}"
        model.save(model_path)
        
        console.print(f"\n✅ [bold green]Training completed successfully![/bold green]")
        console.print(f"💾 Model saved to: [green]{model_path}.zip[/green]")
        
        return model_path
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Training interrupted by user[/yellow]")
        # Save current model state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"trading_bot_interrupted_{timestamp}"
        model_path = f"models/{model_name}"
        model.save(model_path)
        console.print(f"💾 Model saved to: [green]{model_path}.zip[/green]")
        return model_path
        
    except Exception as e:
        console.print(f"[red]❌ Training failed: {str(e)}[/red]")
        return None

def main():
    """Main training function"""
    display_welcome()
    
    try:
        # Step 1: Select data file
        data_file = select_data_file()
        if not data_file:
            return
        
        # Step 2: Load and validate data
        df = load_data(data_file)
        if df is None:
            return
        
        # Step 3: Check for existing models
        existing_model = check_existing_models()
        
        # Step 4: Select model architecture
        model_config = select_model_architecture()
        if not model_config:
            return
        
        # Step 5: Select RL algorithm
        algorithm_config = select_algorithm()
        if not algorithm_config:
            return
        
        # Step 6: Get training parameters
        training_params = get_training_parameters()
        
        # Step 7: Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/trades_{timestamp}.csv"
        
        # Create environment function
        def env_fn():
            return create_environment(
                df=df,
                params=training_params,
                log_file=log_file,
                training_iteration=0
            )
        
        # Step 8: Save configuration
        config = {
            "data_file": data_file,
            "model_architecture": model_config[0],
            "algorithm": algorithm_config[0],
            "training_params": training_params,
            "timestamp": timestamp
        }
        save_config(config, f"config_{timestamp}.json")
        
        # Step 9: Train the model
        model_path = train_model(
            model_class=algorithm_config,
            env_fn=env_fn,
            model_config=model_config,
            training_params=training_params,
            existing_model_path=existing_model
        )
        
        if model_path:
            console.print("\n[bold green]🎉 Training session completed![/bold green]")
            console.print(f"📊 Trade log: [blue]{log_file}[/blue]")
            console.print(f"📈 TensorBoard logs: [blue]tensorboard_logs/[/blue]")
            console.print("\n[yellow]Run 'tensorboard --logdir=tensorboard_logs' to view training progress[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[red]❌ Program interrupted by user[/red]")
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {str(e)}[/red]")

if __name__ == "__main__":
    main()
