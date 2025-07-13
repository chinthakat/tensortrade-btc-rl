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
from rich.prompt import Prompt, IntPrompt, FloatPrompt
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
from stable_baselines3.common.vec_env import VecNormalize

# Local imports
from trading_environment import FuturesTradingEnv
from model_architectures import CNNLSTMFeatureExtractor, AttentionCNNLSTMExtractor, ResNetLSTMExtractor
from action_space_wrapper import DictToBoxActionWrapper, wrap_environment_for_algorithm
from log_archiver import archive_startup_logs

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
        default=5.0,  # Reduced from 25x to 5x for safer training
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
    
    # Enhanced Liquidation Parameters
    params['maintenance_margin_rate'] = FloatPrompt.ask(
        "⚖️ Maintenance margin rate (0.004 = 0.4%)",
        default=0.004,
        show_default=True
    )
    
    params['liquidation_fee_rate'] = FloatPrompt.ask(
        "💸 Liquidation fee rate (0.005 = 0.5%)",
        default=0.005,
        show_default=True
    )
    
    # Number of parallel environments
    params['n_envs'] = IntPrompt.ask(
        "🔄 Number of parallel environments",
        default=4,
        show_default=True
    )
    
    # Train/validation split
    params['train_ratio'] = FloatPrompt.ask(
        "📊 Training data ratio (0.7 = 70% train, 30% validation)",
        default=0.7,
        show_default=True
    )
    
    return params

def get_hyperparameters(algorithm: str) -> Dict[str, Any]:
    """Get algorithm-specific hyperparameters from user"""
    console.print(f"\n[bold]🎛️  {algorithm.upper()} Hyperparameters:[/bold]")
    
    params = {}
    
    # Common hyperparameters
    params['learning_rate'] = FloatPrompt.ask(
        "📈 Learning rate",
        default=3e-4,
        show_default=True
    )
    
    params['batch_size'] = IntPrompt.ask(
        "🎯 Batch size",
        default=64,
        show_default=True
    )
    
    # Algorithm-specific parameters
    if algorithm.lower() == "ppo":
        params['n_steps'] = IntPrompt.ask(
            "👣 Steps per environment per update",
            default=2048,
            show_default=True
        )
        
        params['n_epochs'] = IntPrompt.ask(
            "🔄 Training epochs per update",
            default=10,
            show_default=True
        )
        
        params['clip_range'] = FloatPrompt.ask(
            "✂️ PPO clip range",
            default=0.2,
            show_default=True
        )
        
        params['gamma'] = FloatPrompt.ask(
            "💰 Discount factor (gamma)",
            default=0.99,
            show_default=True
        )
        
        params['gae_lambda'] = FloatPrompt.ask(
            "🎯 GAE lambda",
            default=0.95,
            show_default=True
        )
        
    elif algorithm.lower() == "a2c":
        params['n_steps'] = IntPrompt.ask(
            "👣 Steps per environment per update",
            default=5,
            show_default=True
        )
        
        params['gamma'] = FloatPrompt.ask(
            "💰 Discount factor (gamma)",
            default=0.99,
            show_default=True
        )
        
        params['gae_lambda'] = FloatPrompt.ask(
            "🎯 GAE lambda",
            default=1.0,
            show_default=True
        )
        
        params['ent_coef'] = FloatPrompt.ask(
            "🎲 Entropy coefficient",
            default=0.0,
            show_default=True
        )
        
    elif algorithm.lower() == "sac":
        params['buffer_size'] = IntPrompt.ask(
            "📦 Replay buffer size",
            default=1000000,
            show_default=True
        )
        
        params['train_freq'] = IntPrompt.ask(
            "🚂 Training frequency",
            default=1,
            show_default=True
        )
        
        params['gradient_steps'] = IntPrompt.ask(
            "📈 Gradient steps per training",
            default=1,
            show_default=True
        )
        
        params['tau'] = FloatPrompt.ask(
            "🎯 Target network update rate (tau)",
            default=0.005,
            show_default=True
        )
        
        params['gamma'] = FloatPrompt.ask(
            "💰 Discount factor (gamma)",
            default=0.99,
            show_default=True
        )
    
    # Environment normalization menu
    norm_table = Table(title="Environment Normalization Options")
    norm_table.add_column("Option", style="cyan", no_wrap=True, width=4)
    norm_table.add_column("Description", style="white")
    
    norm_options = [
        ("1", "🔧 Use full normalization (observations + rewards) - Recommended"),
        ("2", "📊 Normalize observations only"),
        ("3", "🎁 Normalize rewards only"),
        ("4", "❌ No normalization"),
    ]
    
    for option, description in norm_options:
        norm_table.add_row(option, description)
    
    console.print(norm_table)
    
    norm_choice = IntPrompt.ask("\nSelect normalization option", default=1)
    
    if norm_choice == 1:
        params['use_normalization'] = True
        params['norm_obs'] = True
        params['norm_reward'] = True
    elif norm_choice == 2:
        params['use_normalization'] = True
        params['norm_obs'] = True
        params['norm_reward'] = False
    elif norm_choice == 3:
        params['use_normalization'] = True
        params['norm_obs'] = False
        params['norm_reward'] = True
    else:
        params['use_normalization'] = False
        params['norm_obs'] = False
        params['norm_reward'] = False
        
        params['clip_obs'] = FloatPrompt.ask(
            "✂️ Observation clipping value",
            default=10.0,
            show_default=True
        )
        
        params['clip_reward'] = FloatPrompt.ask(
            "🎁 Reward clipping value",
            default=10.0,
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
    
    # Create model selection menu
    model_menu_table = Table(title="Model Training Options")
    model_menu_table.add_column("Option", style="cyan", no_wrap=True, width=4)
    model_menu_table.add_column("Description", style="white")
    
    model_options = [
        ("1", "🔄 Continue training an existing model"),
        ("2", "🆕 Start training a new model"),
    ]
    
    for option, description in model_options:
        model_menu_table.add_row(option, description)
    
    console.print(model_menu_table)
    
    choice = IntPrompt.ask("\nSelect option", default=2)
    
    if choice == 1:
        while True:
            try:
                model_choice = IntPrompt.ask(
                    "🎯 Select model to continue (enter number)",
                    default=1,
                    show_default=True
                )
                if 1 <= model_choice <= len(model_files):
                    selected_model = model_files[model_choice-1]
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

def create_train_val_environments(df: pd.DataFrame, params: Dict[str, Any], log_file: str = None, training_iteration: int = 0, train_ratio: float = 0.7):
    """Create separate training and validation environments with proper data splitting"""
    
    # Calculate validation ratio to ensure they don't exceed 1.0
    val_ratio = min(0.3, 1.0 - train_ratio)  # Use up to 30% for validation, but not more than what's left
    
    console.print(f"📊 Data split: {train_ratio:.1%} training, {val_ratio:.1%} validation")
    
    # Use the class method to create properly split environments
    train_env, val_env = FuturesTradingEnv.create_train_val_environments(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        initial_equity=params['initial_equity'],
        max_leverage=params['max_leverage'],
        window_size=params['window_size'],
        stop_loss_pct=params['stop_loss_pct'],
        take_profit_pct=params['take_profit_pct'],
        maintenance_margin_rate=params['maintenance_margin_rate'],
        liquidation_fee_rate=params['liquidation_fee_rate'],
        log_file=log_file,
        training_iteration=training_iteration,
        use_advanced_action_space=True
    )
    
    # Wrap both environments for algorithm compatibility
    wrapped_train_env = wrap_environment_for_algorithm(train_env, "PPO")
    wrapped_val_env = wrap_environment_for_algorithm(val_env, "PPO")
    
    console.print(f"📊 Training data: {len(train_env.df)} samples")
    console.print(f"📊 Validation data: {len(val_env.df)} samples")
    
    return wrapped_train_env, wrapped_val_env

def create_environment(df: pd.DataFrame, params: Dict[str, Any], log_file: str = None, training_iteration: int = 0):
    """Create trading environment with specified parameters (legacy function for backward compatibility)"""
    env = FuturesTradingEnv(
        df=df,
        initial_equity=params['initial_equity'],
        max_leverage=params['max_leverage'],
        window_size=params['window_size'],
        stop_loss_pct=params['stop_loss_pct'],
        take_profit_pct=params['take_profit_pct'],
        maintenance_margin_rate=params['maintenance_margin_rate'],
        liquidation_fee_rate=params['liquidation_fee_rate'],
        log_file=log_file,
        training_iteration=training_iteration,
        use_advanced_action_space=True  # Enable advanced action space by default
    )
    
    # Wrap environment for PPO compatibility (Dict → Box conversion)
    wrapped_env = wrap_environment_for_algorithm(env, "PPO")
    return wrapped_env

def create_vectorized_environment(env_fn, n_envs: int, hyperparams: Dict[str, Any]) -> Any:
    """Create vectorized environment with optional normalization"""
    # Create vectorized environment
    vec_env = make_vec_env(env_fn, n_envs=n_envs)
    
    # Apply normalization if requested
    if hyperparams.get('use_normalization', False):
        console.print("🔧 Applying environment normalization...")
        
        vec_env = VecNormalize(
            vec_env,
            norm_obs=hyperparams.get('norm_obs', True),
            norm_reward=hyperparams.get('norm_reward', True),
            clip_obs=hyperparams.get('clip_obs', 10.0),
            clip_reward=hyperparams.get('clip_reward', 10.0),
            gamma=hyperparams.get('gamma', 0.99),
            training=True  # Enable training mode
        )
        
        console.print("✅ Environment normalization applied")
    
    return vec_env

def save_config(config: Dict[str, Any], filename: str = None, interactive: bool = True):
    """Save training configuration with optional interactive naming"""
    os.makedirs("configs", exist_ok=True)
    
    if interactive and not filename:
        # Ask user for a meaningful name
        console.print("\n[bold]💾 Save Configuration:[/bold]")
        
        config_name = Prompt.ask(
            "🏷️  Configuration name (for easy identification)",
            default=f"{config.get('algorithm', 'unknown')}_{config.get('model_architecture', 'model')}"
        )
        
        config_description = Prompt.ask(
            "📝 Brief description (optional)",
            default="Custom training configuration"
        )
        
        # Add metadata to config
        config['name'] = config_name
        config['description'] = config_description
        config['created_date'] = datetime.now().isoformat()
        
        # Clean filename
        clean_name = "".join(c for c in config_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_').lower()
        filename = f"{clean_name}.json"
    
    elif not filename:
        # Auto-generate filename with timestamp for non-interactive mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_session_{timestamp}.json"
        config['auto_generated'] = True
        config['created_date'] = datetime.now().isoformat()
    
    config_path = f"configs/{filename}"
    
    # Ensure unique filename
    counter = 1
    original_path = config_path
    while os.path.exists(config_path):
        name, ext = os.path.splitext(original_path)
        config_path = f"{name}_{counter}{ext}"
        counter += 1
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    console.print(f"💾 Configuration saved to: [green]{config_path}[/green]")
    return config_path

def load_model_with_normalization(model_path: str, env=None):
    """Load a trained model along with its normalization statistics"""
    from stable_baselines3 import PPO, A2C, SAC
    
    # Determine algorithm from filename
    if "ppo" in model_path.lower():
        model_class = PPO
    elif "a2c" in model_path.lower():
        model_class = A2C
    elif "sac" in model_path.lower():
        model_class = SAC
    else:
        # Default to PPO
        model_class = PPO
    
    # Load the model
    model = model_class.load(model_path, env=env)
    
    # Try to load normalization statistics
    norm_path = model_path.replace('.zip', '_vecnormalize.pkl')
    vec_normalize = None
    
    if os.path.exists(norm_path):
        try:
            vec_normalize = VecNormalize.load(norm_path, env)
            console.print(f"✅ Loaded normalization stats from: [green]{norm_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not load normalization stats: {str(e)}[/yellow]")
    
    return model, vec_normalize

def load_config_from_file() -> Optional[Dict[str, Any]]:
    """Load configuration from existing config file"""
    configs_dir = Path("configs")
    if not configs_dir.exists():
        return None
    
    config_files = list(configs_dir.glob("*.json"))
    if not config_files:
        return None
    
    console.print("\n[bold]📁 Available Configuration Files:[/bold]")
    
    table = Table(title="Configuration Files")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Description", style="yellow")
    table.add_column("Modified", style="magenta")
    
    config_info = []
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            name = config_data.get('name', config_file.stem)
            description = config_data.get('description', 'No description')
            use_case = config_data.get('use_case', '')
            if use_case:
                description = f"{description} - {use_case}"
            
            mod_time = datetime.fromtimestamp(config_file.stat().st_mtime)
            config_info.append((config_file, name, description, mod_time))
        except:
            # Fallback for configs without name/description
            mod_time = datetime.fromtimestamp(config_file.stat().st_mtime)
            config_info.append((config_file, config_file.stem, "Legacy config", mod_time))
    
    # Sort by modification time (newest first)
    config_info.sort(key=lambda x: x[3], reverse=True)
    
    for i, (config_file, name, description, mod_time) in enumerate(config_info):
        table.add_row(
            str(i+1), 
            name, 
            description[:60] + "..." if len(description) > 60 else description,
            mod_time.strftime("%Y-%m-%d %H:%M")
        )
    
    console.print(table)
    
    # Create configuration loading menu
    config_menu_table = Table(title="Configuration Options")
    config_menu_table.add_column("Option", style="cyan", no_wrap=True, width=4)
    config_menu_table.add_column("Description", style="white")
    
    config_options = [
        ("1", "📥 Load configuration from existing file"),
        ("2", "⚙️ Continue with manual configuration"),
    ]
    
    for option, description in config_options:
        config_menu_table.add_row(option, description)
    
    console.print(config_menu_table)
    
    choice = IntPrompt.ask("\nSelect option", default=2)
    
    if choice == 1:
        while True:
            try:
                config_choice = IntPrompt.ask(
                    "🎯 Select config file (enter number)",
                    default=1,
                    show_default=True
                )
                if 1 <= config_choice <= len(config_info):
                    selected_config = config_info[config_choice-1][0]
                    with open(selected_config, 'r') as f:
                        config = json.load(f)
                    console.print(f"✅ Loaded config: [green]{config_info[config_choice-1][1]}[/green]")
                    return config
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
            except KeyboardInterrupt:
                console.print("\n[red]Cancelled by user[/red]")
                break
    
    return None

def train_model(
    model_class,
    train_env_fn,
    val_env_fn,
    model_config: Dict[str, Any],
    training_params: Dict[str, Any],
    hyperparams: Dict[str, Any],
    existing_model_path: Optional[str] = None
):
    """Train the RL model with proper train/validation split"""
    console.print("\n[bold]🚀 Starting Training Process...[/bold]")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Setup vectorized training environment with normalization
    with console.status("[bold green]Setting up training environment..."):
        train_vec_env = create_vectorized_environment(train_env_fn, training_params['n_envs'], hyperparams)
    
    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": model_config[1]["class"],
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 128]  # Additional network layers
    }
    
    # Prepare algorithm-specific parameters
    algo_params = {
        "policy": "MultiInputPolicy",
        "env": train_vec_env,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "tensorboard_log": "./tensorboard_logs/",
        "learning_rate": hyperparams.get('learning_rate', 3e-4),
        "batch_size": hyperparams.get('batch_size', 64),
        "device": "auto"
    }
    
    # Add algorithm-specific parameters
    algorithm_name = model_class[0].lower()
    if algorithm_name == "ppo":
        algo_params.update({
            "n_steps": hyperparams.get('n_steps', 2048),
            "n_epochs": hyperparams.get('n_epochs', 10),
            "clip_range": hyperparams.get('clip_range', 0.2),
            "gamma": hyperparams.get('gamma', 0.99),
            "gae_lambda": hyperparams.get('gae_lambda', 0.95)
        })
    elif algorithm_name == "a2c":
        algo_params.update({
            "n_steps": hyperparams.get('n_steps', 5),
            "gamma": hyperparams.get('gamma', 0.99),
            "gae_lambda": hyperparams.get('gae_lambda', 1.0),
            "ent_coef": hyperparams.get('ent_coef', 0.0)
        })
    elif algorithm_name == "sac":
        algo_params.update({
            "buffer_size": hyperparams.get('buffer_size', 1000000),
            "train_freq": hyperparams.get('train_freq', 1),
            "gradient_steps": hyperparams.get('gradient_steps', 1),
            "tau": hyperparams.get('tau', 0.005),
            "gamma": hyperparams.get('gamma', 0.99)
        })
    
    # Initialize or load model
    if existing_model_path and os.path.exists(existing_model_path):
        console.print(f"🔄 Loading existing model: [green]{existing_model_path}[/green]")
        model = model_class[1]["class"].load(existing_model_path, env=train_vec_env)
        # Update tensorboard log
        model.tensorboard_log = "./tensorboard_logs/"
    else:
        console.print("🆕 Creating new model...")
        model = model_class[1]["class"](**algo_params)
    
    # Setup callbacks
    progress_callback = TradingProgressCallback(check_freq=1000)
    
    # Create evaluation environment - using proper validation data
    console.print("📊 Setting up validation environment...")
    
    if hyperparams.get('use_normalization', False):
        # Create validation environment with the same normalization settings but in evaluation mode
        val_vec_env = make_vec_env(val_env_fn, n_envs=1)
        val_vec_env = VecNormalize(
            val_vec_env,
            norm_obs=hyperparams.get('norm_obs', True),
            norm_reward=hyperparams.get('norm_reward', True),
            clip_obs=hyperparams.get('clip_obs', 10.0),
            clip_reward=hyperparams.get('clip_reward', 10.0),
            gamma=hyperparams.get('gamma', 0.99),
            training=False  # Disable training for evaluation
        )
        
        eval_callback = EvalCallback(
            val_vec_env,
            best_model_save_path="./models/",
            log_path="./logs/",
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
        
        console.print("✅ Using normalized validation environment")
        
    else:
        # Non-normalized validation environment
        val_env = val_env_fn()
        val_env = Monitor(val_env)
        
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path="./models/",
            log_path="./logs/",
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
        
        console.print("✅ Using standard validation environment")
    
    # Start training
    try:
        # Synchronize normalization statistics between training and validation if needed
        if hyperparams.get('use_normalization', False):
            console.print("🔄 Training will auto-sync normalization statistics...")
        
        model.learn(
            total_timesteps=training_params['total_timesteps'],
            callback=[progress_callback, eval_callback],
            progress_bar=False  # We use our custom progress bar
        )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"trading_bot_{model_config[0]}_{algorithm_name}_{timestamp}"
        model_path = f"models/{model_name}"
        model.save(model_path)
        
        # Save normalization statistics if used
        if hyperparams.get('use_normalization', False) and hasattr(train_vec_env, 'save'):
            norm_path = f"models/{model_name}_vecnormalize.pkl"
            train_vec_env.save(norm_path)
            console.print(f"💾 Normalization stats saved to: [green]{norm_path}[/green]")
        
        console.print(f"\n✅ [bold green]Training completed successfully![/bold green]")
        console.print(f"💾 Model saved to: [green]{model_path}.zip[/green]")
        
        return model_path
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Training interrupted by user[/yellow]")
        # Save current model state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"trading_bot_interrupted_{algorithm_name}_{timestamp}"
        model_path = f"models/{model_name}"
        model.save(model_path)
        console.print(f"💾 Model saved to: [green]{model_path}.zip[/green]")
        
        # Save normalization statistics if used
        if hyperparams.get('use_normalization', False) and hasattr(train_vec_env, 'save'):
            norm_path = f"models/{model_name}_vecnormalize.pkl"
            train_vec_env.save(norm_path)
            console.print(f"💾 Normalization stats saved to: [green]{norm_path}[/green]")
        
        return model_path
        
    except Exception as e:
        console.print(f"[red]❌ Training failed: {str(e)}[/red]")
        return None

def main():
    """Main training function"""
    display_welcome()
    
    # Archive old logs before starting training
    console.print("[bold]🗂️  Archiving old logs before training...[/bold]")
    try:
        archive_startup_logs(
            base_dir=".",
            log_age_days=2,      # More aggressive for training sessions
            model_age_days=10,   # Keep fewer old models
            tensorboard_age_days=5
        )
    except Exception as e:
        console.print(f"[yellow]⚠️  Log archiving failed: {str(e)}[/yellow]")
        console.print("[yellow]Continuing with training startup...[/yellow]")
    
    try:
        # Check if user wants to load from config file
        loaded_config = load_config_from_file()
        
        if loaded_config:
            # Use loaded configuration
            data_file = loaded_config.get('data_file')
            model_architecture = loaded_config.get('model_architecture')
            algorithm = loaded_config.get('algorithm')
            training_params = loaded_config.get('training_params', {})
            hyperparams = loaded_config.get('hyperparameters', {})
            
            # Validate loaded data
            if not data_file or not os.path.exists(data_file):
                console.print(f"[red]❌ Data file not found: {data_file}[/red]")
                data_file = select_data_file()
                if not data_file:
                    return
            
            # Find model and algorithm configs
            model_config = None
            algorithm_config = None
            
            for key, model_info in ModelConfig.AVAILABLE_MODELS.items():
                if key == model_architecture:
                    model_config = (key, model_info)
                    break
            
            for key, algo_info in ModelConfig.AVAILABLE_ALGORITHMS.items():
                if key == algorithm:
                    algorithm_config = (key, algo_info)
                    break
            
            if not model_config:
                console.print(f"[yellow]⚠️  Model architecture '{model_architecture}' not found, selecting manually[/yellow]")
                model_config = select_model_architecture()
                if not model_config:
                    return
            
            if not algorithm_config:
                console.print(f"[yellow]⚠️  Algorithm '{algorithm}' not found, selecting manually[/yellow]")
                algorithm_config = select_algorithm()
                if not algorithm_config:
                    return
            
            console.print(f"✅ Using loaded configuration with {algorithm_config[1]['name']} and {model_config[1]['name']}")
            
        else:
            # Interactive configuration
            # Step 1: Select data file
            data_file = select_data_file()
            if not data_file:
                return
            
            # Step 2: Check for existing models
            existing_model = check_existing_models()
            
            # Step 3: Select model architecture
            model_config = select_model_architecture()
            if not model_config:
                return
            
            # Step 4: Select RL algorithm
            algorithm_config = select_algorithm()
            if not algorithm_config:
                return
            
            # Step 5: Get training parameters
            training_params = get_training_parameters()
            
            # Step 6: Get hyperparameters
            hyperparams = get_hyperparameters(algorithm_config[0])
        
        # Load and validate data
        df = load_data(data_file)
        if df is None:
            return
        
        # Check for existing models (if not already done)
        if 'existing_model' not in locals():
            existing_model = check_existing_models()
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/trades_{timestamp}.csv"
        
        # Create train and validation environment functions with proper data splitting
        console.print("\n[bold]📊 Creating training and validation environments...[/bold]")
        
        train_env, val_env = create_train_val_environments(
            df=df,
            params=training_params,
            log_file=log_file,
            training_iteration=0,
            train_ratio=training_params.get('train_ratio', 0.7)
        )
        
        # Create environment functions for training
        def train_env_fn():
            return create_train_val_environments(
                df=df,
                params=training_params,
                log_file=log_file,
                training_iteration=0,
                train_ratio=training_params.get('train_ratio', 0.7)
            )[0]  # Return only training environment
            
        def val_env_fn():
            return create_train_val_environments(
                df=df,
                params=training_params,
                log_file=log_file,
                training_iteration=0,
                train_ratio=training_params.get('train_ratio', 0.7)
            )[1]  # Return only validation environment
        
        # Save configuration
        config = {
            "data_file": data_file,
            "model_architecture": model_config[0],
            "algorithm": algorithm_config[0],
            "training_params": training_params,
            "hyperparameters": hyperparams,
            "timestamp": timestamp
        }
        
        # Configuration saving menu
        save_table = Table(title="Configuration Saving Options")
        save_table.add_column("Option", style="cyan", no_wrap=True, width=4)
        save_table.add_column("Description", style="white")
        
        save_options = [
            ("1", "💾 Save with custom name for future reuse"),
            ("2", "📁 Auto-save for session tracking only"),
        ]
        
        for option, description in save_options:
            save_table.add_row(option, description)
        
        console.print(save_table)
        
        save_choice = IntPrompt.ask("\nSelect saving option", default=1)
        
        if save_choice == 1:
            save_config(config, interactive=True)
        else:
            # Save with auto-generated name for session tracking
            save_config(config, interactive=False)
        
        # Train the model
        model_path = train_model(
            model_class=algorithm_config,
            train_env_fn=train_env_fn,
            val_env_fn=val_env_fn,
            model_config=model_config,
            training_params=training_params,
            hyperparams=hyperparams,
            existing_model_path=existing_model
        )
        
        if model_path:
            console.print("\n[bold green]🎉 Training session completed![/bold green]")
            console.print(f"📊 Trade log: [blue]{log_file}[/blue]")
            console.print(f"📈 TensorBoard logs: [blue]tensorboard_logs/[/blue]")
            console.print("\n[yellow]Run 'tensorboard --logdir=tensorboard_logs' to view training progress[/yellow]")
            
            # Display hyperparameter summary
            console.print("\n[bold]📋 Hyperparameters Used:[/bold]")
            hp_table = Table(title="Final Hyperparameters")
            hp_table.add_column("Parameter", style="cyan")
            hp_table.add_column("Value", style="green")
            
            for key, value in hyperparams.items():
                hp_table.add_row(key, str(value))
            
            console.print(hp_table)
        
    except KeyboardInterrupt:
        console.print("\n[red]❌ Program interrupted by user[/red]")
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {str(e)}[/red]")

if __name__ == "__main__":
    main()
