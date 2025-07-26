"""
Multi-Episode Training System with Model Persistence
Supports continuous learning and model retraining
"""

import os
import json
import numpy as np
import pandas as pd
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, FloatPrompt
from rich import print as rprint

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from trading_environment import FuturesTradingEnv
from model_architectures import CNNLSTMFeatureExtractor, AttentionCNNLSTMExtractor, ResNetLSTMExtractor
from backtest import TradingBacktester
from action_space_wrapper import wrap_environment_for_algorithm

console = Console()

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)

def timeout_confirmation(prompt: str, timeout_seconds: int = 60, default: bool = True) -> bool:
    """
    Ask for confirmation with automatic timeout (Windows-compatible).
    
    Args:
        prompt: The confirmation prompt
        timeout_seconds: Seconds to wait before auto-continuing
        default: Default response when timeout occurs
    
    Returns:
        True to continue, False to stop
    """
    import threading
    import sys
    import select
    
    console.print(f"\n[yellow]{prompt}[/yellow]")
    console.print(f"[dim]Auto-continuing in {timeout_seconds} seconds...[/dim]")
    console.print(f"[dim]Press Enter to continue now, or type 'n' + Enter to stop[/dim]")
    
    result = [default]  # Use list to allow modification in nested function
    input_received = [False]
    stop_countdown = [False]
    
    def get_input():
        try:
            user_input = input().strip().lower()
            input_received[0] = True
            stop_countdown[0] = True
            if user_input in ['n', 'no', 'stop', 'quit']:
                result[0] = False
                console.print("[yellow]⛔ User chose to stop training[/yellow]")
            else:
                result[0] = True
                console.print("[green]✅ User chose to continue[/green]")
        except Exception as e:
            console.print(f"[dim]Input error: {e}[/dim]")
            result[0] = default
            stop_countdown[0] = True
    
    # Start input thread
    input_thread = threading.Thread(target=get_input, daemon=True)
    input_thread.start()
    
    # Countdown with progress
    try:
        for i in range(timeout_seconds, 0, -1):
            if stop_countdown[0]:
                break
            console.print(f"[dim]Auto-continuing in {i:2d} seconds... (Press Enter or type 'n')[/dim]", end="\r")
            time.sleep(1)
        
        console.print("")  # Clear the countdown line
        
        if not input_received[0]:
            console.print(f"[green]✅ Auto-continuing to next episode (timeout)[/green]")
            result[0] = default
        
        return result[0]
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]❌ Training stopped by user (Ctrl+C)[/yellow]")
        return False

console = Console()

class EpisodeTracker:
    """Track training episodes and performance"""
    
    def __init__(self, save_path: str = "episode_tracking"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        self.episodes = []
        self.performance_history = []
        
        # Load existing tracking data if available
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing episode tracking data"""
        tracking_file = os.path.join(self.save_path, "episode_tracking.json")
        if os.path.exists(tracking_file):
            try:
                with open(tracking_file, 'r') as f:
                    data = json.load(f)
                    self.episodes = data.get('episodes', [])
                    self.performance_history = data.get('performance_history', [])
                
                console.print(f"📊 Loaded {len(self.episodes)} previous episodes")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load tracking data: {str(e)}[/yellow]")
    
    def add_episode(self, episode_data: Dict):
        """Add new episode data"""
        episode_data['timestamp'] = datetime.now().isoformat()
        self.episodes.append(episode_data)
        self._save_data()
    
    def add_performance(self, performance_data: Dict):
        """Add performance evaluation data"""
        performance_data['timestamp'] = datetime.now().isoformat()
        self.performance_history.append(performance_data)
        self._save_data()
    
    def _save_data(self):
        """Save tracking data to file"""
    
        tracking_file = os.path.join(self.save_path, "episode_tracking.json")
        data = {
            'episodes': self.episodes,
            'performance_history': self.performance_history
        }
        
        with open(tracking_file, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    def get_best_episode(self) -> Optional[Dict]:
        """Get the best performing episode"""
        if not self.performance_history:
            return None
        
        best_episode = max(self.performance_history, key=lambda x: x.get('total_return_pct', -float('inf')))
        return best_episode
    
    def display_episode_summary(self):
        """Display summary of all episodes"""
        if not self.episodes:
            console.print("[yellow]No episodes recorded yet[/yellow]")
            return
        
        table = Table(title="Episode Training Summary")
        table.add_column("Episode", style="cyan", no_wrap=True)
        table.add_column("Timesteps", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Date", style="blue")
        table.add_column("Status", style="magenta")
        
        for i, episode in enumerate(self.episodes):
            status = "✅ Completed" if episode.get('completed', False) else "⚠️  Incomplete"
            table.add_row(
                str(i + 1),
                str(episode.get('total_timesteps', 'N/A')),
                episode.get('model_architecture', 'N/A'),
                episode.get('timestamp', 'N/A')[:10],
                status
            )
        
        console.print(table)
        
        # Performance summary
        if self.performance_history:
            perf_table = Table(title="Performance History")
            perf_table.add_column("Episode", style="cyan")
            perf_table.add_column("Return %", style="green")
            perf_table.add_column("Sharpe", style="yellow")
            perf_table.add_column("Max DD %", style="red")
            perf_table.add_column("Trades", style="blue")
            
            for i, perf in enumerate(self.performance_history):
                perf_table.add_row(
                    str(i + 1),
                    f"{perf.get('total_return_pct', 0):.2f}",
                    f"{perf.get('sharpe_ratio', 0):.3f}",
                    f"{perf.get('max_drawdown', 0)*100:.2f}",
                    str(perf.get('total_trades', 0))
                )
            
            console.print(perf_table)

class MultiEpisodeTrainer:
    """Multi-episode training system with model persistence"""
    
    def __init__(self, data_path: str, base_config: Dict, starting_model_path: Optional[str] = None, validation_pct: float = 0.05, use_simple_split: bool = True, num_episodes: int = 1, reward_config=None):
        self.data_path = data_path
        self.base_config = base_config
        self.starting_model_path = starting_model_path
        self.validation_pct = validation_pct
        self.use_simple_split = use_simple_split
        self.num_episodes_requested = num_episodes
        self.reward_config = reward_config  # Store the reward configuration
        self.episode_tracker = EpisodeTracker()
        
        # Load data
        self.df = pd.read_csv(data_path)
        console.print(f"📊 Loaded data with {len(self.df)} rows")
        
        # Clean and validate data
        self._clean_and_validate_data()
        
        # Split data for episodes
        self.data_splits = self._create_data_splits(
            validation_pct=self.validation_pct, 
            use_walk_forward=not self.use_simple_split,
            num_episodes=num_episodes
        )
        
        # Training state
        self.current_episode = 0
        self.best_model_path = starting_model_path  # Initialize with starting model
        self.best_performance = None
        
        # If we have a starting model, it's considered our current best
        if starting_model_path:
            console.print(f"🎯 Starting from model: {starting_model_path}")
    
    def _clean_and_validate_data(self):
        """Clean and validate the dataset, removing corrupted data"""
        original_len = len(self.df)
        
        # Check for negative prices in any price column
        price_columns = ['open', 'high', 'low', 'close']
        
        # Find the first row where any price goes negative
        for col in price_columns:
            if col in self.df.columns:
                negative_mask = self.df[col] <= 0
                if negative_mask.any():
                    first_negative_idx = negative_mask.idxmax()
                    console.print(f"⚠️  [yellow]Found negative/zero {col} prices starting at row {first_negative_idx}[/yellow]")
                    
                    # Truncate dataset before negative prices
                    self.df = self.df.iloc[:first_negative_idx].copy()
                    console.print(f"🔧 [cyan]Truncated dataset: {original_len} → {len(self.df)} rows[/cyan]")
                    console.print(f"📉 [cyan]Removed {original_len - len(self.df)} corrupted rows with negative prices[/cyan]")
                    break
        
        # Additional validation
        if len(self.df) == 0:
            raise ValueError("No valid data remaining after cleaning!")
        
        if len(self.df) < 1000:
            console.print(f"[yellow]⚠️  Warning: Only {len(self.df)} rows remaining after cleaning[/yellow]")
        
        # Ensure timestamp column exists and is properly formatted
        if 'timestamp' in self.df.columns:
            try:
                # Convert timestamp to datetime for validation
                import pandas as pd
                timestamps = pd.to_datetime(self.df['timestamp'], unit='s')
                console.print(f"📅 Data range: {timestamps.iloc[0]} → {timestamps.iloc[-1]}")
            except Exception as e:
                console.print(f"[yellow]⚠️  Timestamp validation warning: {str(e)}[/yellow]")
        
        console.print(f"✅ [green]Data cleaning completed: {len(self.df)} valid rows[/green]")
    
    def _create_data_splits(self, validation_pct: float = 0.05, use_walk_forward: bool = False, num_episodes: int = 1) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create train/validation split - simple split (default) or walk-forward splits"""
        
        if use_walk_forward:
            # Legacy walk-forward approach
            return self._create_walk_forward_splits()
        
        # Simple approach: use last X% for validation
        total_rows = len(self.df)
        
        # Calculate split point - use last X% for validation
        val_size = int(total_rows * validation_pct)
        train_size = total_rows - val_size
        
        # Create single split using majority for training, small portion for validation
        train_data = self.df.iloc[:train_size].copy()
        val_data = self.df.iloc[train_size:].copy()
        
        console.print(f"📊 Created simple train/validation split:")
        console.print(f"   📈 Training data: {len(train_data):,} rows ({100*(1-validation_pct):.0f}%)")
        console.print(f"   📋 Validation data: {len(val_data):,} rows ({validation_pct*100:.0f}%)")
        console.print(f"   🎯 Training will use {len(train_data):,} rows instead of ~10,000!")
        
        # For multi-episode training with simple split, replicate the same split
        # This allows different episodes to train on the same data with different random seeds/hyperparameters
        splits = []
        for episode in range(num_episodes):
            splits.append((train_data.copy(), val_data.copy()))
            
        if num_episodes > 1:
            console.print(f"   🔄 Replicated split for {num_episodes} episodes (each episode will use the same data)")
        
        return splits
    
    def _create_walk_forward_splits(self, min_train_size: int = 10000, validation_size: int = 2000) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create walk-forward data splits for episodes (legacy approach)"""
        splits = []
        total_rows = len(self.df)
        
        # Calculate number of possible splits
        max_splits = (total_rows - min_train_size) // validation_size
        
        for i in range(min(max_splits, 10)):  # Limit to 10 episodes max
            train_end = min_train_size + (i * validation_size)
            val_start = train_end
            val_end = min(val_start + validation_size, total_rows)
            
            if val_end <= total_rows:
                train_data = self.df.iloc[:train_end].copy()
                val_data = self.df.iloc[val_start:val_end].copy()
                splits.append((train_data, val_data))
        
        console.print(f"📈 Created {len(splits)} walk-forward data splits (legacy mode)")
        return splits
    
    def train_episode(
        self,
        episode_num: int,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        model_architecture: str = "cnn_lstm",
        algorithm: str = "ppo",
        timesteps: int = 500000,
        continue_from_best: bool = True
    ) -> Optional[str]:
        """Train a single episode"""
        
        console.print(f"\n[bold]🎯 Starting Episode {episode_num + 1}[/bold]")
        console.print(f"📊 Training data: {len(train_data)} rows")
        console.print(f"📊 Validation data: {len(val_data)} rows")
        
        # Create timestamp for this episode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_id = f"episode_{episode_num + 1:02d}_{timestamp}"
        
        # Setup directories
        episode_dir = f"episodes/{episode_id}"
        os.makedirs(episode_dir, exist_ok=True)
        os.makedirs(f"{episode_dir}/models", exist_ok=True)
        os.makedirs(f"{episode_dir}/logs", exist_ok=True)
        
        # Log file for this episode
        base_log_file = f"{episode_dir}/logs/trades_{episode_id}"
        
        # Create environment function
        env_counter = 0
        def env_fn():
            nonlocal env_counter
            # Filter out parameters that shouldn't be passed to individual environment
            env_params = {k: v for k, v in self.base_config["training_params"].items() 
                         if k not in ['n_envs', 'total_timesteps', 'train_ratio']}  # Remove vectorization and training-specific params
            
            # Create unique log file for each environment instance to avoid conflicts
            instance_log_file = f"{base_log_file}_env{env_counter}.csv"
            env_counter += 1
            
            env = FuturesTradingEnv(
                df=train_data,
                log_file=instance_log_file,
                training_iteration=episode_num,
                use_advanced_action_space=True,  # Enable advanced action space by default
                reward_config=self.reward_config,  # Pass the reward configuration
                **env_params
            )
            # Wrap environment for PPO compatibility
            return wrap_environment_for_algorithm(env, "PPO")
        
        # Setup vectorized environment
        vec_env = make_vec_env(env_fn, n_envs=self.base_config["training_params"].get("n_envs", 4))
        
        # Model configuration
        model_configs = {
            "cnn_lstm": CNNLSTMFeatureExtractor,
            "attention_cnn_lstm": AttentionCNNLSTMExtractor,
            "resnet_lstm": ResNetLSTMExtractor
        }
        
        algorithm_classes = {
            "ppo": PPO,
            "a2c": A2C,
            "sac": SAC
        }
        
        policy_kwargs = {
            "features_extractor_class": model_configs[model_architecture],
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": [256, 128]
        }
        
        # Load existing model or create new one
        model_class = algorithm_classes[algorithm]
        model_to_load = None
        
        # Determine which model to load
        if continue_from_best and self.best_model_path:
            if os.path.exists(self.best_model_path):
                model_to_load = self.best_model_path
                console.print(f"🔄 Continuing from best model: {self.best_model_path}")
            else:
                console.print(f"[yellow]⚠️  Best model path not found: {self.best_model_path}[/yellow]")
        
        # Try to load the selected model
        if model_to_load:
            try:
                model = model_class.load(model_to_load, env=vec_env)
                model.tensorboard_log = f"{episode_dir}/tensorboard/"
                console.print(f"✅ Successfully loaded model from: {model_to_load}")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load model: {str(e)}[/yellow]")
                console.print("Creating new model instead...")
                model = model_class(
                    "MultiInputPolicy",
                    vec_env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=f"{episode_dir}/tensorboard/",
                    device="auto"
                )
        else:
            console.print("🆕 Creating new model")
            model = model_class(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=f"{episode_dir}/tensorboard/",
                device="auto"
            )
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"{episode_dir}/models/",
            name_prefix=f"checkpoint_{episode_id}"
        )
        
        # Custom progress callback
        class EpisodeProgressCallback(BaseCallback):
            def __init__(self, check_freq: int = 1000):
                super().__init__()
                self.check_freq = check_freq
                self.progress = None
                self.task_id = None
            
            def _on_training_start(self):
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                )
                self.progress.start()
                self.task_id = self.progress.add_task(
                    f"Training Episode {episode_num + 1}...", 
                    total=timesteps
                )
            
            def _on_step(self):
                if self.n_calls % self.check_freq == 0:
                    if self.progress and self.task_id is not None:
                        self.progress.update(self.task_id, completed=self.num_timesteps)
                return True
            
            def _on_training_end(self):
                if self.progress:
                    self.progress.stop()
        
        progress_callback = EpisodeProgressCallback()
        
        # Start training
        try:
            model.learn(
                total_timesteps=timesteps,
                callback=[checkpoint_callback, progress_callback],
                progress_bar=False
            )
            
            # Save final model
            final_model_path = f"{episode_dir}/models/final_model_{episode_id}.zip"
            model.save(final_model_path)
            
            # Record episode completion
            episode_data = {
                'episode_id': episode_id,
                'episode_number': episode_num + 1,
                'total_timesteps': timesteps,
                'model_architecture': model_architecture,
                'algorithm': algorithm,
                'train_data_size': len(train_data),
                'val_data_size': len(val_data),
                'completed': True,
                'model_path': final_model_path
            }
            
            self.episode_tracker.add_episode(episode_data)
            
            console.print(f"✅ [green]Episode {episode_num + 1} completed![/green]")
            console.print(f"💾 Model saved to: {final_model_path}")
            
            # Evaluate on validation data
            self._evaluate_episode(final_model_path, val_data, episode_num)
            
            return final_model_path
            
        except KeyboardInterrupt:
            console.print(f"\n[yellow]⚠️  Episode {episode_num + 1} interrupted by user[/yellow]")
            
            # Save current state
            interrupted_model_path = f"{episode_dir}/models/interrupted_model_{episode_id}.zip"
            model.save(interrupted_model_path)
            
            episode_data = {
                'episode_id': episode_id,
                'episode_number': episode_num + 1,
                'total_timesteps': timesteps,
                'model_architecture': model_architecture,
                'algorithm': algorithm,
                'train_data_size': len(train_data),
                'val_data_size': len(val_data),
                'completed': False,
                'model_path': interrupted_model_path,
                'interrupted': True
            }
            
            self.episode_tracker.add_episode(episode_data)
            return interrupted_model_path
            
        except Exception as e:
            console.print(f"[red]❌ Episode {episode_num + 1} failed: {str(e)}[/red]")
            return None
    
    def _evaluate_episode(self, model_path: str, val_data: pd.DataFrame, episode_num: int):
        """Evaluate episode performance on validation data"""
        console.print(f"📊 Evaluating Episode {episode_num + 1} on validation data...")
        
        try:
            # Create temporary CSV file for validation data
            temp_val_file = f"temp_val_episode_{episode_num + 1}.csv"
            val_data.to_csv(temp_val_file, index=False)
            
            # Run backtest
            backtester = TradingBacktester(model_path, temp_val_file)
            results = backtester.run_backtest(deterministic=True)
            
            # Add to performance history
            results['episode_number'] = episode_num + 1
            results['model_path'] = model_path
            self.episode_tracker.add_performance(results)
            
            # Check if this is the best model so far
            if (self.best_performance is None or 
                results['total_return_pct'] > self.best_performance.get('total_return_pct', -float('inf'))):
                
                self.best_model_path = model_path
                self.best_performance = results
                
                console.print(f"🏆 [bold green]New best model! Return: {results['total_return_pct']:.2f}%[/bold green]")
            
            # Display results
            perf_table = Table(title=f"Episode {episode_num + 1} Validation Results")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="green")
            
            perf_table.add_row("Total Return", f"{results['total_return_pct']:.2f}%")
            perf_table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.3f}")
            perf_table.add_row("Max Drawdown", f"{results['max_drawdown']*100:.2f}%")
            perf_table.add_row("Win Rate", f"{results['win_rate']:.1f}%")
            perf_table.add_row("Total Trades", str(results['total_trades']))
            
            console.print(perf_table)
            
            # Clean up temp file
            if os.path.exists(temp_val_file):
                os.remove(temp_val_file)
            
        except Exception as e:
            console.print(f"[red]❌ Evaluation failed: {str(e)}[/red]")
    
    def run_multi_episode_training(
        self,
        num_episodes: int = None,
        model_architecture: str = "cnn_lstm",
        algorithm: str = "ppo",
        timesteps_per_episode: int = 500000
    ):
        """Run multiple training episodes"""
        
        if num_episodes is None:
            num_episodes = len(self.data_splits)
        
        num_episodes = min(num_episodes, len(self.data_splits))
        
        console.print(f"🚀 [bold]Starting Multi-Episode Training[/bold]")
        console.print(f"📊 Episodes to train: {num_episodes}")
        console.print(f"🧠 Model architecture: {model_architecture}")
        console.print(f"⚡ Algorithm: {algorithm}")
        console.print(f"🎯 Timesteps per episode: {timesteps_per_episode:,}")
        
        if self.starting_model_path:
            console.print(f"🏁 Starting from: {Path(self.starting_model_path).name}")
        
        # Display existing episode history
        self.episode_tracker.display_episode_summary()
        
        try:
            for episode_num in range(num_episodes):
                console.print(f"\n[bold cyan]🔄 Starting Episode {episode_num + 1} of {num_episodes}[/bold cyan]")
                
                # Check if we have enough data splits
                if episode_num >= len(self.data_splits):
                    console.print(f"[red]❌ No data split available for episode {episode_num + 1}[/red]")
                    console.print(f"[yellow]Available splits: {len(self.data_splits)}, Requested episode: {episode_num + 1}[/yellow]")
                    break
                
                train_data, val_data = self.data_splits[episode_num]
                
                # Validate data
                if train_data is None or val_data is None:
                    console.print(f"[red]❌ Invalid data for episode {episode_num + 1}[/red]")
                    break
                
                if len(train_data) == 0 or len(val_data) == 0:
                    console.print(f"[red]❌ Empty data for episode {episode_num + 1}[/red]")
                    console.print(f"Train data: {len(train_data)} rows, Val data: {len(val_data)} rows")
                    break
                
                # For the first episode, use starting model if available
                # For subsequent episodes, continue from best
                continue_from_best = episode_num > 0 or self.starting_model_path is not None
                console.print(f"[cyan]📚 Continue from best: {continue_from_best}[/cyan]")
                
                # Train episode
                console.print(f"[cyan]🎯 Training episode {episode_num + 1}...[/cyan]")
                model_path = self.train_episode(
                    episode_num=episode_num,
                    train_data=train_data,
                    val_data=val_data,
                    model_architecture=model_architecture,
                    algorithm=algorithm,
                    timesteps=timesteps_per_episode,
                    continue_from_best=continue_from_best
                )
                
                if model_path is None:
                    console.print(f"[red]❌ Episode {episode_num + 1} failed, stopping training[/red]")
                    break
                
                console.print(f"[green]✅ Episode {episode_num + 1} completed successfully[/green]")
                console.print(f"[cyan]💾 Model saved to: {model_path}[/cyan]")
                
                # Ask if user wants to continue to next episode with 60-second timeout
                if episode_num < num_episodes - 1:
                    console.print(f"\n[yellow]🎉 Episode {episode_num + 1} completed[/yellow]")
                    console.print(f"[cyan]💤 60-second break before next episode...[/cyan]")
                    console.print(f"[dim]Next episode: {episode_num + 2} of {num_episodes}[/dim]")
                    
                    should_continue = timeout_confirmation("Continue to next episode?", timeout_seconds=60, default=True)
                    console.print(f"[cyan]👀 User decision: {'Continue' if should_continue else 'Stop'}[/cyan]")
                    
                    if not should_continue:
                        console.print("[yellow]⛔ Training stopped by user[/yellow]")
                        break
                else:
                    console.print(f"\n[green]🎉 All {num_episodes} episodes completed![/green]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Multi-episode training interrupted[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Unexpected error in multi-episode training: {str(e)}[/red]")
            import traceback
            console.print(f"[dim]Traceback: {traceback.format_exc()}[/dim]")
        
        # Final summary
        console.print("\n[bold]🎉 Multi-Episode Training Complete![/bold]")
        self.episode_tracker.display_episode_summary()
        
        if self.best_model_path:
            console.print(f"\n🏆 [bold green]Best model: {self.best_model_path}[/bold green]")
            if self.best_performance:
                console.print(f"📈 Best return: {self.best_performance['total_return_pct']:.2f}%")
            else:
                console.print(f"📈 Performance: [yellow]Not evaluated yet[/yellow]")
            
            # Save the best model to the general models folder
            self._save_best_model_to_general_folder()
    
    def _save_best_model_to_general_folder(self):
        """Save the best model from multi-episode training to the general models folder"""
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            console.print("[yellow]⚠️  No best model found to save to general folder[/yellow]")
            return
        
        try:
            import shutil
            from pathlib import Path
            
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Generate filename for the best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = Path(self.best_model_path).name
            
            # Extract useful info from the original path for naming
            if "episode_" in self.best_model_path:
                episode_info = self.best_model_path.split("episode_")[1].split(os.sep)[0]
            else:
                episode_info = "multi_episode"
            
            # Create descriptive filename
            new_filename = f"best_multi_episode_{episode_info}_{timestamp}.zip"
            destination_path = models_dir / new_filename
            
            # Debug: Print paths
            console.print(f"[cyan]Debug - Source path: {self.best_model_path}[/cyan]")
            console.print(f"[cyan]Debug - Destination path: {destination_path}[/cyan]")
            console.print(f"[cyan]Debug - Episode info: {episode_info}[/cyan]")
            
            # Ensure source file exists
            if not os.path.exists(self.best_model_path):
                console.print(f"[red]❌ Source file does not exist: {self.best_model_path}[/red]")
                return
            
            # Ensure destination directory exists
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if source and destination are different files to avoid "same file" error
            source_path = Path(self.best_model_path).resolve()
            dest_path = destination_path.resolve()
            
            if source_path != dest_path:
                # Copy the best model to general models folder
                shutil.copy2(self.best_model_path, destination_path)
                
                console.print(f"\n[bold green]✅ Best model saved to general models folder![/bold green]")
                console.print(f"📁 Saved as: [green]{new_filename}[/green]")
            else:
                console.print(f"\n[bold yellow]⚠️ Source and destination are the same file, skipping copy[/bold yellow]")
                console.print(f"📁 File: [green]{new_filename}[/green]")
            
            if self.best_performance:
                console.print(f"📊 Model performance: {self.best_performance['total_return_pct']:.2f}% return")
            else:
                console.print(f"📊 Model performance: [yellow]Not evaluated yet[/yellow]")
            
            # Also update the best_model.zip (traditional location)
            best_model_path = models_dir / "best_model.zip"
            
            # Check if source and destination are different files to avoid "same file" error
            source_path = Path(self.best_model_path).resolve()
            dest_path = best_model_path.resolve()
            
            if source_path != dest_path:
                shutil.copy2(self.best_model_path, best_model_path)
                console.print(f"🔄 Also updated: [green]best_model.zip[/green]")
            else:
                console.print(f"🔄 [yellow]best_model.zip is already the current best model[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Error saving best model to general folder: {str(e)}[/red]")

def get_existing_models():
    """Find all existing trained models, prioritizing important ones"""
    models = {}
    
    # Search in models directory first (usually contains best models)
    models_dir = Path("models")
    if models_dir.exists():
        for model_file in models_dir.glob("*.zip"):
            rel_path = str(model_file.relative_to(models_dir.parent))
            model_info = {
                'path': str(model_file.absolute()),
                'name': model_file.name,
                'episode': 'models',
                'size_mb': model_file.stat().st_size / 1024 / 1024,
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime),
                'priority': 1  # High priority for models in main directory
            }
            models[rel_path] = model_info
    
    # Search in episodes directory - but only for important models
    episodes_dir = Path("episodes")
    if episodes_dir.exists():
        for episode_dir in episodes_dir.iterdir():
            if episode_dir.is_dir():
                models_dir = episode_dir / "models"
                if models_dir.exists():
                    # Get all model files for this episode
                    episode_models = list(models_dir.glob("*.zip"))
                    
                    # Priority 1: Final models (most important)
                    final_models = [f for f in episode_models if f.name.startswith("final_")]
                    for model_file in final_models:
                        rel_path = str(model_file.relative_to(episodes_dir.parent))
                        model_info = {
                            'path': str(model_file.absolute()),
                            'name': model_file.name,
                            'episode': episode_dir.name,
                            'size_mb': model_file.stat().st_size / 1024 / 1024,
                            'modified': datetime.fromtimestamp(model_file.stat().st_mtime),
                            'priority': 2  # High priority for final models
                        }
                        models[rel_path] = model_info
                    
                    # Priority 2: Interrupted models (if no final model exists)
                    if not final_models:
                        interrupted_models = [f for f in episode_models if f.name.startswith("interrupted_")]
                        for model_file in interrupted_models:
                            rel_path = str(model_file.relative_to(episodes_dir.parent))
                            model_info = {
                                'path': str(model_file.absolute()),
                                'name': model_file.name,
                                'episode': episode_dir.name,
                                'size_mb': model_file.stat().st_size / 1024 / 1024,
                                'modified': datetime.fromtimestamp(model_file.stat().st_mtime),
                                'priority': 3  # Medium priority for interrupted models
                            }
                            models[rel_path] = model_info
                    
                    # Priority 3: Only the latest checkpoint per episode (fallback)
                    checkpoint_models = [f for f in episode_models if f.name.startswith("checkpoint_")]
                    if checkpoint_models and not final_models:
                        # Get the most recent checkpoint only
                        latest_checkpoint = max(checkpoint_models, key=lambda f: f.stat().st_mtime)
                        rel_path = str(latest_checkpoint.relative_to(episodes_dir.parent))
                        model_info = {
                            'path': str(latest_checkpoint.absolute()),
                            'name': latest_checkpoint.name,
                            'episode': episode_dir.name,
                            'size_mb': latest_checkpoint.stat().st_size / 1024 / 1024,
                            'modified': datetime.fromtimestamp(latest_checkpoint.stat().st_mtime),
                            'priority': 4  # Lower priority for checkpoints
                        }
                        models[rel_path] = model_info
    
    # Filter and sort models by priority and recency
    important_models = {}
    
    # Always include models from main models directory
    for path, info in models.items():
        if info['priority'] == 1:  # Models directory
            important_models[path] = info
    
    # Include final models from episodes
    for path, info in models.items():
        if info['priority'] == 2:  # Final models
            important_models[path] = info
    
    # If we have fewer than 8 important models, add some others
    if len(important_models) < 8:
        remaining_models = [(path, info) for path, info in models.items() 
                           if path not in important_models]
        # Sort by priority (lower is better) then by modification time (newer first)
        remaining_models.sort(key=lambda x: (x[1]['priority'], -x[1]['modified'].timestamp()))
        
        # Add up to fill 8 slots total
        for path, info in remaining_models[:8 - len(important_models)]:
            important_models[path] = info
    
    return important_models

def cleanup_all_models():
    """Completely clean up ALL models from both episodes and models directories"""
    console.print("\n[bold red]🗑️ CLEANUP ALL MODELS[/bold red]")
    console.print("[yellow]⚠️  This will permanently delete ALL trained models![/yellow]")
    console.print("[yellow]This includes:[/yellow]")
    console.print("  • All episode models (checkpoints, final models)")
    console.print("  • All models in the models/ directory")
    console.print("  • best_model.zip and all other saved models")
    console.print("  • Model files will NOT be archived - they will be deleted!")
    
    from rich.prompt import Confirm
    
    # Double confirmation
    if not Confirm.ask("\n[red]Are you absolutely sure you want to delete ALL models?[/red]"):
        console.print("[green]Cleanup cancelled[/green]")
        return False
    
    if not Confirm.ask("[red]This cannot be undone! Final confirmation - DELETE ALL MODELS?[/red]"):
        console.print("[green]Cleanup cancelled[/green]")
        return False
    
    try:
        deleted_count = 0
        total_size_mb = 0
        
        # Clean up episodes directory
        episodes_dir = Path("episodes")
        if episodes_dir.exists():
            console.print("\n[yellow]🗂️ Cleaning episodes directory...[/yellow]")
            for episode_dir in episodes_dir.iterdir():
                if episode_dir.is_dir():
                    models_dir = episode_dir / "models"
                    if models_dir.exists():
                        for model_file in models_dir.glob("*.zip"):
                            file_size = model_file.stat().st_size / 1024 / 1024
                            model_file.unlink()
                            console.print(f"  ❌ Deleted: {model_file.relative_to(episodes_dir.parent)}")
                            deleted_count += 1
                            total_size_mb += file_size
                        
                        # Remove empty models directory
                        if not any(models_dir.iterdir()):
                            models_dir.rmdir()
                            console.print(f"  📁 Removed empty directory: {models_dir.relative_to(episodes_dir.parent)}")
        
        # Clean up main models directory
        models_dir = Path("models")
        if models_dir.exists():
            console.print("\n[yellow]🤖 Cleaning models directory...[/yellow]")
            for model_file in models_dir.glob("*.zip"):
                file_size = model_file.stat().st_size / 1024 / 1024
                model_file.unlink()
                console.print(f"  ❌ Deleted: {model_file.name}")
                deleted_count += 1
                total_size_mb += file_size
            
            # Also clean other model formats
            for model_file in models_dir.glob("*.pkl"):
                file_size = model_file.stat().st_size / 1024 / 1024
                model_file.unlink()
                console.print(f"  ❌ Deleted: {model_file.name}")
                deleted_count += 1
                total_size_mb += file_size
        
        if deleted_count > 0:
            console.print(f"\n[bold green]✅ Cleanup completed![/bold green]")
            console.print(f"[green]   🗑️ Deleted {deleted_count} model files[/green]")
            console.print(f"[green]   💾 Freed {total_size_mb:.1f} MB of disk space[/green]")
        else:
            console.print("\n[yellow]📭 No model files found to delete[/yellow]")
        
        return True
        
    except Exception as e:
        console.print(f"\n[red]❌ Error during cleanup: {str(e)}[/red]")
        return False

def setup_multi_episode_training(reward_config=None):
    """Setup and run multi-episode training
    
    Args:
        reward_config (dict, optional): Custom reward configuration to use instead of default
    """
    console.print("[bold]🎯 Multi-Episode Training Setup[/bold]")
    
    # Show reward configuration status
    if reward_config is not None:
        console.print("[bold green]🎯 Using Custom Reward Configuration[/bold green]")
        console.print("[dim]Enhanced trading behavior improvements active[/dim]\n")
    else:
        console.print("[bold]🎯 Using Default Reward Configuration[/bold]\n")
    
    # Get data file
    from pathlib import Path
    data_files = {}
    data_dir = Path("data")
    if data_dir.exists():
        for file in data_dir.rglob("*.csv"):
            relative_path = str(file.relative_to(data_dir))
            data_files[relative_path] = str(file)
    
    if not data_files:
        console.print("[red]❌ No data files found![/red]")
        return
    
    # Display available files
    table = Table(title="Available Data Files")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("File", style="green")
    
    file_list = list(data_files.items())
    for i, (name, path) in enumerate(file_list):
        table.add_row(str(i+1), name)
    
    console.print(table)
    
    choice = IntPrompt.ask("Select data file", default=1)
    if 1 <= choice <= len(file_list):
        data_path = file_list[choice-1][1]
    else:
        console.print("[red]Invalid selection[/red]")
        return
    
    # Model selection - existing or new
    console.print("\n[bold]🤖 Model Selection[/bold]")
    existing_models = get_existing_models()
    
    if existing_models:
        console.print("\n[cyan]Found existing models:[/cyan]")
        
        # Display existing models with improved formatting
        model_table = Table(title="Important Available Models")
        model_table.add_column("Index", style="cyan", no_wrap=True)
        model_table.add_column("Model Name", style="green")
        model_table.add_column("Type", style="yellow")
        model_table.add_column("Episode/Location", style="blue")
        model_table.add_column("Size (MB)", style="magenta")
        model_table.add_column("Modified", style="white")
        
        # Sort models by priority (lower is better) and modification time (newer first)
        model_list = list(existing_models.items())
        model_list.sort(key=lambda x: (x[1]['priority'], -x[1]['modified'].timestamp()))
        
        # Add "Create New Model" option
        model_table.add_row("0", "[bold]🆕 Create New Model[/bold]", "New", "New Training", "-", "-")
        
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
        
        # Add cleanup option at the bottom
        cleanup_index = len(model_list) + 1
        model_table.add_row(str(cleanup_index), "[bold red]🗑️ CLEANUP ALL MODELS[/bold red]", "🗑️ Delete", "Delete Everything", "-", "-")
        
        console.print(model_table)
        
        # Show helpful info about model types
        console.print("\n[dim]Model Types:[/dim]")
        console.print("[dim]🏆 Best = Highest performing models[/dim]")
        console.print("[dim]✅ Final = Completed episode models[/dim]")
        console.print("[dim]⚠️ Interrupted = Partially trained models[/dim]")
        console.print("[dim]📝 Checkpoint = Latest checkpoint per episode[/dim]")
        
        model_choice = IntPrompt.ask(
            f"Select model (0 = new, 1-{len(model_list)} = existing, {cleanup_index} = cleanup all)", 
            default=0
        )
        
        if model_choice == cleanup_index:
            # Handle cleanup
            if cleanup_all_models():
                console.print("[green]All models have been deleted. You can now create a new model.[/green]")
                starting_model_path = None
            else:
                console.print("[yellow]Cleanup cancelled. Returning to main menu.[/yellow]")
                return
        elif model_choice == 0:
            # Create new model
            starting_model_path = None
            console.print("[green]✅ Creating new model from scratch[/green]")
        elif 1 <= model_choice <= len(model_list):
            # Use existing model
            selected_model = model_list[model_choice-1][1]
            starting_model_path = selected_model['path']
            console.print(f"[green]✅ Using existing model:[/green]")
            console.print(f"   📁 Path: {selected_model['path']}")
            console.print(f"   📦 Size: {selected_model['size_mb']:.1f} MB")
            console.print(f"   📅 Modified: {selected_model['modified'].strftime('%Y-%m-%d %H:%M')}")
        else:
            console.print("[red]Invalid model selection[/red]")
            return
    else:
        console.print("[yellow]No existing models found.[/yellow]")
        console.print("[cyan]Options:[/cyan]")
        console.print("1. 🆕 Create new model")
        console.print("2. 🗑️ Cleanup any hidden/corrupted models and create new")
        
        choice = IntPrompt.ask("Select option", default=1)
        if choice == 2:
            if cleanup_all_models():
                console.print("[green]Cleanup completed. Creating new model.[/green]")
            else:
                console.print("[yellow]Cleanup cancelled. Creating new model anyway.[/yellow]")
        
        starting_model_path = None
    
    # Data configuration
    console.print("\n[bold]📊 Data Split Configuration[/bold]")
    console.print("Choose validation approach:")
    console.print("1. 🎯 Simple split - Use last 5% for validation (RECOMMENDED)")
    console.print("2. 📈 Walk-forward splits - Multiple small training datasets")
    
    split_choice = IntPrompt.ask("Select data split approach", default=1)
    
    if split_choice == 1:
        validation_pct = FloatPrompt.ask("Validation percentage (0.05 = 5%)", default=0.05)
        use_simple_split = True
    else:
        validation_pct = 0.05  # Default for walk-forward
        use_simple_split = False
    
    # Training configuration
    num_episodes = IntPrompt.ask("Number of episodes to train", default=3)
    timesteps_per_episode = IntPrompt.ask("Timesteps per episode", default=50000)
    
    # Model architecture selection
    architectures = ["cnn_lstm", "attention_cnn_lstm", "resnet_lstm"]
    console.print("\nAvailable architectures:")
    for i, arch in enumerate(architectures):
        console.print(f"{i+1}. {arch}")
    
    arch_choice = IntPrompt.ask("Select architecture", default=2)
    if 1 <= arch_choice <= len(architectures):
        model_architecture = architectures[arch_choice-1]
    else:
        model_architecture = "attention_cnn_lstm"
    
    # Algorithm selection
    algorithms = ["ppo", "a2c", "sac"]
    console.print("\nAvailable algorithms:")
    for i, algo in enumerate(algorithms):
        console.print(f"{i+1}. {algo}")
    
    algo_choice = IntPrompt.ask("Select algorithm", default=1)
    if 1 <= algo_choice <= len(algorithms):
        algorithm = algorithms[algo_choice-1]
    else:
        algorithm = "ppo"
    
    # Base configuration
    base_config = {
        "training_params": {
            "initial_equity": FloatPrompt.ask("Initial equity", default=10000.0),
            "max_leverage": FloatPrompt.ask("Max leverage", default=10.0),
            "window_size": IntPrompt.ask("Window size", default=20),
            "stop_loss_pct": FloatPrompt.ask("Stop loss %", default=0.02),
            "take_profit_pct": FloatPrompt.ask("Take profit %", default=0.04),
            "max_risk_per_trade": FloatPrompt.ask("Max risk per trade (0.02 = 2%)", default=0.02),
            "maintenance_margin_rate": FloatPrompt.ask("Maintenance margin rate (0.004 = 0.4%)", default=0.004),
            "liquidation_fee_rate": FloatPrompt.ask("Liquidation fee rate (0.005 = 0.5%)", default=0.005),
            "n_envs": IntPrompt.ask("Parallel environments", default=1)
        }
    }
    
    # Create and run trainer
    trainer = MultiEpisodeTrainer(
        data_path, 
        base_config, 
        starting_model_path=starting_model_path,
        validation_pct=validation_pct,
        use_simple_split=use_simple_split,
        num_episodes=num_episodes,
        reward_config=reward_config  # Pass the reward configuration
    )
    trainer.run_multi_episode_training(
        num_episodes=num_episodes,
        model_architecture=model_architecture,
        algorithm=algorithm,
        timesteps_per_episode=timesteps_per_episode
    )

if __name__ == "__main__":
    setup_multi_episode_training()
