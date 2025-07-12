"""
Multi-Episode Training System with Model Persistence
Supports continuous learning and model retraining
"""

import os
import json
import numpy as np
import pandas as pd
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
            json.dump(data, f, indent=2)
    
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
    
    def __init__(self, data_path: str, base_config: Dict):
        self.data_path = data_path
        self.base_config = base_config
        self.episode_tracker = EpisodeTracker()
        
        # Load data
        self.df = pd.read_csv(data_path)
        console.print(f"📊 Loaded data with {len(self.df)} rows")
        
        # Split data for episodes (walk-forward style)
        self.data_splits = self._create_data_splits()
        
        # Training state
        self.current_episode = 0
        self.best_model_path = None
        self.best_performance = None
    
    def _create_data_splits(self, min_train_size: int = 10000, validation_size: int = 2000) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create walk-forward data splits for episodes"""
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
        
        console.print(f"📈 Created {len(splits)} data splits for training episodes")
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
        log_file = f"{episode_dir}/logs/trades_{episode_id}.csv"
        
        # Create environment function
        def env_fn():
            env = FuturesTradingEnv(
                df=train_data,
                log_file=log_file,
                training_iteration=episode_num,
                use_advanced_action_space=True,  # Enable advanced action space by default
                **self.base_config["training_params"]
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
        
        if continue_from_best and self.best_model_path and os.path.exists(self.best_model_path):
            console.print(f"🔄 Continuing from best model: {self.best_model_path}")
            try:
                model = model_class.load(self.best_model_path, env=vec_env)
                model.tensorboard_log = f"{episode_dir}/tensorboard/"
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load existing model: {str(e)}[/yellow]")
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
        
        # Display existing episode history
        self.episode_tracker.display_episode_summary()
        
        try:
            for episode_num in range(num_episodes):
                train_data, val_data = self.data_splits[episode_num]
                
                # Train episode
                model_path = self.train_episode(
                    episode_num=episode_num,
                    train_data=train_data,
                    val_data=val_data,
                    model_architecture=model_architecture,
                    algorithm=algorithm,
                    timesteps=timesteps_per_episode,
                    continue_from_best=True
                )
                
                if model_path is None:
                    console.print(f"[red]Episode {episode_num + 1} failed, stopping training[/red]")
                    break
                
                # Ask if user wants to continue to next episode
                if episode_num < num_episodes - 1:
                    console.print(f"\n[yellow]Episode {episode_num + 1} completed[/yellow]")
                    if not Confirm.ask("Continue to next episode?", default=True):
                        console.print("[yellow]Training stopped by user[/yellow]")
                        break
        
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Multi-episode training interrupted[/yellow]")
        
        # Final summary
        console.print("\n[bold]🎉 Multi-Episode Training Complete![/bold]")
        self.episode_tracker.display_episode_summary()
        
        if self.best_model_path:
            console.print(f"\n🏆 [bold green]Best model: {self.best_model_path}[/bold green]")
            console.print(f"📈 Best return: {self.best_performance['total_return_pct']:.2f}%")

def setup_multi_episode_training():
    """Setup and run multi-episode training"""
    console.print("[bold]🎯 Multi-Episode Training Setup[/bold]")
    
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
    
    # Training configuration
    num_episodes = IntPrompt.ask("Number of episodes to train", default=5)
    timesteps_per_episode = IntPrompt.ask("Timesteps per episode", default=500000)
    
    # Model architecture selection
    architectures = ["cnn_lstm", "attention_cnn_lstm", "resnet_lstm"]
    console.print("\nAvailable architectures:")
    for i, arch in enumerate(architectures):
        console.print(f"{i+1}. {arch}")
    
    arch_choice = IntPrompt.ask("Select architecture", default=1)
    if 1 <= arch_choice <= len(architectures):
        model_architecture = architectures[arch_choice-1]
    else:
        model_architecture = "cnn_lstm"
    
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
            "max_leverage": FloatPrompt.ask("Max leverage", default=25.0),
            "window_size": IntPrompt.ask("Window size", default=60),
            "stop_loss_pct": FloatPrompt.ask("Stop loss %", default=0.02),
            "take_profit_pct": FloatPrompt.ask("Take profit %", default=0.04),
            "maintenance_margin_rate": FloatPrompt.ask("Maintenance margin rate (0.004 = 0.4%)", default=0.004),
            "liquidation_fee_rate": FloatPrompt.ask("Liquidation fee rate (0.005 = 0.5%)", default=0.005),
            "n_envs": IntPrompt.ask("Parallel environments", default=4)
        }
    }
    
    # Create and run trainer
    trainer = MultiEpisodeTrainer(data_path, base_config)
    trainer.run_multi_episode_training(
        num_episodes=num_episodes,
        model_architecture=model_architecture,
        algorithm=algorithm,
        timesteps_per_episode=timesteps_per_episode
    )

if __name__ == "__main__":
    setup_multi_episode_training()
