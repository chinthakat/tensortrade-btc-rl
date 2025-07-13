"""
Hyperparameter Optimization Script for Trading Bot
Uses Optuna for automated hyperparameter tuning
"""

import os
import json
import optuna
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

# Local imports
from trading_environment import FuturesTradingEnv
from action_space_wrapper import wrap_environment_for_algorithm
from train_model import create_vectorized_environment, ModelConfig
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.evaluation import evaluate_policy

console = Console()

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, data_file: str, algorithm: str = "ppo", n_trials: int = 50):
        self.data_file = data_file
        self.algorithm = algorithm.lower()
        self.n_trials = n_trials
        self.df = pd.read_csv(data_file)
        
        # Base parameters
        self.base_params = {
            'initial_equity': 10000.0,
            'max_leverage': 25.0,
            'window_size': 60,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'maintenance_margin_rate': 0.004,
            'liquidation_fee_rate': 0.005,
            'n_envs': 2  # Reduced for optimization
        }
        
        console.print(f"🎯 Optimizing {algorithm.upper()} hyperparameters")
        console.print(f"📊 Using data: {Path(data_file).name}")
        console.print(f"🔬 Running {n_trials} trials")
    
    def create_env_fn(self, trial_params: Dict[str, Any]):
        """Create environment function for optimization"""
        def env_fn():
            env = FuturesTradingEnv(
                df=self.df,
                initial_equity=self.base_params['initial_equity'],
                max_leverage=self.base_params['max_leverage'],
                window_size=self.base_params['window_size'],
                stop_loss_pct=self.base_params['stop_loss_pct'],
                take_profit_pct=self.base_params['take_profit_pct'],
                maintenance_margin_rate=self.base_params['maintenance_margin_rate'],
                liquidation_fee_rate=self.base_params['liquidation_fee_rate'],
                use_advanced_action_space=True
            )
            return wrap_environment_for_algorithm(env, "PPO")
        return env_fn
    
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        try:
            # Suggest hyperparameters based on algorithm
            if self.algorithm == "ppo":
                hyperparams = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                    'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
                    'n_epochs': trial.suggest_int('n_epochs', 3, 20),
                    'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.999),
                    'use_normalization': trial.suggest_categorical('use_normalization', [True, False])
                }
                
                if hyperparams['use_normalization']:
                    hyperparams.update({
                        'norm_obs': True,
                        'norm_reward': trial.suggest_categorical('norm_reward', [True, False]),
                        'clip_obs': trial.suggest_float('clip_obs', 5.0, 20.0),
                        'clip_reward': trial.suggest_float('clip_reward', 5.0, 20.0)
                    })
                
                model_class = PPO
                
            elif self.algorithm == "a2c":
                hyperparams = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'n_steps': trial.suggest_categorical('n_steps', [5, 8, 16, 32]),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
                    'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
                    'use_normalization': trial.suggest_categorical('use_normalization', [True, False])
                }
                
                if hyperparams['use_normalization']:
                    hyperparams.update({
                        'norm_obs': True,
                        'norm_reward': trial.suggest_categorical('norm_reward', [True, False]),
                        'clip_obs': trial.suggest_float('clip_obs', 5.0, 20.0),
                        'clip_reward': trial.suggest_float('clip_reward', 5.0, 20.0)
                    })
                
                model_class = A2C
                
            elif self.algorithm == "sac":
                hyperparams = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                    'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 500000, 1000000]),
                    'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
                    'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2, 4]),
                    'tau': trial.suggest_float('tau', 0.001, 0.02),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'use_normalization': trial.suggest_categorical('use_normalization', [True, False])
                }
                
                if hyperparams['use_normalization']:
                    hyperparams.update({
                        'norm_obs': True,
                        'norm_reward': False,  # SAC typically doesn't normalize rewards
                        'clip_obs': trial.suggest_float('clip_obs', 5.0, 20.0),
                        'clip_reward': 10.0
                    })
                
                model_class = SAC
            
            # Create environment
            env_fn = self.create_env_fn(hyperparams)
            vec_env = create_vectorized_environment(env_fn, self.base_params['n_envs'], hyperparams)
            
            # Create model with suggested hyperparameters
            policy_kwargs = {
                "net_arch": [256, 128]
            }
            
            # Prepare algorithm parameters
            algo_params = {
                "policy": "MultiInputPolicy",
                "env": vec_env,
                "policy_kwargs": policy_kwargs,
                "verbose": 0,
                "learning_rate": hyperparams['learning_rate'],
                "device": "auto"
            }
            
            # Add algorithm-specific parameters
            if self.algorithm == "ppo":
                algo_params.update({
                    "batch_size": hyperparams['batch_size'],
                    "n_steps": hyperparams['n_steps'],
                    "n_epochs": hyperparams['n_epochs'],
                    "clip_range": hyperparams['clip_range'],
                    "gamma": hyperparams['gamma'],
                    "gae_lambda": hyperparams['gae_lambda']
                })
            elif self.algorithm == "a2c":
                algo_params.update({
                    "n_steps": hyperparams['n_steps'],
                    "gamma": hyperparams['gamma'],
                    "gae_lambda": hyperparams['gae_lambda'],
                    "ent_coef": hyperparams['ent_coef']
                })
            elif self.algorithm == "sac":
                algo_params.update({
                    "batch_size": hyperparams['batch_size'],
                    "buffer_size": hyperparams['buffer_size'],
                    "train_freq": hyperparams['train_freq'],
                    "gradient_steps": hyperparams['gradient_steps'],
                    "tau": hyperparams['tau'],
                    "gamma": hyperparams['gamma']
                })
            
            # Create and train model
            model = model_class(**algo_params)
            
            # Train for a shorter time for optimization
            model.learn(total_timesteps=50000)
            
            # Evaluate model
            eval_env = env_fn()
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=10, deterministic=True
            )
            
            # Clean up
            vec_env.close()
            eval_env.close()
            del model
            
            return mean_reward
            
        except Exception as e:
            console.print(f"[red]Trial failed: {str(e)}[/red]")
            return -1000.0  # Return very low reward for failed trials
    
    def optimize(self):
        """Run hyperparameter optimization"""
        # Create study
        study_name = f"trading_bot_{self.algorithm}_optimization"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Display optimization panel
        opt_text = f"🔬 Optimizing {self.algorithm.upper()} Hyperparameters\n"
        opt_text += f"📊 Trials: {self.n_trials}\n"
        opt_text += f"🎯 Objective: Maximize mean reward"
        
        console.print(Panel(opt_text, title="Hyperparameter Optimization", border_style="blue"))
        
        # Run optimization with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Optimizing hyperparameters...", total=self.n_trials)
            
            def callback(study, trial):
                progress.update(task, advance=1)
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    progress.update(
                        task, 
                        description=f"Trial {trial.number + 1}/{self.n_trials} - Best: {study.best_value:.4f}"
                    )
            
            study.optimize(self.objective, n_trials=self.n_trials, callbacks=[callback])
        
        # Save results
        self.save_optimization_results(study)
        
        return study
    
    def save_optimization_results(self, study):
        """Save optimization results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("optimization_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save best parameters
        best_params = study.best_params.copy()
        best_params['best_value'] = study.best_value
        best_params['algorithm'] = self.algorithm
        best_params['data_file'] = self.data_file
        best_params['timestamp'] = timestamp
        
        best_file = results_dir / f"best_params_{self.algorithm}_{timestamp}.json"
        with open(best_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save study results
        df_trials = study.trials_dataframe()
        trials_file = results_dir / f"trials_{self.algorithm}_{timestamp}.csv"
        df_trials.to_csv(trials_file, index=False)
        
        console.print(f"\n✅ [bold green]Optimization completed![/bold green]")
        console.print(f"🏆 Best value: {study.best_value:.4f}")
        console.print(f"💾 Best parameters saved to: [green]{best_file}[/green]")
        console.print(f"📊 All trials saved to: [green]{trials_file}[/green]")
        
        # Display best parameters
        console.print("\n[bold]🏆 Best Hyperparameters:[/bold]")
        from rich.table import Table
        
        params_table = Table(title="Optimal Parameters")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="green")
        
        for param, value in study.best_params.items():
            params_table.add_row(param, str(value))
        
        console.print(params_table)

def main():
    """Main optimization function"""
    console.print("🎯 [bold]Trading Bot Hyperparameter Optimization[/bold]")
    
    # Get data files
    from train_model import get_data_files
    data_files = get_data_files()
    
    if not data_files:
        console.print("[red]❌ No data files found![/red]")
        return
    
    # Select data file
    console.print("\n📊 Available data files:")
    file_list = list(data_files.items())
    for i, (name, path) in enumerate(file_list):
        console.print(f"  {i+1}. {name}")
    
    choice = int(input("\nSelect data file (enter number): ")) - 1
    if 0 <= choice < len(file_list):
        data_file = file_list[choice][1]
    else:
        console.print("[red]Invalid choice![/red]")
        return
    
    # Select algorithm
    algorithms = ["ppo", "a2c", "sac"]
    console.print("\n⚡ Available algorithms:")
    for i, algo in enumerate(algorithms):
        console.print(f"  {i+1}. {algo.upper()}")
    
    algo_choice = int(input("\nSelect algorithm (enter number): ")) - 1
    if 0 <= algo_choice < len(algorithms):
        algorithm = algorithms[algo_choice]
    else:
        console.print("[red]Invalid choice![/red]")
        return
    
    # Number of trials
    n_trials = int(input("\nNumber of optimization trials (default 50): ") or "50")
    
    # Run optimization
    optimizer = HyperparameterOptimizer(data_file, algorithm, n_trials)
    study = optimizer.optimize()

if __name__ == "__main__":
    main()
