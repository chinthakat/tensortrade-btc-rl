"""
Action Space Wrapper for Dict to Box conversion
Enables using Dict action spaces with algorithms that only support Box action spaces (like PPO)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any


class DictToBoxActionWrapper(gym.ActionWrapper):
    """
    Wrapper that converts Dict action space to Box action space for compatibility with PPO.
    
    The Dict action space {'leverage': Box(-25.0, 25.0), 'risk_percentage': Box(0.01, 1.0)}
    is flattened to a Box(2,) where:
    - action[0] = leverage (mapped from [-1, 1] to [-25, 25])
    - action[1] = risk_percentage (mapped from [-1, 1] to [0.01, 1.0])
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Check if the environment uses advanced action space
        if hasattr(env, 'use_advanced_action_space') and env.use_advanced_action_space:
            # Check if it's the new enhanced action space with action_type
            if hasattr(env.action_space, 'spaces') and 'action_type' in env.action_space.spaces:
                # New enhanced action space: action_type + leverage + risk_percentage
                self.action_space = spaces.Box(
                    low=-1.0, 
                    high=1.0, 
                    shape=(3,), 
                    dtype=np.float32
                )
                
                # Store original action space bounds for conversion
                self.action_type_count = 4  # 0=HOLD, 1=BUY, 2=SELL, 3=CANCEL
                self.leverage_low = 0.1
                self.leverage_high = env.action_space.spaces['leverage'].high[0]
                self.risk_low = 0.01
                self.risk_high = 1.0
                
                print("🔄 Enhanced action space wrapped: Dict(3) → Box(3,) for PPO compatibility")
                print(f"   Box action[0] → action_type [0-3] (HOLD/BUY/SELL/CANCEL)")
                print(f"   Box action[1] → leverage [{self.leverage_low}, {self.leverage_high}]")
                print(f"   Box action[2] → risk_percentage [{self.risk_low}, {self.risk_high}]")
                
                self.enhanced_actions = True
            else:
                # Legacy advanced action space: leverage + risk_percentage only
                self.action_space = spaces.Box(
                    low=-1.0, 
                    high=1.0, 
                    shape=(2,), 
                    dtype=np.float32
                )
                
                # Store original Dict action space bounds for conversion
                self.leverage_low = -25.0
                self.leverage_high = 25.0
                self.risk_low = 0.01
                self.risk_high = 1.0
                
                print("🔄 Action space wrapped: Dict → Box(2,) for PPO compatibility")
                print(f"   Box action[0] → leverage [{self.leverage_low}, {self.leverage_high}]")
                print(f"   Box action[1] → risk_percentage [{self.risk_low}, {self.risk_high}]")
                
                self.enhanced_actions = False
        else:
            # Environment uses simple action space, no wrapping needed
            print("✅ Simple action space detected, no wrapping needed")
            self.enhanced_actions = False
    
    def action(self, action):
        """
        Convert Box action to Dict action for the environment.
        
        Args:
            action: np.array with values in [-1, 1]
            - For enhanced actions: shape (3,) = [action_type, leverage, risk_percentage]
            - For legacy actions: shape (2,) = [leverage, risk_percentage]
        
        Returns:
            Dict action with appropriate keys
        """
        if hasattr(self.env, 'use_advanced_action_space') and self.env.use_advanced_action_space:
            if self.enhanced_actions:
                # Enhanced action space: 3 parameters
                action_type_norm = np.clip(action[0], -1.0, 1.0)
                leverage_norm = np.clip(action[1], -1.0, 1.0)
                risk_norm = np.clip(action[2], -1.0, 1.0)
                
                # Convert action_type from [-1, 1] to [0, 3] (discrete)
                action_type = int((action_type_norm + 1.0) / 2.0 * (self.action_type_count - 1))
                action_type = np.clip(action_type, 0, self.action_type_count - 1)
                
                # Convert leverage from [-1, 1] to [leverage_low, leverage_high]
                leverage_mapped = (leverage_norm + 1.0) / 2.0 * (self.leverage_high - self.leverage_low) + self.leverage_low
                
                # Convert risk_percentage from [-1, 1] to [0.01, 1.0]
                risk_mapped = (risk_norm + 1.0) / 2.0 * (self.risk_high - self.risk_low) + self.risk_low
                
                return {
                    'action_type': action_type,  # Keep as scalar int
                    'leverage': np.array([leverage_mapped], dtype=np.float32),
                    'risk_percentage': np.array([risk_mapped], dtype=np.float32)
                }
            else:
                # Legacy advanced action space: 2 parameters
                leverage = np.clip(action[0], -1.0, 1.0)
                risk_norm = np.clip(action[1], -1.0, 1.0)
                
                # Map [-1, 1] to actual ranges
                leverage_mapped = leverage * (self.leverage_high - self.leverage_low) / 2.0
                # For risk_percentage, map [-1, 1] to [0.01, 1.0]
                risk_mapped = (risk_norm + 1.0) / 2.0 * (self.risk_high - self.risk_low) + self.risk_low
                
                return {
                    'leverage': np.array([leverage_mapped], dtype=np.float32),
                    'risk_percentage': np.array([risk_mapped], dtype=np.float32)
                }
        else:
            # Pass through for simple action space
            return action
    
    def __getattr__(self, name):
        """
        Forward attribute access to the underlying environment.
        This allows accessing attributes like price_data, current_step, etc.
        """
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class BoxToDictActionWrapper(gym.ActionWrapper):
    """
    Alternative wrapper that converts Box action space to Dict action space.
    Use this when you want to convert a simple Box action to Dict format.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Check if environment uses simple action space but we want to convert to Dict
        if not (hasattr(env, 'use_advanced_action_space') and env.use_advanced_action_space):
            # Convert simple Box action space to Dict action space
            self.action_space = spaces.Dict({
                'leverage': spaces.Box(-25.0, 25.0, (1,), dtype=np.float32),
                'risk_percentage': spaces.Box(0.01, 1.0, (1,), dtype=np.float32)
            })
            
            print("🔄 Action space wrapped: Box → Dict for advanced features")
        else:
            print("✅ Advanced action space already enabled")
    
    def action(self, action):
        """
        Convert Dict action to Box action for environments expecting simple actions.
        
        Args:
            action: Dict with 'leverage' and 'risk_percentage' keys
        
        Returns:
            Box action (leverage value)
        """
        if isinstance(action, dict):
            # Extract leverage for simple action space environments
            return action['leverage']
        else:
            # Convert simple action to dict format
            leverage = np.clip(action[0] if hasattr(action, '__len__') else action, -25.0, 25.0)
            
            # For simple action, derive risk_percentage from leverage magnitude
            risk_percentage = min(0.8, max(0.1, abs(leverage) / 25.0 * 0.8))
            
            return {
                'leverage': np.array([leverage], dtype=np.float32),
                'risk_percentage': np.array([risk_percentage], dtype=np.float32)
            }
    
    def __getattr__(self, name):
        """
        Forward attribute access to the underlying environment.
        This allows accessing attributes like price_data, current_step, etc.
        """
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def wrap_environment_for_algorithm(env, algorithm_name: str = "PPO"):
    """
    Automatically wrap environment based on algorithm requirements.
    
    Args:
        env: The trading environment
        algorithm_name: Name of the RL algorithm ("PPO", "SAC", etc.)
    
    Returns:
        Wrapped environment compatible with the algorithm
    """
    
    # Algorithms that don't support Dict action spaces
    box_only_algorithms = ["PPO", "A2C", "DQN"]
    
    if algorithm_name.upper() in box_only_algorithms:
        if hasattr(env, 'use_advanced_action_space') and env.use_advanced_action_space:
            print(f"🎯 Wrapping environment for {algorithm_name} (Dict → Box)")
            return DictToBoxActionWrapper(env)
        else:
            print(f"✅ Environment already compatible with {algorithm_name}")
            return env
    else:
        # Algorithms that support Dict action spaces (SAC, TD3, etc.)
        print(f"✅ {algorithm_name} supports Dict action spaces")
        return env


if __name__ == "__main__":
    # Test the wrapper
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from trading_environment import FuturesTradingEnv
    import pandas as pd
    
    # Create test data
    np.random.seed(42)
    n_samples = 100
    data = {
        'timestamp': range(n_samples),
        'open': np.random.uniform(45000, 55000, n_samples),
        'high': np.random.uniform(45000, 55000, n_samples),
        'low': np.random.uniform(45000, 55000, n_samples),
        'close': np.random.uniform(45000, 55000, n_samples),
        'volume': np.random.uniform(100, 1000, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Test Dict action space environment
    env = FuturesTradingEnv(df=df, use_advanced_action_space=True, window_size=10)
    wrapped_env = DictToBoxActionWrapper(env)
    
    print(f"Original action space: {env.action_space}")
    print(f"Wrapped action space: {wrapped_env.action_space}")
    
    # Test action conversion
    test_action = np.array([0.5, -0.3], dtype=np.float32)  # Box action
    converted_action = wrapped_env.action(test_action)
    print(f"Box action {test_action} → Dict action {converted_action}")
