"""
Observation space wrapper to convert Dict observations to Box format
for compatibility with models trained on flattened observations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any


class DictToBoxObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that converts Dict observation space to Box observation space
    by flattening the dictionary values into a single array.
    
    This is needed when the environment returns Dict observations but
    the trained model expects Box (flat array) observations.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get the original Dict observation space
        original_space = env.observation_space
        assert isinstance(original_space, spaces.Dict), "Environment must have Dict observation space"
        
        # Calculate total flattened size
        total_size = 0
        self._space_info = {}
        
        for key, space in original_space.spaces.items():
            if isinstance(space, spaces.Box):
                space_size = np.prod(space.shape)
                self._space_info[key] = {
                    'start_idx': total_size,
                    'end_idx': total_size + space_size,
                    'shape': space.shape,
                    'size': space_size
                }
                total_size += space_size
            else:
                raise ValueError(f"Unsupported space type for key '{key}': {type(space)}")
        
        # Create new Box observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )
        
        print(f"DictToBoxWrapper: Converting Dict observation to Box({total_size},)")
        for key, info in self._space_info.items():
            print(f"  {key}: indices {info['start_idx']}:{info['end_idx']} (shape {info['shape']})")
    
    def observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert Dict observation to flattened Box observation.
        
        Args:
            observation: Dict with keys 'market_features' and 'portfolio_features'
            
        Returns:
            Flattened numpy array containing all observation data
        """
        # Create output array
        total_size = self.observation_space.shape[0]
        flattened_obs = np.zeros(total_size, dtype=np.float32)
        
        # Fill the flattened array
        for key, data in observation.items():
            if key in self._space_info:
                info = self._space_info[key]
                start_idx = info['start_idx']
                end_idx = info['end_idx']
                
                # Flatten the data and place it in the correct position
                flattened_data = data.flatten()
                
                # Ensure the size matches expected
                if len(flattened_data) != info['size']:
                    raise ValueError(
                        f"Size mismatch for key '{key}': expected {info['size']}, "
                        f"got {len(flattened_data)} (shape {data.shape})"
                    )
                
                flattened_obs[start_idx:end_idx] = flattened_data
            else:
                print(f"Warning: Unknown observation key '{key}' ignored")
        
        return flattened_obs
    
    def get_original_observation_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about how the original Dict observation is mapped to indices."""
        return self._space_info.copy()


def test_wrapper():
    """Test the wrapper with a mock observation."""
    # Create mock observation like the trading environment
    mock_observation = {
        'market_features': np.random.randn(30, 54).astype(np.float32),  # 30 timesteps, 54 features
        'portfolio_features': np.random.randn(13).astype(np.float32)    # 13 portfolio features
    }
    
    # Create mock observation space
    mock_obs_space = spaces.Dict({
        'market_features': spaces.Box(low=-np.inf, high=np.inf, shape=(30, 54), dtype=np.float32),
        'portfolio_features': spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
    })
    
    # Create mock environment that inherits from gym.Env
    class MockEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = mock_obs_space
            self.action_space = spaces.Discrete(3)  # Dummy action space
        
        def step(self, action):
            return mock_observation, 0, False, False, {}
        
        def reset(self, **kwargs):
            return mock_observation, {}
    
    # Test the wrapper
    env = MockEnv()
    wrapped_env = DictToBoxObservationWrapper(env)
    
    print("\nOriginal observation:")
    print(f"  market_features shape: {mock_observation['market_features'].shape}")
    print(f"  portfolio_features shape: {mock_observation['portfolio_features'].shape}")
    
    flattened = wrapped_env.observation(mock_observation)
    print(f"\nFlattened observation shape: {flattened.shape}")
    print(f"Expected total size: {30 * 54 + 13} = {30 * 54 + 13}")
    
    # Verify we can reconstruct
    info = wrapped_env.get_original_observation_info()
    print(f"\nMapping info:")
    for key, mapping in info.items():
        print(f"  {key}: {mapping}")
    
    return wrapped_env, flattened


if __name__ == "__main__":
    test_wrapper()
