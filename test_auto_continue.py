"""
Test script for automatic episode continuation functionality
"""
import sys
from multi_episode_training import setup_multi_episode_training

def test_auto_continue():
    """Test the automatic continuation feature"""
    print("🧪 Testing Multi-Episode Training with Auto-Continue")
    print("=" * 60)
    
    try:
        # Set up multi-episode training
        trainer = setup_multi_episode_training()
        
        if trainer is None:
            print("❌ Failed to set up trainer")
            return False
        
        print("✅ Trainer setup successful")
        print("📋 Available episodes:", len(trainer.data_splits))
        
        # Test with just 2 episodes and minimal timesteps for quick testing
        print("\n🚀 Starting test run with 2 episodes...")
        print("⏰ Each episode will auto-continue after 10 seconds")
        
        # Temporarily modify the timeout for testing
        import multi_episode_training
        original_function = multi_episode_training.timeout_confirmation
        
        def quick_timeout(prompt, timeout_seconds=60, default=True):
            return original_function(prompt, timeout_seconds=10, default=default)
        
        # Monkey patch for testing
        multi_episode_training.timeout_confirmation = quick_timeout
        
        try:
            trainer.run_training(
                num_episodes=2,
                model_architecture='attention_cnn_lstm',
                algorithm='PPO',
                timesteps_per_episode=100  # Very small for testing
            )
            print("✅ Auto-continue test completed successfully!")
            return True
            
        finally:
            # Restore original function
            multi_episode_training.timeout_confirmation = original_function
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_auto_continue()
    sys.exit(0 if success else 1)
