# Auto-Continue Multi-Episode Training

## Overview
The multi-episode training system now supports automatic episode continuation with a configurable timeout, eliminating the need for manual confirmation between episodes.

## Key Features

### ✅ Automatic Episode Continuation
- **60-second timeout** by default between episodes
- **Countdown display** showing remaining time
- **Manual override** option: Press Enter to continue immediately or 'n' + Enter to stop
- **Windows-compatible** implementation using threading

### ✅ Enhanced User Experience
- Clear visual feedback with Rich console formatting
- Real-time countdown with overwrite display
- Graceful handling of user interruptions (Ctrl+C)
- Fallback to default behavior on timeout

## How It Works

### Timeout Confirmation Function
```python
def timeout_confirmation(prompt: str, timeout_seconds: int = 60, default: bool = True) -> bool:
```

**Parameters:**
- `prompt`: Message displayed to user
- `timeout_seconds`: Seconds to wait (default: 60)
- `default`: Action when timeout occurs (default: True = continue)

**Behavior:**
1. Displays prompt and instructions
2. Shows countdown from timeout_seconds to 0
3. Monitors for user input in background thread
4. Auto-continues if no input received within timeout
5. Respects user choice if input provided

### Episode Flow
```
Episode 1 Complete → 60s countdown → Auto-continue to Episode 2
Episode 2 Complete → 60s countdown → Auto-continue to Episode 3
...etc
```

## Usage Examples

### Standard Multi-Episode Training
```python
# Will auto-continue every 60 seconds
trainer.run_training(
    num_episodes=5,
    model_architecture='attention_cnn_lstm',
    algorithm='PPO',
    timesteps_per_episode=50000
)
```

### Manual Override Options
During countdown:
- **Press Enter**: Continue immediately to next episode
- **Type 'n' + Enter**: Stop training 
- **Ctrl+C**: Emergency stop
- **Wait**: Auto-continue after timeout

## Visual Output Example
```
Episode 1 completed
💤 60-second break before next episode...

Continue to next episode?
Auto-continuing in 60 seconds...
Press Enter to continue now, or type 'n' + Enter to stop
Auto-continuing in 45 seconds...
Auto-continuing in 44 seconds...
...
✅ Auto-continuing to next episode
```

## Benefits

1. **Unattended Training**: Can run overnight or for extended periods
2. **Flexible Control**: Manual override when monitoring
3. **System Recovery**: Prevents hanging on user input
4. **Progress Visibility**: Clear countdown and status

## Configuration

The timeout can be customized by modifying the call in `multi_episode_training.py`:

```python
if not timeout_confirmation("Continue to next episode?", timeout_seconds=120, default=True):
```

## Compatibility

- ✅ **Windows**: Full support with threading-based implementation
- ✅ **Linux/Mac**: Full support 
- ✅ **PowerShell**: Native compatibility
- ✅ **Command Prompt**: Full support

## Testing

The `test_auto_continue.py` script referenced here was never written — the file
was committed empty and has since been removed. Verify the behaviour by starting
a short multi-episode run and leaving the confirmation prompt unanswered; it
should time out and continue on its own.
