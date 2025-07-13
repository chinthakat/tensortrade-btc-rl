# Archive Directory

This directory contains archived logs, models, and TensorBoard files to keep the project organized.

## Automatic Archiving

The system automatically archives old files when `main.py` or `train_model.py` is started:

### Archive Schedule:
- **Logs** (trades_*.csv, *.log, *.npz): Archived after 2-3 days
- **Models** (*.zip, *.pkl): Archived after 10-14 days (except `best_model.zip`)
- **TensorBoard Logs**: Archived after 5-7 days

### Archive Cleanup:
- Maximum 15 archive files are kept
- Older archives are automatically deleted

## Archive Contents

Archive files are organized by type and timestamp:
- `logs_archive_YYYYMMDD_HHMMSS.zip` - Contains old log files
- `models_archive_YYYYMMDD_HHMMSS.zip` - Contains old model files  
- `tensorboard_archive_YYYYMMDD_HHMMSS.zip` - Contains old TensorBoard logs

## Manual Archiving

You can manually run archiving:

```python
from log_archiver import archive_startup_logs

# Archive with custom settings
archive_startup_logs(
    base_dir=".",
    log_age_days=1,      # More aggressive
    model_age_days=7,    
    tensorboard_age_days=3
)
```

## Restoring Files

To restore archived files:
1. Extract the desired archive file
2. Copy files back to their original locations
3. Maintain directory structure (logs/, models/, tensorboard_logs/)

## Benefits

- **Cleaner workspace**: Keeps active directories uncluttered
- **Performance**: Faster file operations with fewer files
- **Organization**: Easy to find and manage historical data
- **Backup**: Archives serve as compressed backups
