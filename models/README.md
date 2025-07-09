# Models Directory

This directory stores trained machine learning models.

## Directory Structure

```
models/
├── README.md                    # This file
├── *.pth                       # PyTorch model files
├── *.pkl                       # Pickled scikit-learn models
└── *.joblib                    # Joblib serialized models
```

## Model Files

- **PyTorch Models**: `.pth` files containing trained neural networks
- **Scalers**: `.pkl` files containing fitted StandardScaler objects
- **Metadata**: JSON files with model configuration and metrics

## Usage

Models are automatically saved during training and loaded during:
- Backtesting
- Live trading
- Model evaluation

## Note

Model files are excluded from Git tracking due to their large size. Train your models locally using the training scripts.
