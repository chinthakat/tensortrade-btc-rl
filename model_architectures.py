"""
CNN-LSTM Feature Extractor for Market Data
TensorTrade-inspired deep learning architecture for financial time series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

class CNNLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN-LSTM hybrid architecture for extracting features from market data
    Inspired by TensorTrade's deep learning components
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Get dimensions from observation space
        market_shape = observation_space['market_features'].shape  # (window_size, n_features)
        portfolio_shape = observation_space['portfolio_features'].shape  # (n_portfolio_features,)
        
        self.window_size = market_shape[0]
        self.n_market_features = market_shape[1]
        self.n_portfolio_features = portfolio_shape[0]
        
        # CNN layers for market features
        self.conv1d_1 = nn.Conv1d(
            in_channels=self.n_market_features,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv1d_3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Max pooling
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Portfolio features processing
        self.portfolio_fc = nn.Sequential(
            nn.Linear(self.n_portfolio_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Combined features processing
        # LSTM output is bidirectional, so 128 * 2 = 256
        # Portfolio features contribute 32
        combined_features = 256 + 32
        
        self.final_fc = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            observations: Dict containing 'market_features' and 'portfolio_features'
        
        Returns:
            Extracted features tensor
        """
        market_features = observations['market_features']  # Shape: (batch, window_size, n_features)
        portfolio_features = observations['portfolio_features']  # Shape: (batch, n_portfolio_features)
        
        batch_size = market_features.shape[0]
        
        # Process market features through CNN
        # Transpose for Conv1d: (batch, n_features, window_size)
        x = market_features.transpose(1, 2)
        
        # First CNN block
        x = self.conv1d_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second CNN block
        x = self.conv1d_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        
        # Third CNN block
        x = self.conv1d_3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Transpose back for LSTM: (batch, sequence_length, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from LSTM (many-to-one)
        # For bidirectional LSTM, concatenate forward and backward hidden states
        market_features_processed = lstm_out[:, -1, :]  # Shape: (batch, 256)
        
        # Process portfolio features
        portfolio_features_processed = self.portfolio_fc(portfolio_features)  # Shape: (batch, 32)
        
        # Combine market and portfolio features
        combined_features = torch.cat([market_features_processed, portfolio_features_processed], dim=1)
        
        # Final processing
        output = self.final_fc(combined_features)
        
        return output

class AttentionCNNLSTMExtractor(BaseFeaturesExtractor):
    """
    Advanced CNN-LSTM with attention mechanism for better temporal modeling
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        market_shape = observation_space['market_features'].shape
        portfolio_shape = observation_space['portfolio_features'].shape
        
        self.window_size = market_shape[0]
        self.n_market_features = market_shape[1]
        self.n_portfolio_features = portfolio_shape[0]
        
        # CNN feature extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(self.n_market_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM with attention
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_hidden_size * 2,  # Bidirectional LSTM
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Portfolio processing
        self.portfolio_layers = nn.Sequential(
            nn.Linear(self.n_portfolio_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, features_dim)
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        market_features = observations['market_features']
        portfolio_features = observations['portfolio_features']
        
        # CNN processing
        x = market_features.transpose(1, 2)  # (batch, features, time)
        cnn_out = self.cnn_layers(x)
        cnn_out = cnn_out.transpose(1, 2)  # (batch, time, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_out)
        
        # Self-attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over time dimension
        market_features_final = torch.mean(attended_out, dim=1)
        
        # Portfolio processing
        portfolio_out = self.portfolio_layers(portfolio_features)
        
        # Combine and final processing
        combined = torch.cat([market_features_final, portfolio_out], dim=1)
        output = self.final_layers(combined)
        
        return output

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)

class ResNetLSTMExtractor(BaseFeaturesExtractor):
    """
    ResNet-style CNN with LSTM for robust feature extraction
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        market_shape = observation_space['market_features'].shape
        portfolio_shape = observation_space['portfolio_features'].shape
        
        self.n_market_features = market_shape[1]
        self.n_portfolio_features = portfolio_shape[0]
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(self.n_market_features, 64, kernel_size=7, padding=3)
        self.initial_bn = nn.BatchNorm1d(64)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.res_block3 = ResidualBlock(128, 256)
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(30)  # Fixed output length
        
        # LSTM
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, dropout=0.2)
        
        # Portfolio processing
        self.portfolio_fc = nn.Sequential(
            nn.Linear(self.n_portfolio_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Final processing
        self.final_fc = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, features_dim)
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        market_features = observations['market_features']
        portfolio_features = observations['portfolio_features']
        
        # Initial convolution
        x = market_features.transpose(1, 2)
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Adaptive pooling and LSTM
        x = self.adaptive_pool(x)
        x = x.transpose(1, 2)
        
        lstm_out, (hidden, _) = self.lstm(x)
        market_out = hidden[-1]  # Last layer, last time step
        
        # Portfolio processing
        portfolio_out = self.portfolio_fc(portfolio_features)
        
        # Combine and final processing
        combined = torch.cat([market_out, portfolio_out], dim=1)
        output = self.final_fc(combined)
        
        return output
