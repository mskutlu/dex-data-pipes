"""
Temporal sequence models for optimal-stopping sell signal prediction
TCN and Temporal Fusion Transformer implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import polars as pl_df
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import math

from ..config import MODEL_CONFIG, OPTIMAL_STOPPING_CONFIG

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for sequence modeling
    Optimized for DEX trading data with dilated convolutions
    """
    
    def __init__(self, num_inputs: int, num_channels: List[int], 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size, 
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    """Basic building block for TCN"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from temporal dimension"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHorizonTransformer(nn.Module):
    """
    Transformer model for multi-horizon upside-left prediction
    Predicts quantiles of return distribution
    """
    
    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, num_horizons: int = 5, num_quantiles: int = 9):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_horizons = num_horizons
        self.num_quantiles = num_quantiles
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4, dropout=MODEL_CONFIG.dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Multi-horizon output heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(MODEL_CONFIG.dropout),
                nn.Linear(d_model // 2, num_quantiles)
            ) for _ in range(num_horizons)
        ])
        
        # Quantile levels for pinball loss
        self.quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size, seq_len = x.shape[:2]
        
        # Project to model dimension
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Use last timestep for prediction
        last_encoded = encoded[:, -1, :]  # (batch, d_model)
        
        # Multi-horizon predictions
        horizon_outputs = []
        for head in self.horizon_heads:
            quantile_pred = head(last_encoded)  # (batch, num_quantiles)
            horizon_outputs.append(quantile_pred)
            
        return torch.stack(horizon_outputs, dim=1)  # (batch, num_horizons, num_quantiles)
        
    def quantile_loss(self, y_pred, y_true, quantiles):
        """Quantile (pinball) loss for distributional prediction"""
        # y_pred: (batch, num_horizons, num_quantiles)
        # y_true: (batch, num_horizons)
        
        batch_size, num_horizons, num_quantiles = y_pred.shape
        quantiles = quantiles.view(1, 1, -1).expand(batch_size, num_horizons, -1)
        
        y_true_expanded = y_true.unsqueeze(-1).expand(-1, -1, num_quantiles)
        
        error = y_true_expanded - y_pred
        loss = torch.max(quantiles * error, (quantiles - 1) * error)
        
        return loss.mean()


class DEXSequenceDataset(Dataset):
    """Dataset for temporal sequence modeling of DEX data"""
    
    def __init__(self, data: pl_df.DataFrame, sequence_length: int = 60, 
                 target_col: str = 'upside_left', feature_cols: List[str] = None):
        
        self.sequence_length = sequence_length
        self.target_col = target_col
        
        # Convert to pandas for easier processing
        df = data.to_pandas().sort_values(['token', 'pool', 'timestamp'])
        
        # Prepare features
        if feature_cols is None:
            exclude_cols = {'timestamp', 'token', 'pool', 'bucket', target_col, 
                           'net_return_current', 'entry_price', 'current_price'}
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
        self.feature_cols = feature_cols
        
        # Normalize features
        self.scaler = RobustScaler()
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols].fillna(0))
        
        # Build sequences grouped by (token, pool, entry_time)
        self.sequences = []
        self.targets = []
        
        for (token, pool), group in df.groupby(['token', 'pool']):
            group_sorted = group.sort_values('timestamp').reset_index(drop=True)
            
            # Create overlapping sequences
            for i in range(len(group_sorted) - sequence_length + 1):
                seq_data = group_sorted.iloc[i:i+sequence_length][feature_cols].values
                target_data = group_sorted.iloc[i+sequence_length-1][target_col]
                
                self.sequences.append(seq_data.astype(np.float32))
                self.targets.append(target_data)
                
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets, dtype=np.float32)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])


class TCNSellPredictor(pl.LightningModule):
    """Lightning module for TCN-based sell signal prediction"""
    
    def __init__(self, input_dim: int, tcn_channels: List[int] = None, 
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        super().__init__()
        
        self.save_hyperparameters()
        
        if tcn_channels is None:
            tcn_channels = [64, 128, 128, 64]
            
        # TCN backbone
        self.tcn = TemporalConvNet(input_dim, tcn_channels, dropout=MODEL_CONFIG.dropout)
        
        # Output heads
        self.upside_head = nn.Sequential(
            nn.Linear(tcn_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG.dropout),
            nn.Linear(64, 1)
        )
        
        self.drawdown_head = nn.Sequential(
            nn.Linear(tcn_channels[-1], 64),
            nn.ReLU(), 
            nn.Dropout(MODEL_CONFIG.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # TCN expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # TCN encoding
        tcn_out = self.tcn(x)  # (batch, channels, seq_len)
        
        # Use last timestep
        last_out = tcn_out[:, :, -1]  # (batch, channels)
        
        # Predictions
        upside_pred = self.upside_head(last_out).squeeze(-1)
        drawdown_pred = self.drawdown_head(last_out).squeeze(-1)
        
        return upside_pred, drawdown_pred
        
    def training_step(self, batch, batch_idx):
        x, y_upside = batch
        
        upside_pred, drawdown_pred = self(x)
        
        # Upside regression loss
        upside_loss = F.mse_loss(upside_pred, y_upside)
        
        # Create drawdown labels (simplified: negative upside = high risk)
        y_drawdown = (y_upside < -0.02).float()  # 2% negative upside = drawdown risk
        drawdown_loss = F.binary_cross_entropy(drawdown_pred, y_drawdown)
        
        # Combined loss
        total_loss = upside_loss + 0.5 * drawdown_loss
        
        self.log('train_loss', total_loss)
        self.log('train_upside_loss', upside_loss)
        self.log('train_drawdown_loss', drawdown_loss)
        
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        x, y_upside = batch
        
        upside_pred, drawdown_pred = self(x)
        
        upside_loss = F.mse_loss(upside_pred, y_upside)
        y_drawdown = (y_upside < -0.02).float()
        drawdown_loss = F.binary_cross_entropy(drawdown_pred, y_drawdown)
        
        total_loss = upside_loss + 0.5 * drawdown_loss
        
        self.log('val_loss', total_loss)
        self.log('val_upside_loss', upside_loss) 
        self.log('val_drawdown_loss', drawdown_loss)
        
        return total_loss
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay)


class TransformerSellPredictor(pl.LightningModule):
    """Lightning module for Transformer-based multi-horizon prediction"""
    
    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, learning_rate: float = 1e-4):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.transformer = MultiHorizonTransformer(
            input_dim, d_model, nhead, num_layers,
            num_horizons=len(OPTIMAL_STOPPING_CONFIG.horizon_hours),
            num_quantiles=9
        )
        
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.transformer(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Multi-horizon target (replicate for all horizons for now)
        y_multi = y.unsqueeze(1).repeat(1, len(OPTIMAL_STOPPING_CONFIG.horizon_hours))
        
        y_pred = self(x)
        
        # Quantile loss
        loss = self.transformer.quantile_loss(y_pred, y_multi, self.transformer.quantiles)
        
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_multi = y.unsqueeze(1).repeat(1, len(OPTIMAL_STOPPING_CONFIG.horizon_hours))
        y_pred = self(x)
        
        loss = self.transformer.quantile_loss(y_pred, y_multi, self.transformer.quantiles)
        
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
    def get_upside_prediction(self, x):
        """Extract upside-left prediction from quantile outputs"""
        quantile_preds = self(x)
        
        # Use median quantile (index 4) as point prediction
        upside_pred = quantile_preds[:, 0, 4]  # First horizon, median quantile
        
        # Calculate uncertainty (IQR)
        q25 = quantile_preds[:, 0, 2]  # 25th percentile
        q75 = quantile_preds[:, 0, 6]  # 75th percentile
        uncertainty = q75 - q25
        
        return upside_pred, uncertainty


def create_sequence_data_loaders(train_data: pl_df.DataFrame, 
                               val_data: pl_df.DataFrame = None,
                               sequence_length: int = 60,
                               batch_size: int = 256) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for sequence models"""
    
    # Create datasets
    train_dataset = DEXSequenceDataset(train_data, sequence_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # Utilize your multi-core CPU
        pin_memory=True,  # Speed up GPU transfer
        persistent_workers=True
    )
    
    val_loader = None
    if val_data is not None:
        val_dataset = DEXSequenceDataset(val_data, sequence_length)
        val_dataset.scaler = train_dataset.scaler  # Use same scaler
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        
    return train_loader, val_loader
