# coding: utf-8
"""
WaveNet-style Dilated Causal Convolutions for Time Series Forecasting

Uses dilated causal convolutions to capture long-range dependencies efficiently.

Reference: "WaveNet: A Generative Model for Raw Audio"

Author: xingqiang chen
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_forecaster import BaseForecaster


class CausalConv1d(nn.Module):
    """Causal convolution layer with proper padding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class WaveNetBlock(nn.Module):
    """Single WaveNet block with dilated convolutions and gated activations."""
    
    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Dilated causal convolution
        self.filter_conv = CausalConv1d(
            residual_channels,
            residual_channels,
            kernel_size,
            dilation
        )
        self.gate_conv = CausalConv1d(
            residual_channels,
            residual_channels,
            kernel_size,
            dilation
        )
        
        # 1x1 convolutions
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Gated activation
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x_gated = filter_out * gate_out
        x_gated = self.dropout(x_gated)
        
        # Skip connection
        skip = self.skip_conv(x_gated)
        
        # Residual connection
        residual = self.residual_conv(x_gated) + x
        
        return residual, skip


class WaveNetModel(nn.Module):
    """WaveNet model for time series forecasting."""
    
    def __init__(
        self,
        input_dim: int,
        residual_channels: int = 32,
        skip_channels: int = 32,
        num_blocks: int = 3,
        num_layers_per_block: int = 4,
        kernel_size: int = 2,
        dropout: float = 0.1,
        prediction_length: int = 7
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.prediction_length = prediction_length
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        
        # Input embedding
        self.input_conv = nn.Conv1d(input_dim, residual_channels, 1)
        
        # WaveNet blocks with exponentially increasing dilations
        self.blocks = nn.ModuleList()
        for block in range(num_blocks):
            for layer in range(num_layers_per_block):
                dilation = 2 ** layer
                self.blocks.append(
                    WaveNetBlock(
                        residual_channels,
                        skip_channels,
                        kernel_size,
                        dilation,
                        dropout
                    )
                )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, prediction_length, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Transpose for conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Input embedding
        x = self.input_conv(x)
        
        # WaveNet blocks
        skip_sum = None
        for block in self.blocks:
            x, skip = block(x)
            if skip_sum is None:
                skip_sum = skip
            else:
                skip_sum = skip_sum + skip
        
        # Output
        output = self.output_layers(skip_sum)
        
        # Take last time step and transpose back
        output = output[:, :, -1]  # (batch, prediction_length)
        
        return output


class WaveNetForecaster(BaseForecaster):
    """
    WaveNet-style forecaster using dilated causal convolutions.
    
    Features:
    - Dilated causal convolutions for large receptive fields
    - Gated activations for learning complex patterns
    - Skip connections for gradient flow
    - Efficient training with parallel computation
    """
    
    def __init__(
        self,
        prediction_length: int = 7,
        context_length: int = 30,
        residual_channels: int = 32,
        skip_channels: int = 32,
        num_blocks: int = 3,
        num_layers_per_block: int = 4,
        kernel_size: int = 2,
        dropout: float = 0.1,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto',
        random_seed: int = 42
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for WaveNet. Install with: pip install torch")
        
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            device=device,
            random_seed=random_seed
        )
        
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        self.optimizer = None
        self.scheduler = None
        
    def _build_model(self, input_dim: int):
        """Build the WaveNet model."""
        self.model = WaveNetModel(
            input_dim=input_dim,
            residual_channels=self.residual_channels,
            skip_channels=self.skip_channels,
            num_blocks=self.num_blocks,
            num_layers_per_block=self.num_layers_per_block,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            prediction_length=self.prediction_length
        ).to(self.device)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
    def _train_step(self, batch: Dict) -> float:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        X = torch.FloatTensor(batch['X']).to(self.device)
        y = torch.FloatTensor(batch['y']).to(self.device)
        
        # Reshape for sequence input
        if X.dim() == 2:
            X = X.unsqueeze(1).expand(-1, self.context_length, -1)
        
        predictions = self.model(X)
        
        # Huber loss for robustness
        loss = F.smooth_l1_loss(predictions, y.expand(-1, self.prediction_length))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _predict_step(self, batch: Dict) -> np.ndarray:
        """Perform a single prediction step."""
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(batch['X']).to(self.device)
            
            if X.dim() == 2:
                X = X.unsqueeze(1).expand(-1, self.context_length, -1)
            
            predictions = self.model(X)
            
        return predictions.cpu().numpy()[:, -1]
    
    def get_receptive_field(self) -> int:
        """Calculate the receptive field of the model."""
        layers_total = self.num_blocks * self.num_layers_per_block
        receptive_field = 1
        
        for layer in range(layers_total):
            dilation = 2 ** (layer % self.num_layers_per_block)
            receptive_field += (self.kernel_size - 1) * dilation
        
        return receptive_field
    
    def _get_model_state(self):
        """Get model state for saving."""
        if self.model is not None:
            return {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            }
        return None
    
    def _set_model_state(self, state):
        """Set model state for loading."""
        if state is not None and self.model is not None:
            self.model.load_state_dict(state['model'])
            if state['optimizer'] and self.optimizer:
                self.optimizer.load_state_dict(state['optimizer'])
