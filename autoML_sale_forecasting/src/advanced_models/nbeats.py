# coding: utf-8
"""
N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting

A deep neural architecture designed for univariate time series forecasting
with interpretable decomposition into trend and seasonality.

Reference: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"

Author: xingqiang chen
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_forecaster import BaseForecaster


class NBeatsBlock(nn.Module):
    """Basic N-BEATS block with fully connected layers."""
    
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        basis_function: nn.Module,
        num_layers: int = 4,
        layer_size: int = 256
    ):
        super().__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        
        # Stack of fully connected layers
        layers = []
        layers.append(nn.Linear(input_size, layer_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())
        
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta layers for backcast and forecast
        self.theta_backcast = nn.Linear(layer_size, theta_size)
        self.theta_forecast = nn.Linear(layer_size, theta_size)
        
    def forward(self, x):
        # Pass through FC stack
        h = self.fc_stack(x)
        
        # Get theta parameters
        theta_b = self.theta_backcast(h)
        theta_f = self.theta_forecast(h)
        
        # Apply basis function
        backcast = self.basis_function(theta_b, self.input_size)
        forecast = self.basis_function(theta_f, self.input_size)
        
        return backcast, forecast


class GenericBasis(nn.Module):
    """Generic basis function using learnable linear transformations."""
    
    def __init__(self, theta_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(theta_size, output_size, bias=False)
        
    def forward(self, theta, output_size):
        return self.linear(theta)


class TrendBasis(nn.Module):
    """Trend basis function using polynomial expansion."""
    
    def __init__(self, degree: int, output_size: int):
        super().__init__()
        self.degree = degree
        self.output_size = output_size
        
        # Create polynomial basis
        self.register_buffer(
            'time_vector',
            torch.linspace(0, 1, output_size).unsqueeze(0)
        )
        
    def forward(self, theta, output_size):
        # theta shape: (batch, degree)
        # Polynomial: sum(theta_i * t^i)
        powers = torch.arange(self.degree, device=theta.device).float()
        basis = self.time_vector.unsqueeze(-1) ** powers  # (1, output_size, degree)
        
        output = torch.einsum('bd,tod->bo', theta, basis)
        return output


class SeasonalityBasis(nn.Module):
    """Seasonality basis using Fourier terms."""
    
    def __init__(self, harmonics: int, output_size: int):
        super().__init__()
        self.harmonics = harmonics
        self.output_size = output_size
        
        # Create Fourier basis
        time = torch.linspace(0, 2 * np.pi, output_size)
        frequencies = torch.arange(1, harmonics + 1).float()
        
        # Sin and cos terms
        sin_terms = torch.sin(time.unsqueeze(1) * frequencies.unsqueeze(0))
        cos_terms = torch.cos(time.unsqueeze(1) * frequencies.unsqueeze(0))
        
        # Combine: (output_size, 2 * harmonics)
        self.register_buffer(
            'fourier_basis',
            torch.cat([sin_terms, cos_terms], dim=1)
        )
        
    def forward(self, theta, output_size):
        # theta shape: (batch, 2 * harmonics)
        output = torch.matmul(theta, self.fourier_basis.T)  # (batch, output_size)
        return output


class NBeatsStack(nn.Module):
    """Stack of N-BEATS blocks with residual connections."""
    
    def __init__(
        self,
        input_size: int,
        prediction_length: int,
        stack_type: str = 'generic',
        num_blocks: int = 3,
        num_layers: int = 4,
        layer_size: int = 256,
        degree: int = 3,
        harmonics: int = 4
    ):
        super().__init__()
        
        self.input_size = input_size
        self.prediction_length = prediction_length
        
        # Create basis function
        if stack_type == 'trend':
            theta_size = degree
            basis_function = TrendBasis(degree, prediction_length)
        elif stack_type == 'seasonality':
            theta_size = 2 * harmonics
            basis_function = SeasonalityBasis(harmonics, prediction_length)
        else:  # generic
            theta_size = layer_size
            basis_function = GenericBasis(theta_size, prediction_length)
        
        # Create blocks
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                theta_size=theta_size,
                basis_function=basis_function,
                num_layers=num_layers,
                layer_size=layer_size
            )
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        residual = x
        forecast = torch.zeros(x.shape[0], self.prediction_length, device=x.device)
        
        block_forecasts = []
        
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)
        
        return forecast, residual, block_forecasts


class NBeatsModel(nn.Module):
    """Full N-BEATS model with multiple stacks."""
    
    def __init__(
        self,
        input_size: int,
        prediction_length: int,
        stack_types: List[str] = ['trend', 'seasonality', 'generic'],
        num_blocks_per_stack: int = 3,
        num_layers: int = 4,
        layer_size: int = 256,
        degree: int = 3,
        harmonics: int = 4
    ):
        super().__init__()
        
        self.input_size = input_size
        self.prediction_length = prediction_length
        
        self.stacks = nn.ModuleList([
            NBeatsStack(
                input_size=input_size,
                prediction_length=prediction_length,
                stack_type=stack_type,
                num_blocks=num_blocks_per_stack,
                num_layers=num_layers,
                layer_size=layer_size,
                degree=degree,
                harmonics=harmonics
            )
            for stack_type in stack_types
        ])
        
    def forward(self, x, return_decomposition=False):
        residual = x
        forecast = torch.zeros(x.shape[0], self.prediction_length, device=x.device)
        
        stack_forecasts = []
        
        for stack in self.stacks:
            stack_forecast, residual, block_forecasts = stack(residual)
            forecast = forecast + stack_forecast
            stack_forecasts.append({
                'stack_forecast': stack_forecast,
                'block_forecasts': block_forecasts
            })
        
        if return_decomposition:
            return forecast, stack_forecasts
        
        return forecast


class NBeatsForecaster(BaseForecaster):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting.
    
    Features:
    - Interpretable decomposition into trend and seasonality
    - Deep fully connected architecture
    - Residual learning between blocks
    - No need for feature engineering
    """
    
    def __init__(
        self,
        prediction_length: int = 7,
        context_length: int = 30,
        stack_types: List[str] = ['trend', 'seasonality', 'generic'],
        num_blocks_per_stack: int = 3,
        num_layers: int = 4,
        layer_size: int = 256,
        degree: int = 3,
        harmonics: int = 4,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto',
        random_seed: int = 42
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for N-BEATS. Install with: pip install torch")
        
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
        
        self.stack_types = stack_types
        self.num_blocks_per_stack = num_blocks_per_stack
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.degree = degree
        self.harmonics = harmonics
        
        self.optimizer = None
        self.scheduler = None
        
    def _build_model(self, input_dim: int):
        """Build the N-BEATS model."""
        self.model = NBeatsModel(
            input_size=input_dim,
            prediction_length=self.prediction_length,
            stack_types=self.stack_types,
            num_blocks_per_stack=self.num_blocks_per_stack,
            num_layers=self.num_layers,
            layer_size=self.layer_size,
            degree=self.degree,
            harmonics=self.harmonics
        ).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
    def _train_step(self, batch: Dict) -> float:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        X = torch.FloatTensor(batch['X']).to(self.device)
        y = torch.FloatTensor(batch['y']).to(self.device)
        
        predictions = self.model(X)
        
        # Use MAPE loss for better scale invariance
        loss = F.mse_loss(predictions, y.expand(-1, self.prediction_length))
        
        # Add SMAPE component
        smape = 2 * torch.abs(predictions - y.expand(-1, self.prediction_length))
        smape = smape / (torch.abs(predictions) + torch.abs(y.expand(-1, self.prediction_length)) + 1e-8)
        loss = loss + 0.1 * smape.mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def _predict_step(self, batch: Dict) -> np.ndarray:
        """Perform a single prediction step."""
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(batch['X']).to(self.device)
            predictions = self.model(X)
            
        return predictions.cpu().numpy()[:, -1]  # Return last prediction
    
    def get_decomposition(self, X: np.ndarray) -> Dict:
        """Get trend/seasonality decomposition."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, stack_forecasts = self.model(X_tensor, return_decomposition=True)
        
        decomposition = {}
        for i, (stack_type, stack_data) in enumerate(zip(self.stack_types, stack_forecasts)):
            decomposition[stack_type] = stack_data['stack_forecast'].cpu().numpy()
        
        return decomposition
    
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
