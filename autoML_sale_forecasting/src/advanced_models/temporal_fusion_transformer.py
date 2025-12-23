# coding: utf-8
"""
Temporal Fusion Transformer (TFT)

State-of-the-art interpretable deep learning model for time series forecasting.
Combines LSTM layers with multi-head attention for capturing temporal patterns.

Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"

Author: xingqiang chen
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_forecaster import BaseForecaster


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for feature gating."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        sig = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(sig * x)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) for flexible nonlinear processing."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim, dropout)
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
        if input_dim != output_dim:
            self.skip_connection = nn.Linear(input_dim, output_dim)
        else:
            self.skip_connection = None
            
    def forward(self, x, context=None):
        residual = x
        
        x = self.fc1(x)
        
        if context is not None and self.context_dim is not None:
            context = self.context_fc(context)
            x = x + context
            
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.glu(x)
        
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
            
        return self.layer_norm(x + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for automatic feature selection."""
    
    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        
        # Single variable GRNs
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_inputs)
        ])
        
        # Variable selection weights
        self.grn_var_selection = GatedResidualNetwork(
            input_dim * num_inputs,
            hidden_dim,
            num_inputs,
            dropout,
            context_dim
        )
        
    def forward(self, x, context=None):
        # x shape: (batch, num_inputs, input_dim)
        batch_size = x.shape[0]
        
        # Apply individual GRNs
        processed = []
        for i in range(self.num_inputs):
            processed.append(self.grns[i](x[:, i]))
        processed = torch.stack(processed, dim=1)  # (batch, num_inputs, hidden_dim)
        
        # Compute variable selection weights
        flattened = x.reshape(batch_size, -1)
        weights = self.grn_var_selection(flattened, context)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)  # (batch, num_inputs, 1)
        
        # Apply weights
        output = (processed * weights).sum(dim=1)  # (batch, hidden_dim)
        
        return output, weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention with separate value projections."""
    
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(context), attn_weights


class TFTModel(nn.Module):
    """Temporal Fusion Transformer Model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        prediction_length: int = 7,
        context_length: int = 30
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            input_dim=1,
            num_inputs=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # LSTM Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # LSTM Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Static context GRNs
        self.static_enrichment_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        
        # Temporal self-attention
        self.self_attention = InterpretableMultiHeadAttention(
            n_heads=num_heads,
            d_model=hidden_dim,
            dropout=dropout
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_dim)
        
        # Position-wise feedforward
        self.positionwise_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Quantile outputs for uncertainty estimation
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in [0.1, 0.5, 0.9]
        ])
        
    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        # Variable selection
        x_reshaped = x.unsqueeze(-1)  # (batch, seq, input_dim, 1)
        x_selected_list = []
        var_weights_list = []
        
        for t in range(seq_len):
            selected, weights = self.vsn(x_reshaped[:, t])
            x_selected_list.append(selected)
            var_weights_list.append(weights)
        
        x_selected = torch.stack(x_selected_list, dim=1)  # (batch, seq, hidden_dim)
        var_weights = torch.stack(var_weights_list, dim=1)  # (batch, seq, input_dim)
        
        # LSTM encoding
        encoder_output, (h_n, c_n) = self.encoder_lstm(x_selected)
        
        # Static enrichment
        static_context = self.static_enrichment_grn(h_n[-1])
        
        # Expand static context for attention
        enriched = encoder_output + static_context.unsqueeze(1)
        
        # Self-attention
        attention_output, attention_weights = self.self_attention(
            enriched, enriched, enriched
        )
        
        # Add & Norm
        x = self.attention_layer_norm(encoder_output + attention_output)
        
        # Position-wise feedforward
        output = self.positionwise_grn(x)
        
        # Final output - take last time steps for prediction
        predictions = self.output_layer(output[:, -self.prediction_length:])
        
        # Quantile predictions
        quantiles = [q(output[:, -self.prediction_length:]) for q in self.quantile_outputs]
        quantiles = torch.cat(quantiles, dim=-1)
        
        if return_attention:
            return predictions, quantiles, attention_weights, var_weights
        
        return predictions, quantiles


class TemporalFusionTransformer(BaseForecaster):
    """
    Temporal Fusion Transformer for sales forecasting.
    
    State-of-the-art model that combines:
    - Variable selection networks for automatic feature selection
    - LSTM layers for temporal patterns
    - Multi-head attention for long-range dependencies
    - Interpretable attention weights
    """
    
    def __init__(
        self,
        prediction_length: int = 7,
        context_length: int = 30,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto',
        random_seed: int = 42
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TFT. Install with: pip install torch")
        
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
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        
        self.optimizer = None
        self.scheduler = None
        
    def _build_model(self, input_dim: int):
        """Build the TFT model."""
        self.model = TFTModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_lstm_layers=self.num_lstm_layers,
            dropout=self.dropout,
            prediction_length=self.prediction_length,
            context_length=self.context_length
        ).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
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
        
        predictions, quantiles = self.model(X)
        
        # MSE loss for point predictions
        loss = F.mse_loss(predictions.squeeze(-1), y.expand(-1, self.prediction_length))
        
        # Quantile loss for uncertainty
        quantile_values = [0.1, 0.5, 0.9]
        for i, q in enumerate(quantile_values):
            errors = y.expand(-1, self.prediction_length) - quantiles[:, :, i]
            quantile_loss = torch.max((q - 1) * errors, q * errors).mean()
            loss = loss + 0.1 * quantile_loss
        
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
            
            predictions, _ = self.model(X)
            
        return predictions.cpu().numpy().squeeze(-1)[:, -1]  # Return last prediction
    
    def _predict_uncertainty(self, batch: Dict) -> np.ndarray:
        """Predict with uncertainty quantiles."""
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(batch['X']).to(self.device)
            
            if X.dim() == 2:
                X = X.unsqueeze(1).expand(-1, self.context_length, -1)
            
            _, quantiles = self.model(X)
            
        return quantiles.cpu().numpy()[:, -1, :]  # Return last time step quantiles
    
    def get_attention_weights(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get attention weights for interpretability."""
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            
            if X.dim() == 2:
                X = X.unsqueeze(1).expand(-1, self.context_length, -1)
            
            _, _, attention_weights, var_weights = self.model(X, return_attention=True)
            
        return (
            attention_weights.cpu().numpy(),
            var_weights.cpu().numpy()
        )
    
    def get_feature_importance(self) -> 'pd.DataFrame':
        """Get feature importance from variable selection weights."""
        import pandas as pd
        
        if not hasattr(self, '_last_var_weights'):
            return super().get_feature_importance()
        
        importance = self._last_var_weights.mean(axis=(0, 1))
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
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
