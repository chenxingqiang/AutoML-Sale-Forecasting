# coding: utf-8
"""
Attention-Enhanced LSTM for Time Series Forecasting

Combines LSTM with multi-head self-attention for improved temporal modeling.

Author: xingqiang chen
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_forecaster import BaseForecaster


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence position information."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn
        return out


class AttentionLSTMModel(nn.Module):
    """LSTM with multi-head attention."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
        prediction_length: int = 7
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.prediction_length = prediction_length
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Layer normalization after LSTM
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            d_model=lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(lstm_output_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, lstm_output_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(lstm_output_dim)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_length)
        )
        
        # Attention pooling for variable-length sequences
        self.attention_pool = nn.Sequential(
            nn.Linear(lstm_output_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Self-attention with residual connection
        if return_attention:
            attn_out, attn_weights = self.attention(lstm_out, return_attention=True)
        else:
            attn_out = self.attention(lstm_out)
            attn_weights = None
        
        x = self.attn_norm(lstm_out + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        
        # Attention pooling
        pool_weights = self.attention_pool(x)
        pooled = (x * pool_weights).sum(dim=1)
        
        # Output prediction
        output = self.output_layer(pooled)
        
        if return_attention:
            return output, attn_weights
        return output


class AttentionLSTM(BaseForecaster):
    """
    Attention-Enhanced LSTM for time series forecasting.
    
    Features:
    - Bidirectional LSTM for capturing past and future context
    - Multi-head self-attention for global dependencies
    - Positional encoding for sequence position
    - Attention pooling for variable-length sequences
    """
    
    def __init__(
        self,
        prediction_length: int = 7,
        context_length: int = 30,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto',
        random_seed: int = 42
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
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
        self.num_lstm_layers = num_lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.optimizer = None
        self.scheduler = None
        
    def _build_model(self, input_dim: int):
        """Build the Attention LSTM model."""
        self.model = AttentionLSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_lstm_layers=self.num_lstm_layers,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            prediction_length=self.prediction_length
        ).to(self.device)
        
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
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
        
        # Combined MSE and MAE loss
        mse_loss = F.mse_loss(predictions, y.expand(-1, self.prediction_length))
        mae_loss = F.l1_loss(predictions, y.expand(-1, self.prediction_length))
        loss = mse_loss + 0.1 * mae_loss
        
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
            
            if X.dim() == 2:
                X = X.unsqueeze(1).expand(-1, self.context_length, -1)
            
            predictions = self.model(X)
            
        return predictions.cpu().numpy()[:, -1]
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Get attention weights for interpretability."""
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            
            if X.dim() == 2:
                X = X.unsqueeze(1).expand(-1, self.context_length, -1)
            
            _, attn_weights = self.model(X, return_attention=True)
            
        return attn_weights.cpu().numpy()
    
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
