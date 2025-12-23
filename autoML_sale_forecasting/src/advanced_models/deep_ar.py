# coding: utf-8
"""
DeepAR: Probabilistic Forecasting with Autoregressive RNNs

A probabilistic forecasting model using LSTM/GRU with autoregressive decoding.
Produces prediction distributions rather than point estimates.

Reference: "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"

Author: xingqiang chen
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, NegativeBinomial, StudentT
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_forecaster import BaseForecaster


class DeepARModel(nn.Module):
    """DeepAR model with probabilistic outputs."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        distribution: str = 'normal',
        cell_type: str = 'lstm'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.distribution = distribution
        self.cell_type = cell_type
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for lagged target
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # RNN cell
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Distribution parameters output
        if distribution == 'normal':
            # Mean and standard deviation
            self.dist_params = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)  # mu, sigma
            )
        elif distribution == 'negative_binomial':
            # Total count and probability
            self.dist_params = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)  # n, p
            )
        elif distribution == 'student_t':
            # Location, scale, and degrees of freedom
            self.dist_params = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)  # loc, scale, df
            )
        
        # Scale output for ensuring positive scale
        self.softplus = nn.Softplus()
        
    def forward(
        self,
        x: torch.Tensor,
        y_past: torch.Tensor,
        hidden: Optional[Tuple] = None
    ):
        """
        Forward pass.
        
        Args:
            x: Covariates (batch, seq_len, input_dim)
            y_past: Past target values (batch, seq_len, 1)
            hidden: Initial hidden state
            
        Returns:
            Distribution parameters and hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Concatenate covariates with lagged target
        inputs = torch.cat([x, y_past], dim=-1)
        
        # Embed inputs
        embedded = self.input_embedding(inputs)
        
        # RNN forward pass
        output, hidden = self.rnn(embedded, hidden)
        
        # Get distribution parameters
        params = self.dist_params(output)
        
        return params, hidden
    
    def get_distribution(self, params: torch.Tensor):
        """Create distribution from parameters."""
        if self.distribution == 'normal':
            mu = params[..., 0]
            sigma = self.softplus(params[..., 1]) + 1e-6
            return Normal(mu, sigma)
        elif self.distribution == 'negative_binomial':
            total_count = self.softplus(params[..., 0]) + 1e-6
            probs = torch.sigmoid(params[..., 1])
            return NegativeBinomial(total_count, probs)
        elif self.distribution == 'student_t':
            df = self.softplus(params[..., 0]) + 2  # Ensure df > 2
            loc = params[..., 1]
            scale = self.softplus(params[..., 2]) + 1e-6
            return StudentT(df, loc, scale)
    
    def sample(
        self,
        x: torch.Tensor,
        y_past: torch.Tensor,
        prediction_length: int,
        num_samples: int = 100,
        hidden: Optional[Tuple] = None
    ):
        """
        Generate samples from the predictive distribution.
        
        Args:
            x: Future covariates (batch, prediction_length, input_dim)
            y_past: Last observed target value (batch, 1)
            prediction_length: Number of steps to forecast
            num_samples: Number of samples to generate
            hidden: Initial hidden state from encoding phase
            
        Returns:
            Samples (batch, num_samples, prediction_length)
        """
        batch_size = x.shape[0]
        device = x.device
        
        samples = torch.zeros(batch_size, num_samples, prediction_length, device=device)
        
        for s in range(num_samples):
            current_y = y_past.clone()
            current_hidden = hidden
            
            for t in range(prediction_length):
                # Get covariates for current time step
                x_t = x[:, t:t+1, :]
                
                # Forward pass
                params, current_hidden = self.forward(x_t, current_y.unsqueeze(1), current_hidden)
                
                # Sample from distribution
                dist = self.get_distribution(params.squeeze(1))
                sample = dist.sample()
                
                # Store sample and update current_y
                samples[:, s, t] = sample
                current_y = sample.unsqueeze(-1)
        
        return samples


class DeepARForecaster(BaseForecaster):
    """
    DeepAR: Probabilistic Forecasting with Autoregressive RNNs.
    
    Features:
    - Probabilistic predictions with uncertainty estimates
    - Multiple distribution options (Normal, Negative Binomial, Student-t)
    - Autoregressive decoding for multi-step forecasting
    - Handles covariates and time features
    """
    
    def __init__(
        self,
        prediction_length: int = 7,
        context_length: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        distribution: str = 'normal',
        cell_type: str = 'lstm',
        num_samples: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto',
        random_seed: int = 42
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepAR. Install with: pip install torch")
        
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
        self.num_layers = num_layers
        self.dropout = dropout
        self.distribution = distribution
        self.cell_type = cell_type
        self.num_samples = num_samples
        
        self.optimizer = None
        self.scheduler = None
        
    def _build_model(self, input_dim: int):
        """Build the DeepAR model."""
        self.model = DeepARModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            distribution=self.distribution,
            cell_type=self.cell_type
        ).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
    def _train_step(self, batch: Dict) -> float:
        """Perform a single training step using negative log-likelihood."""
        self.model.train()
        self.optimizer.zero_grad()
        
        X = torch.FloatTensor(batch['X']).to(self.device)
        y = torch.FloatTensor(batch['y']).to(self.device)
        
        # Create lagged target (shift by 1)
        y_lagged = torch.zeros_like(y).unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        # Reshape X if needed
        if X.dim() == 2:
            X = X.unsqueeze(1).expand(-1, self.context_length, -1)
        
        # Forward pass
        params, _ = self.model(X, y_lagged.expand(-1, self.context_length, -1))
        
        # Get distribution and compute negative log-likelihood
        dist = self.model.get_distribution(params[:, -1, :])
        
        if y.dim() == 2:
            y = y[:, 0]
        
        nll = -dist.log_prob(y)
        loss = nll.mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _predict_step(self, batch: Dict) -> np.ndarray:
        """Perform prediction returning mean of predictive distribution."""
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(batch['X']).to(self.device)
            
            if X.dim() == 2:
                X = X.unsqueeze(1).expand(-1, self.context_length, -1)
            
            # Use zero as initial y (will be scaled)
            y_init = torch.zeros(X.shape[0], 1, device=self.device)
            
            # Encode
            y_lagged = torch.zeros(X.shape[0], self.context_length, 1, device=self.device)
            params, hidden = self.model(X, y_lagged)
            
            # Get mean prediction
            dist = self.model.get_distribution(params[:, -1, :])
            predictions = dist.mean
            
        return predictions.cpu().numpy()
    
    def predict_samples(
        self,
        X: np.ndarray,
        num_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate samples from predictive distribution.
        
        Args:
            X: Feature matrix
            num_samples: Number of samples (uses default if None)
            
        Returns:
            Samples array (n_samples, num_samples, prediction_length)
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            
            if X.dim() == 2:
                X = X.unsqueeze(1).expand(-1, self.context_length, -1)
            
            # Encode
            y_lagged = torch.zeros(X.shape[0], self.context_length, 1, device=self.device)
            _, hidden = self.model(X, y_lagged)
            
            # Sample
            y_init = torch.zeros(X.shape[0], 1, device=self.device)
            samples = self.model.sample(
                X[:, -self.prediction_length:, :],
                y_init,
                self.prediction_length,
                num_samples,
                hidden
            )
            
        return samples.cpu().numpy()
    
    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[float, np.ndarray]:
        """
        Predict quantiles of the predictive distribution.
        
        Args:
            X: Feature matrix
            quantiles: List of quantiles to compute
            
        Returns:
            Dictionary mapping quantile to predictions
        """
        samples = self.predict_samples(X)
        
        result = {}
        for q in quantiles:
            result[q] = np.percentile(samples, q * 100, axis=1)[:, -1]
        
        return result
    
    def _predict_uncertainty(self, batch: Dict) -> np.ndarray:
        """Return prediction intervals."""
        X = batch['X']
        quantiles = self.predict_quantiles(X, [0.1, 0.5, 0.9])
        
        return np.stack([quantiles[0.1], quantiles[0.5], quantiles[0.9]], axis=-1)
    
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
