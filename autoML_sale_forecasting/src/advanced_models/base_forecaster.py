# coding: utf-8
"""
Base Forecaster Class

Provides common interface and utilities for all forecasting models.

Author: xingqiang chen
"""

import os
import json
import pickle
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(
        self,
        prediction_length: int = 7,
        context_length: int = 30,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto',
        random_seed: int = 42
    ):
        """
        Initialize base forecaster.
        
        Args:
            prediction_length: Number of future time steps to predict
            context_length: Number of historical time steps to use
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            epochs: Maximum number of training epochs
            early_stopping_patience: Patience for early stopping
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
            random_seed: Random seed for reproducibility
        """
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed
        
        # Set device
        self.device = self._get_device(device)
        
        # Initialize scalers
        self.target_scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        
        # Model state
        self.model = None
        self.is_fitted = False
        self.training_history = []
        self.feature_names = []
        
        # Set random seeds
        self._set_random_seeds()
        
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device != 'auto':
            return device
            
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except ImportError:
            pass
        return 'cpu'
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_seed)
        try:
            import torch
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)
        except ImportError:
            pass
    
    @abstractmethod
    def _build_model(self, input_dim: int) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def _train_step(self, batch: Dict) -> float:
        """Perform a single training step."""
        pass
    
    @abstractmethod
    def _predict_step(self, batch: Dict) -> np.ndarray:
        """Perform a single prediction step."""
        pass
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
        static_features: Optional[pd.DataFrame] = None,
        time_features: Optional[pd.DataFrame] = None
    ) -> 'BaseForecaster':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix with temporal features
            y: Target variable(s)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            static_features: Time-invariant features (e.g., store info)
            time_features: Time-varying known features (e.g., holidays)
            
        Returns:
            self
        """
        logger.info(f"Training {self.__class__.__name__}...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale data
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))
        
        # Build model if not already built
        if self.model is None:
            self._build_model(X_scaled.shape[1])
        
        # Prepare data loaders
        train_loader = self._create_data_loader(X_scaled, y_scaled, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.feature_scaler.transform(X_val)
            y_val_scaled = self.target_scaler.transform(y_val.values.reshape(-1, 1))
            val_loader = self._create_data_loader(X_val_scaled, y_val_scaled, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader)
            
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_best_model()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        self._load_best_model()
                        break
                        
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                self.training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss
                })
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions.
        
        Args:
            X: Feature matrix
            return_uncertainty: Whether to return prediction intervals
            
        Returns:
            Predictions (and optionally uncertainty estimates)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.feature_scaler.transform(X)
        loader = self._create_data_loader(X_scaled, None, shuffle=False)
        
        predictions = []
        uncertainties = []
        
        for batch in loader:
            pred = self._predict_step(batch)
            predictions.append(pred)
            
            if return_uncertainty and hasattr(self, '_predict_uncertainty'):
                uncert = self._predict_uncertainty(batch)
                uncertainties.append(uncert)
        
        predictions = np.concatenate(predictions, axis=0)
        predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        if return_uncertainty and uncertainties:
            uncertainties = np.concatenate(uncertainties, axis=0)
            return predictions, uncertainties
        
        return predictions
    
    def _create_data_loader(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        shuffle: bool = True
    ):
        """Create data loader for training/prediction."""
        dataset = list(zip(X, y if y is not None else [None] * len(X)))
        
        if shuffle:
            np.random.shuffle(dataset)
        
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            X_batch = np.array([b[0] for b in batch])
            if y is not None:
                y_batch = np.array([b[1] for b in batch])
                yield {'X': X_batch, 'y': y_batch}
            else:
                yield {'X': X_batch}
    
    def _train_epoch(self, loader) -> float:
        """Train for one epoch."""
        total_loss = 0
        n_batches = 0
        
        for batch in loader:
            loss = self._train_step(batch)
            total_loss += loss
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _validate_epoch(self, loader) -> float:
        """Validate for one epoch."""
        total_loss = 0
        n_batches = 0
        
        for batch in loader:
            loss = self._compute_validation_loss(batch)
            total_loss += loss
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _compute_validation_loss(self, batch: Dict) -> float:
        """Compute validation loss (default implementation)."""
        predictions = self._predict_step(batch)
        targets = batch['y']
        return np.mean((predictions.flatten() - targets.flatten()) ** 2)
    
    def _save_best_model(self):
        """Save the best model state."""
        self._best_model_state = self._get_model_state()
    
    def _load_best_model(self):
        """Load the best model state."""
        if hasattr(self, '_best_model_state'):
            self._set_model_state(self._best_model_state)
    
    def _get_model_state(self):
        """Get current model state."""
        return None  # Override in subclasses
    
    def _set_model_state(self, state):
        """Set model state."""
        pass  # Override in subclasses
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        state = {
            'model_class': self.__class__.__name__,
            'config': {
                'prediction_length': self.prediction_length,
                'context_length': self.context_length,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'early_stopping_patience': self.early_stopping_patience,
                'random_seed': self.random_seed,
            },
            'scalers': {
                'target': self.target_scaler,
                'feature': self.feature_scaler,
            },
            'feature_names': self.feature_names,
            'model_state': self._get_model_state(),
            'training_history': self.training_history,
            'is_fitted': self.is_fitted,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseForecaster':
        """Load model from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(**state['config'])
        model.target_scaler = state['scalers']['target']
        model.feature_scaler = state['scalers']['feature']
        model.feature_names = state['feature_names']
        model.training_history = state['training_history']
        model.is_fitted = state['is_fitted']
        
        if state['model_state'] is not None:
            model._build_model(len(state['feature_names']))
            model._set_model_state(state['model_state'])
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (override in subclasses)."""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': [1.0 / len(self.feature_names)] * len(self.feature_names)
        })
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_splits: int = 5,
        gap: int = 0
    ) -> Dict:
        """
        Perform time series cross-validation.
        
        Args:
            X: Features
            y: Target
            n_splits: Number of CV splits
            gap: Gap between train and test sets
            
        Returns:
            Dictionary with CV results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        
        scores = []
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model
            self._reset_model()
            self.fit(X_train, y_train, X_val, y_val)
            
            # Predict
            predictions = self.predict(X_val)
            
            # Calculate metrics
            mse = np.mean((predictions - y_val.values.flatten()) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_val.values.flatten()))
            mape = np.mean(np.abs((y_val.values.flatten() - predictions) / 
                                   np.maximum(y_val.values.flatten(), 1e-8))) * 100
            
            scores.append({
                'fold': fold + 1,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            })
            
            fold_predictions.append({
                'fold': fold + 1,
                'predictions': predictions,
                'actuals': y_val.values.flatten()
            })
        
        return {
            'scores': pd.DataFrame(scores),
            'mean_rmse': np.mean([s['rmse'] for s in scores]),
            'std_rmse': np.std([s['rmse'] for s in scores]),
            'mean_mape': np.mean([s['mape'] for s in scores]),
            'fold_predictions': fold_predictions
        }
    
    def _reset_model(self):
        """Reset model for cross-validation."""
        self.model = None
        self.is_fitted = False
        self.training_history = []
