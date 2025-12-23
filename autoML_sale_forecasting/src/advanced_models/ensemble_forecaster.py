# coding: utf-8
"""
Ensemble Forecaster: Combining Multiple Models for Superior Predictions

Implements various ensemble strategies:
- Simple averaging
- Weighted averaging with learned weights
- Stacking with meta-learner
- Boosting-style sequential correction

Author: xingqiang chen
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

from .base_forecaster import BaseForecaster

logger = logging.getLogger(__name__)


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble Forecaster combining multiple base models.
    
    Strategies:
    - 'mean': Simple averaging of predictions
    - 'weighted': Weighted average with validation-based weights
    - 'stacking': Train meta-learner on base model predictions
    - 'boosting': Sequential correction of predictions
    """
    
    def __init__(
        self,
        base_models: List[BaseForecaster],
        ensemble_method: str = 'stacking',
        meta_learner: str = 'ridge',
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
        Initialize ensemble forecaster.
        
        Args:
            base_models: List of base forecaster instances
            ensemble_method: 'mean', 'weighted', 'stacking', or 'boosting'
            meta_learner: Type of meta-learner for stacking ('ridge', 'elasticnet', 'gbm')
            Other args: Inherited from BaseForecaster
        """
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
        
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.meta_learner_type = meta_learner
        
        self.weights = None
        self.meta_model = None
        self.model_names = [m.__class__.__name__ for m in base_models]
        
    def _build_model(self, input_dim: int):
        """Build ensemble components."""
        # Build base models
        for model in self.base_models:
            if model.model is None:
                model._build_model(input_dim)
        
        # Initialize meta-learner for stacking
        if self.ensemble_method == 'stacking':
            if self.meta_learner_type == 'ridge':
                self.meta_model = Ridge(alpha=1.0)
            elif self.meta_learner_type == 'elasticnet':
                self.meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
            elif self.meta_learner_type == 'gbm':
                self.meta_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1
                )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
        static_features: Optional[pd.DataFrame] = None,
        time_features: Optional[pd.DataFrame] = None
    ) -> 'EnsembleForecaster':
        """
        Fit ensemble model.
        
        For stacking, trains base models on part of training data
        and meta-learner on held-out validation predictions.
        """
        logger.info(f"Training ensemble with {len(self.base_models)} base models...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Fit scalers
        self.feature_scaler.fit(X)
        self.target_scaler.fit(y.values.reshape(-1, 1))
        
        # Build models if needed
        self._build_model(X.shape[1])
        
        if self.ensemble_method in ['stacking', 'boosting']:
            # Split data for stacking/boosting
            split_idx = int(len(X) * 0.8)
            X_train, X_stack = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_stack = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train base models
            base_predictions = []
            for i, model in enumerate(self.base_models):
                logger.info(f"Training base model {i+1}/{len(self.base_models)}: {self.model_names[i]}")
                model.fit(X_train, y_train, X_val, y_val)
                pred = model.predict(X_stack)
                base_predictions.append(pred)
            
            # Stack predictions
            stacked_preds = np.column_stack(base_predictions)
            y_stack_values = y_stack.values.flatten()
            
            if self.ensemble_method == 'stacking':
                # Train meta-learner
                logger.info("Training meta-learner...")
                self.meta_model.fit(stacked_preds, y_stack_values)
                
            elif self.ensemble_method == 'boosting':
                # Learn residual corrections
                self._fit_boosting(stacked_preds, y_stack_values)
                
        elif self.ensemble_method == 'weighted':
            # Train models and compute weights based on validation performance
            val_scores = []
            for i, model in enumerate(self.base_models):
                logger.info(f"Training base model {i+1}/{len(self.base_models)}: {self.model_names[i]}")
                model.fit(X, y, X_val, y_val)
                
                if X_val is not None:
                    pred = model.predict(X_val)
                    mse = np.mean((pred - y_val.values.flatten()) ** 2)
                    val_scores.append(1.0 / (mse + 1e-8))
                else:
                    val_scores.append(1.0)
            
            # Normalize weights
            self.weights = np.array(val_scores) / np.sum(val_scores)
            logger.info(f"Model weights: {dict(zip(self.model_names, self.weights))}")
            
        else:  # mean
            # Simple training of all models
            for i, model in enumerate(self.base_models):
                logger.info(f"Training base model {i+1}/{len(self.base_models)}: {self.model_names[i]}")
                model.fit(X, y, X_val, y_val)
            
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        self.is_fitted = True
        return self
    
    def _fit_boosting(self, predictions: np.ndarray, targets: np.ndarray):
        """Fit boosting-style residual correction."""
        self.residual_models = []
        current_pred = predictions.mean(axis=1)
        
        for i in range(3):  # 3 boosting iterations
            residuals = targets - current_pred
            
            # Train residual corrector
            residual_model = Ridge(alpha=1.0)
            residual_model.fit(predictions, residuals)
            self.residual_models.append(residual_model)
            
            # Update predictions
            correction = residual_model.predict(predictions)
            current_pred = current_pred + 0.1 * correction
    
    def predict(
        self,
        X: pd.DataFrame,
        return_all_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature matrix
            return_all_predictions: Whether to return individual model predictions
            
        Returns:
            Ensemble predictions (and optionally individual predictions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get base model predictions
        base_predictions = {}
        for i, model in enumerate(self.base_models):
            pred = model.predict(X)
            base_predictions[self.model_names[i]] = pred
        
        stacked_preds = np.column_stack(list(base_predictions.values()))
        
        # Combine predictions based on method
        if self.ensemble_method == 'stacking':
            final_pred = self.meta_model.predict(stacked_preds)
            
        elif self.ensemble_method == 'boosting':
            final_pred = stacked_preds.mean(axis=1)
            for residual_model in self.residual_models:
                correction = residual_model.predict(stacked_preds)
                final_pred = final_pred + 0.1 * correction
                
        else:  # mean or weighted
            final_pred = np.average(stacked_preds, axis=1, weights=self.weights)
        
        if return_all_predictions:
            return final_pred, base_predictions
        
        return final_pred
    
    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        confidence: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates from model disagreement.
        
        Args:
            X: Feature matrix
            confidence: Confidence level for intervals
            
        Returns:
            Tuple of (mean prediction, lower bound, upper bound)
        """
        # Get all predictions
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        stacked_preds = np.column_stack(base_predictions)
        
        # Compute statistics
        mean_pred = np.mean(stacked_preds, axis=1)
        std_pred = np.std(stacked_preds, axis=1)
        
        # Confidence intervals
        z = 1.645 if confidence == 0.9 else 1.96  # 90% or 95%
        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred
        
        return mean_pred, lower, upper
    
    def get_model_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze contribution of each model to the ensemble.
        
        Returns:
            DataFrame with model predictions and final ensemble prediction
        """
        final_pred, base_predictions = self.predict(X, return_all_predictions=True)
        
        result = pd.DataFrame(base_predictions)
        result['ensemble'] = final_pred
        
        if self.weights is not None:
            result['weights'] = pd.Series(
                dict(zip(self.model_names, self.weights))
            )
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Aggregate feature importance from all models.
        """
        all_importances = []
        
        for model in self.base_models:
            try:
                imp = model.get_feature_importance()
                imp['model'] = model.__class__.__name__
                all_importances.append(imp)
            except Exception:
                pass
        
        if not all_importances:
            return super().get_feature_importance()
        
        combined = pd.concat(all_importances, ignore_index=True)
        
        # Aggregate by feature
        aggregated = combined.groupby('feature')['importance'].agg(['mean', 'std'])
        aggregated = aggregated.reset_index()
        aggregated.columns = ['feature', 'importance', 'importance_std']
        
        return aggregated.sort_values('importance', ascending=False)
    
    def _train_step(self, batch: Dict) -> float:
        """Not used - ensemble trains base models separately."""
        raise NotImplementedError("Ensemble uses fit() directly")
    
    def _predict_step(self, batch: Dict) -> np.ndarray:
        """Not used - ensemble uses predict() directly."""
        raise NotImplementedError("Ensemble uses predict() directly")
    
    def _get_model_state(self):
        """Get state of all models."""
        return {
            'base_model_states': [m._get_model_state() for m in self.base_models],
            'weights': self.weights,
            'meta_model': self.meta_model,
        }
    
    def _set_model_state(self, state):
        """Set state of all models."""
        for i, model_state in enumerate(state['base_model_states']):
            self.base_models[i]._set_model_state(model_state)
        self.weights = state['weights']
        self.meta_model = state['meta_model']


def create_default_ensemble(
    prediction_length: int = 7,
    context_length: int = 30,
    device: str = 'auto'
) -> EnsembleForecaster:
    """
    Create a default ensemble with recommended model combinations.
    
    Args:
        prediction_length: Forecast horizon
        context_length: Historical context length
        device: Device for training
        
    Returns:
        Configured EnsembleForecaster
    """
    from .temporal_fusion_transformer import TemporalFusionTransformer
    from .nbeats import NBeatsForecaster
    from .deep_ar import DeepARForecaster
    from .attention_lstm import AttentionLSTM
    
    base_models = [
        TemporalFusionTransformer(
            prediction_length=prediction_length,
            context_length=context_length,
            hidden_dim=32,
            num_heads=4,
            device=device
        ),
        NBeatsForecaster(
            prediction_length=prediction_length,
            context_length=context_length,
            layer_size=128,
            device=device
        ),
        DeepARForecaster(
            prediction_length=prediction_length,
            context_length=context_length,
            hidden_dim=32,
            device=device
        ),
        AttentionLSTM(
            prediction_length=prediction_length,
            context_length=context_length,
            hidden_dim=64,
            device=device
        ),
    ]
    
    return EnsembleForecaster(
        base_models=base_models,
        ensemble_method='stacking',
        meta_learner='gbm',
        prediction_length=prediction_length,
        context_length=context_length,
        device=device
    )
