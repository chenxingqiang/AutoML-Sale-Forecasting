# coding: utf-8
"""
Advanced Sales Forecasting Models

State-of-the-art deep learning models for time series forecasting:
- Temporal Fusion Transformer (TFT)
- N-BEATS
- DeepAR
- Transformer-based models
- Attention-enhanced LSTM
- Ensemble methods

Author: xingqiang chen
Version: 2.0
"""

from .base_forecaster import BaseForecaster
from .temporal_fusion_transformer import TemporalFusionTransformer
from .nbeats import NBeatsForecaster
from .deep_ar import DeepARForecaster
from .wavenet import WaveNetForecaster
from .attention_lstm import AttentionLSTM
from .ensemble_forecaster import EnsembleForecaster
from .feature_engineering import AdvancedFeatureEngineer

__all__ = [
    'BaseForecaster',
    'TemporalFusionTransformer',
    'NBeatsForecaster',
    'DeepARForecaster',
    'WaveNetForecaster',
    'AttentionLSTM',
    'EnsembleForecaster',
    'AdvancedFeatureEngineer',
]
