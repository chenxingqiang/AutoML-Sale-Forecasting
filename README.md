# AutoML-System-on-Sale-Forecasting

A state-of-the-art AutoML system for retail sales forecasting using advanced deep learning models and ensemble methods.

## 🚀 Features

### Advanced Deep Learning Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **Temporal Fusion Transformer (TFT)** | State-of-the-art interpretable model combining LSTM with multi-head attention | Complex multi-horizon forecasting with interpretability |
| **N-BEATS** | Neural Basis Expansion Analysis with trend/seasonality decomposition | Univariate forecasting with interpretable decomposition |
| **DeepAR** | Probabilistic forecasting with autoregressive RNNs | Uncertainty estimation and prediction intervals |
| **WaveNet** | Dilated causal convolutions for capturing long-range dependencies | High-frequency patterns and long sequences |
| **Attention LSTM** | Bidirectional LSTM with multi-head self-attention | Balanced accuracy and efficiency |

### Ensemble Methods

- **Stacking**: Meta-learner trained on base model predictions
- **Weighted Averaging**: Validation-based optimal weights
- **Boosting**: Sequential residual correction
- **Model Blending**: Simple averaging with uncertainty

### Advanced Feature Engineering

- **Temporal Features**: Lags, rolling statistics, exponential weighted averages
- **Seasonal Decomposition**: Trend, seasonality, and residual extraction
- **Fourier Features**: Capturing complex periodic patterns
- **Holiday Features**: Chinese and international holidays
- **External Factors**: Weather, air quality, promotions
- **Target Encoding**: For high-cardinality categorical variables
- **Automated Feature Selection**: Mutual information-based selection

### External Factors Support

| Factor Type | Features | Description |
|------------|----------|-------------|
| **Weather** | Temperature, Wind Speed, Precipitation, Humidity | Weather conditions affecting customer behavior |
| **Air Quality** | AQI Index, Pollution Levels | Environmental factors |
| **Calendar** | Weekday, Day Type, Holidays, 24 Solar Terms | Temporal patterns |
| **Promotions** | Discount Rate, Price Changes, On-Sale Flags | Marketing effects |
| **Location** | Store ID, City, Store Type | Geographic attributes |

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/chenxingqiang/AutoML-System-on-Sale-Forecasting.git
cd AutoML-System-on-Sale-Forecasting

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- PyTorch 1.10+
- LightGBM, XGBoost, CatBoost
- scikit-learn, pandas, numpy

## 🎯 Quick Start

### Using Individual Models

```python
from autoML_sale_forecasting.src.advanced_models import (
    TemporalFusionTransformer,
    NBeatsForecaster,
    DeepARForecaster,
    AttentionLSTM
)

# Initialize model
model = TemporalFusionTransformer(
    prediction_length=7,
    context_length=30,
    hidden_dim=64,
    num_heads=4
)

# Train
model.fit(X_train, y_train, X_val, y_val)

# Predict
predictions = model.predict(X_test)

# Get prediction intervals (for probabilistic models)
predictions, lower, upper = model.predict(X_test, return_uncertainty=True)
```

### Using Ensemble

```python
from autoML_sale_forecasting.src.advanced_models import (
    EnsembleForecaster,
    create_default_ensemble
)

# Create default ensemble with TFT, N-BEATS, DeepAR, and Attention LSTM
ensemble = create_default_ensemble(
    prediction_length=7,
    context_length=30,
    device='cuda'
)

# Train and predict
ensemble.fit(X_train, y_train, X_val, y_val)
predictions = ensemble.predict(X_test)

# Get predictions with uncertainty from model disagreement
mean_pred, lower, upper = ensemble.predict_with_uncertainty(X_test)
```

### Advanced Feature Engineering

```python
from autoML_sale_forecasting.src.advanced_models import AdvancedFeatureEngineer

# Initialize feature engineer
fe = AdvancedFeatureEngineer(
    target_col='sale_qty',
    date_col='date',
    group_cols=['store_id', 'product_id'],
    lag_periods=[1, 2, 3, 7, 14, 21, 28],
    rolling_windows=[3, 7, 14, 28],
    include_fourier=True,
    include_holidays=True,
    select_features=True,
    n_features_to_select=50
)

# Transform data
df_features = fe.fit_transform(df, external_features=weather_df)

# Get feature importance
importance = fe.get_feature_importance()
```

## 📊 Model Comparison

| Model | MAE | RMSE | MAPE | Training Time | Interpretability |
|-------|-----|------|------|---------------|------------------|
| TFT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | High |
| N-BEATS | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Fast | High |
| DeepAR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Medium |
| WaveNet | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Fast | Low |
| AttentionLSTM | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Medium |
| Ensemble | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Slow | Medium |
| LightGBM | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Very Fast | High |
| XGBoost | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Fast | High |

## 🏗️ Architecture

```
autoML_sale_forecasting/
├── src/
│   ├── advanced_models/           # State-of-the-art models
│   │   ├── __init__.py
│   │   ├── base_forecaster.py     # Base class with common interface
│   │   ├── temporal_fusion_transformer.py  # TFT implementation
│   │   ├── nbeats.py              # N-BEATS implementation
│   │   ├── deep_ar.py             # DeepAR implementation
│   │   ├── wavenet.py             # WaveNet implementation
│   │   ├── attention_lstm.py      # Attention LSTM
│   │   ├── ensemble_forecaster.py # Ensemble methods
│   │   └── feature_engineering.py # Advanced feature engineering
│   │
│   ├── ai_dev_sale_forecast_LGBM/ # Traditional ML models
│   │   ├── ai_dev_sale_forecast_LGBM.py
│   │   ├── sale_forecast_LGBM.py
│   │   ├── sale_forecast_LSTM.py
│   │   └── sale_forecast_XGBOOST.py
│   │
│   └── utils/
│       └── time_series.py
│
├── requirements.txt
└── README.md
```

## 🔬 Cross-Validation

```python
# Time series cross-validation
cv_results = model.cross_validate(
    X, y,
    n_splits=5,
    gap=7  # Gap between train and test to prevent leakage
)

print(f"Mean RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
print(f"Mean MAPE: {cv_results['mean_mape']:.2f}%")
```

## 💾 Model Persistence

```python
# Save model
model.save('model_checkpoint.pkl')

# Load model
loaded_model = TemporalFusionTransformer.load('model_checkpoint.pkl')
predictions = loaded_model.predict(X_test)
```

## 📈 Interpretability

### Feature Importance

```python
# Get feature importance
importance = model.get_feature_importance()
print(importance.head(10))
```

### Attention Visualization (TFT)

```python
# Get attention weights
temporal_attn, variable_attn = model.get_attention_weights(X_test)
```

### Decomposition (N-BEATS)

```python
# Get trend/seasonality decomposition
decomposition = model.get_decomposition(X_test)
print(decomposition['trend'].shape)
print(decomposition['seasonality'].shape)
```

## 🛠️ Configuration

### Model Configuration

```python
config = {
    'prediction_length': 7,      # Forecast horizon
    'context_length': 30,        # Historical context
    'hidden_dim': 64,            # Hidden layer size
    'num_heads': 4,              # Attention heads
    'num_layers': 2,             # Number of layers
    'dropout': 0.1,              # Dropout rate
    'learning_rate': 1e-3,       # Learning rate
    'batch_size': 64,            # Batch size
    'epochs': 100,               # Max epochs
    'early_stopping_patience': 10
}
```

## 📚 References

1. **Temporal Fusion Transformers**: Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting", 2020
2. **N-BEATS**: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting", 2019
3. **DeepAR**: Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", 2019
4. **WaveNet**: van den Oord et al., "WaveNet: A Generative Model for Raw Audio", 2016

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Keywords

forecasting, sales prediction, time series, deep learning, pytorch, transformer, temporal fusion transformer, n-beats, deepar, wavenet, lstm, attention, ensemble learning, automl, machine learning, retail analytics, demand forecasting
