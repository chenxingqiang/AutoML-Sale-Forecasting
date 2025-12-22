# AutoML-System-on-Sale-Forecasting

An AutoML system for retail sales forecasting using machine learning models (LightGBM, XGBoost, GBDT, etc.) with support for external factors.

## Overview

This repository contains a Python-based AutoML system for sales forecasting in retail environments. It supports multiple machine learning models and integrates external factors to improve prediction accuracy.

## Features

- **Multiple ML Models**: LightGBM, XGBoost, GBDT, Random Forest, KNN, Extra Trees
- **External Factors Support**: Weather, Air Quality (AQI), Calendar/Holiday data integration
- **Time Series Features**: Automatic feature engineering for time series data
- **Store-level Forecasting**: Multi-store sales prediction with store-specific attributes
- **Automated Data Pipeline**: SQL-based data ingestion and preprocessing

## Supported External Factors

The system supports importing and utilizing the following external factors for prediction:

| Factor Type | Features | Description |
|------------|----------|-------------|
| **Weather** | Temperature (`tem`), Wind Speed (`windspeed`), Precipitation (`pre1h`), Humidity (`rhu`) | Weather conditions that affect customer behavior |
| **Air Quality** | AQI Index, AQI Level | Air quality index and pollution levels |
| **Calendar** | Weekday, Day Type (workday/weekend), Holiday, 24 Solar Terms | Temporal patterns and special dates |
| **Location** | Store ID, City, Store Type | Store-specific geographic attributes |

## Installation

```bash
# Clone the repository
git clone https://github.com/chenxingqiang/AutoML-System-on-Sale-Forecasting.git
cd AutoML-System-on-Sale-Forecasting

# Install dependencies
pip install numpy pandas scikit-learn lightgbm xgboost pymysql sqlalchemy
```

## Usage

```python
from autoML_sale_forecasting.src.ai_dev_sale_forecast_LGBM import ai_dev_sale_forecast_LGBM

# Run the forecasting pipeline
# The system will automatically fetch data from the configured database
# and generate predictions with external factors included
```

## Data Requirements

The system expects the following data tables:

- `mid_aggrbydate_saleqty_his`: Historical sales quantity data
- `mid_aggrbydate_saleprice_his`: Historical price and promotion data
- `mid_aggrbydate_weather_his`: Weather history (temperature, wind, precipitation, humidity)
- `mid_aggrbydate_airq_his`: Air quality index history
- `calendar`: Calendar information (weekday, holidays, solar terms)
- `raw_cleaned_goods_his`: Product information
- `raw_cleaned_category_his`: Product category hierarchy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Keywords

forecasting, system, python, automl, sale, jupyter notebook, machine learning, lightgbm, xgboost, weather, external factors
