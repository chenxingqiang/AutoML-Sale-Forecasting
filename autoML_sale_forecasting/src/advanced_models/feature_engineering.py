# coding: utf-8
"""
Advanced Feature Engineering for Sales Forecasting

Automated feature engineering pipeline with:
- Temporal features (lags, rolling statistics, trends)
- Seasonal decomposition
- Holiday and event features
- External factor integration
- Target encoding
- Feature selection

Author: xingqiang chen
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering pipeline for time series forecasting.
    
    Features:
    - Lag features with multiple time horizons
    - Rolling statistics (mean, std, min, max, skew, kurtosis)
    - Exponential weighted features
    - Trend and seasonality decomposition
    - Date/time features
    - Holiday and special event markers
    - External factor integration
    - Target encoding for categorical variables
    - Automated feature selection
    """
    
    def __init__(
        self,
        target_col: str = 'sale_qty',
        date_col: str = 'date',
        group_cols: Optional[List[str]] = None,
        lag_periods: List[int] = [1, 2, 3, 7, 14, 21, 28],
        rolling_windows: List[int] = [3, 7, 14, 28],
        ewm_spans: List[int] = [7, 14, 28],
        include_fourier: bool = True,
        fourier_terms: int = 4,
        include_holidays: bool = True,
        select_features: bool = True,
        n_features_to_select: int = 50,
        feature_selection_method: str = 'mutual_info'
    ):
        """
        Initialize feature engineer.
        
        Args:
            target_col: Name of target column
            date_col: Name of date column
            group_cols: Columns to group by (e.g., store_id, product_id)
            lag_periods: List of lag periods to create
            rolling_windows: Window sizes for rolling statistics
            ewm_spans: Spans for exponential weighted features
            include_fourier: Whether to include Fourier terms for seasonality
            fourier_terms: Number of Fourier term pairs
            include_holidays: Whether to generate holiday features
            select_features: Whether to perform feature selection
            n_features_to_select: Number of features to select
            feature_selection_method: 'mutual_info' or 'correlation'
        """
        self.target_col = target_col
        self.date_col = date_col
        self.group_cols = group_cols or []
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.ewm_spans = ewm_spans
        self.include_fourier = include_fourier
        self.fourier_terms = fourier_terms
        self.include_holidays = include_holidays
        self.select_features = select_features
        self.n_features_to_select = n_features_to_select
        self.feature_selection_method = feature_selection_method
        
        # Fitted components
        self.encoders = {}
        self.target_encoders = {}
        self.scalers = {}
        self.selected_features = None
        self.feature_importance = None
        
        # Chinese holidays (simplified)
        self.chinese_holidays = self._get_chinese_holidays()
        
    def _get_chinese_holidays(self) -> Dict[str, List[str]]:
        """Define major Chinese holidays."""
        return {
            'spring_festival': [],  # Dates vary by year
            'qingming': ['04-04', '04-05', '04-06'],
            'labor_day': ['05-01', '05-02', '05-03'],
            'dragon_boat': [],  # Dates vary
            'mid_autumn': [],  # Dates vary
            'national_day': ['10-01', '10-02', '10-03', '10-04', '10-05', '10-06', '10-07'],
        }
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        external_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fit the feature engineer and transform data.
        
        Args:
            df: Input DataFrame with target and features
            external_features: Optional external features (weather, etc.)
            
        Returns:
            Transformed DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Sort by date
        df = df.sort_values(self.date_col)
        
        # Generate all features
        df = self._create_date_features(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_ewm_features(df)
        df = self._create_diff_features(df)
        df = self._create_trend_features(df)
        
        if self.include_fourier:
            df = self._create_fourier_features(df)
        
        if self.include_holidays:
            df = self._create_holiday_features(df)
        
        if external_features is not None:
            df = self._merge_external_features(df, external_features)
        
        # Encode categorical variables
        df = self._encode_categoricals(df)
        
        # Target encoding
        df = self._create_target_encoding(df)
        
        # Handle missing values
        df = self._handle_missing(df)
        
        # Feature selection
        if self.select_features and self.target_col in df.columns:
            df = self._select_features(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def transform(
        self,
        df: pd.DataFrame,
        external_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Transform new data using fitted components."""
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(self.date_col)
        
        # Apply same transformations
        df = self._create_date_features(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_ewm_features(df)
        df = self._create_diff_features(df)
        df = self._create_trend_features(df)
        
        if self.include_fourier:
            df = self._create_fourier_features(df)
        
        if self.include_holidays:
            df = self._create_holiday_features(df)
        
        if external_features is not None:
            df = self._merge_external_features(df, external_features)
        
        # Use fitted encoders
        df = self._encode_categoricals(df, fit=False)
        df = self._create_target_encoding(df, fit=False)
        df = self._handle_missing(df)
        
        # Select same features
        if self.selected_features is not None:
            available = [f for f in self.selected_features if f in df.columns]
            df = df[available]
        
        return df
    
    def _create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date/time features."""
        date_series = df[self.date_col]
        
        # Basic date features
        df['year'] = date_series.dt.year
        df['month'] = date_series.dt.month
        df['day'] = date_series.dt.day
        df['day_of_week'] = date_series.dt.dayofweek
        df['day_of_year'] = date_series.dt.dayofyear
        df['week_of_year'] = date_series.dt.isocalendar().week.astype(int)
        df['quarter'] = date_series.dt.quarter
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Month start/end
        df['is_month_start'] = date_series.dt.is_month_start.astype(int)
        df['is_month_end'] = date_series.dt.is_month_end.astype(int)
        
        # Week of month
        df['week_of_month'] = (df['day'] - 1) // 7 + 1
        
        # Season
        df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for target variable."""
        if self.target_col not in df.columns:
            return df
        
        for lag in self.lag_periods:
            if self.group_cols:
                df[f'lag_{lag}'] = df.groupby(self.group_cols)[self.target_col].shift(lag)
            else:
                df[f'lag_{lag}'] = df[self.target_col].shift(lag)
        
        # Same day last week, 2 weeks ago, etc.
        for weeks in [1, 2, 3, 4]:
            if self.group_cols:
                df[f'lag_week_{weeks}'] = df.groupby(self.group_cols)[self.target_col].shift(weeks * 7)
            else:
                df[f'lag_week_{weeks}'] = df[self.target_col].shift(weeks * 7)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics."""
        if self.target_col not in df.columns:
            return df
        
        for window in self.rolling_windows:
            if self.group_cols:
                grouped = df.groupby(self.group_cols)[self.target_col]
            else:
                grouped = df[self.target_col]
            
            # Shift by 1 to avoid data leakage
            rolled = grouped.shift(1).rolling(window=window, min_periods=1)
            
            df[f'rolling_mean_{window}'] = rolled.mean()
            df[f'rolling_std_{window}'] = rolled.std()
            df[f'rolling_min_{window}'] = rolled.min()
            df[f'rolling_max_{window}'] = rolled.max()
            df[f'rolling_median_{window}'] = rolled.median()
            
            # Range and coefficient of variation
            df[f'rolling_range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
            df[f'rolling_cv_{window}'] = df[f'rolling_std_{window}'] / (df[f'rolling_mean_{window}'] + 1e-8)
        
        return df
    
    def _create_ewm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create exponentially weighted moving average features."""
        if self.target_col not in df.columns:
            return df
        
        for span in self.ewm_spans:
            if self.group_cols:
                grouped = df.groupby(self.group_cols)[self.target_col]
            else:
                grouped = df[self.target_col]
            
            ewm = grouped.shift(1).ewm(span=span, min_periods=1)
            
            df[f'ewm_mean_{span}'] = ewm.mean()
            df[f'ewm_std_{span}'] = ewm.std()
        
        return df
    
    def _create_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create difference features."""
        if self.target_col not in df.columns:
            return df
        
        # First and second order differences
        for diff_order in [1, 7]:
            if self.group_cols:
                df[f'diff_{diff_order}'] = df.groupby(self.group_cols)[self.target_col].diff(diff_order)
            else:
                df[f'diff_{diff_order}'] = df[self.target_col].diff(diff_order)
        
        # Percent change
        if self.group_cols:
            df['pct_change_1'] = df.groupby(self.group_cols)[self.target_col].pct_change(1)
            df['pct_change_7'] = df.groupby(self.group_cols)[self.target_col].pct_change(7)
        else:
            df['pct_change_1'] = df[self.target_col].pct_change(1)
            df['pct_change_7'] = df[self.target_col].pct_change(7)
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend-related features."""
        if self.target_col not in df.columns:
            return df
        
        # Moving average crossovers
        if 'rolling_mean_7' in df.columns and 'rolling_mean_28' in df.columns:
            df['ma_7_28_ratio'] = df['rolling_mean_7'] / (df['rolling_mean_28'] + 1e-8)
            df['ma_crossover'] = (df['rolling_mean_7'] > df['rolling_mean_28']).astype(int)
        
        # Momentum
        if 'lag_7' in df.columns and self.target_col in df.columns:
            df['momentum_7'] = df[self.target_col] - df['lag_7']
        
        return df
    
    def _create_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Fourier terms for capturing seasonality."""
        day_of_year = df['day_of_year']
        
        for k in range(1, self.fourier_terms + 1):
            # Yearly seasonality
            df[f'sin_yearly_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
            df[f'cos_yearly_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)
            
            # Weekly seasonality
            df[f'sin_weekly_{k}'] = np.sin(2 * np.pi * k * df['day_of_week'] / 7)
            df[f'cos_weekly_{k}'] = np.cos(2 * np.pi * k * df['day_of_week'] / 7)
            
            # Monthly seasonality
            df[f'sin_monthly_{k}'] = np.sin(2 * np.pi * k * df['day'] / 30)
            df[f'cos_monthly_{k}'] = np.cos(2 * np.pi * k * df['day'] / 30)
        
        return df
    
    def _create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday and special event features."""
        date_series = df[self.date_col]
        
        # Generic indicators
        df['is_holiday'] = 0
        
        # Days before/after holiday
        df['days_to_holiday'] = 999
        df['days_from_holiday'] = 999
        
        # National Day (Oct 1-7)
        national_day_mask = (date_series.dt.month == 10) & (date_series.dt.day <= 7)
        df.loc[national_day_mask, 'is_holiday'] = 1
        
        # Labor Day (May 1-3)
        labor_day_mask = (date_series.dt.month == 5) & (date_series.dt.day <= 3)
        df.loc[labor_day_mask, 'is_holiday'] = 1
        
        # Major shopping days
        df['is_singles_day'] = ((date_series.dt.month == 11) & 
                                (date_series.dt.day == 11)).astype(int)
        df['is_618'] = ((date_series.dt.month == 6) & 
                        (date_series.dt.day == 18)).astype(int)
        df['is_double_12'] = ((date_series.dt.month == 12) & 
                              (date_series.dt.day == 12)).astype(int)
        
        # Pre-holiday period (shopping spike)
        df['is_pre_holiday'] = 0
        for month, day in [(10, 1), (5, 1), (1, 1)]:
            pre_holiday = (
                (date_series.dt.month == month - 1 if month > 1 else 12) & 
                (date_series.dt.day >= 25)
            )
            df.loc[pre_holiday, 'is_pre_holiday'] = 1
        
        return df
    
    def _merge_external_features(
        self,
        df: pd.DataFrame,
        external: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge external features (weather, etc.)."""
        if self.date_col in external.columns:
            external[self.date_col] = pd.to_datetime(external[self.date_col])
        
        merge_cols = [self.date_col] + [c for c in self.group_cols if c in external.columns]
        
        df = df.merge(external, on=merge_cols, how='left')
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_cols = [c for c in cat_cols if c != self.date_col]
        
        for col in cat_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            elif col in self.encoders:
                # Handle unseen categories
                known = set(self.encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known else 'unknown')
                if 'unknown' not in known:
                    self.encoders[col].classes_ = np.append(
                        self.encoders[col].classes_, 'unknown'
                    )
                df[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def _create_target_encoding(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create target encoding for high-cardinality categoricals."""
        if self.target_col not in df.columns:
            return df
        
        for col in self.group_cols:
            if col not in df.columns:
                continue
            
            if fit:
                # Calculate mean target per category with smoothing
                global_mean = df[self.target_col].mean()
                counts = df.groupby(col).size()
                means = df.groupby(col)[self.target_col].mean()
                
                # Smoothing factor
                m = 10
                smooth_means = (counts * means + m * global_mean) / (counts + m)
                
                self.target_encoders[col] = smooth_means.to_dict()
            
            if col in self.target_encoders:
                df[f'{col}_target_enc'] = df[col].map(self.target_encoders[col])
                df[f'{col}_target_enc'] = df[f'{col}_target_enc'].fillna(
                    np.mean(list(self.target_encoders[col].values()))
                )
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        # Fill numeric with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top features based on importance."""
        feature_cols = [c for c in df.columns 
                       if c not in [self.target_col, self.date_col] + self.group_cols]
        
        if len(feature_cols) <= self.n_features_to_select:
            self.selected_features = feature_cols + [self.target_col]
            return df
        
        X = df[feature_cols].values
        y = df[self.target_col].values
        
        # Remove rows with NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if self.feature_selection_method == 'mutual_info':
            # Mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            importance = pd.Series(mi_scores, index=feature_cols)
        else:
            # Correlation-based
            correlations = []
            for i, col in enumerate(feature_cols):
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                correlations.append(corr if not np.isnan(corr) else 0)
            importance = pd.Series(correlations, index=feature_cols)
        
        # Select top features
        top_features = importance.nlargest(self.n_features_to_select).index.tolist()
        self.selected_features = top_features + [self.target_col]
        self.feature_importance = importance.sort_values(ascending=False)
        
        logger.info(f"Selected {len(top_features)} features")
        
        # Keep only selected features plus target
        keep_cols = list(set(self.selected_features + [self.date_col] + self.group_cols))
        keep_cols = [c for c in keep_cols if c in df.columns]
        
        return df[keep_cols]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores."""
        if self.feature_importance is None:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_importance.index,
            'importance': self.feature_importance.values
        })
