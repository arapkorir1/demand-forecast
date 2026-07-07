import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from src.util.config_loader import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureBuilder:
    """
    Builds time-series features for demand forecasting.
    
    Features:
    - Calendar: day_of_week, month, is_holiday, weekend, etc.
    - Lags: sales_t-1, t-7, t-14, t-28
    - Rolling stats: 7d/14d/28d mean, std, min, max
    - Price features: sell_price, price_change
    
    Note: Always sorts by (id, day_num) before feature creation.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Merged DataFrame with columns: id, item_id, store_id, 
                day_num, sales, date, sell_price, and calendar columns.
        """
        self.df = df.copy()
        self.config = {
            'lags': get_config('features.lags'),
            'rolling_windows': get_config('features.rolling_windows'),
            'rolling_aggs': get_config('features.rolling_aggregations'),
        }
        logger.info(f"FeatureBuilder initialized with config: {self.config}")
        logger.info(f"Input shape: {self.df.shape}")
        
    def _sort_data(self) -> None:
        """Ensure data is sorted per item-store combination."""
        logger.info("Sorting data by (id, day_num)...")
        self.df = self.df.sort_values(['id', 'day_num'])
        
    def add_calendar_features(self) -> None:
        """Extract calendar features from date/datetime columns."""
        logger.info("Adding calendar features...")
        
        # Ensure date is datetime
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['year'] = self.df['date'].dt.year
            self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype('int8')
            
            # Handle holiday column safely
            if 'is_holiday' in self.df.columns:
                self.df['is_holiday'] = self.df['is_holiday'].fillna(0).astype('int8')
            else:
                self.df['is_holiday'] = 0
            
            # Cyclical encoding (sine/cosine) for day and month
            self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
            self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
            
        logger.info(f"Calendar features added. Shape: {self.df.shape}")
        
    def add_lag_features(self) -> None:
        """Add lag features for each item-store combination."""
        logger.info(f"Adding lag features: {self.config['lags']}")
        
        # Group by unique ID (item-store combination)
        grouped = self.df.groupby('id')
        
        for lag in self.config['lags']:
            col_name = f'sales_lag_{lag}'
            self.df[col_name] = grouped['sales'].shift(lag)
            
        logger.info(f"Lag features added. Shape: {self.df.shape}")
        
    def add_rolling_features(self) -> None:
        """Add rolling statistics (mean, std, min, max)."""
        logger.info(f"Adding rolling features: {self.config['rolling_windows']}")
        
        grouped = self.df.groupby('id')
        
        for window in self.config['rolling_windows']:
            # Rolling mean
            col_name = f'sales_roll_mean_{window}'
            self.df[col_name] = grouped['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Rolling std
            col_name = f'sales_roll_std_{window}'
            self.df[col_name] = grouped['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            
            # Rolling min
            col_name = f'sales_roll_min_{window}'
            self.df[col_name] = grouped['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            
            # Rolling max
            col_name = f'sales_roll_max_{window}'
            self.df[col_name] = grouped['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
            
        logger.info(f"Rolling features added. Shape: {self.df.shape}")
        
    def add_price_features(self) -> None:
        """Add price-based features."""
        logger.info("Adding price features...")
        
        if 'sell_price' not in self.df.columns:
            logger.warning("No sell_price column found. Skipping price features.")
            return
        
        # Price change vs previous day
        grouped = self.df.groupby('id')
        self.df['price_change'] = grouped['sell_price'].pct_change()
        self.df['price_change'] = self.df['price_change'].fillna(0)
        
        # Price relative to item average (optional)
        item_avg_price = self.df.groupby('item_id')['sell_price'].transform('mean')
        self.df['price_relative_to_item'] = self.df['sell_price'] / item_avg_price
        
        logger.info("Price features added.")
        
    def transform(self) -> pd.DataFrame:
        """Execute all feature engineering steps in order."""
        logger.info("=" * 60)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        # Step 1: Sort (critical!)
        self._sort_data()
        
        # Step 2: Calendar
        self.add_calendar_features()
        
        # Step 3: Lags
        self.add_lag_features()
        
        # Step 4: Rolling stats
        self.add_rolling_features()
        
        # Step 5: Price features
        self.add_price_features()
        
        # Step 6: Drop NaN rows (caused by lags/rolling at beginning of series)
        rows_before = len(self.df)
        self.df = self.df.dropna()
        rows_after = len(self.df)
        logger.info(f"Dropped {rows_before - rows_after:,} rows with NaN features")
        
        logger.info(f"✅ Feature engineering complete! Final shape: {self.df.shape}")
        logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        return self.df


# Quick test on sample data
if __name__ == "__main__":
    import time
    from pathlib import Path
    from src.data.loader import DataLoader
    
    # Load sample or full data
    sample_path = Path('data/processed/sample_ca1_50items.parquet')
    
    if sample_path.exists():
        logger.info(f"Loading sample from: {sample_path}")
        df = pd.read_parquet(sample_path)
    else:
        logger.info("Sample not found. Loading full dataset...")
        loader = DataLoader()
        data = loader.load_all()
        df = data['merged']
        # Take a small subset
        df = df[df['store_id'] == 'CA_1']
        items = df['item_id'].unique()[:50]
        df = df[df['item_id'].isin(items)]
    
    # Build features
    builder = FeatureBuilder(df)
    start = time.time()
    featured_df = builder.transform()
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print("FEATURE BUILDING RESULTS")
    print("=" * 60)
    print(f"✅ Rows: {len(featured_df):,}")
    print(f"✅ Columns: {len(featured_df.columns)}")
    print(f"✅ Time elapsed: {elapsed:.2f} seconds")
    print(f"✅ New features: {', '.join([col for col in featured_df.columns if col.startswith('sales_') or col.startswith('price_') or col in ['day_of_week', 'month', 'is_holiday']])}")
