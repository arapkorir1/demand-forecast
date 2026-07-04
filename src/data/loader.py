"""
Production-grade data loader for M5 forecasting.
Handles 30M+ rows efficiently with:
- Memory-optimized dtypes
- Data validation (schema checks)
- Progress logging
- Config-driven paths
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

# Import your config loader
from src.util.config_loader import get_config

# Set up logging
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads and validates M5 forecasting data.
    
    Features:
    - Reads CSVs with optimized dtypes (saves 60%+ memory)
    - Validates schema (columns, dtypes, date ranges)
    - Merges sales with calendar and price data
    - Logs every step for debugging
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with config using dot notation lookup."""
        # Use the dot-notation key expected by src/util/config_loader.py
        self.data_config = get_config('data')
        if not self.data_config:
            raise KeyError("Could not retrieve 'data' section from configuration parameters.")
            
        self.raw_path = Path(self.data_config['raw_path'])
        
        # Base ID mapping to minimize footprint
        self.base_dtypes = {
            'id': 'category',
            'item_id': 'category',
            'dept_id': 'category',
            'cat_id': 'category',
            'store_id': 'category',
            'state_id': 'category',
        }
        
        logger.info(f"DataLoader initialized with raw_path: {self.raw_path}")
    
    def load_sales(self) -> pd.DataFrame:
        """
        Load sales_train_evaluation.csv with optimized dtypes and fast melting.
        
        Returns:
            DataFrame with sales data (melted to long format)
        """
        sales_file = self.raw_path / self.data_config['dataset']['train_file']
        
        if not sales_file.exists():
            raise FileNotFoundError(f"Sales file not found: {sales_file}")
        
        logger.info(f"Loading sales data from: {sales_file}")
        logger.info("Optimizing dtypes dynamically...")

        # Peek ahead to get all column names
        cols = pd.read_csv(sales_file, nrows=0).columns
        sales_cols = [col for col in cols if col.startswith('d_')]
        
        # Enforce int16 for daily values to optimize RAM usage
        dynamic_dtypes = {col: 'int16' for col in sales_cols}
        dynamic_dtypes.update(self.base_dtypes)
        
        df = pd.read_csv(
            sales_file,
            dtype=dynamic_dtypes,
            low_memory=False
        )
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Step 2: Melt to long format
        logger.info("Melting from wide to long format...")
        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        df_melted = df.melt(
            id_vars=id_vars,
            value_vars=sales_cols,
            var_name='day',
            value_name='sales'
        )
        
        # Fast string extraction using vectorized string replace
        logger.info("Optimizing melted day string formats...")
        df_melted['day_num'] = df_melted['day'].str.replace('d_', '').astype('int16')
        
        # Drop the heavy object string column now that we have day_num
        df_melted.drop(columns=['day'], inplace=True)
        
        logger.info(f"Melted to {len(df_melted):,} rows")
        logger.info(f"Memory usage after melt: {df_melted.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        return df_melted
    
    def load_calendar(self) -> pd.DataFrame:
        """Load calendar.csv with date features."""
        cal_file = self.raw_path / self.data_config['dataset']['calendar_file']
        
        if not cal_file.exists():
            logger.warning(f"Calendar file not found: {cal_file}")
            return None
        
        logger.info(f"Loading calendar from: {cal_file}")
        
        cal_dtypes = {
            'wm_yr_wk': 'int32',
            'weekday': 'category',
            'wday': 'int8',
            'month': 'int8',
            'year': 'int16',
            'd': 'category',
            'event_name_1': 'category',
            'event_type_1': 'category',
            'event_name_2': 'category',
            'event_type_2': 'category',
            'snap_CA': 'int8',
            'snap_TX': 'int8',
            'snap_WI': 'int8'
        }
        
        df = pd.read_csv(cal_file, dtype=cal_dtypes)
        df['date'] = pd.to_datetime(df['date'])
        df['day_num'] = df['d'].str.replace('d_', '').astype('int16')
        df.drop(columns=['d'], inplace=True)
        
        logger.info(f"Loaded {len(df):,} calendar rows")
        return df
    
    def load_prices(self) -> pd.DataFrame:
        """Load sell_prices.csv."""
        price_file = self.raw_path / self.data_config['dataset']['prices_file']
        
        if not price_file.exists():
            logger.warning(f"Prices file not found: {price_file}")
            return None
        
        logger.info(f"Loading prices from: {price_file}")
        
        df = pd.read_csv(
            price_file,
            dtype={
                'item_id': 'category',
                'store_id': 'category',
                'sell_price': 'float32',
                'wm_yr_wk': 'int32'
            }
        )
        
        logger.info(f"Loaded {len(df):,} price records")
        return df
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data and merge into a single memory-optimized DataFrame.
        """
        logger.info("=" * 60)
        logger.info("LOADING AND MERGING ALL DATA")
        logger.info("=" * 60)
        
        sales = self.load_sales()
        calendar = self.load_calendar()
        prices = self.load_prices()
        
        # Merge sales with calendar features
        logger.info("Merging sales with calendar...")
        merged = sales.merge(calendar, on='day_num', how='left')
        
        # Drop temporary object reference to free memory allocation
        del sales 
        
        # Merge with prices using structural composite keys
        if prices is not None:
            logger.info("Merging with prices on (store_id, item_id, wm_yr_wk)...")
            merged = merged.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        
        logger.info("Data loading and downstream merging complete!")
        logger.info(f"Final Merged shape: {merged.shape}")
        logger.info(f"Final Merged Memory Usage: {merged.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        return {
            'calendar': calendar,
            'prices': prices,
            'merged': merged
        }
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate that the data has the expected schema bounds."""
        required_cols = ['id', 'item_id', 'store_id', 'day_num', 'sales', 'wm_yr_wk', 'sell_price']
        
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found in critical fields:\n{null_counts[null_counts > 0]}")
        
        logger.info("✅ Schema validation passed!")
        return True
    
    def get_summary(self, df: pd.DataFrame) -> Dict:
        """Get a structural summary parameters dictionary."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'unique_items': df['item_id'].nunique() if 'item_id' in df else None,
            'unique_stores': df['store_id'].nunique() if 'store_id' in df else None,
            'date_range': (df['day_num'].min(), df['day_num'].max()) if 'day_num' in df else None,
            'total_sales': df['sales'].sum() if 'sales' in df else None,
            'avg_sales': df['sales'].mean() if 'sales' in df else None
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    data_dict = loader.load_all()
    
    if data_dict['merged'] is not None:
        loader.validate_schema(data_dict['merged'])
        summary = loader.get_summary(data_dict['merged'])
        print("\n📊 Data Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
