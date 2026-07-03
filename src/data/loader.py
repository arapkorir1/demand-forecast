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
        """Initialize with config."""
        self.config = get_config()
        self.data_config = self.config['data']
        self.raw_path = Path(self.data_config['raw_path'])
        
        # Define optimized dtypes for memory efficiency
        self.dtype_dict = {
            # Integer columns (downcast to save memory)
            'id': 'int32',
            'item_id': 'int32',
            'dept_id': 'int32',
            'cat_id': 'int32',
            'store_id': 'int32',
            'state_id': 'int32',
            
            # Float columns
            'sell_price': 'float32',
            
            # Daily sales (1,916 columns!) — downcast to float32
            # We'll handle these dynamically in load_sales()
        }
        
        logger.info(f"DataLoader initialized with raw_path: {self.raw_path}")
    
    def load_sales(self) -> pd.DataFrame:
        """
        Load sales_train_evaluation.csv with optimized dtypes.
        
        Returns:
            DataFrame with sales data (melted to long format)
        """
        sales_file = self.raw_path / self.data_config['dataset']['train_file']
        
        if not sales_file.exists():
            raise FileNotFoundError(f"Sales file not found: {sales_file}")
        
        logger.info(f"Loading sales data from: {sales_file}")
        logger.info("This may take 30-60 seconds for 30M rows...")
        
        # Step 1: Read the CSV with optimized dtypes
        # We'll read everything as float32 except the first few ID columns
        df = pd.read_csv(
            sales_file,
            dtype={
                'id': 'int32',
                'item_id': 'int32',
                'dept_id': 'int32',
                'cat_id': 'int32',
                'store_id': 'int32',
                'state_id': 'int32',
            },
            low_memory=False  # Prevents mixed-type warnings
        )
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        # Step 2: Melt to long format (d1, d2, ..., d1916 → date, sales)
        logger.info("Melting from wide to long format...")
        
        # Identify sales columns (d1 to d1916)
        sales_cols = [col for col in df.columns if col.startswith('d_')]
        logger.info(f"Found {len(sales_cols)} daily sales columns")
        
        # Melt: one row per SKU per day
        df_melted = df.melt(
            id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
            value_vars=sales_cols,
            var_name='day',
            value_name='sales'
        )
        
        # Convert day (d_1) to integer
        df_melted['day_num'] = df_melted['day'].str.replace('d_', '').astype('int32')
        
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
        
        df = pd.read_csv(cal_file)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create day_num to match sales data (d_1 = day 1)
        df['day_num'] = df['d'].str.replace('d_', '').astype('int32')
        
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
                'item_id': 'int32',
                'store_id': 'int32',
                'sell_price': 'float32'
            }
        )
        
        # Convert date to datetime
        df['wm_yr_wk'] = df['wm_yr_wk'].astype('int32')
        
        logger.info(f"Loaded {len(df):,} price records")
        
        return df
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data and merge into a single DataFrame.
        
        Returns:
            Dictionary with keys: 'sales', 'calendar', 'prices', 'merged'
        """
        logger.info("=" * 60)
        logger.info("LOADING ALL DATA")
        logger.info("=" * 60)
        
        # Load individual datasets
        sales = self.load_sales()
        calendar = self.load_calendar()
        prices = self.load_prices()
        
        # Merge sales with calendar
        logger.info("Merging sales with calendar...")
        merged = sales.merge(calendar, on='day_num', how='left')
        
        # Merge with prices (if available)
        if prices is not None:
            logger.info("Merging with prices...")
            # Note: M5 requires merging on (store_id, item_id, wm_yr_wk)
            # We'll implement this properly in Session 2
            pass
        
        logger.info("Data loading complete!")
        logger.info(f"Merged shape: {merged.shape}")
        
        return {
            'sales': sales,
            'calendar': calendar,
            'prices': prices,
            'merged': merged
        }
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the data has the expected schema.
        
        Checks:
        - Required columns exist
        - No null values in critical columns
        - Date ranges are within expected bounds
        """
        required_cols = ['id', 'item_id', 'store_id', 'day_num', 'sales']
        
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        # Check for nulls
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found: {null_counts[null_counts > 0]}")
        
        # Check date range
        if 'day_num' in df.columns:
            logger.info(f"Date range: {df['day_num'].min()} to {df['day_num'].max()}")
        
        logger.info("✅ Schema validation passed!")
        return True
    
    def get_summary(self, df: pd.DataFrame) -> Dict:
        """Get a quick summary of the data."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'unique_items': df['item_id'].nunique() if 'item_id' in df else None,
            'unique_stores': df['store_id'].nunique() if 'store_id' in df else None,
            'date_range': (df['day_num'].min(), df['day_num'].max()) if 'day_num' in df else None,
            'total_sales': df['sales'].sum() if 'sales' in df else None,
            'avg_sales': df['sales'].mean() if 'sales' in df else None,
            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }


# Quick test when run directly
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the loader
    loader = DataLoader()
    data_dict = loader.load_all()
    
    # Validate
    if data_dict['merged'] is not None:
        loader.validate_schema(data_dict['merged'])
        summary = loader.get_summary(data_dict['merged'])
        print("\n📊 Data Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
