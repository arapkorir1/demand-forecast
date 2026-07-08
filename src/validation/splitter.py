import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
from src.util.config_loader import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationSplit:
    """Container for a single walk-forward validation split."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    
    def __repr__(self):
        return f"Split(train={self.train_start}-{self.train_end}, test={self.test_start}-{self.test_end})"


class WalkForwardValidator:
    """
    Walk-forward (expanding window) validation for time-series.
    
    For each split:
        - Train on all data up to day T
        - Validate on the next `test_size` days
        - Expanding window: each split uses more historical data
    
    Attributes:
        n_splits: Number of validation splits
        test_size: Number of days to predict in each split
        gap: Optional gap between train and test (for realistic prediction)
        min_train_size: Minimum days of training data required
    """
    
    def __init__(
        self,
        n_splits: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        min_train_size: int = 30,
    ):
        """
        Args:
            n_splits: Number of validation splits (from config if not provided)
            test_size: Number of days to predict (from config if not provided)
            gap: Gap between train and test (default 0)
            min_train_size: Minimum historical days required (default 30)
        """
        config = get_config('training.validation')
        
        self.n_splits = n_splits or config.get('n_splits', 5)
        self.test_size = test_size or config.get('test_size', 28)
        self.gap = gap or config.get('gap', 0)
        self.min_train_size = min_train_size
        
        logger.info(f"WalkForwardValidator initialized:")
        logger.info(f"  - n_splits: {self.n_splits}")
        logger.info(f"  - test_size: {self.test_size}")
        logger.info(f"  - gap: {self.gap}")
        logger.info(f"  - min_train_size: {self.min_train_size}")
    
    def get_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure data is sorted and has required columns.
        
        Returns a DataFrame with 'day_num' and unique ID columns.
        """
        # Ensure data is sorted
        if 'day_num' not in df.columns:
            raise ValueError("DataFrame must have 'day_num' column")
        
        df_sorted = df.sort_values(['id', 'day_num']).copy()
        return df_sorted
    
    def get_split_boundaries(
        self, 
        df: pd.DataFrame
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate the start/end days for each split.
        
        Returns:
            List of tuples: (train_start, train_end, test_start, test_end)
        """
        # Get global day range
        all_days = sorted(df['day_num'].unique())
        min_day = all_days[0]
        max_day = all_days[-1]
        
        logger.info(f"Data covers days {min_day} to {max_day} ({len(all_days)} days)")
        
        # Calculate split boundaries (backwards from the end)
        splits = []
        
        # Start from the latest possible test window
        latest_test_end = max_day
        latest_test_start = latest_test_end - self.test_size + 1
        
        # Walk backwards through test windows
        for i in range(self.n_splits):
            # Test window
            test_end = latest_test_end - (i * self.test_size)
            test_start = test_end - self.test_size + 1
            
            # Training window (all data up to train_end)
            train_end = test_start - self.gap - 1
            train_start = min_day
            
            # Ensure we have enough training data
            if train_end - train_start + 1 < self.min_train_size:
                logger.warning(
                    f"Split {i}: Training data too small "
                    f"({train_end - train_start + 1} days < {self.min_train_size}). "
                    f"Skipping split."
                )
                continue
            
            # Ensure test window is valid
            if test_start < min_day:
                logger.warning(f"Split {i}: Test window starts before data begins. Skipping.")
                continue
            
            splits.append((train_start, train_end, test_start, test_end))
            
            # Stop if we've run out of data
            if test_start - self.test_size < min_day:
                break
        
        # Reverse so splits are in chronological order
        splits = splits[::-1]
        
        logger.info(f"Generated {len(splits)} valid splits")
        for i, (t_start, t_end, val_start, val_end) in enumerate(splits):
            logger.info(
                f"  Split {i}: train {t_start}-{t_end} "
                f"({t_end - t_start + 1} days), "
                f"test {val_start}-{val_end} ({self.test_size} days)"
            )
        
        return splits
    
    def split(
        self, 
        df: pd.DataFrame,
        return_indices: bool = True
    ) -> List[ValidationSplit]:
        """
        Generate walk-forward validation splits.
        
        Args:
            df: DataFrame with columns ['id', 'day_num', 'sales'] and others
            return_indices: If True, returns indices for training/testing
            
        Returns:
            List of ValidationSplit objects
        """
        df_sorted = self.get_time_series_features(df)
        
        # Get split boundaries
        split_boundaries = self.get_split_boundaries(df_sorted)
        
        # Build splits
        splits = []
        
        for train_start, train_end, test_start, test_end in split_boundaries:
            # Create masks for train and test
            train_mask = (
                (df_sorted['day_num'] >= train_start) & 
                (df_sorted['day_num'] <= train_end)
            )
            test_mask = (
                (df_sorted['day_num'] >= test_start) & 
                (df_sorted['day_num'] <= test_end)
            )
            
            # Get indices
            train_indices = df_sorted.index[train_mask].values
            test_indices = df_sorted.index[test_mask].values
            
            # Create split object
            split = ValidationSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_indices=train_indices if return_indices else None,
                test_indices=test_indices if return_indices else None,
            )
            splits.append(split)
        
        logger.info(f"Created {len(splits)} validation splits")
        
        # Log data sizes
        for i, split in enumerate(splits):
            logger.info(
                f"  Split {i}: train {len(split.train_indices):,} rows, "
                f"test {len(split.test_indices):,} rows"
            )
        
        return splits
    
    def get_split_summary(self, splits: List[ValidationSplit]) -> Dict:
        """Get summary statistics for the splits."""
        return {
            'n_splits': len(splits),
            'total_train_rows': sum(len(s.train_indices) for s in splits),
            'total_test_rows': sum(len(s.test_indices) for s in splits),
            'avg_train_rows': np.mean([len(s.train_indices) for s in splits]),
            'avg_test_rows': np.mean([len(s.test_indices) for s in splits]),
        }


# Quick test on sample data
if __name__ == "__main__":
    import time
    from pathlib import Path
    
    # Load sample data
    sample_path = Path('data/processed/sample_ca1_50items.parquet')
    
    if sample_path.exists():
        logger.info(f"Loading sample from: {sample_path}")
        df = pd.read_parquet(sample_path)
    else:
        logger.error("Sample not found. Please create sample first.")
        exit(1)
    
    # Create validator
    validator = WalkForwardValidator(
        n_splits=5,
        test_size=28,
        gap=0,
        min_train_size=30
    )
    
    # Generate splits
    start = time.time()
    splits = validator.split(df, return_indices=True)
    elapsed = time.time() - start
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SPLITS RESULTS")
    print("=" * 60)
    print(f"✅ Generated {len(splits)} splits in {elapsed:.2f} seconds")
    
    for i, split in enumerate(splits):
        print(f"  Split {i}: train {len(split.train_indices):,} rows, test {len(split.test_indices):,} rows")
    
    # Show example of a split
    if splits:
        print("\n📊 Example Split 0:")
        print(f"  Train days: {splits[0].train_start} - {splits[0].train_end}")
        print(f"  Test days:  {splits[0].test_start} - {splits[0].test_end}")
        print(f"  Train rows: {len(splits[0].train_indices):,}")
        print(f"  Test rows:  {len(splits[0].test_indices):,}")
        
        # Show first few and last few days
        train_days = df.loc[splits[0].train_indices, 'day_num'].unique()
        test_days = df.loc[splits[0].test_indices, 'day_num'].unique()
        print(f"\n  Train days range: {train_days.min()} - {train_days.max()}")
        print(f"  Test days range:  {test_days.min()} - {test_days.max()}")
