"""
API Dependencies: Pre-load models and historical data at startup.
This ensures sub-200ms latency for predictions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException
from src.util.config_loader import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastDependencies:
    """
    Singleton class that loads and caches all dependencies at API startup.
    
    Caches:
    - Trained LightGBM model
    - Historical sales data (last 30 days per item-store)
    - Feature column names
    - Scaler (if used)
    """
    
    _instance = None
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._loaded:
            self._load_dependencies()
    
    def _load_dependencies(self) -> None:
        """Load all dependencies from disk."""
        logger.info("=" * 60)
        logger.info("LOADING API DEPENDENCIES")
        logger.info("=" * 60)
        
        # Load model
        model_dir = Path('models/artifacts')
        model_path = model_dir / 'lgbm_split_0.pkl'  # Use first split for now
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Loaded model from: {model_path}")
        else:
            logger.warning("No model found. Please train first.")
            self.model = None
        
        # Load feature columns from training summary
        summary_path = model_dir / 'training_summary.json'
        if summary_path.exists():
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            self.feature_cols = summary.get('feature_cols', [])
            logger.info(f"✅ Loaded {len(self.feature_cols)} feature columns")
        else:
            self.feature_cols = []
        
        # Load historical data (for building lags)
        # In production, this would be a database query
        # For now, use a precomputed parquet file
        history_path = Path('data/processed/historical_data.parquet')
        if history_path.exists():
            self.historical_data = pd.read_parquet(history_path)
            logger.info(f"✅ Loaded historical data: {len(self.historical_data):,} rows")
        else:
            logger.warning("No historical data found. Using sample data...")
            sample_path = Path('data/processed/sample_ca1_50items.parquet')
            if sample_path.exists():
                self.historical_data = pd.read_parquet(sample_path)
                logger.info(f"✅ Loaded sample data: {len(self.historical_data):,} rows")
            else:
                self.historical_data = None
        
        self._loaded = True
        logger.info("=" * 60)
    
    def get_model(self):
        """Return loaded model."""
        if not self._loaded:
            self._load_dependencies()
        return self.model
    
    def get_historical_data(self) -> pd.DataFrame:
        """Return cached historical data."""
        if not self._loaded:
            self._load_dependencies()
        return self.historical_data
    
    def get_feature_cols(self) -> list:
        """Return feature column names."""
        if not self._loaded:
            self._load_dependencies()
        return self.feature_cols

# Global singleton instance
deps = ForecastDependencies()
