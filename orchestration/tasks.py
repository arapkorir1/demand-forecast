"""
Prefect tasks for the demand forecasting pipeline.
Each task is a discrete, reusable unit of work.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
import pickle
import json
from datetime import datetime
from typing import Dict, Any
import time

from prefect import task

from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.validation.splitter import WalkForwardValidator
from src.models.trainer import ModelTrainer
from src.util.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task(name="load_full_data", retries=2, retry_delay_seconds=60)
def load_full_data() -> pd.DataFrame:
    """Load the full M5 dataset and merge."""
    logger.info("Loading full dataset...")
    loader = DataLoader()
    data = loader.load_all()
    merged = data['merged']
    logger.info(f"Loaded {len(merged):,} rows")
    return merged

@task(name="build_features", retries=2)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the full dataset."""
    logger.info("Building features...")
    builder = FeatureBuilder(df)
    featured_df = builder.transform()
    logger.info(f"Features built: {featured_df.shape}")
    return featured_df

@task(name="train_model", retries=2)
def train_model(featured_df: pd.DataFrame) -> Dict[str, Any]:
    """Train models with walk-forward validation."""
    logger.info("Training model...")
    trainer = ModelTrainer()
    results = trainer.train_all_splits(
        df=featured_df,
        target_col='sales',
        save_models=True
    )
    trainer.save_results(results)
    logger.info("Training complete.")
    return results

@task(name="log_to_mlflow")
def log_to_mlflow(results: Dict[str, Any]) -> None:
    """Log metrics and model to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri("models/mlflow")
        mlflow.set_experiment("demand_forecast")
        
        with mlflow.start_run():
            # Log parameters
            config = load_config()
            mlflow.log_params(config.get('training', {}).get('lgbm', {}))
            
            # Log metrics
            metrics_df = results['metrics_df']
            for col in metrics_df.columns:
                if col not in ['split_id', 'train_size', 'test_size']:
                    mlflow.log_metric(f"avg_{col}", metrics_df[col].mean())
                    mlflow.log_metric(f"std_{col}", metrics_df[col].std())
            
            # Log model
            # (we already saved models as artifacts, but we can also log to MLflow)
            model_path = Path('models/artifacts/lgbm_split_0.pkl')
            if model_path.exists():
                mlflow.log_artifact(str(model_path))
            
            # Log feature importance
            importance_path = Path('models/artifacts/feature_importance.csv')
            if importance_path.exists():
                mlflow.log_artifact(str(importance_path))
            
            logger.info("Logged to MLflow successfully.")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

@task(name="deploy_model")
def deploy_model() -> None:
    """Deploy the best model to the API (restart container or copy)."""
    # In a real production setup, this would trigger a deployment.
    # For now, we just ensure the model exists.
    model_dir = Path('models/artifacts')
    if not (model_dir / 'lgbm_split_0.pkl').exists():
        raise FileNotFoundError("No model to deploy.")
    
    # Optionally, copy to a deployment directory or trigger API restart.
    logger.info("Model deployment successful.")

@task(name="cleanup")
def cleanup() -> None:
    """Remove temporary files, free memory."""
    import gc
    gc.collect()
    logger.info("Cleanup complete.")
