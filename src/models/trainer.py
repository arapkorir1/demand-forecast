import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import logging
import pickle
import json
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.util.config_loader import load_config
from src.validation.splitter import WalkForwardValidator, ValidationSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Production-grade LightGBM trainer with walk-forward validation.
    
    Features:
    - Config-driven hyperparameters
    - Walk-forward validation (no leakage)
    - Early stopping with multiple metrics
    - Model persistence (pickle + MLflow)
    - Feature importance logging
    - Prediction and evaluation per split
    """
    
    def __init__(
        self,
        model_params: Optional[Dict] = None,
        n_splits: Optional[int] = None,
        test_size: Optional[int] = None,
    ):
        """
        Args:
            model_params: LightGBM parameters (from config if not provided)
            n_splits: Number of validation splits (from config if not provided)
            test_size: Number of days to predict (from config if not provided)
        """
        # Load config
        config = load_config()
        lgbm_config = config['training']['lgbm']
        val_config = config['training']['validation']
        
        # Set model parameters
        self.model_params = model_params or {
            'objective': lgbm_config.get('objective', 'regression'),
            'metric': lgbm_config.get('metric', 'rmse'),
            'boosting_type': lgbm_config.get('boosting_type', 'gbdt'),
            'num_leaves': lgbm_config.get('num_leaves', 31),
            'max_depth': lgbm_config.get('max_depth', -1),
            'learning_rate': lgbm_config.get('learning_rate', 0.05),
            'n_estimators': lgbm_config.get('n_estimators', 1000),
            'feature_fraction': lgbm_config.get('feature_fraction', 0.8),
            'bagging_fraction': lgbm_config.get('bagging_fraction', 0.8),
            'bagging_freq': lgbm_config.get('bagging_freq', 5),
            'min_child_samples': lgbm_config.get('min_child_samples', 20),
            'num_threads': config.get('hardware', {}).get('cpu_cores', 4),
            'random_state': lgbm_config.get('random_state', 42),
            'verbose': -1,
        }
        
        self.n_splits = n_splits or val_config.get('n_splits', 5)
        self.test_size = test_size or val_config.get('test_size', 28)
        self.early_stopping_rounds = lgbm_config.get('early_stopping_rounds', 50)
        
        logger.info("ModelTrainer initialized with:")
        logger.info(f"  - n_splits: {self.n_splits}")
        logger.info(f"  - test_size: {self.test_size}")
        logger.info(f"  - early_stopping_rounds: {self.early_stopping_rounds}")
        logger.info(f"  - model_params: {self.model_params}")
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'sales',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature columns (auto-detect if None)
        
        Returns:
            X: Feature DataFrame
            y: Target Series
            feature_cols: List of feature column names
        """
        logger.info(f"Preparing data with target: '{target_col}'")
        
        # Remove columns that should not be used as features
        exclude_cols = [
            target_col, 
            'id', 'day', 'date', 'd',  # Identifiers/dates
            'day_num', 'wm_yr_wk',  # Already encoded
            'state_id', 'cat_id', 'dept_id',  # IDs (handle as categorical)
        ]
        
        if feature_cols is None:
            # Auto-detect features: all numeric columns except target and excluded
            feature_cols = [
                col for col in df.columns 
                if col not in exclude_cols 
                and col != target_col
                and not col.startswith('d_')  # Exclude original sales columns
            ]
            
            # Identify categorical columns (string or object)
            categorical_cols = [
                col for col in feature_cols 
                if df[col].dtype in ['object', 'category', 'string']
            ]
            
            # Convert categorical columns to 'category' dtype
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                    logger.info(f"Converted '{col}' to categorical")
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        logger.info(f"Prepared {len(X):,} rows with {len(feature_cols)} features")
        logger.info(f"Feature columns: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Features: {feature_cols}")
        
        return X, y, feature_cols
    
    def train_split(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        split_id: int,
    ) -> Dict[str, Any]:
        """
        Train a single model on one split. Handles empty test windows gracefully.
        
        Returns:
            Dict with model, predictions, metrics, feature_importance
        """
        logger.info(f"Training split {split_id}...")
        logger.info(f"  Train: {len(X_train):,} rows")
        logger.info(f"  Test:  {len(X_test):,} rows")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            categorical_feature='auto'  # Auto-detect categorical columns
        )
        
        # Train
        model = lgb.train(
            self.model_params,
            train_data,
            valid_sets=[train_data],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),  # Silent
            ]
        )
        
        # Predict train
        y_pred_train = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_wape = np.sum(np.abs(y_train - y_pred_train)) / np.sum(y_train) * 100 if np.sum(y_train) > 0 else 0
        
        # Handle empty or zero row validation splits defensively
        if len(X_test) == 0:
            logger.warning(f"⚠️ Split {split_id} has an empty test set! Skipping validation metrics for this split.")
            y_pred_test = np.array([])
            test_rmse, test_mae, test_wape = np.nan, np.nan, np.nan
        else:
            y_pred_test = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_wape = np.sum(np.abs(y_test - y_pred_test)) / np.sum(y_test) * 100 if np.sum(y_test) > 0 else 0
            
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        metrics = {
            'split_id': split_id,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_wape': train_wape,
            'test_wape': test_wape,
            'n_estimators': model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators,
        }
        
        logger.info(f"Split {split_id} results:")
        logger.info(f"  Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        logger.info(f"  Train WAPE: {train_wape:.2f}% | Test WAPE: {test_wape:.2f}%")
        
        return {
            'model': model,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'metrics': metrics,
            'feature_importance': feature_importance,
        }
    
    def train_all_splits(
        self,
        df: pd.DataFrame,
        target_col: str = 'sales',
        feature_cols: Optional[List[str]] = None,
        save_models: bool = True,
        model_dir: Path = Path('models/artifacts'),
    ) -> Dict[str, Any]:
        """
        Train models on all walk-forward splits.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            feature_cols: List of feature columns (auto-detect if None)
            save_models: Whether to save models to disk
            model_dir: Directory to save models
        
        Returns:
            Dict with results for all splits
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING (WALK-FORWARD)")
        logger.info("=" * 60)
        
        # Prepare features
        X, y, feature_cols = self.prepare_data(df, target_col, feature_cols)
        
        # Generate splits
        validator = WalkForwardValidator(
            n_splits=self.n_splits,
            test_size=self.test_size,
        )
        splits = validator.split(df, return_indices=True)
        
        logger.info(f"Training on {len(splits)} validation splits")
        
        # Train on each split
        results = {
            'models': [],
            'predictions': [],
            'metrics': [],
            'feature_importance': [],
            'feature_cols': feature_cols,
        }
        
        for i, split in enumerate(splits):
            logger.info(f"\n{'='*40}")
            logger.info(f"SPLIT {i+1}/{len(splits)}")
            logger.info(f"{'='*40}")
            
            # Get train/test indices
            train_idx = split.train_indices
            test_idx = split.test_indices
            
            # Split data
            X_train = X.loc[X.index.intersection(train_idx)]
            X_test = X.loc[X.index.intersection(test_idx)]
            y_train = y.loc[y.index.intersection(train_idx)]
            y_test = y.loc[y.index.intersection(test_idx)]
            
            # Train
            split_result = self.train_split(
                X_train, y_train,
                X_test, y_test,
                split_id=i
            )
            
            # Store results
            results['models'].append(split_result['model'])
            results['predictions'].append({
                'split_id': i,
                'y_pred_train': split_result['y_pred_train'],
                'y_pred_test': split_result['y_pred_test'],
            })
            results['metrics'].append(split_result['metrics'])
            results['feature_importance'].append(split_result['feature_importance'])
            
            # Save model
            if save_models:
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f'lgbm_split_{i}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(split_result['model'], f)
                logger.info(f"✅ Saved model to: {model_path}")
        
        # Summarize results
        metrics_df = pd.DataFrame(results['metrics'])
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Average Test RMSE: {metrics_df['test_rmse'].nanmean() if hasattr(metrics_df['test_rmse'], 'nanmean') else metrics_df['test_rmse'].mean(skipna=True):.4f}")
        logger.info(f"Average Test WAPE: {metrics_df['test_wape'].mean(skipna=True):.2f}%")
        
        results['metrics_df'] = metrics_df
        
        return results
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: Path = Path('models/artifacts'),
    ) -> None:
        """Save training results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_df = results['metrics_df']
        metrics_path = output_dir / 'training_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"✅ Saved metrics to: {metrics_path}")
        
        # Save feature importance
        importance_df = pd.concat(results['feature_importance'])
        importance_summary = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        importance_path = output_dir / 'feature_importance.csv'
        importance_summary.to_csv(importance_path)
        logger.info(f"✅ Saved feature importance to: {importance_path}")
        
        # Save summary
        summary = {
            'n_splits': len(results['models']),
            'avg_test_rmse': float(metrics_df['test_rmse'].mean(skipna=True)),
            'avg_test_wape': float(metrics_df['test_wape'].mean(skipna=True)),
            'feature_cols': results['feature_cols'],
            'model_params': self.model_params,
            'timestamp': datetime.now().isoformat(),
        }
        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Saved summary to: {summary_path}")


# Quick test on sample data
if __name__ == "__main__":
    from pathlib import Path
    
    # Load sample data
    sample_path = Path('data/processed/sample_ca1_50items.parquet')
    
    if sample_path.exists():
        logger.info(f"Loading sample from: {sample_path}")
        df = pd.read_parquet(sample_path)
    else:
        logger.error("Sample not found. Please create sample first.")
        exit(1)
    
    # Build features (if not already built)
    from src.features.builder import FeatureBuilder
    builder = FeatureBuilder(df)
    df_featured = builder.transform()
    
    # Train model
    trainer = ModelTrainer(
        n_splits=3,  # Use 3 splits for quick test
        test_size=28,
    )
    
    results = trainer.train_all_splits(
        df=df_featured,
        target_col='sales',
        save_models=True,
    )
    
    trainer.save_results(results)
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models saved in: models/artifacts/")
    print(f"Training metrics saved in: models/artifacts/training_metrics.csv")
