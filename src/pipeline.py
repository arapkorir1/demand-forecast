
"""
Master pipeline that orchestrates: load → process → feature engineer → train → evaluate
This is the ENTRY POINT for Phase 1.
"""
import sys
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import get_config
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.features.builder import FeatureBuilder
from src.features.calendar import CalendarFeatures
from src.validation.splitter import TimeSeriesSplitter
from src.models.trainer import ModelTrainer
from src.models.fallback import FallbackModel
from src.evaluation.metrics import Evaluator

# Configure logging (we'll set up proper logging in Session 2)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForecastPipeline:
    """End-to-end demand forecasting pipeline."""
    
    def __init__(self):
        self.config = {
            'lags': get_config('features.lags'),
            'rolling_windows': get_config('features.rolling_windows'),
            'n_splits': get_config('training.validation.n_splits'),
            'test_size': get_config('training.validation.test_size'),
        }
        logger.info(f"Pipeline initialized with config: {self.config}")
    
    def run(self):
        """Execute the full pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING DEMAND FORECAST PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Load data
        logger.info("Step 1: Loading raw data...")
        # loader = DataLoader()
        # raw_data = loader.load()
        raw_data = None  # Placeholder - will implement in Session 2
        
        # Step 2: Process data (handle missing dates, outliers)
        logger.info("Step 2: Processing data...")
        # processor = DataProcessor()
        # processed_data = processor.process(raw_data)
        processed_data = None  # Placeholder
        
        # Step 3: Feature engineering
        logger.info("Step 3: Building features...")
        # calendar = CalendarFeatures()
        # calendar_features = calendar.transform(processed_data)
        # feature_builder = FeatureBuilder(lags=self.config['lags'])
        # X, y = feature_builder.build(processed_data)
        X, y = None, None  # Placeholder
        
        # Step 4: Time-series split
        logger.info("Step 4: Creating walk-forward validation splits...")
        # splitter = TimeSeriesSplitter(n_splits=self.config['n_splits'])
        # splits = splitter.split(X, y)
        splits = None  # Placeholder
        
        # Step 5: Train models
        logger.info("Step 5: Training LightGBM...")
        # trainer = ModelTrainer()
        # model = trainer.train(X, y, splits)
        model = None  # Placeholder
        
        # Step 6: Fallback for cold-start SKUs
        logger.info("Step 6: Training fallback SARIMA model...")
        # fallback = FallbackModel()
        # fallback_model = fallback.train(processed_data)
        fallback_model = None  # Placeholder
        
        # Step 7: Evaluate
        logger.info("Step 7: Evaluating model...")
        # evaluator = Evaluator()
        # metrics = evaluator.evaluate(model, X, y, splits)
        metrics = None  # Placeholder
        
        # Step 8: Save artifacts
        logger.info("Step 8: Saving model artifacts...")
        # Save to models/artifacts/
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return model, metrics

if __name__ == "__main__":
    # Run the pipeline directly
    pipeline = ForecastPipeline()
    model, metrics = pipeline.run()
    
    print("\n✅ Pipeline finished!")
    print(f"Metrics: {metrics}")

