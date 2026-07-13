"""
Prefect flow that orchestrates the full demand forecasting pipeline.
Runs nightly via schedule.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prefect import flow, get_run_logger
from prefect.schedules import IntervalSchedule
from datetime import timedelta
import logging

from orchestration.tasks import (
    load_full_data,
    build_features,
    train_model,
    log_to_mlflow,
    deploy_model,
    cleanup
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define schedule (run every night at 2 AM)
schedule = IntervalSchedule(
    interval=timedelta(days=1),
    anchor_time="02:00:00"  # UTC time
)

@flow(
    name="demand-forecast-pipeline",
    description="End-to-end retraining pipeline for demand forecasting",
    schedule=schedule,
    log_prints=True
)
def demand_forecast_pipeline():
    """Main Prefect flow for demand forecasting."""
    logger.info("=" * 60)
    logger.info("🚀 STARTING DEMAND FORECAST PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load data
        raw_df = load_full_data()
        logger.info(f"Data loaded: {len(raw_df):,} rows")
        
        # Step 2: Build features
        featured_df = build_features(raw_df)
        logger.info(f"Features built: {featured_df.shape}")
        
        # Step 3: Train model
        results = train_model(featured_df)
        logger.info(f"Training results: {results['metrics_df']}")
        
        # Step 4: Log to MLflow
        log_to_mlflow(results)
        
        # Step 5: Deploy model
        deploy_model()
        
        # Step 6: Cleanup
        cleanup()
        
        logger.info("✅ Pipeline completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the flow (for testing)
    demand_forecast_pipeline()
