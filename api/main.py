"""
Main FastAPI application for demand forecasting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from api.routers import forecast
from api.dependencies import deps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Demand Forecasting API",
    description="Production-grade time-series demand forecasting for M5 dataset",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecast.router)

@app.on_event("startup")
async def startup_event():
    """Load dependencies on startup."""
    logger.info("=" * 60)
    logger.info("🚀 STARTING DEMAND FORECAST API")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    # Pre-load dependencies (ensures models are loaded)
    try:
        model = deps.get_model()
        data = deps.get_historical_data()
        logger.info(f"✅ Model loaded: {model is not None}")
        logger.info(f"✅ Data loaded: {data is not None}")
    except Exception as e:
        logger.error(f"Error loading dependencies: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Demand Forecasting API",
        "docs": "/docs",
        "health": "/forecast/health",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
