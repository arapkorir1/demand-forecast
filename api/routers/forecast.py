"""
Forecast endpoints for the demand forecasting API.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime, timedelta
import time

from api.schemas import (
    ForecastRequest, ForecastBatchRequest,
    ForecastResponse, ForecastBatchResponse
)
from api.dependencies import deps

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/forecast", tags=["forecast"])

def build_features_for_prediction(
    store_id: str,
    item_id: str,
    date: str,
    historical_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Build features for a single prediction point.
    
    This recreates the feature engineering pipeline for a single row:
    - Looks up historical sales for this item-store
    - Builds lags (t-1, t-7, t-14, t-28)
    - Builds rolling stats (7d, 14d, 28d mean/std)
    - Adds calendar features for the date
    
    Returns a DataFrame with one row and all required features.
    """
    # Convert date to datetime
    target_date = pd.to_datetime(date)
    day_num = (target_date - pd.Timestamp('2011-01-29')).days  # M5 day 1 = 2011-01-29
    day_num = int(day_num)
    
    # Get historical data for this item-store
    item_hist = historical_data[
        (historical_data['store_id'] == store_id) &
        (historical_data['item_id'] == item_id)
    ].copy()
    
    if item_hist.empty:
        raise ValueError(f"No historical data found for store '{store_id}', item '{item_id}'")
    
    # Sort by day
    item_hist = item_hist.sort_values('day_num')
    
    # Get last 30 days of sales (for lags)
    hist_mask = item_hist['day_num'] < day_num
    hist_sales = item_hist[hist_mask]['sales'].tail(30)
    
    if len(hist_sales) < 28:
        raise ValueError(f"Insufficient historical data: only {len(hist_sales)} days available")
    
    # Extract any existing string attributes safely from the matching item data
    base_row = item_hist.iloc[0]
    
    # Build features
    features = {
        'item_id': item_id,
        'store_id': store_id,
        'weekday': target_date.day_name(),
        'wday': int(target_date.strftime('%w')) + 1,  # M5 maps Sunday=1, Saturday=7
        'day_num': day_num,
        'day_of_week': target_date.dayofweek,
        'month': target_date.month,
        'quarter': target_date.quarter,
        'year': target_date.year,
        'is_weekend': int(target_date.dayofweek >= 5),
        'is_holiday': int(base_row.get('is_holiday', 0)),
        'event_name_1': base_row.get('event_name_1', 'none'),
        'event_type_1': base_row.get('event_type_1', 'none'),
        'event_name_2': base_row.get('event_name_2', 'none'),
        'event_type_2': base_row.get('event_type_2', 'none'),
        'day_sin': np.sin(2 * np.pi * target_date.dayofweek / 7),
        'day_cos': np.cos(2 * np.pi * target_date.dayofweek / 7),
        'month_sin': np.sin(2 * np.pi * target_date.month / 12),
        'month_cos': np.cos(2 * np.pi * target_date.month / 12),
    }
    
    # Lags
    lags = [1, 2, 3, 7, 14, 28]
    for lag in lags:
        if len(hist_sales) >= lag:
            features[f'sales_lag_{lag}'] = hist_sales.iloc[-lag]
        else:
            features[f'sales_lag_{lag}'] = 0
    
    # Rolling stats
    windows = [7, 14, 28]
    for window in windows:
        if len(hist_sales) >= window:
            window_data = hist_sales.iloc[-window:]
            features[f'sales_roll_mean_{window}'] = window_data.mean()
            features[f'sales_roll_std_{window}'] = window_data.std()
            features[f'sales_roll_min_{window}'] = window_data.min()
            features[f'sales_roll_max_{window}'] = window_data.max()
        else:
            features[f'sales_roll_mean_{window}'] = 0
            features[f'sales_roll_std_{window}'] = 0
            features[f'sales_roll_min_{window}'] = 0
            features[f'sales_roll_max_{window}'] = 0
    
    # Price features
    features['sell_price'] = float(base_row.get('sell_price', 1.0))
    features['price_change'] = float(base_row.get('price_change', 0.0))
    features['price_relative_to_item'] = float(base_row.get('price_relative_to_item', 1.0))
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Ensure all feature columns exist
    expected_cols = deps.get_feature_cols()
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Align features exactly to training columns order
    df = df[expected_cols]
    
    # Explicitly cast the known pipeline categorical features
    categorical_cols = [
        'item_id', 'store_id', 'weekday', 
        'event_name_1', 'event_type_1', 
        'event_name_2', 'event_type_2'
    ]
    for cat_col in categorical_cols:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype('category')
                
    return df

@router.post("/single", response_model=ForecastResponse)
async def forecast_single(request: ForecastRequest):
    """
    Get a single demand forecast for a specific item-store-date.
    """
    start_time = time.time()
    
    # Get dependencies
    model = deps.get_model()
    historical_data = deps.get_historical_data()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Historical data not loaded")
    
    try:
        # Build features
        X = build_features_for_prediction(
            request.store_id,
            request.item_id,
            request.date,
            historical_data
        )
        
        # Predict
        forecast = model.predict(X)[0]
        
        response = ForecastResponse(
            store_id=request.store_id,
            item_id=request.item_id,
            date=request.date,
            forecast=float(forecast),
            model_version="1.0.0"
        )
        
        logger.info(f"Forecast: {request.store_id}/{request.item_id} -> {forecast:.2f} on {request.date}")
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/batch", response_model=ForecastBatchResponse)
async def forecast_batch(request: ForecastBatchRequest, background_tasks: BackgroundTasks):
    """
    Get batch forecasts (up to 100 requests).
    """
    start_time = time.time()
    predictions = []
    
    model = deps.get_model()
    historical_data = deps.get_historical_data()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Historical data not loaded")
        
    for req in request.requests:
        try:
            X = build_features_for_prediction(
                req.store_id,
                req.item_id,
                req.date,
                historical_data
            )
            
            forecast = model.predict(X)[0]
            
            predictions.append(ForecastResponse(
                store_id=req.store_id,
                item_id=req.item_id,
                date=req.date,
                forecast=float(forecast),
                model_version="1.0.0"
            ))
            
        except Exception as e:
            logger.error(f"Batch prediction error for {req}: {str(e)}")
            
    elapsed_ms = (time.time() - start_time) * 1000
    
    return ForecastBatchResponse(
        predictions=predictions,
        total_time_ms=elapsed_ms
    )

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": deps.get_model() is not None,
        "data_loaded": deps.get_historical_data() is not None
    }
