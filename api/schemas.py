"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime, date

class ForecastRequest(BaseModel):
    """Request body for single forecast."""
    store_id: str = Field(..., description="Store ID (e.g., 'CA_1')")
    item_id: str = Field(..., description="Item ID (e.g., 'HOBBIES_1_001')")
    date: str = Field(..., description="Date to forecast (YYYY-MM-DD)")
    
    @validator('date')
    def validate_date(cls, v):
        """Validate date format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

class ForecastBatchRequest(BaseModel):
    """Request body for batch forecasts."""
    requests: List[ForecastRequest] = Field(..., min_items=1, max_items=100)

class ForecastResponse(BaseModel):
    """Response for a single forecast."""
    store_id: str
    item_id: str
    date: str
    forecast: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    model_version: str = "1.0.0"

class ForecastBatchResponse(BaseModel):
    """Response for batch forecasts."""
    predictions: List[ForecastResponse]
    total_time_ms: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    data_loaded: bool
    uptime_seconds: float
    version: str = "1.0.0"
