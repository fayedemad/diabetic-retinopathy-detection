from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    stage: int = Field(..., description="Predicted DR stage (0-4)")
    stage_name: str = Field(..., description="Name of the predicted DR stage")
    confidence: float = Field(..., description="Confidence score for the prediction", ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(..., description="Probability scores for all stages")

    class Config:
        schema_extra = {
            "example": {
                "stage": 1,
                "stage_name": "Mild Non-proliferative DR",
                "confidence": 0.95,
                "probabilities": {
                    "No DR": 0.02,
                    "Mild Non-proliferative DR": 0.95,
                    "Moderate Non-proliferative DR": 0.02,
                    "Severe Non-proliferative DR": 0.005,
                    "Proliferative DR": 0.005
                }
            }
        }

class ErrorResponse(BaseModel):
    """Response model for error cases."""
    error: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")

    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid image format",
                "details": {"supported_formats": ["JPEG", "PNG"]}
            }
        }

class APIInfo(BaseModel):
    """Response model for API information."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints and their descriptions")

    class Config:
        schema_extra = {
            "example": {
                "name": "Diabetic Retinopathy Detection API",
                "version": "1.0.0",
                "endpoints": {
                    "/predict": "POST - Upload a retinal image to get DR stage prediction",
                    "/": "GET - API information"
                }
            }
        } 