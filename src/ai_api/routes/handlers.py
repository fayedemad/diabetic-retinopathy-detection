from fastapi import APIRouter, UploadFile, File, HTTPException
from ai_api.routes.models import PredictionResponse, APIInfo
from ai_api.validators.models import ImageUploadValidator
from ai_api.services.prediction import predict, get_checkpoint
from PIL import Image
import io
from pathlib import Path

router = APIRouter()

async def validate_upload(file: UploadFile) -> Image.Image:
    """
    Validate and process uploaded file.
    
    Args:
        file: Uploaded file
        
    Returns:
        PIL Image object
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        content = await file.read()
        validator = ImageUploadValidator(
            content=content,
            filename=file.filename,
            content_type=file.content_type
        )
        image = Image.open(io.BytesIO(content))
        
        return image
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Predict DR stage from retinal image.
    
    Args:
        file: Uploaded retinal image file
        
    Returns:
        PredictionResponse containing stage, confidence, and probabilities
    """
    try:
        image = await validate_upload(file)
        result = predict(image)
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/accuracy")
async def _get_model_accuracy():
    return {"model_accuracy": get_checkpoint(Path(__file__).parent.parent.parent / "models" / "best_model.pth")["val_acc"]}

@router.get("/", response_model=APIInfo)
async def get_info():
    """Get API information."""
    return APIInfo(
        name="Diabetic Retinopathy Detection API",
        version="1.0.0",
        endpoints={
            "/predict": "POST - Upload a retinal image to get DR stage prediction",
            "/": "GET - API information"
        }
    ) 