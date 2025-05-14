import torch
from PIL import Image
from fastapi import HTTPException
from ai_model.models.model import get_model
from ai_api.services.preprocessing import preprocess_image

# DR stages mapping
DR_STAGES = {
    0: "No DR",
    1: "Mild Non-proliferative DR",
    2: "Moderate Non-proliferative DR",
    3: "Severe Non-proliferative DR",
    4: "Proliferative DR"
}

def get_model_instance():
    """Get model instance for prediction."""
    model = get_model()
    model.eval()
    return model

def predict(image: Image.Image) -> dict:
    """
    Make prediction for a retinal image.
    
    Args:
        image: PIL Image to predict
        
    Returns:
        Dictionary containing prediction results
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Get model instance
        model = get_model_instance()
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_stage = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_stage].item()
        
        # Format probabilities
        prob_dict = {
            DR_STAGES[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "stage": predicted_stage,
            "stage_name": DR_STAGES[predicted_stage],
            "confidence": confidence,
            "probabilities": prob_dict
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        ) 