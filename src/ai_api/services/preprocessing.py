from PIL import Image
import torch
import numpy as np
from fastapi import HTTPException
from ai_model.utils.transforms import get_transforms

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image to preprocess
        
    Returns:
        Preprocessed image tensor on CUDA device
        
    Raises:
        HTTPException: If preprocessing fails
    """
    try:
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Apply transforms with validation phase
        transform = get_transforms(phase='val')
        image_tensor = transform(image=image_np)["image"]
        
        # Move tensor to CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_tensor = image_tensor.to(device)
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error preprocessing image: {str(e)}"
        )

def validate_image_dimensions(image: Image.Image) -> None:
    """
    Validate image dimensions.
    
    Args:
        image: PIL Image to validate
        
    Raises:
        HTTPException: If dimensions are invalid
    """
    if image.size[0] < 100 or image.size[1] < 100:
        raise HTTPException(
            status_code=400,
            detail="Image dimensions too small. Minimum size is 100x100 pixels"
        ) 