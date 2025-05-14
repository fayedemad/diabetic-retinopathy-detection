from pydantic import BaseModel, Field, validator
from PIL import Image
import io

class ImageUploadValidator(BaseModel):
    """Validator for image upload requests."""
    content: bytes
    filename: str
    content_type: str
    
    @validator('content_type')
    def validate_content_type(cls, v):
        if v not in ['image/jpeg', 'image/png']:
            raise ValueError('Only JPEG and PNG images are supported')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if len(v) > 10 * 1024 * 1024:  # 10MB
            raise ValueError('File size must be less than 10MB')
        
        try:
            image = Image.open(io.BytesIO(v))
            image.verify()
            image = Image.open(io.BytesIO(v))
            
            if image.size[0] < 100 or image.size[1] < 100:
                raise ValueError('Image dimensions must be at least 100x100 pixels')
                
        except Exception as e:
            raise ValueError(f'Invalid image file: {str(e)}')
            
        return v 