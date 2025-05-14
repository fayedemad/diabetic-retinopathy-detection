# Diabetic Retinopathy Detection API

A FastAPI-based API for detecting diabetic retinopathy stages from retinal images using deep learning.

## Project Structure

```
.
├── src/
│   ├── ai_api/                    # API application code
│   │   ├── routes/               # API route handlers
│   │   ├── services/             # Business logic services
│   │   ├── validators/           # Input validation
│   │   ├── __init__.py          # Package initialization
│   │   └── main.py              # FastAPI application setup
│   ├── ai_model/                 # Model-related code
│   │   ├── models/              # Model architecture definitions
│   │   ├── utils/               # Utility functions
│   │   ├── data/                # Data handling utilities
│   │   ├── config.py            # Model configuration
│   │   ├── main.py              # Model training entry point
│   │   └── trainer.py           # Training utilities
│   ├── data/                    # Data directory
│   ├── logs/                    # Log files
│   └── models/                  # Model checkpoints
├── data/                        # Project data directory
├── .gitignore                  # Git ignore file
├── requirements.txt            # Python dependencies
└── README.md                  # Project documentation
```

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 4GB+ RAM
- 2GB+ free disk space

### Python Dependencies
```
fastapi>=0.68.0
uvicorn>=0.15.0
python-multipart>=0.0.5
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.3.1
numpy>=1.21.0
albumentations>=1.0.0
pydantic>=1.8.0
```

## Model Requirements

The model expects retinal images with the following specifications:
- Input size: 380x380 pixels (will be resized if different)
- Color format: RGB
- File formats: JPEG, PNG
- Minimum dimensions: 100x100 pixels

The model classifies images into 5 stages of diabetic retinopathy:
1. No DR
2. Mild Non-proliferative DR
3. Moderate Non-proliferative DR
4. Severe Non-proliferative DR
5. Proliferative DR

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the model:
   - The model file should be placed in the `models` directory
   - The model file should be named `best_model.pth`
   - Contact the maintainers for access to the model file

5. Run the API:
```bash
python -m uvicorn src.ai_api.main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Endpoints

1. `GET /`
   - Returns API information
   - No authentication required

2. `POST /predict`
   - Accepts retinal image upload
   - Returns prediction results
   - Content-Type: multipart/form-data
   - Parameter: `file` (image file)

### Example Request

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("retinal_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Example Response

```json
{
    "stage": 2,
    "stage_name": "Moderate Non-proliferative DR",
    "confidence": 0.95,
    "probabilities": {
        "No DR": 0.01,
        "Mild Non-proliferative DR": 0.02,
        "Moderate Non-proliferative DR": 0.95,
        "Severe Non-proliferative DR": 0.01,
        "Proliferative DR": 0.01
    }
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request (invalid input)
- 500: Internal Server Error (processing/prediction error)

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
The project follows PEP 8 style guidelines. Use `black` for code formatting:
```bash
black src/
```

## License

[Specify your license here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

[Add contact information] 