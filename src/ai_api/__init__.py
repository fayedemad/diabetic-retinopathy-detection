from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.handlers import router
from .main import app

__all__ = ['app']

app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="API for detecting diabetic retinopathy stages from retinal images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routes
app.include_router(router) 