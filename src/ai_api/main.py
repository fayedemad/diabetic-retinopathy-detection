from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_api.routes.handlers import router

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title="Diabetic Retinopathy Detection API",
        description="API for detecting diabetic retinopathy stages from retinal images",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"], 
    )

    # Include routes
    app.include_router(router)

    return app

app = create_app() 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)