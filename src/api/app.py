"""
Main FastAPI application for the face recognition API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

from .routes.face_routes import router as face_router
from .middleware import setup_middleware
from ..config.settings import settings
from ..core.exceptions import FaceRecognitionError, ModelLoadError

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Face Recognition API...")
    
    # Check if models are loaded
    try:
        from ..core.face_recognition import face_recognition_engine
        if not face_recognition_engine.models_loaded:
            logger.error("Face recognition models failed to load")
        else:
            logger.info("Face recognition models loaded successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Face Recognition API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        description="A modern, modular face recognition API built with FastAPI and InsightFace",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Add HTTPS redirect middleware (uncomment for production)
    # app.add_middleware(HTTPSRedirectMiddleware)
    
    # Include routers
    app.include_router(face_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Face Recognition API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/face/health"
        }
    
    # Global exception handler
    @app.exception_handler(FaceRecognitionError)
    async def face_recognition_exception_handler(request, exc):
        """Handle face recognition exceptions."""
        logger.error(f"Face recognition error: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(exc),
                "error_code": "FACE_RECOGNITION_ERROR"
            }
        )
    
    @app.exception_handler(ModelLoadError)
    async def model_load_exception_handler(request, exc):
        """Handle model loading exceptions."""
        logger.error(f"Model load error: {str(exc)}")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": str(exc),
                "error_code": "MODEL_LOAD_ERROR"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Handle general exceptions."""
        logger.error(f"Unexpected error: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR"
            }
        )
    
    return app


# Create the application instance
app = create_app() 