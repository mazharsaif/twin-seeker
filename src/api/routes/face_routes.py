"""
Face recognition API routes.
Handles HTTP endpoints for face detection, comparison, and embedding extraction.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse

from ...services.face_service import face_service
from ...models.schemas import (
    FaceMatchResult, FaceDetectionResult, FaceEmbeddingResult,
    BatchFaceComparisonResult, FaceComparisonRequest, FaceDetectionRequest,
    BatchFaceComparisonRequest, HealthCheckResponse, ErrorResponse
)
from ...core.exceptions import (
    NoFacesDetectedError, MultipleFacesDetectedError, FaceRecognitionError,
    ImageLoadError, ValidationError
)
from ...config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/face", tags=["face-recognition"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    import datetime
    
    health_status = face_service.get_health_status()
    
    return HealthCheckResponse(
        status="healthy" if health_status["models_loaded"] else "unhealthy",
        timestamp=datetime.datetime.now().isoformat(),
        version="1.0.0",
        models_loaded=health_status["models_loaded"]
    )


@router.post("/compare", response_model=FaceMatchResult)
async def compare_faces(request: FaceComparisonRequest):
    """
    Compare faces in two images.
    
    Supports various image sources (URL, base64, file paths).
    """
    try:
        result = face_service.compare_faces(
            image1=request.image1_data,
            image2=request.image2_data,
            threshold=request.threshold
        )
        
        if not result.success:
            return JSONResponse(
                status_code=400,
                content=result.dict()
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in compare_faces endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-upload", response_model=FaceMatchResult)
async def compare_faces_upload(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    """
    Compare faces in two uploaded images.
    """
    try:
        # Validate file types
        if not image1.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="First file must be an image")
        if not image2.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Second file must be an image")
        
        # Read file contents
        image1_content = await image1.read()
        image2_content = await image2.read()
        
        result = face_service.compare_faces(
            image1=image1_content,
            image2=image2_content,
            threshold=threshold
        )
        
        if not result.success:
            return JSONResponse(
                status_code=400,
                content=result.dict()
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in compare_faces_upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect", response_model=FaceDetectionResult)
async def detect_faces(request: FaceDetectionRequest):
    """
    Detect faces in an image.
    
    Supports various image sources (URL, base64, file paths).
    """
    try:
        result = face_service.detect_faces(
            image=request.image_data,
            return_landmarks=request.return_landmarks,
            return_embeddings=request.return_embeddings
        )
        
        if not result.success:
            return JSONResponse(
                status_code=400,
                content=result.dict()
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in detect_faces endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-upload", response_model=FaceDetectionResult)
async def detect_faces_upload(
    image: UploadFile = File(...),
    return_landmarks: bool = Form(True),
    return_embeddings: bool = Form(False)
):
    """
    Detect faces in an uploaded image.
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        image_content = await image.read()
        
        result = face_service.detect_faces(
            image=image_content,
            return_landmarks=return_landmarks,
            return_embeddings=return_embeddings
        )
        
        if not result.success:
            return JSONResponse(
                status_code=400,
                content=result.dict()
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_faces_upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embedding", response_model=FaceEmbeddingResult)
async def extract_face_embedding(image: UploadFile = File(...)):
    """
    Extract face embedding from the primary face in an uploaded image.
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        image_content = await image.read()
        
        result = face_service.extract_embedding(image=image_content)
        
        if not result.success:
            return JSONResponse(
                status_code=400,
                content=result.dict()
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_face_embedding endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-compare", response_model=BatchFaceComparisonResult)
async def batch_compare_faces(request: BatchFaceComparisonRequest):
    """
    Compare a reference image against multiple comparison images.
    """
    try:
        result = face_service.batch_compare_faces(
            reference_image=request.reference_image,
            comparison_images=request.comparison_images,
            threshold=request.threshold
        )
        
        if not result.success:
            return JSONResponse(
                status_code=400,
                content=result.dict()
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in batch_compare_faces endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-url", response_model=FaceMatchResult)
async def compare_faces_url(
    url1: str = Form(...),
    url2: str = Form(...),
    threshold: float = Form(0.6)
):
    """
    Compare faces in two images from URLs.
    """
    try:
        result = face_service.compare_faces(
            image1=url1,
            image2=url2,
            threshold=threshold
        )
        
        if not result.success:
            return JSONResponse(
                status_code=400,
                content=result.dict()
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in compare_faces_url endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-url", response_model=FaceDetectionResult)
async def detect_faces_url(
    url: str = Form(...),
    return_landmarks: bool = Form(True),
    return_embeddings: bool = Form(False)
):
    """
    Detect faces in an image from URL.
    """
    try:
        result = face_service.detect_faces(
            image=url,
            return_landmarks=return_landmarks,
            return_embeddings=return_embeddings
        )
        
        if not result.success:
            return JSONResponse(
                status_code=400,
                content=result.dict()
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in detect_faces_url endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Legacy endpoints for backward compatibility
@router.post("/compare-legacy")
async def compare_faces_legacy(
    password: str = Form(...),
    url1: str = Form(...),
    url2: str = Form(...)
):
    """
    Legacy endpoint for face comparison (maintains backward compatibility).
    """
    try:
        # Simple password validation (in production, use proper authentication)
        if password != "your-secure-password":  # Replace with proper auth
            raise HTTPException(status_code=401, detail="Invalid password")
        
        result = face_service.compare_faces(
            image1=url1,
            image2=url2
        )
        
        return result.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in compare_faces_legacy endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 