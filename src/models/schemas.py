"""
Pydantic schemas for request/response validation and API documentation.
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl
from enum import Enum


class FaceMatchResult(BaseModel):
    """Schema for face match comparison result."""
    
    success: bool = Field(description="Whether the operation was successful")
    similarity_score: float = Field(description="Cosine similarity score between faces")
    is_match: bool = Field(description="Whether faces are considered a match")
    confidence: str = Field(description="Confidence level: high, medium, low")
    message: str = Field(description="Human-readable message about the result")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "similarity_score": 0.85,
                "is_match": True,
                "confidence": "high",
                "message": "Faces are very similar"
            }
        }


class FaceDetectionResult(BaseModel):
    """Schema for face detection result."""
    
    success: bool = Field(description="Whether the operation was successful")
    faces_detected: int = Field(description="Number of faces detected")
    faces: List[Dict[str, Any]] = Field(description="List of detected faces with details")
    message: str = Field(description="Human-readable message about the result")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "faces_detected": 2,
                "faces": [
                    {
                        "bbox": [100, 100, 200, 200],
                        "confidence": 0.95,
                        "landmarks": [[120, 130], [180, 130], [150, 180]]
                    }
                ],
                "message": "Successfully detected 2 faces"
            }
        }


class FaceEmbeddingResult(BaseModel):
    """Schema for face embedding extraction result."""
    
    success: bool = Field(description="Whether the operation was successful")
    embedding: Optional[List[float]] = Field(description="Face embedding vector")
    face_count: int = Field(description="Number of faces detected")
    message: str = Field(description="Human-readable message about the result")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "embedding": [0.1, 0.2, 0.3, ...],
                "face_count": 1,
                "message": "Successfully extracted face embedding"
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code for client handling")
    details: Optional[Dict[str, Any]] = Field(description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "No faces detected in the image",
                "error_code": "NO_FACES_DETECTED",
                "details": {"image_size": "1920x1080"}
            }
        }


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(description="Service status")
    timestamp: str = Field(description="Current timestamp")
    version: str = Field(description="API version")
    models_loaded: bool = Field(description="Whether face recognition models are loaded")


class ImageSourceType(str, Enum):
    """Enum for image source types."""
    URL = "url"
    FILE = "file"
    BASE64 = "base64"


class FaceComparisonRequest(BaseModel):
    """Schema for face comparison request."""
    
    image1_source: ImageSourceType = Field(description="Type of first image source")
    image1_data: str = Field(description="First image data (URL, base64, or file path)")
    image2_source: ImageSourceType = Field(description="Type of second image source")
    image2_data: str = Field(description="Second image data (URL, base64, or file path)")
    threshold: Optional[float] = Field(default=0.6, description="Custom similarity threshold")
    
    @validator("threshold")
    def validate_threshold(cls, v):
        """Validate threshold is between 0 and 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Threshold must be between 0 and 1")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "image1_source": "url",
                "image1_data": "https://example.com/image1.jpg",
                "image2_source": "url",
                "image2_data": "https://example.com/image2.jpg",
                "threshold": 0.7
            }
        }


class FaceDetectionRequest(BaseModel):
    """Schema for face detection request."""
    
    image_source: ImageSourceType = Field(description="Type of image source")
    image_data: str = Field(description="Image data (URL, base64, or file path)")
    return_landmarks: bool = Field(default=True, description="Whether to return facial landmarks")
    return_embeddings: bool = Field(default=False, description="Whether to return face embeddings")
    
    class Config:
        schema_extra = {
            "example": {
                "image_source": "url",
                "image_data": "https://example.com/image.jpg",
                "return_landmarks": True,
                "return_embeddings": False
            }
        }


class BatchFaceComparisonRequest(BaseModel):
    """Schema for batch face comparison request."""
    
    reference_image: str = Field(description="Reference image URL or base64")
    comparison_images: List[str] = Field(description="List of comparison image URLs or base64")
    threshold: Optional[float] = Field(default=0.6, description="Custom similarity threshold")
    
    @validator("comparison_images")
    def validate_comparison_images(cls, v):
        """Validate at least one comparison image is provided."""
        if not v:
            raise ValueError("At least one comparison image is required")
        if len(v) > 10:
            raise ValueError("Maximum 10 comparison images allowed")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "reference_image": "https://example.com/reference.jpg",
                "comparison_images": [
                    "https://example.com/compare1.jpg",
                    "https://example.com/compare2.jpg"
                ],
                "threshold": 0.7
            }
        }


class BatchFaceComparisonResult(BaseModel):
    """Schema for batch face comparison result."""
    
    success: bool = Field(description="Whether the operation was successful")
    reference_face_count: int = Field(description="Number of faces in reference image")
    results: List[Dict[str, Any]] = Field(description="Comparison results for each image")
    message: str = Field(description="Human-readable message about the result")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "reference_face_count": 1,
                "results": [
                    {
                        "image_index": 0,
                        "similarity_score": 0.85,
                        "is_match": True,
                        "confidence": "high"
                    }
                ],
                "message": "Successfully compared 2 images"
            }
        } 