"""
Face recognition service layer.
Handles business logic for face detection, comparison, and embedding extraction.
"""

import logging
from typing import List, Dict, Any, Optional, Union

from ..core.face_recognition import face_recognition_engine
from ..core.exceptions import (
    NoFacesDetectedError, MultipleFacesDetectedError,
    FaceRecognitionError, ImageLoadError
)
from ..models.schemas import (
    FaceMatchResult, FaceDetectionResult, FaceEmbeddingResult,
    BatchFaceComparisonResult
)
from ..config.settings import settings

logger = logging.getLogger(__name__)


class FaceService:
    """Service layer for face recognition operations."""
    
    def __init__(self):
        """Initialize the face service."""
        self.engine = face_recognition_engine
    
    def compare_faces(self, image1: Union[str, bytes], image2: Union[str, bytes], 
                     threshold: Optional[float] = None) -> FaceMatchResult:
        """
        Compare faces in two images.
        
        Args:
            image1: First image (URL, base64, or file path)
            image2: Second image (URL, base64, or file path)
            threshold: Custom similarity threshold
            
        Returns:
            FaceMatchResult with comparison details
        """
        try:
            # Use default threshold if not provided
            if threshold is None:
                threshold = settings.face_match_threshold
            
            # Compare images using the engine
            result = self.engine.compare_images(image1, image2, threshold)
            
            # Create response
            return FaceMatchResult(
                success=True,
                similarity_score=result["similarity_score"],
                is_match=result["is_match"],
                confidence=result["confidence"],
                message=self._get_comparison_message(result)
            )
            
        except NoFacesDetectedError as e:
            logger.warning(f"No faces detected during comparison: {str(e)}")
            return FaceMatchResult(
                success=False,
                similarity_score=0.0,
                is_match=False,
                confidence="none",
                message=str(e)
            )
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return FaceMatchResult(
                success=False,
                similarity_score=0.0,
                is_match=False,
                confidence="none",
                message=f"Comparison failed: {str(e)}"
            )
    
    def detect_faces(self, image: Union[str, bytes], 
                    return_landmarks: bool = True,
                    return_embeddings: bool = False) -> FaceDetectionResult:
        """
        Detect faces in an image.
        
        Args:
            image: Image source
            return_landmarks: Whether to return facial landmarks
            return_embeddings: Whether to return face embeddings
            
        Returns:
            FaceDetectionResult with detection details
        """
        try:
            # Process image
            faces = self.engine.process_image(image)
            
            # Format face data
            formatted_faces = []
            for face in faces:
                face_data = {
                    "bbox": face["bbox"].tolist(),
                    "confidence": face["confidence"]
                }
                
                if return_landmarks and face["landmarks"] is not None:
                    face_data["landmarks"] = face["landmarks"].tolist()
                
                if return_embeddings and "embedding" in face:
                    face_data["embedding"] = face["embedding"].tolist()
                
                formatted_faces.append(face_data)
            
            return FaceDetectionResult(
                success=True,
                faces_detected=len(faces),
                faces=formatted_faces,
                message=f"Successfully detected {len(faces)} face(s)"
            )
            
        except NoFacesDetectedError:
            return FaceDetectionResult(
                success=True,
                faces_detected=0,
                faces=[],
                message="No faces detected in the image"
            )
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return FaceDetectionResult(
                success=False,
                faces_detected=0,
                faces=[],
                message=f"Face detection failed: {str(e)}"
            )
    
    def extract_embedding(self, image: Union[str, bytes]) -> FaceEmbeddingResult:
        """
        Extract face embedding from the primary face in an image.
        
        Args:
            image: Image source
            
        Returns:
            FaceEmbeddingResult with embedding details
        """
        try:
            # Get primary face embedding
            embedding = self.engine.get_primary_face_embedding(image)
            
            return FaceEmbeddingResult(
                success=True,
                embedding=embedding.tolist(),
                face_count=1,
                message="Successfully extracted face embedding"
            )
            
        except NoFacesDetectedError:
            return FaceEmbeddingResult(
                success=False,
                embedding=None,
                face_count=0,
                message="No faces detected in the image"
            )
        except MultipleFacesDetectedError as e:
            return FaceEmbeddingResult(
                success=False,
                embedding=None,
                face_count=0,
                message=str(e)
            )
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            return FaceEmbeddingResult(
                success=False,
                embedding=None,
                face_count=0,
                message=f"Embedding extraction failed: {str(e)}"
            )
    
    def batch_compare_faces(self, reference_image: str, 
                           comparison_images: List[str],
                           threshold: Optional[float] = None) -> BatchFaceComparisonResult:
        """
        Compare a reference image against multiple comparison images.
        
        Args:
            reference_image: Reference image URL or base64
            comparison_images: List of comparison image URLs or base64
            threshold: Custom similarity threshold
            
        Returns:
            BatchFaceComparisonResult with comparison results
        """
        try:
            # Use default threshold if not provided
            if threshold is None:
                threshold = settings.face_match_threshold
            
            # Process reference image
            reference_faces = self.engine.process_image(reference_image)
            reference_face_count = len(reference_faces)
            
            if reference_face_count == 0:
                return BatchFaceComparisonResult(
                    success=False,
                    reference_face_count=0,
                    results=[],
                    message="No faces detected in reference image"
                )
            
            # Get reference embedding (use largest face)
            reference_embedding = reference_faces[0]["embedding"]
            
            # Compare with each comparison image
            results = []
            for i, comparison_image in enumerate(comparison_images):
                try:
                    # Process comparison image
                    comparison_faces = self.engine.process_image(comparison_image)
                    
                    if len(comparison_faces) == 0:
                        results.append({
                            "image_index": i,
                            "similarity_score": 0.0,
                            "is_match": False,
                            "confidence": "none",
                            "error": "No faces detected"
                        })
                        continue
                    
                    # Compare embeddings
                    comparison_embedding = comparison_faces[0]["embedding"]
                    similarity_score = self.engine.compare_faces(
                        reference_embedding, comparison_embedding
                    )
                    
                    # Determine match status and confidence
                    is_match = similarity_score >= threshold
                    
                    if similarity_score >= 0.8:
                        confidence = "high"
                    elif similarity_score >= 0.6:
                        confidence = "medium"
                    else:
                        confidence = "low"
                    
                    results.append({
                        "image_index": i,
                        "similarity_score": similarity_score,
                        "is_match": is_match,
                        "confidence": confidence
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing comparison image {i}: {str(e)}")
                    results.append({
                        "image_index": i,
                        "similarity_score": 0.0,
                        "is_match": False,
                        "confidence": "none",
                        "error": str(e)
                    })
            
            return BatchFaceComparisonResult(
                success=True,
                reference_face_count=reference_face_count,
                results=results,
                message=f"Successfully compared {len(comparison_images)} images"
            )
            
        except Exception as e:
            logger.error(f"Error in batch comparison: {str(e)}")
            return BatchFaceComparisonResult(
                success=False,
                reference_face_count=0,
                results=[],
                message=f"Batch comparison failed: {str(e)}"
            )
    
    def _get_comparison_message(self, result: Dict[str, Any]) -> str:
        """Generate human-readable message for comparison result."""
        similarity = result["similarity_score"]
        is_match = result["is_match"]
        confidence = result["confidence"]
        
        if is_match:
            if confidence == "high":
                return f"Faces are very similar (similarity: {similarity:.3f})"
            elif confidence == "medium":
                return f"Faces are similar (similarity: {similarity:.3f})"
            else:
                return f"Faces are weakly similar (similarity: {similarity:.3f})"
        else:
            if similarity < 0.3:
                return f"Faces are very different (similarity: {similarity:.3f})"
            elif similarity < 0.5:
                return f"Faces are different (similarity: {similarity:.3f})"
            else:
                return f"Faces are somewhat similar but below threshold (similarity: {similarity:.3f})"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the face recognition service."""
        return {
            "models_loaded": self.engine.models_loaded,
            "service_status": "healthy" if self.engine.models_loaded else "unhealthy",
            "face_match_threshold": settings.face_match_threshold,
            "max_faces_per_image": settings.max_faces_per_image
        }


# Global instance
face_service = FaceService() 