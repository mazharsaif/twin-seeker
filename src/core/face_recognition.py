"""
Core face recognition engine using InsightFace.
Handles face detection, alignment, and embedding extraction.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import cv2
import insightface
import numpy as np
from insightface.utils.face_align import norm_crop
from sklearn.metrics.pairwise import cosine_similarity

from ..config.settings import settings
from .exceptions import (
    ModelLoadError, NoFacesDetectedError, MultipleFacesDetectedError,
    InvalidImageError, FaceRecognitionError
)
from .image_processor import image_processor

logger = logging.getLogger(__name__)


class FaceRecognitionEngine:
    """Main face recognition engine using InsightFace models."""
    
    def __init__(self):
        """Initialize the face recognition engine."""
        self.detector = None
        self.recognizer = None
        self._models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load face detection and recognition models."""
        try:
            logger.info("Loading face recognition models...")
            
            # Get model paths
            model_dir = Path(settings.model_path)
            detector_path = model_dir / settings.detector_model
            recognizer_path = model_dir / settings.recognizer_model
            
            # Validate model files exist
            if not detector_path.exists():
                logger.warning(f"Detector model not found: {detector_path}")
                logger.warning("Models not loaded. Please download models to models/buffalo_l/")
                logger.warning("You can use: python scripts/setup_models.py")
                return
            
            if not recognizer_path.exists():
                logger.warning(f"Recognizer model not found: {recognizer_path}")
                logger.warning("Models not loaded. Please download models to models/buffalo_l/")
                logger.warning("You can use: python scripts/setup_models.py")
                return
            
            # Load detector
            logger.info(f"Loading detector model: {detector_path}")
            self.detector = insightface.model_zoo.get_model(str(detector_path))
            self.detector.prepare(
                ctx_id=0,
                input_size=(640, 640),
                det_thresh=0.5
            )
            
            # Load recognizer
            logger.info(f"Loading recognizer model: {recognizer_path}")
            self.recognizer = insightface.model_zoo.get_model(str(recognizer_path))
            
            self._models_loaded = True
            logger.info("Face recognition models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load face recognition models: {str(e)}")
            logger.warning("Models not loaded. Please download models to models/buffalo_l/")
            logger.warning("You can use: python scripts/setup_models.py")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected faces with bounding boxes, landmarks, and confidence scores
        """
        if not self._models_loaded:
            raise ModelLoadError("Models not loaded. Please download models to models/buffalo_l/")
        
        try:
            # Preprocess image
            image = image_processor.preprocess_for_face_recognition(image)
            
            # Detect faces
            bboxes, kpss = self.detector.detect(
                image,
                max_num=settings.max_faces_per_image,
                metric='default'
            )
            
            faces = []
            for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
                det_score = bbox[-1]
                bbox = bbox[:-1]
                
                # Filter by minimum face size
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                if face_width < settings.min_face_size or face_height < settings.min_face_size:
                    logger.warning(f"Face {i} too small, skipping")
                    continue
                
                faces.append({
                    "bbox": bbox.astype(np.int32),
                    "confidence": float(det_score),
                    "landmarks": kps.astype(np.int32) if kps is not None else None,
                    "face_idx": i
                })
            
            # Sort faces by area (largest first)
            faces = sorted(faces, key=lambda x: self._get_face_area(x["bbox"]), reverse=True)
            
            logger.info(f"Detected {len(faces)} faces in image")
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            raise FaceRecognitionError(f"Face detection failed: {str(e)}")
    
    def extract_embeddings(self, image: np.ndarray, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract face embeddings for detected faces.
        
        Args:
            image: Input image
            faces: List of detected faces
            
        Returns:
            List of faces with embeddings added
        """
        if not self._models_loaded:
            raise ModelLoadError("Models not loaded. Please download models to models/buffalo_l/")
        
        try:
            for face in faces:
                landmarks = face["landmarks"]
                if landmarks is None:
                    logger.warning("No landmarks available for face alignment")
                    continue
                
                # Align and crop face
                face_aligned = norm_crop(image, landmarks, 112, mode="arcface")
                
                # Extract embedding
                embedding = self.recognizer.get_feat(face_aligned).flatten()
                
                # Add to face data
                face["embedding"] = embedding
                face["face_aligned"] = face_aligned
            
            logger.info(f"Extracted embeddings for {len(faces)} faces")
            return faces
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {str(e)}")
            raise FaceRecognitionError(f"Embedding extraction failed: {str(e)}")
    
    def process_image(self, image: Union[str, np.ndarray, bytes]) -> List[Dict[str, Any]]:
        """
        Process an image to detect faces and extract embeddings.
        
        Args:
            image: Image source (URL, numpy array, or bytes)
            
        Returns:
            List of detected faces with embeddings
        """
        try:
            # Load and validate image
            image_array = image_processor.load_image(image)
            
            # Validate image quality
            is_valid, message = image_processor.validate_image_quality(image_array)
            if not is_valid:
                logger.warning(f"Image quality warning: {message}")
            
            # Detect faces
            faces = self.detect_faces(image_array)
            
            if not faces:
                raise NoFacesDetectedError("No faces detected in the image")
            
            # Extract embeddings
            faces_with_embeddings = self.extract_embeddings(image_array, faces)
            
            return faces_with_embeddings
            
        except Exception as e:
            if isinstance(e, (NoFacesDetectedError, ModelLoadError)):
                raise
            logger.error(f"Error processing image: {str(e)}")
            raise FaceRecognitionError(f"Image processing failed: {str(e)}")
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            ).item()
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            raise FaceRecognitionError(f"Face comparison failed: {str(e)}")
    
    def compare_images(self, image1: Union[str, np.ndarray, bytes], 
                      image2: Union[str, np.ndarray, bytes],
                      threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Compare faces in two images.
        
        Args:
            image1: First image
            image2: Second image
            threshold: Custom similarity threshold (uses default if None)
            
        Returns:
            Comparison result with similarity score and match status
        """
        try:
            # Use default threshold if not provided
            if threshold is None:
                threshold = settings.face_match_threshold
            
            # Process both images
            faces1 = self.process_image(image1)
            faces2 = self.process_image(image2)
            
            # Check for single faces
            if len(faces1) > 1:
                logger.warning(f"Multiple faces ({len(faces1)}) detected in first image, using largest")
            if len(faces2) > 1:
                logger.warning(f"Multiple faces ({len(faces2)}) detected in second image, using largest")
            
            # Get primary faces (largest by area)
            face1 = faces1[0]
            face2 = faces2[0]
            
            # Compare embeddings
            similarity_score = self.compare_faces(face1["embedding"], face2["embedding"])
            
            # Determine match status
            is_match = similarity_score >= threshold
            
            # Determine confidence level
            if similarity_score >= 0.8:
                confidence = "high"
            elif similarity_score >= 0.6:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "similarity_score": similarity_score,
                "is_match": is_match,
                "confidence": confidence,
                "threshold": threshold,
                "face1_confidence": face1["confidence"],
                "face2_confidence": face2["confidence"]
            }
            
        except NoFacesDetectedError as e:
            logger.error(f"No faces detected: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error comparing images: {str(e)}")
            raise FaceRecognitionError(f"Image comparison failed: {str(e)}")
    
    def get_primary_face_embedding(self, image: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """
        Get embedding for the primary (largest) face in an image.
        
        Args:
            image: Image source
            
        Returns:
            Face embedding vector
            
        Raises:
            NoFacesDetectedError: If no faces are detected
            MultipleFacesDetectedError: If multiple faces are detected
        """
        try:
            faces = self.process_image(image)
            
            if len(faces) == 0:
                raise NoFacesDetectedError("No faces detected in the image")
            elif len(faces) > 1:
                raise MultipleFacesDetectedError(f"Multiple faces ({len(faces)}) detected, expected single face")
            
            return faces[0]["embedding"]
            
        except (NoFacesDetectedError, MultipleFacesDetectedError):
            raise
        except Exception as e:
            logger.error(f"Error getting primary face embedding: {str(e)}")
            raise FaceRecognitionError(f"Failed to get face embedding: {str(e)}")
    
    def _get_face_area(self, bbox: np.ndarray) -> float:
        """Calculate face area from bounding box."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def models_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_loaded


# Global instance
face_recognition_engine = FaceRecognitionEngine() 