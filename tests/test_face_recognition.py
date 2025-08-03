"""
Unit tests for face recognition functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.core.face_recognition import FaceRecognitionEngine
from src.core.image_processor import ImageProcessor
from src.services.face_service import FaceService
from src.core.exceptions import (
    NoFacesDetectedError, MultipleFacesDetectedError,
    ModelLoadError, ImageLoadError
)


class TestImageProcessor:
    """Test image processing functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ImageProcessor()
    
    def test_validate_numpy_array_valid(self):
        """Test validation of valid numpy array."""
        # Create a valid RGB image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.processor._validate_numpy_array(image)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
    
    def test_validate_numpy_array_grayscale(self):
        """Test conversion of grayscale to RGB."""
        # Create grayscale image
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = self.processor._validate_numpy_array(image)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
    
    def test_validate_numpy_array_too_small(self):
        """Test validation fails for too small image."""
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image too small"):
            self.processor._validate_numpy_array(image)
    
    def test_validate_numpy_array_too_large(self):
        """Test validation fails for too large image."""
        image = np.random.randint(0, 255, (15000, 15000, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image too large"):
            self.processor._validate_numpy_array(image)
    
    def test_validate_numpy_array_wrong_channels(self):
        """Test validation fails for wrong number of channels."""
        image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Image must have 3 color channels"):
            self.processor._validate_numpy_array(image)
    
    def test_is_url_valid(self):
        """Test URL validation for valid URLs."""
        assert self.processor._is_url("https://example.com/image.jpg")
        assert self.processor._is_url("http://localhost:8000/test.png")
    
    def test_is_url_invalid(self):
        """Test URL validation for invalid URLs."""
        assert not self.processor._is_url("not-a-url")
        assert not self.processor._is_url("/path/to/file.jpg")
    
    def test_resize_image_no_resize_needed(self):
        """Test image resize when no resize is needed."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.processor.resize_image(image, max_size=200)
        assert result.shape == (100, 100, 3)
    
    def test_resize_image_with_resize(self):
        """Test image resize when resize is needed."""
        image = np.random.randint(0, 255, (2000, 1000, 3), dtype=np.uint8)
        result = self.processor.resize_image(image, max_size=1000)
        assert result.shape[0] <= 1000 or result.shape[1] <= 1000
        assert result.shape[2] == 3


class TestFaceRecognitionEngine:
    """Test face recognition engine functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = FaceRecognitionEngine()
    
    @patch('src.core.face_recognition.insightface.model_zoo.get_model')
    def test_load_models_success(self, mock_get_model):
        """Test successful model loading."""
        # Mock the model loading
        mock_detector = Mock()
        mock_recognizer = Mock()
        mock_get_model.side_effect = [mock_detector, mock_recognizer]
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            self.engine._load_models()
        
        assert self.engine.models_loaded is True
        assert self.engine.detector is not None
        assert self.engine.recognizer is not None
    
    @patch('pathlib.Path.exists')
    def test_load_models_missing_files(self, mock_exists):
        """Test model loading with missing files."""
        mock_exists.return_value = False
        
        with pytest.raises(ModelLoadError, match="Detector model not found"):
            self.engine._load_models()
    
    def test_get_face_area(self):
        """Test face area calculation."""
        bbox = np.array([10, 20, 50, 60])  # x1, y1, x2, y2
        area = self.engine._get_face_area(bbox)
        expected_area = (50 - 10) * (60 - 20)  # 40 * 40 = 1600
        assert area == expected_area


class TestFaceService:
    """Test face service functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.service = FaceService()
    
    @patch('src.services.face_service.face_recognition_engine')
    def test_compare_faces_success(self, mock_engine):
        """Test successful face comparison."""
        # Mock engine response
        mock_engine.compare_images.return_value = {
            "similarity_score": 0.85,
            "is_match": True,
            "confidence": "high",
            "threshold": 0.6,
            "face1_confidence": 0.95,
            "face2_confidence": 0.92
        }
        
        result = self.service.compare_faces("image1", "image2", threshold=0.6)
        
        assert result.success is True
        assert result.similarity_score == 0.85
        assert result.is_match is True
        assert result.confidence == "high"
    
    @patch('src.services.face_service.face_recognition_engine')
    def test_compare_faces_no_faces(self, mock_engine):
        """Test face comparison with no faces detected."""
        mock_engine.compare_images.side_effect = NoFacesDetectedError("No faces detected")
        
        result = self.service.compare_faces("image1", "image2")
        
        assert result.success is False
        assert result.similarity_score == 0.0
        assert result.is_match is False
        assert result.confidence == "none"
    
    @patch('src.services.face_service.face_recognition_engine')
    def test_detect_faces_success(self, mock_engine):
        """Test successful face detection."""
        # Mock engine response
        mock_faces = [
            {
                "bbox": np.array([10, 20, 50, 60]),
                "confidence": 0.95,
                "landmarks": np.array([[15, 25], [45, 55]]),
                "embedding": np.random.rand(512)
            }
        ]
        mock_engine.process_image.return_value = mock_faces
        
        result = self.service.detect_faces("image", return_landmarks=True, return_embeddings=True)
        
        assert result.success is True
        assert result.faces_detected == 1
        assert len(result.faces) == 1
        assert result.faces[0]["confidence"] == 0.95
    
    @patch('src.services.face_service.face_recognition_engine')
    def test_extract_embedding_success(self, mock_engine):
        """Test successful embedding extraction."""
        # Mock engine response
        mock_embedding = np.random.rand(512)
        mock_engine.get_primary_face_embedding.return_value = mock_embedding
        
        result = self.service.extract_embedding("image")
        
        assert result.success is True
        assert result.embedding == mock_embedding.tolist()
        assert result.face_count == 1
    
    @patch('src.services.face_service.face_recognition_engine')
    def test_extract_embedding_multiple_faces(self, mock_engine):
        """Test embedding extraction with multiple faces."""
        mock_engine.get_primary_face_embedding.side_effect = MultipleFacesDetectedError("Multiple faces")
        
        result = self.service.extract_embedding("image")
        
        assert result.success is False
        assert result.embedding is None
        assert result.face_count == 0
    
    def test_get_comparison_message_high_confidence_match(self):
        """Test message generation for high confidence match."""
        result = {
            "similarity_score": 0.9,
            "is_match": True,
            "confidence": "high"
        }
        message = self.service._get_comparison_message(result)
        assert "very similar" in message
        assert "0.900" in message
    
    def test_get_comparison_message_no_match(self):
        """Test message generation for no match."""
        result = {
            "similarity_score": 0.2,
            "is_match": False,
            "confidence": "low"
        }
        message = self.service._get_comparison_message(result)
        assert "very different" in message
        assert "0.200" in message
    
    def test_get_health_status(self):
        """Test health status retrieval."""
        status = self.service.get_health_status()
        assert "models_loaded" in status
        assert "service_status" in status
        assert "face_match_threshold" in status
        assert "max_faces_per_image" in status


# Integration tests
class TestIntegration:
    """Integration tests for the complete face recognition pipeline."""
    
    @pytest.mark.integration
    def test_complete_pipeline_with_mock_models(self):
        """Test complete pipeline with mocked models."""
        # This would test the complete pipeline with mocked models
        # In a real scenario, you'd use actual model files
        pass
    
    @pytest.mark.integration
    def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline."""
        # This would test error handling in the complete pipeline
        pass 