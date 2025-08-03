"""
Image processing utilities for face recognition.
Handles various image formats, validation, and preprocessing.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Union, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from requests.exceptions import RequestException

from ..config.settings import settings
from .exceptions import (
    ImageLoadError, InvalidImageError, ImageTooLargeError,
    UnsupportedImageFormatError, NetworkError
)

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image loading, validation, and preprocessing."""
    
    def __init__(self):
        self.max_file_size = settings.max_file_size
        self.allowed_image_types = settings.allowed_image_types
    
    def load_image(self, image_source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Load and validate an image from various sources.
        
        Args:
            image_source: Image source (URL, file path, bytes, numpy array, or PIL Image)
            
        Returns:
            numpy.ndarray: RGB image as numpy array
            
        Raises:
            ImageLoadError: If image cannot be loaded
            InvalidImageError: If image is invalid or corrupted
            ImageTooLargeError: If image exceeds size limits
            UnsupportedImageFormatError: If image format is not supported
        """
        try:
            if isinstance(image_source, str):
                return self._load_from_string(image_source)
            elif isinstance(image_source, bytes):
                return self._load_from_bytes(image_source)
            elif isinstance(image_source, np.ndarray):
                return self._validate_numpy_array(image_source)
            elif isinstance(image_source, Image.Image):
                return self._convert_pil_to_numpy(image_source)
            else:
                raise ImageLoadError(f"Unsupported image source type: {type(image_source)}")
                
        except Exception as e:
            if isinstance(e, ImageLoadError):
                raise
            logger.error(f"Failed to load image: {str(e)}")
            raise ImageLoadError(f"Failed to load image: {str(e)}")
    
    def _load_from_string(self, image_source: str) -> np.ndarray:
        """Load image from string (URL or file path)."""
        if self._is_url(image_source):
            return self._load_from_url(image_source)
        else:
            return self._load_from_file_path(image_source)
    
    def _load_from_url(self, url: str) -> np.ndarray:
        """Load image from URL."""
        try:
            logger.info(f"Loading image from URL: {url}")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'webp']):
                raise UnsupportedImageFormatError(f"Unsupported content type: {content_type}")
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size:
                raise ImageTooLargeError(f"Image size ({content_length} bytes) exceeds limit")
            
            # Load image
            image_data = response.content
            if len(image_data) > self.max_file_size:
                raise ImageTooLargeError(f"Image size ({len(image_data)} bytes) exceeds limit")
            
            return self._load_from_bytes(image_data)
            
        except RequestException as e:
            logger.error(f"Network error loading image from URL {url}: {str(e)}")
            raise NetworkError(f"Failed to load image from URL: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading image from URL {url}: {str(e)}")
            raise ImageLoadError(f"Failed to load image from URL: {str(e)}")
    
    def _load_from_file_path(self, file_path: str) -> np.ndarray:
        """Load image from file path."""
        try:
            logger.info(f"Loading image from file: {file_path}")
            path = Path(file_path)
            
            if not path.exists():
                raise ImageLoadError(f"File does not exist: {file_path}")
            
            if path.stat().st_size > self.max_file_size:
                raise ImageTooLargeError(f"File size ({path.stat().st_size} bytes) exceeds limit")
            
            with open(path, 'rb') as f:
                image_data = f.read()
            
            return self._load_from_bytes(image_data)
            
        except Exception as e:
            logger.error(f"Error loading image from file {file_path}: {str(e)}")
            raise ImageLoadError(f"Failed to load image from file: {str(e)}")
    
    def _load_from_bytes(self, image_data: bytes) -> np.ndarray:
        """Load image from bytes."""
        try:
            # Try to load with PIL first
            image = Image.open(io.BytesIO(image_data))
            return self._convert_pil_to_numpy(image)
            
        except UnidentifiedImageError:
            raise InvalidImageError("Image format not recognized")
        except Exception as e:
            logger.error(f"Error loading image from bytes: {str(e)}")
            raise ImageLoadError(f"Failed to load image from bytes: {str(e)}")
    
    def _convert_pil_to_numpy(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array."""
        try:
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            return self._validate_numpy_array(image_array)
            
        except Exception as e:
            logger.error(f"Error converting PIL image to numpy: {str(e)}")
            raise ImageLoadError(f"Failed to convert image: {str(e)}")
    
    def _validate_numpy_array(self, image_array: np.ndarray) -> np.ndarray:
        """Validate and normalize numpy array."""
        try:
            # Check if it's a valid image array
            if image_array.ndim not in [2, 3]:
                raise InvalidImageError("Invalid image dimensions")
            
            # Convert grayscale to RGB if necessary
            if image_array.ndim == 2:
                image_array = np.stack((image_array,) * 3, axis=-1)
            
            # Ensure it's RGB (3 channels)
            if image_array.shape[2] != 3:
                raise InvalidImageError("Image must have 3 color channels (RGB)")
            
            # Check data type and convert if necessary
            if image_array.dtype != np.uint8:
                if image_array.dtype in [np.float32, np.float64]:
                    # Normalize to 0-255 range
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    else:
                        image_array = image_array.astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # Validate image size
            height, width = image_array.shape[:2]
            if height < 20 or width < 20:
                raise InvalidImageError("Image too small (minimum 20x20 pixels)")
            if height > 10000 or width > 10000:
                raise InvalidImageError("Image too large (maximum 10000x10000 pixels)")
            
            return image_array
            
        except Exception as e:
            if isinstance(e, InvalidImageError):
                raise
            logger.error(f"Error validating numpy array: {str(e)}")
            raise InvalidImageError(f"Invalid image array: {str(e)}")
    
    def _is_url(self, string: str) -> bool:
        """Check if string is a valid URL."""
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def preprocess_for_face_recognition(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for face recognition.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Ensure image is in the correct format
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Convert BGR to RGB if necessary (OpenCV uses BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Check if it's BGR by comparing with RGB conversion
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if not np.array_equal(image, rgb_image):
                    image = rgb_image
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ImageLoadError(f"Failed to preprocess image: {str(e)}")
    
    def resize_image(self, image: np.ndarray, max_size: int = 1920) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            max_size: Maximum dimension size
            
        Returns:
            numpy.ndarray: Resized image
        """
        try:
            height, width = image.shape[:2]
            
            if height <= max_size and width <= max_size:
                return image
            
            # Calculate new dimensions
            if height > width:
                new_height = max_size
                new_width = int(width * max_size / height)
            else:
                new_width = max_size
                new_height = int(height * max_size / width)
            
            # Resize using OpenCV
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return resized_image
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise ImageLoadError(f"Failed to resize image: {str(e)}")
    
    def validate_image_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image quality for face recognition.
        
        Args:
            image: Input image
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            height, width = image.shape[:2]
            
            # Check minimum size
            if height < 50 or width < 50:
                return False, "Image too small for reliable face detection"
            
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                return False, "Image aspect ratio too extreme"
            
            # Check for blur (simple variance-based blur detection)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score < 50:
                return False, "Image appears to be blurry"
            
            return True, "Image quality is acceptable"
            
        except Exception as e:
            logger.error(f"Error validating image quality: {str(e)}")
            return False, f"Error validating image quality: {str(e)}"


# Global instance
image_processor = ImageProcessor() 