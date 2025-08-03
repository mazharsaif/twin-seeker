"""
Custom exceptions for the face recognition application.
"""


class FaceRecognitionError(Exception):
    """Base exception for face recognition errors."""
    pass


class ModelLoadError(FaceRecognitionError):
    """Raised when face recognition models fail to load."""
    pass


class ImageLoadError(FaceRecognitionError):
    """Raised when an image cannot be loaded or processed."""
    pass


class NoFacesDetectedError(FaceRecognitionError):
    """Raised when no faces are detected in an image."""
    pass


class MultipleFacesDetectedError(FaceRecognitionError):
    """Raised when multiple faces are detected when only one is expected."""
    pass


class InvalidImageError(FaceRecognitionError):
    """Raised when an image is invalid or corrupted."""
    pass


class ImageTooLargeError(FaceRecognitionError):
    """Raised when an image exceeds size limits."""
    pass


class UnsupportedImageFormatError(FaceRecognitionError):
    """Raised when an image format is not supported."""
    pass


class NetworkError(FaceRecognitionError):
    """Raised when network operations fail."""
    pass


class ValidationError(FaceRecognitionError):
    """Raised when input validation fails."""
    pass


class RateLimitError(FaceRecognitionError):
    """Raised when rate limits are exceeded."""
    pass


class AuthenticationError(FaceRecognitionError):
    """Raised when authentication fails."""
    pass 