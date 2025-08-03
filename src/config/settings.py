"""
Configuration settings for the face recognition application.
Uses Pydantic BaseSettings for environment variable management.
"""

import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = Field(default="Face Recognition API", env="APP_NAME")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8081, env="PORT")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-here-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Face recognition settings
    face_match_threshold: float = Field(default=0.6, env="FACE_MATCH_THRESHOLD")
    max_faces_per_image: int = Field(default=10, env="MAX_FACES_PER_IMAGE")
    min_face_size: int = Field(default=20, env="MIN_FACE_SIZE")
    
    # Model settings
    model_path: str = Field(default="models/buffalo_l", env="MODEL_PATH")
    detector_model: str = Field(default="det_10g.onnx", env="DETECTOR_MODEL")
    recognizer_model: str = Field(default="w600k_r50.onnx", env="RECOGNIZER_MODEL")
    
    # CORS settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="ALLOWED_ORIGINS"
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        env="ALLOWED_METHODS"
    )
    allowed_headers: List[str] = Field(
        default=["*"],
        env="ALLOWED_HEADERS"
    )
    
    # File upload settings
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_image_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"],
        env="ALLOWED_IMAGE_TYPES"
    )
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Database (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Redis (for caching)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    @validator("allowed_origins", "allowed_methods", "allowed_headers", pre=True)
    def parse_list_fields(cls, v):
        """Parse comma-separated strings into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v
    
    @validator("allowed_image_types", pre=True)
    def parse_image_types(cls, v):
        """Parse comma-separated image types into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings() 