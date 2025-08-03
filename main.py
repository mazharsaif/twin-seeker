"""
Main entry point for the Face Recognition API.
"""

import uvicorn
from src.api.app import app
from src.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 