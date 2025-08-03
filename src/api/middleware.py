"""
API middleware for authentication, rate limiting, and CORS.
"""

import time
import logging
from typing import Dict, Set
from fastapi import Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config.settings import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
        self.max_requests = settings.rate_limit_per_minute
        self.window_seconds = 60
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for the client."""
        now = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests outside the window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_id = request.client.host
    
    if not rate_limiter.is_allowed(client_id):
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": f"Maximum {settings.rate_limit_per_minute} requests per minute"
            }
        )
    
    response = await call_next(request)
    return response


async def authentication_middleware(request: Request, call_next):
    """Simple authentication middleware."""
    # Skip authentication for health check and documentation
    if request.url.path in ["/api/v1/face/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # Check for API key in headers
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        # For now, allow requests without API key (implement proper auth as needed)
        logger.warning(f"No API key provided for request: {request.url.path}")
        pass
    
    response = await call_next(request)
    return response


async def logging_middleware(request: Request, call_next):
    """Request logging middleware."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
        
        return response
        
    except Exception as e:
        # Log errors
        process_time = time.time() - start_time
        logger.error(f"Error processing request: {str(e)} in {process_time:.3f}s")
        raise


def setup_cors_middleware(app):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=settings.allowed_methods,
        allow_headers=settings.allowed_headers,
    )


def setup_middleware(app):
    """Setup all middleware."""
    # Add custom middleware
    app.middleware("http")(logging_middleware)
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(authentication_middleware)
    
    # Setup CORS
    setup_cors_middleware(app) 