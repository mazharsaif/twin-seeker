#!/usr/bin/env python3
"""
Startup script for the Face Recognition API.
Loads environment variables and starts the server.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

def check_models():
    """Check if required models exist."""
    models_dir = project_root / "models" / "buffalo_l"
    required_models = ["det_10g.onnx", "w600k_r50.onnx"]
    
    missing_models = []
    for model in required_models:
        if not (models_dir / model).exists():
            missing_models.append(model)
    
    if missing_models:
        print("⚠ Warning: Missing required models:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nPlease download the models to models/buffalo_l/")
        print("You can use: python scripts/setup_models.py")
        return False
    
    print("✓ All required models found")
    return True

def main():
    """Main startup function."""
    print("Starting Face Recognition API...")
    
    # Load environment variables
    load_env_file()
    
    # Check models
    if not check_models():
        print("\nModels not found. Please setup models first.")
        sys.exit(1)
    
    # Import and run the application
    try:
        from src.api.app import app
        import uvicorn
        from src.config.settings import settings
        
        print(f"Starting server on {settings.host}:{settings.port}")
        print(f"API Documentation: http://{settings.host}:{settings.port}/docs")
        
        uvicorn.run(
            "src.api.app:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level=settings.log_level.lower()
        )
        
    except ImportError as e:
        print(f"Error importing application: {str(e)}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 