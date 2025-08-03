#!/usr/bin/env python3
"""
Script to run tests with proper configuration.
"""

import subprocess
import sys
import os

def run_tests():
    """Run the test suite."""
    print("Running face recognition API tests...")
    
    # Set test environment variables
    env = os.environ.copy()
    env.update({
        "TESTING": "true",
        "LOG_LEVEL": "WARNING",
        "FACE_MATCH_THRESHOLD": "0.6",
        "MAX_FACES_PER_IMAGE": "5"
    })
    
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v"
        ], env=env, check=True)
        
        print("\n✓ Tests completed successfully!")
        print("Coverage report generated in htmlcov/")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Tests failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error running tests: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests() 