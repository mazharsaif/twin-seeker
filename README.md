# Twin-Seeker: Advanced Face Recognition API

A powerful, production-ready face recognition API built with FastAPI and InsightFace. **Twin-Seeker** enables you to verify if two people are the same person by comparing their facial features with high accuracy. Perfect for identity verification, duplicate detection, and facial similarity analysis. *Coming soon: Find your celebrity twin with our vector database search feature!*

## 🎭 Project Vision

**Twin-Seeker** was born from the idea of finding your celebrity twin! Upload a picture of your face and discover which celebrities you most resemble. The name reflects our mission to help you find your "twin" in the world of celebrity faces.

## 🎯 Key Features

### **Face Matching Verification**
- **High-Accuracy Comparison**: Compare two face images to determine if they belong to the same person
- **Configurable Thresholds**: Adjustable similarity thresholds for different use cases
- **Multiple Input Formats**: Support for URLs, file uploads, and base64 encoded images
- **Confidence Scoring**: Get confidence levels (high, medium, low) for match results

### **Comprehensive Face Analysis**
- **Face Detection**: Detect and locate faces in images with bounding boxes
- **Facial Landmarks**: Extract 68-point facial landmarks for detailed analysis
- **Embedding Extraction**: Generate unique face embeddings for advanced applications
- **Batch Processing**: Compare one reference face against multiple images

### **Production-Ready Features**
- **Modular Architecture**: Clean separation of concerns with config, core, services, and API layers
- **Rate Limiting**: Built-in protection against API abuse
- **Health Monitoring**: Real-time service status and model loading checks
- **Comprehensive Testing**: Unit tests and integration tests
- **Docker Support**: Easy deployment with Docker and Docker Compose

## 🚀 Quick Start

### 1. Installation

#### **Option A: Using UV (Recommended)**

```bash
# Clone the repository
git clone <repository-url>
cd twin-seeker

# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies using UV
uv sync

# Or install in development mode
uv sync --dev
```

#### **Option B: Using pip**

```bash
# Clone the repository
git clone <repository-url>
cd twin-seeker

# Install dependencies
pip install -r requirements.txt

# Or use the Makefile
make install
```

### 2. Model Setup

Download the InsightFace models and place them in the `models/buffalo_l/` directory:

```bash
mkdir -p models/buffalo_l
# Download det_10g.onnx and w600k_r50.onnx to models/buffalo_l/
```

### 3. Run the Application

#### **Using UV**
```bash
# Development mode
uv run python main.py

# Or run with specific Python version
uv run --python 3.11 python main.py

# Run tests
uv run pytest

# Run with hot reload
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### **Using pip/Makefile**
```bash
# Development mode
python main.py

# Or using the Makefile
make run-dev

# Production mode
make run
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/face/health
- **Root Endpoint**: http://localhost:8000/

## 🔍 Face Matching Verification Examples

### **Example 1: Verify if Two People are the Same**

```bash
# Compare faces from URLs
curl -X POST "http://localhost:8000/api/v1/face/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "image1_source": "url",
    "image1_data": "https://example.com/person1.jpg",
    "image2_source": "url", 
    "image2_data": "https://example.com/person2.jpg",
    "threshold": 0.7
  }'
```

**Response:**
```json
{
  "success": true,
  "similarity_score": 0.85,
  "is_match": true,
  "confidence": "high",
  "message": "Faces are very similar - likely the same person"
}
```

### **Example 2: Upload Images for Verification**

```bash
# Compare uploaded images
curl -X POST "http://localhost:8000/api/v1/face/compare-upload" \
  -F "image1=@person1.jpg" \
  -F "image2=@person2.jpg" \
  -F "threshold=0.7"
```

### **Example 3: Batch Verification**

```bash
# Compare one reference image against multiple images
curl -X POST "http://localhost:8000/api/v1/face/batch-compare" \
  -H "Content-Type: application/json" \
  -d '{
    "reference_image": "https://example.com/reference_person.jpg",
    "comparison_images": [
      "https://example.com/candidate1.jpg",
      "https://example.com/candidate2.jpg",
      "https://example.com/candidate3.jpg"
    ],
    "threshold": 0.7
  }'
```

## 📋 All Use Cases

### **1. Identity Verification**
- **Use Case**: Verify if a person's ID photo matches their current photo
- **Endpoint**: `/api/v1/face/compare`
- **Threshold**: 0.7-0.8 for high security

### **2. Duplicate Detection**
- **Use Case**: Find duplicate photos in a photo library
- **Endpoint**: `/api/v1/face/batch-compare`
- **Threshold**: 0.6-0.7 for general use

### **3. Access Control**
- **Use Case**: Verify identity for building/device access
- **Endpoint**: `/api/v1/face/compare-upload`
- **Threshold**: 0.8+ for high security

### **4. Social Media Analysis**
- **Use Case**: Find similar faces across social media platforms
- **Endpoint**: `/api/v1/face/batch-compare`
- **Threshold**: 0.6-0.7

### **5. Law Enforcement**
- **Use Case**: Match surveillance photos with suspect databases
- **Endpoint**: `/api/v1/face/compare`
- **Threshold**: 0.7-0.8

### **6. Customer Service**
- **Use Case**: Verify customer identity for account access
- **Endpoint**: `/api/v1/face/compare-upload`
- **Threshold**: 0.7

### **7. Photo Organization**
- **Use Case**: Group photos by person automatically
- **Endpoint**: `/api/v1/face/batch-compare`
- **Threshold**: 0.6

### **8. Event Management**
- **Use Case**: Track attendance at events
- **Endpoint**: `/api/v1/face/compare`
- **Threshold**: 0.7

### **9. Celebrity Twin Finder** *(Coming Soon)*
- **Use Case**: Find your celebrity twin from a database of celebrity faces
- **Endpoint**: `/api/v1/face/celebrity-search`
- **Features**: Vector database search with top-K similar faces
- **Datasets**: WikiFaces, IMDB Celebrity Dataset
- **Threshold**: 0.6-0.8 for celebrity matching

## 🔧 API Endpoints

### **Face Comparison**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/face/compare` | POST | Compare faces from URLs/base64 |
| `/api/v1/face/compare-upload` | POST | Compare uploaded images |
| `/api/v1/face/compare-url` | POST | Compare faces from URLs (form data) |
| `/api/v1/face/batch-compare` | POST | Compare one face against multiple images |

### **Face Detection**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/face/detect` | POST | Detect faces from URL/base64 |
| `/api/v1/face/detect-upload` | POST | Detect faces in uploaded image |
| `/api/v1/face/detect-url` | POST | Detect faces from URL (form data) |

### **Face Analysis**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/face/embedding` | POST | Extract face embedding vector |

### **Celebrity Search** *(Coming Soon)*

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/face/celebrity-search` | POST | Find celebrity twins for uploaded face |

### **System**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/face/health` | GET | Health check and status |

## ⚙️ Configuration

The application uses environment variables for configuration. Create a `.env` file:

```bash
# Server Configuration
DEBUG=true
HOST=0.0.0.0
PORT=8000
SECRET_KEY=your-secret-key-here

# Face Recognition Settings
FACE_MATCH_THRESHOLD=0.6
MAX_FACES_PER_IMAGE=10
MIN_FACE_SIZE=20

# API Limits
RATE_LIMIT_PER_MINUTE=60
MAX_FILE_SIZE=10485760  # 10MB

# Vector Database Settings (for future celebrity twin feature)
VECTOR_DB_TYPE=pinecone  # pinecone, weaviate, qdrant, chroma
VECTOR_DB_API_KEY=your_vector_db_api_key
VECTOR_DB_ENVIRONMENT=your_environment

# Logging
LOG_LEVEL=INFO
```

### **Threshold Guidelines**

| Use Case | Threshold | Confidence |
|----------|-----------|------------|
| High Security (Access Control) | 0.8+ | Very Strict |
| Identity Verification | 0.7-0.8 | Strict |
| General Matching | 0.6-0.7 | Balanced |
| Photo Organization | 0.5-0.6 | Lenient |

## 🐳 Docker Deployment

### **Quick Start with Docker Compose**

```bash
# Start all services
docker-compose up --build

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f twin-seeker-api
```

### **Environment Configuration**

Create a `.env` file for Docker environment variables:

```bash
# Server Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000
SECRET_KEY=your-production-secret-key-here

# Face Recognition Settings
FACE_MATCH_THRESHOLD=0.6
MAX_FACES_PER_IMAGE=10
MIN_FACE_SIZE=20

# API Limits
RATE_LIMIT_PER_MINUTE=60
MAX_FILE_SIZE=10485760

# Logging
LOG_LEVEL=INFO

# Vector Database Settings (for future features)
VECTOR_DB_TYPE=pinecone
VECTOR_DB_API_KEY=your-api-key
VECTOR_DB_ENVIRONMENT=your-environment
```

### **Using Docker Directly**

```bash
# Build the image
docker build -t twin-seeker .

# Run with environment variables
docker run -p 8000:8000 \
  -e DEBUG=false \
  -e FACE_MATCH_THRESHOLD=0.6 \
  -e SECRET_KEY=your-secret-key \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs \
  twin-seeker

# Run with custom configuration
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs \
  twin-seeker
```

### **Docker Features**

#### **🔒 Security**
- **Non-root user**: Container runs as `app` user
- **Read-only models**: Models directory mounted as read-only
- **Minimal base image**: Python 3.11 slim for smaller attack surface

#### **⚡ Performance**
- **UV package management**: Faster dependency installation
- **Multi-stage builds**: Optimized for production
- **Health checks**: Automatic container health monitoring

#### **🔧 Development**
- **Hot reload**: Development mode with auto-restart
- **Volume mounting**: Live code changes without rebuild
- **Debug mode**: Easy debugging with environment variables

### **Production Deployment**

```bash
# Build production image
docker build -t twin-seeker:latest .

# Run with production settings
docker run -d \
  --name twin-seeker-api \
  -p 8000:8000 \
  --env-file .env.production \
  -v /path/to/models:/app/models:ro \
  -v /path/to/logs:/app/logs \
  --restart unless-stopped \
  twin-seeker:latest
```

### **Docker Compose Services**

The `docker-compose.yml` includes:

- **twin-seeker-api**: Main application
- **redis**: Caching (commented out)
- **qdrant**: Vector database (commented out)
- **prometheus**: Monitoring (commented out)

### **Health Monitoring**

```bash
# Check container health
docker ps

# View health check logs
docker logs twin-seeker-api

# Test API health
curl http://localhost:8000/api/v1/face/health
```

### **Troubleshooting**

```bash
# View container logs
docker-compose logs twin-seeker-api

# Access container shell
docker-compose exec twin-seeker-api bash

# Rebuild without cache
docker-compose build --no-cache

# Check container resources
docker stats twin-seeker-api
```

## 🧪 Testing

### **Run All Tests**

#### **Using UV**
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_face_recognition.py

# Run with verbose output
uv run pytest -v
```

#### **Using pip/Makefile**
```bash
# Using Makefile
make test

# Using pytest directly
pytest

# With coverage
make test-cov
```

### **Test Face Matching**

```bash
# Test the health endpoint
curl http://localhost:8000/api/v1/face/health

# Test face comparison with sample images
curl -X POST "http://localhost:8000/api/v1/face/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "image1_source": "url",
    "image1_data": "https://example.com/test1.jpg",
    "image2_source": "url",
    "image2_data": "https://example.com/test2.jpg",
    "threshold": 0.7
  }'
```

## 📊 Performance

- **Model Loading**: Models loaded once at startup for optimal performance
- **Image Processing**: Automatic resizing for large images
- **Async Processing**: Non-blocking API endpoints
- **Memory Management**: Efficient image processing and cleanup

## 🔒 Security Features

- **Input Validation**: Comprehensive validation of all inputs
- **Rate Limiting**: Per-client rate limiting to prevent abuse
- **File Size Limits**: Configurable upload size limits
- **CORS Configuration**: Configurable CORS settings
- **Error Sanitization**: Safe error messages without sensitive data

## 📈 Monitoring

### **Health Check**

```bash
curl http://localhost:8000/api/v1/face/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:30:00",
  "version": "1.0.0",
  "models_loaded": true
}
```

### **Logging**

The application uses structured logging with configurable levels:

```bash
export LOG_LEVEL=DEBUG  # For development
export LOG_LEVEL=INFO   # For production
```

## 🚀 Future Features

### **🎭 Celebrity Twin Finder**

**Find your celebrity doppelgänger!** Upload a photo and discover which celebrities you most resemble.

#### **Planned Features:**
- **Vector Database Integration**: Store celebrity face embeddings in a high-performance vector database
- **Top-K Search**: Find the most similar celebrity faces to your uploaded photo
- **Multiple Datasets**: Support for WikiFaces, IMDB Celebrity Dataset, and more
- **Similarity Ranking**: Get ranked results with similarity scores
- **Celebrity Information**: Display celebrity details alongside matches

#### **Technical Implementation:**
```python
# Future API endpoint example
POST /api/v1/face/celebrity-search
{
  "image": "uploaded_face_photo.jpg",
  "top_k": 10,
  "dataset": "wikifaces",  # or "imdb_celebrity"
  "min_similarity": 0.6
}

# Response
{
  "success": true,
  "matches": [
    {
      "celebrity_name": "Tom Hanks",
      "similarity_score": 0.85,
      "dataset": "wikifaces",
      "image_url": "https://...",
      "confidence": "high"
    },
    {
      "celebrity_name": "Morgan Freeman", 
      "similarity_score": 0.78,
      "dataset": "wikifaces",
      "image_url": "https://...",
      "confidence": "medium"
    }
  ]
}
```

#### **Datasets Planned:**
- **WikiFaces**: Wikipedia celebrity faces dataset
- **IMDB Celebrity Dataset**: Large-scale celebrity face dataset
- **Custom Datasets**: Support for custom celebrity databases

#### **Vector Database Options:**
- **Pinecone**: Cloud-based vector database
- **Weaviate**: Open-source vector database
- **Qdrant**: High-performance vector similarity search
- **Chroma**: Embedding database for AI applications

### **🔍 Advanced Search Features**
- **Multi-modal Search**: Search by text description + face
- **Age/Gender Filtering**: Filter results by demographic
- **Style-based Matching**: Match based on facial features and style
- **Batch Celebrity Search**: Compare multiple faces at once

### **🔧 Technical Optimizations**
- **Embedding Normalization**: L2 normalization for improved cosine similarity accuracy
- **Quality Assessment**: Embedding quality checks before comparison
- **Caching**: Cache frequently used embeddings for performance

### **📅 Development Roadmap**

#### **Phase 1: Core Infrastructure** ✅
- [x] Face detection and comparison
- [x] API endpoints and validation
- [x] Docker deployment
- [x] Testing framework

#### **Phase 2: Vector Database Integration** 🚧
- [ ] Vector database setup (Pinecone/Weaviate/Qdrant)
- [ ] Celebrity dataset preprocessing
- [ ] Embedding storage and indexing
- [ ] Search API endpoints

#### **Phase 3: Celebrity Twin Feature** 📋
- [ ] WikiFaces dataset integration
- [ ] IMDB Celebrity dataset support
- [ ] Top-K search implementation
- [ ] Celebrity information API

#### **Phase 4: Advanced Features** 🔮
- [ ] Multi-modal search
- [ ] Demographic filtering
- [ ] Style-based matching
- [ ] Mobile app integration

#### **Phase 5: Performance Optimizations** ⚡
- [ ] L2 embedding normalization for improved cosine similarity
- [ ] Embedding quality assessment and filtering
- [ ] Vector Database Implementation (see detailed steps below)
- [ ] GPU acceleration for large-scale comparisons

## 🗄️ Vector Database Implementation Guide

### **Phase 1: Vector Database Setup**

#### **Step 1: Choose Vector Database**
```bash
# Option 1: Pinecone (Cloud-based, easiest)
uv add pinecone-client

# Option 2: Weaviate (Self-hosted, open-source)
uv add weaviate-client

# Option 3: Qdrant (High-performance, self-hosted)
uv add qdrant-client

# Option 4: Chroma (Lightweight, local)
uv add chromadb
```

#### **Step 2: Database Schema Design**
```python
# Example schema for celebrity embeddings
celebrity_schema = {
    "id": "unique_celebrity_id",
    "name": "celebrity_name",
    "embedding": "512_dim_vector",
    "dataset": "wikifaces|imdb_celebrity",
    "image_url": "celebrity_image_url",
    "metadata": {
        "age": "estimated_age",
        "gender": "male|female",
        "occupation": "actor|singer|politician",
        "nationality": "country"
    }
}
```

#### **Step 3: Data Preprocessing Pipeline**
```python
import numpy as np
import uuid
from pathlib import Path

# 1. Download celebrity datasets
# - WikiFaces: ~100K celebrity faces
# - IMDB Celebrity: ~500K celebrity faces

# 2. Extract embeddings
def process_celebrity_dataset(dataset_path):
    embeddings = []
    for image_path in Path(dataset_path).glob("*.jpg"):
        embedding = face_recognition_engine.extract_embedding(str(image_path))
        embeddings.append({
            "id": str(uuid.uuid4()),
            "embedding": embedding,
            "metadata": extract_metadata(str(image_path))
        })
    return embeddings

# 3. Normalize embeddings (L2 normalization)
def normalize_embeddings(embeddings):
    for emb in embeddings:
        emb["embedding"] = emb["embedding"] / np.linalg.norm(emb["embedding"])
    return embeddings
```

#### **Step 4: Database Population**
```python
# Example with Pinecone
import pinecone

# Initialize
pinecone.init(api_key="your_api_key", environment="your_environment")
index_name = "celebrity-faces"

# Create index
pinecone.create_index(
    name=index_name,
    dimension=512,  # ArcFace embedding dimension
    metric="cosine"
)

# Populate database
index = pinecone.Index(index_name)
for celebrity in processed_celebrities:
    index.upsert(
        vectors=[{
            "id": celebrity["id"],
            "values": celebrity["embedding"].tolist(),
            "metadata": celebrity["metadata"]
        }]
    )
```

### **Phase 2: Search API Implementation**

#### **Step 1: Add New API Endpoints**
```python
# Add to src/api/routes/face_routes.py

@router.post("/celebrity-search", response_model=CelebritySearchResult)
async def search_celebrity_twins(
    image: UploadFile = File(...),
    top_k: int = Form(10),
    dataset: str = Form("all"),
    min_similarity: float = Form(0.6)
):
    """Find celebrity twins for uploaded face image."""
    try:
        # Extract embedding from uploaded image
        image_content = await image.read()
        user_embedding = face_service.extract_embedding(image_content)
        
        # Search vector database
        results = vector_db_service.search_similar_faces(
            query_embedding=user_embedding,
            top_k=top_k,
            dataset=dataset,
            min_similarity=min_similarity
        )
        
        return CelebritySearchResult(
            success=True,
            matches=results,
            query_embedding_quality=assess_embedding_quality(user_embedding)
        )
        
    except Exception as e:
        logger.error(f"Celebrity search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

#### **Step 2: Vector Database Service**
```python
# Create src/services/vector_db_service.py
import numpy as np
import pinecone

class VectorDatabaseService:
    def __init__(self):
        self.index = pinecone.Index("celebrity-faces")
    
    def search_similar_faces(self, query_embedding, top_k=10, 
                           dataset="all", min_similarity=0.6):
        """Search for similar celebrity faces."""
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Perform search
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter={"dataset": dataset} if dataset != "all" else None
        )
        
        # Process results
        matches = []
        for match in results.matches:
            if match.score >= min_similarity:
                matches.append({
                    "celebrity_id": match.id,
                    "celebrity_name": match.metadata.get("name"),
                    "similarity_score": match.score,
                    "dataset": match.metadata.get("dataset"),
                    "image_url": match.metadata.get("image_url"),
                    "metadata": match.metadata
                })
        
        return matches
```

### **Phase 3: Performance Optimizations**

#### **Step 1: Batch Processing**
```python
# Process multiple images at once
def batch_celebrity_search(image_embeddings, top_k=10):
    """Search for multiple faces simultaneously."""
    results = []
    for i, embedding in enumerate(image_embeddings):
        matches = vector_db_service.search_similar_faces(embedding, top_k)
        results.append({
            "image_index": i,
            "matches": matches
        })
    return results
```

#### **Step 2: Caching Layer**
```python
# Add Redis caching for frequent queries
import json
import redis
import numpy as np

class CachedVectorDBService(VectorDatabaseService):
    def __init__(self):
        super().__init__()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def search_similar_faces(self, query_embedding, top_k=10, **kwargs):
        # Generate cache key
        cache_key = f"search:{hash(query_embedding.tobytes())}:{top_k}"
        
        # Check cache first
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Perform search
        results = super().search_similar_faces(query_embedding, top_k, **kwargs)
        
        # Cache result (expire in 1 hour)
        self.redis_client.setex(cache_key, 3600, json.dumps(results))
        
        return results
```

### **Phase 4: Monitoring and Analytics**

#### **Step 1: Search Analytics**
```python
# Track search performance and popular queries
from datetime import datetime
import numpy as np

class SearchAnalytics:
    def __init__(self):
        self.analytics_db = None  # Initialize analytics database
    
    def log_search(self, query_embedding, results, response_time):
        """Log search query for analytics."""
        self.analytics_db.insert({
            "timestamp": datetime.now(),
            "query_hash": hash(query_embedding.tobytes()),
            "results_count": len(results),
            "top_match_score": results[0]["similarity_score"] if results else 0,
            "response_time_ms": response_time
        })
```

#### **Step 2: Database Health Monitoring**
```python
# Monitor vector database performance
import time
import numpy as np

def check_db_health():
    """Check vector database status and performance."""
    try:
        # Test search performance
        test_embedding = np.random.rand(512)
        start_time = time.time()
        results = vector_db_service.search_similar_faces(test_embedding, top_k=1)
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": response_time,
            "total_embeddings": get_total_embeddings_count(),
            "last_updated": get_last_update_time()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### **Implementation Timeline**

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Vector DB Setup | Database configured and populated |
| 3-4 | Search API | `/celebrity-search` endpoint working |
| 5-6 | Performance | Caching and batch processing |
| 7-8 | Monitoring | Analytics and health checks |

---

## 🛠️ Development

### **Package Management**

This project uses **UV** for fast Python package management. UV provides:
- **Faster installation**: Up to 10-100x faster than pip
- **Reliable dependency resolution**: Better conflict resolution
- **Lock file**: Reproducible builds with `uv.lock`
- **Virtual environment management**: Automatic venv creation

#### **UV Commands**
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update dependencies
uv sync --upgrade

# Run commands in UV environment
uv run python script.py
uv run pytest
uv run uvicorn main:app --reload

# Regenerate lock file (rarely needed)
uv lock
```

### **Project Structure**
```
src/
├── config/          # Configuration management
├── core/           # Core face recognition engine
├── models/         # Pydantic schemas and data models
├── services/       # Business logic layer
├── api/           # API routes and middleware
└── tests/         # Test suite
```

### **Adding New Features**

1. Add new schemas in `src/models/schemas.py`
2. Implement business logic in `src/services/`
3. Add API routes in `src/api/routes/`
4. Write tests in `tests/`

### **Development Workflow with UV**

```bash
# 1. Install new dependency
uv add new-package-name

# 2. Run development server
uv run uvicorn main:app --reload

# 3. Run tests
uv run pytest

# 4. Format and lint
uv run black src/ tests/
uv run flake8 src/ tests/

# Note: uv lock is rarely needed - uv add and uv sync handle it automatically
```

### **Code Quality**

#### **Using UV**
```bash
# Format code with black
uv run black src/ tests/

# Lint code with flake8
uv run flake8 src/ tests/

# Type checking with mypy
uv run mypy src/

# Run all checks
uv run black src/ tests/ && uv run flake8 src/ tests/ && uv run pytest
```

#### **Using Makefile**
```bash
# Format code
make format

# Lint code
make lint

# Run all checks
make check
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: http://localhost:8000/docs (when running)
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

---

**Twin-Seeker** - Advanced face recognition for identity verification and facial similarity analysis.
