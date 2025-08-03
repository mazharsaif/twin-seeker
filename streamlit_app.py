import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import numpy as np
import time
from typing import Optional, Tuple

# Configuration
API_BASE_URL = "http://localhost:8081"
API_ENDPOINTS = {
    "health": f"{API_BASE_URL}/api/v1/face/health",
    "compare": f"{API_BASE_URL}/api/v1/face/compare",
    "compare_upload": f"{API_BASE_URL}/api/v1/face/compare-upload",
    "detect": f"{API_BASE_URL}/api/v1/face/detect",
    "detect_upload": f"{API_BASE_URL}/api/v1/face/detect-upload",
    "embedding": f"{API_BASE_URL}/api/v1/face/embedding",
    "batch_compare": f"{API_BASE_URL}/api/v1/face/batch-compare"
}

def check_api_health():
    """Check if the FastAPI server is running."""
    try:
        response = requests.get(API_ENDPOINTS["health"], timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, None

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(",")[1]
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))

def compare_faces_api(image1: bytes, image2: bytes, threshold: float = 0.6) -> dict:
    """Compare two faces using the FastAPI endpoint."""
    try:
        files = {
            "image1": ("image1.jpg", image1, "image/jpeg"),
            "image2": ("image2.jpg", image2, "image/jpeg"),
            "threshold": (None, str(threshold))
        }
        response = requests.post(API_ENDPOINTS["compare_upload"], files=files)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def detect_faces_api(image: bytes, return_landmarks: bool = True, return_embeddings: bool = False) -> dict:
    """Detect faces in an image using the FastAPI endpoint."""
    try:
        files = {
            "image": ("image.jpg", image, "image/jpeg"),
            "return_landmarks": (None, str(return_landmarks)),
            "return_embeddings": (None, str(return_embeddings))
        }
        response = requests.post(API_ENDPOINTS["detect_upload"], files=files)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_embedding_api(image: bytes) -> dict:
    """Extract face embedding using the FastAPI endpoint."""
    try:
        files = {
            "image": ("image.jpg", image, "image/jpeg")
        }
        response = requests.post(API_ENDPOINTS["embedding"], files=files)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def batch_compare_api(reference_image: bytes, comparison_images: list, threshold: float = 0.6) -> dict:
    """Batch compare faces using the FastAPI endpoint."""
    try:
        files = {
            "reference_image": ("reference.jpg", reference_image, "image/jpeg"),
            "threshold": (None, str(threshold))
        }
        
        # Add comparison images
        for i, img in enumerate(comparison_images):
            files[f"comparison_image_{i}"] = (f"comparison_{i}.jpg", img, "image/jpeg")
        
        response = requests.post(API_ENDPOINTS["batch_compare"], files=files)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    st.set_page_config(
        page_title="Twin-Seeker: Face Recognition System",
        page_icon="👥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">👥 Twin-Seeker</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Advanced Face Recognition System</h3>', unsafe_allow_html=True)
    
    # Check API health
    is_healthy, health_data = check_api_health()
    
    if not is_healthy:
        st.error("⚠️ FastAPI server is not running. Please start the server at http://localhost:8000")
        st.info("To start the server, run: `uv run uvicorn main:app --reload`")
        return
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Threshold slider
    threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum similarity score to consider faces as a match"
    )
    
    # API status
    st.sidebar.markdown("### 📊 API Status")
    if health_data:
        st.sidebar.success(f"✅ Server: {health_data.get('status', 'Unknown')}")
        st.sidebar.info(f"🕒 Timestamp: {health_data.get('timestamp', 'Unknown')}")
        st.sidebar.info(f"📦 Models: {'✅ Loaded' if health_data.get('models_loaded', False) else '❌ Not Loaded'}")
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Face Comparison", 
        "👤 Face Detection", 
        "🧬 Embedding Extraction",
        "📊 Batch Comparison",
        "ℹ️ About"
    ])
    
    # Tab 1: Face Comparison
    with tab1:
        st.header("🔍 Face Comparison")
        st.markdown("Compare two face images to determine if they belong to the same person.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Image 1")
            uploaded_file1 = st.file_uploader(
                "Upload first image", 
                type=['jpg', 'jpeg', 'png'], 
                key="image1"
            )
            
            if uploaded_file1:
                image1 = Image.open(uploaded_file1)
                st.image(image1, caption="Image 1", use_container_width=True)
        
        with col2:
            st.subheader("📸 Image 2")
            uploaded_file2 = st.file_uploader(
                "Upload second image", 
                type=['jpg', 'jpeg', 'png'], 
                key="image2"
            )
            
            if uploaded_file2:
                image2 = Image.open(uploaded_file2)
                st.image(image2, caption="Image 2", use_container_width=True)
        
        # Compare button
        if uploaded_file1 and uploaded_file2:
            if st.button("🔍 Compare Faces", type="primary"):
                with st.spinner("Comparing faces..."):
                    # Convert images to bytes
                    img1_bytes = uploaded_file1.getvalue()
                    img2_bytes = uploaded_file2.getvalue()
                    
                    # Call API
                    result = compare_faces_api(img1_bytes, img2_bytes, threshold)
                    
                    # Display results
                    if result.get("success", False):
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ Comparison completed successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Similarity Score", 
                                f"{result.get('similarity_score', 0):.3f}"
                            )
                        
                        with col2:
                            is_match = result.get('is_match', False)
                            st.metric(
                                "Match Status", 
                                "✅ Match" if is_match else "❌ No Match"
                            )
                        
                        with col3:
                            confidence = result.get('confidence', 'unknown')
                            st.metric(
                                "Confidence", 
                                confidence.title()
                            )
                        
                        # Progress bar for similarity
                        similarity = result.get('similarity_score', 0)
                        st.progress(similarity)
                        st.caption(f"Similarity: {similarity:.1%}")
                        
                        # Detailed results
                        st.markdown("### 📋 Detailed Results")
                        st.json(result)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error("❌ Comparison failed!")
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Face Detection
    with tab2:
        st.header("👤 Face Detection")
        st.markdown("Detect and analyze faces in images.")
        
        uploaded_file = st.file_uploader(
            "Upload image for face detection", 
            type=['jpg', 'jpeg', 'png'],
            key="detection_image"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                return_landmarks = st.checkbox("Return facial landmarks", value=True)
            
            with col2:
                return_embeddings = st.checkbox("Return face embeddings", value=False)
            
            if st.button("🔍 Detect Faces", type="primary"):
                with st.spinner("Detecting faces..."):
                    img_bytes = uploaded_file.getvalue()
                    result = detect_faces_api(img_bytes, return_landmarks, return_embeddings)
                    
                    if result.get("success", False):
                        st.success(f"✅ Detected {result.get('faces_detected', 0)} faces!")
                        
                        faces = result.get('faces', [])
                        if faces:
                            st.markdown("### 📊 Detection Results")
                            
                            for i, face in enumerate(faces):
                                with st.expander(f"Face {i+1}"):
                                    st.json(face)
                        
                        st.markdown("### 📋 Full Response")
                        st.json(result)
                    else:
                        st.error("❌ Face detection failed!")
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    # Tab 3: Embedding Extraction
    with tab3:
        st.header("🧬 Face Embedding Extraction")
        st.markdown("Extract unique face embeddings for advanced analysis.")
        
        uploaded_file = st.file_uploader(
            "Upload image for embedding extraction", 
            type=['jpg', 'jpeg', 'png'],
            key="embedding_image"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🧬 Extract Embedding", type="primary"):
                with st.spinner("Extracting face embedding..."):
                    img_bytes = uploaded_file.getvalue()
                    result = extract_embedding_api(img_bytes)
                    
                    if result.get("success", False):
                        st.success("✅ Embedding extracted successfully!")
                        
                        embedding = result.get('embedding', [])
                        if embedding:
                            st.markdown("### 📊 Embedding Statistics")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Vector Length", len(embedding))
                            
                            with col2:
                                st.metric("Min Value", f"{min(embedding):.4f}")
                            
                            with col3:
                                st.metric("Max Value", f"{max(embedding):.4f}")
                            
                            # Embedding visualization
                            st.markdown("### 📈 Embedding Visualization")
                            st.line_chart(embedding[:100])  # Show first 100 values
                            
                            # Embedding data
                            with st.expander("📋 Full Embedding Data"):
                                st.code(json.dumps(embedding[:10], indent=2) + "\n...", language="json")
                        
                        st.markdown("### 📋 Full Response")
                        st.json(result)
                    else:
                        st.error("❌ Embedding extraction failed!")
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    # Tab 4: Batch Comparison
    with tab4:
        st.header("📊 Batch Face Comparison")
        st.markdown("Compare one reference face against multiple images.")
        
        st.subheader("📸 Reference Image")
        reference_file = st.file_uploader(
            "Upload reference image", 
            type=['jpg', 'jpeg', 'png'],
            key="reference_image"
        )
        
        if reference_file:
            reference_image = Image.open(reference_file)
            st.image(reference_image, caption="Reference Image", use_container_width=True)
        
        st.subheader("📸 Comparison Images")
        comparison_files = st.file_uploader(
            "Upload comparison images", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="comparison_images"
        )
        
        if comparison_files:
            cols = st.columns(min(len(comparison_files), 3))
            for i, file in enumerate(comparison_files):
                with cols[i % 3]:
                    image = Image.open(file)
                    st.image(image, caption=f"Image {i+1}", use_container_width=True)
        
        if reference_file and comparison_files:
            if st.button("📊 Batch Compare", type="primary"):
                with st.spinner("Performing batch comparison..."):
                    ref_bytes = reference_file.getvalue()
                    comp_bytes = [f.getvalue() for f in comparison_files]
                    
                    result = batch_compare_api(ref_bytes, comp_bytes, threshold)
                    
                    if result.get("success", False):
                        st.success("✅ Batch comparison completed!")
                        
                        results = result.get('results', [])
                        if results:
                            st.markdown("### 📊 Comparison Results")
                            
                            for i, res in enumerate(results):
                                with st.expander(f"Image {i+1} Results"):
                                    st.metric("Similarity Score", f"{res.get('similarity_score', 0):.3f}")
                                    st.metric("Match Status", "✅ Match" if res.get('is_match', False) else "❌ No Match")
                                    st.metric("Confidence", res.get('confidence', 'unknown').title())
                        
                        st.markdown("### 📋 Full Response")
                        st.json(result)
                    else:
                        st.error("❌ Batch comparison failed!")
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    # Tab 5: About
    with tab5:
        st.header("ℹ️ About Twin-Seeker")
        
        st.markdown("""
        ### 🎯 What is Twin-Seeker?
        
        **Twin-Seeker** is an advanced face recognition system that enables you to:
        
        - **🔍 Compare faces** to determine if they belong to the same person
        - **👤 Detect faces** in images with detailed analysis
        - **🧬 Extract embeddings** for advanced face recognition applications
        - **📊 Batch compare** one face against multiple images
        
        ### 🛠️ Technology Stack
        
        - **Backend**: FastAPI with InsightFace
        - **Frontend**: Streamlit web interface
        - **Face Recognition**: ArcFace model for high accuracy
        - **Package Management**: UV for fast dependency management
        
        ### 🚀 Features
        
        - **High Accuracy**: State-of-the-art face recognition models
        - **Multiple Formats**: Support for various image formats
        - **Configurable Thresholds**: Adjustable similarity thresholds
        - **Real-time Processing**: Fast API responses
        - **Comprehensive Analysis**: Detailed face detection and comparison
        
        ### 📈 Use Cases
        
        1. **Identity Verification**: Verify if ID photos match current photos
        2. **Access Control**: Secure building/device access
        3. **Photo Organization**: Group photos by person
        4. **Duplicate Detection**: Find duplicate photos in libraries
        5. **Law Enforcement**: Match surveillance photos with databases
        
        ### 🔮 Future Features
        
        - **Celebrity Twin Finder**: Find your celebrity doppelgänger
        - **Vector Database Integration**: Large-scale face search
        - **Advanced Analytics**: Detailed face analysis reports
        - **Mobile App**: iOS and Android applications
        
        ### 📞 Support
        
        For issues or questions, please check the API documentation at:
        http://localhost:8000/docs
        """)

if __name__ == "__main__":
    main() 