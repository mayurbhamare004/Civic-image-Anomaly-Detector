#!/usr/bin/env python3
"""
Streamlit App for Civic Anomaly Detection
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from scripts.inference import CivicAnomalyDetector

# Page configuration
st.set_page_config(
    page_title="Civic Anomaly Detector",
    page_icon="🏙️",
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
    .detection-box {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f0f8ff;
    }
    .anomaly-count {
        font-size: 1.5rem;
        font-weight: bold;
        color: #d62728;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """Load the anomaly detector model"""
    try:
        return CivicAnomalyDetector()
    except Exception as e:
        st.error(f"Failed to load detector: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏙️ Civic Anomaly Detector</h1>', unsafe_allow_html=True)
    st.markdown("**Detect urban civic issues like potholes, garbage dumps, waterlogging, and more!**")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model settings
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Class filter
    st.sidebar.subheader("🎯 Detection Classes")
    class_names = [
        "pothole", "garbage_dump", "waterlogging", 
        "broken_streetlight", "damaged_sidewalk", "construction_debris"
    ]
    
    selected_classes = st.sidebar.multiselect(
        "Select classes to detect:",
        class_names,
        default=class_names,
        help="Choose which types of anomalies to detect"
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "📹 Camera", "📊 Statistics"])
    
    with tab1:
        st.header("Upload Image for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect civic anomalies"
        )
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🔍 Detection Results")
                
                # Process image
                with st.spinner("Analyzing image..."):
                    # Load detector
                    detector = load_detector()
                    
                    if detector is None:
                        st.error("❌ AI detector not available. Please check installation.")
                        st.info("Install dependencies: `pip install ultralytics opencv-python`")
                        result_image, detections = None, []
                    else:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            image.save(tmp_file.name)
                            
                            # Run inference
                            result_image, detections = detector.detect_anomalies(
                                tmp_file.name, 
                                conf_threshold=confidence_threshold
                            )
                            
                            # Clean up temp file
                            os.unlink(tmp_file.name)
                
                if result_image is not None:
                    # Convert BGR to RGB for display
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_image_rgb, use_column_width=True)
                    
                    # Display detection summary
                    if detections:
                        st.markdown(f'<div class="anomaly-count">🎯 Found {len(detections)} anomalies</div>', 
                                  unsafe_allow_html=True)
                        
                        # Group detections by class
                        class_counts = {}
                        for det in detections:
                            class_name = det['class_name']
                            if class_name in selected_classes:
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # Display detection details
                        st.subheader("📋 Detection Details")
                        for class_name, count in class_counts.items():
                            st.write(f"**{class_name.replace('_', ' ').title()}**: {count}")
                        
                        # Detailed detection list
                        with st.expander("🔍 Detailed Detection List"):
                            for i, det in enumerate(detections, 1):
                                if det['class_name'] in selected_classes:
                                    st.write(f"{i}. **{det['class_name'].replace('_', ' ').title()}** "
                                           f"(Confidence: {det['confidence']:.2f})")
                    else:
                        st.success("✅ No civic anomalies detected in this image!")
                else:
                    st.error("❌ Error processing image. Please try again.")
    
    with tab2:
        st.header("📹 Real-time Camera Detection")
        st.info("🚧 Camera functionality coming soon! This will allow real-time detection from your webcam.")
        
        # Placeholder for camera functionality
        if st.button("🎥 Start Camera Detection"):
            st.warning("Camera detection is not yet implemented. Coming in the next update!")
    
    with tab3:
        st.header("📊 Detection Statistics")
        
        # Sample statistics (in a real app, this would come from a database)
        st.subheader("🏆 Detection Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Images Processed", "1,234", "12")
        
        with col2:
            st.metric("Anomalies Detected", "567", "23")
        
        with col3:
            st.metric("Detection Accuracy", "94.2%", "1.2%")
        
        # Sample chart
        st.subheader("📈 Anomaly Distribution")
        
        import pandas as pd
        
        # Sample data
        anomaly_data = pd.DataFrame({
            'Anomaly Type': ['Potholes', 'Garbage Dumps', 'Waterlogging', 
                           'Broken Streetlights', 'Damaged Sidewalks', 'Construction Debris'],
            'Count': [45, 32, 28, 15, 22, 18]
        })
        
        st.bar_chart(anomaly_data.set_index('Anomaly Type'))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏙️ Civic Anomaly Detector | Built with YOLOv8 & Streamlit</p>
        <p>Help make cities better by detecting and reporting civic issues!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()