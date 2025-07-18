#!/usr/bin/env python3
"""
FastAPI Backend for Civic Anomaly Detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import sys
from typing import List, Dict, Any
import base64
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.inference import CivicAnomalyDetector

# Initialize FastAPI app
app = FastAPI(
    title="Civic Anomaly Detector API",
    description="API for detecting urban civic issues using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize the detector on startup"""
    global detector
    detector = CivicAnomalyDetector()
    print("✅ Civic Anomaly Detector API started!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Civic Anomaly Detector API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None
    }

@app.get("/classes")
async def get_classes():
    """Get available detection classes"""
    return {
        "classes": [
            {"id": 0, "name": "pothole"},
            {"id": 1, "name": "garbage_dump"},
            {"id": 2, "name": "waterlogging"},
            {"id": 3, "name": "broken_streetlight"},
            {"id": 4, "name": "damaged_sidewalk"},
            {"id": 5, "name": "construction_debris"}
        ]
    }

@app.post("/detect")
async def detect_anomalies(
    file: UploadFile = File(...),
    confidence: float = 0.5
):
    """
    Detect civic anomalies in uploaded image
    
    Args:
        file: Image file to analyze
        confidence: Confidence threshold for detections (0.1-1.0)
    
    Returns:
        JSON response with detection results
    """
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate confidence threshold
    if not 0.1 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.1 and 1.0")
    
    try:
        # Read and save uploaded file temporarily
        contents = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
        
        # Run detection
        result_image, detections = detector.detect_anomalies(
            tmp_file_path, 
            conf_threshold=confidence
        )
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        if result_image is None:
            raise HTTPException(status_code=500, detail="Error processing image")
        
        # Convert result image to base64 for response
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Process detections
        processed_detections = []
        class_counts = {}
        
        for det in detections:
            processed_det = {
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": round(det["confidence"], 3),
                "bbox": {
                    "x1": det["bbox"][0],
                    "y1": det["bbox"][1], 
                    "x2": det["bbox"][2],
                    "y2": det["bbox"][3]
                }
            }
            processed_detections.append(processed_det)
            
            # Count by class
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "success": True,
            "filename": file.filename,
            "total_detections": len(detections),
            "detections": processed_detections,
            "class_counts": class_counts,
            "result_image": result_image_b64,
            "parameters": {
                "confidence_threshold": confidence
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence: float = 0.5
):
    """
    Detect anomalies in multiple images
    
    Args:
        files: List of image files to analyze
        confidence: Confidence threshold for detections
    
    Returns:
        JSON response with batch detection results
    """
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    total_detections = 0
    overall_class_counts = {}
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue
            
            # Process file
            contents = await file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(contents)
                tmp_file_path = tmp_file.name
            
            # Run detection
            result_image, detections = detector.detect_anomalies(
                tmp_file_path, 
                conf_threshold=confidence
            )
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            if result_image is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Error processing image"
                })
                continue
            
            # Process detections
            processed_detections = []
            class_counts = {}
            
            for det in detections:
                processed_det = {
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": round(det["confidence"], 3),
                    "bbox": {
                        "x1": det["bbox"][0],
                        "y1": det["bbox"][1],
                        "x2": det["bbox"][2], 
                        "y2": det["bbox"][3]
                    }
                }
                processed_detections.append(processed_det)
                
                # Count by class
                class_name = det["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                overall_class_counts[class_name] = overall_class_counts.get(class_name, 0) + 1
            
            total_detections += len(detections)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": processed_detections,
                "detection_count": len(detections),
                "class_counts": class_counts
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "batch_size": len(files),
        "total_detections": total_detections,
        "overall_class_counts": overall_class_counts,
        "results": results,
        "parameters": {
            "confidence_threshold": confidence
        }
    }

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    # In a real application, this would come from a database
    return {
        "total_requests": 1234,
        "total_images_processed": 5678,
        "total_anomalies_detected": 2345,
        "average_detections_per_image": 2.1,
        "most_common_anomaly": "pothole",
        "api_uptime": "99.9%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)