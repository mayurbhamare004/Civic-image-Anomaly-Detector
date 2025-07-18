#!/usr/bin/env python3
"""
Test script for civic anomaly detection
"""

import sys
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os

# Add app directory to path
sys.path.append('app')

try:
    from civic_detector import analyze_image_for_civic_issues, load_yolo_model
    print("✅ Successfully imported detection functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_test_image():
    """Create a test image with potential anomalies"""
    # Create a 640x480 test image
    width, height = 640, 480
    
    # Create base road-like image (gray)
    img = Image.new('RGB', (width, height), color=(100, 100, 100))
    draw = ImageDraw.Draw(img)
    
    # Add some road markings (white lines)
    draw.line([(width//2, 0), (width//2, height)], fill=(255, 255, 255), width=5)
    
    # Add a dark spot (potential pothole)
    draw.ellipse([200, 300, 280, 360], fill=(30, 30, 30))
    
    # Add some colorful clutter (potential garbage)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, color in enumerate(colors):
        x = 400 + i * 20
        y = 200 + i * 15
        draw.rectangle([x, y, x+15, y+15], fill=color)
    
    # Add blue area (potential water)
    draw.ellipse([100, 350, 180, 400], fill=(50, 100, 200))
    
    return img

def test_detection():
    """Test the detection system"""
    print("🧪 Testing Civic Anomaly Detection System")
    print("=" * 50)
    
    # Load model
    print("📥 Loading YOLO model...")
    model, model_loaded = load_yolo_model()
    
    if not model_loaded:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully")
    
    # Create test image
    print("🖼️  Creating test image...")
    test_image = create_test_image()
    
    # Save test image temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        test_image.save(tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        # Run detection
        print("🔍 Running anomaly detection...")
        detections = analyze_image_for_civic_issues(test_image, model, confidence=0.2)
        
        # Display results
        print(f"\n🎯 Detection Results:")
        print(f"Found {len(detections)} potential civic issues:")
        
        if detections:
            for i, det in enumerate(detections, 1):
                print(f"\n{i}. {det['type'].replace('_', ' ').title()}")
                print(f"   Confidence: {det['confidence']:.1%}")
                print(f"   Description: {det['description']}")
                print(f"   Location: {det['bbox']}")
        else:
            print("❌ No detections found!")
        
        # Test with different confidence levels
        print(f"\n📊 Testing different confidence levels:")
        for conf in [0.1, 0.3, 0.5, 0.7]:
            test_detections = analyze_image_for_civic_issues(test_image, model, confidence=conf)
            print(f"   Confidence {conf}: {len(test_detections)} detections")
        
    finally:
        # Clean up
        os.unlink(temp_path)
    
    print("\n✅ Detection test completed!")

if __name__ == "__main__":
    test_detection()