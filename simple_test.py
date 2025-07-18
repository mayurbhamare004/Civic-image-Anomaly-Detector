#!/usr/bin/env python3
"""
Simple test for civic anomaly detection without Streamlit dependencies
"""

import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os

try:
    from ultralytics import YOLO
    print("✅ YOLO available")
except ImportError:
    print("❌ YOLO not available - install with: pip install ultralytics")
    exit(1)

def simple_civic_detection(image_path, confidence=0.3):
    """Simple civic anomaly detection"""
    
    # Load YOLO model
    model = YOLO('yolov8n.pt')
    
    # Load image
    image = Image.open(image_path)
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    print(f"📸 Analyzing image: {width}x{height}")
    
    # Run YOLO detection
    results = model(image_path, conf=confidence)
    yolo_detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                yolo_detections.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': bbox
                })
    
    # Simple image analysis for civic issues
    gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    civic_detections = []
    
    # Look for dark spots (potential potholes)
    road_area = gray[height//2:, :]  # Focus on lower half
    if road_area.size > 0:
        road_mean = np.mean(road_area)
        dark_threshold = road_mean - np.std(road_area) * 1.5
        dark_mask = road_area < dark_threshold
        
        if np.sum(dark_mask) > (road_area.size * 0.02):  # >2% dark
            dark_coords = np.where(dark_mask)
            if len(dark_coords[0]) > 0:
                center_y = dark_coords[0][0] + height//2
                center_x = dark_coords[1][0]
                
                civic_detections.append({
                    'type': 'pothole',
                    'confidence': 0.75,
                    'bbox': [max(0, center_x-50), max(0, center_y-40), 
                            min(width, center_x+50), min(height, center_y+40)],
                    'description': 'Dark road area detected'
                })
    
    # Look for high color variation (potential garbage)
    if len(img_array.shape) == 3:
        color_std = np.mean(np.std(img_array, axis=(0,1)))
        if color_std > 30:
            civic_detections.append({
                'type': 'garbage_area',
                'confidence': min(0.8, color_std / 40),
                'bbox': [width//4, height//4, 3*width//4, 3*height//4],
                'description': f'High color variation detected (std: {color_std:.1f})'
            })
    
    # Map YOLO detections to civic context
    civic_mapping = {
        'car': 'parked_vehicle',
        'truck': 'heavy_vehicle', 
        'traffic light': 'traffic_infrastructure',
        'stop sign': 'road_signage',
        'person': 'pedestrian_area'
    }
    
    for det in yolo_detections:
        if det['class_name'] in civic_mapping:
            civic_detections.append({
                'type': civic_mapping[det['class_name']],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'description': f"{det['class_name']} detected",
                'original_yolo': det['class_name']
            })
    
    return civic_detections, yolo_detections

def create_test_image():
    """Create test image with civic issues"""
    img = Image.new('RGB', (640, 480), color=(120, 120, 120))
    draw = ImageDraw.Draw(img)
    
    # Road markings
    draw.line([(320, 0), (320, 480)], fill=(255, 255, 255), width=4)
    
    # Dark pothole
    draw.ellipse([200, 300, 280, 360], fill=(20, 20, 20))
    
    # Colorful garbage area
    colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
    for i, color in enumerate(colors):
        x, y = 400 + i*15, 200 + i*10
        draw.rectangle([x, y, x+12, y+12], fill=color)
    
    return img

def main():
    print("🏙️ Simple Civic Anomaly Detection Test")
    print("=" * 40)
    
    # Create and save test image
    test_img = create_test_image()
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_img.save(tmp.name)
        temp_path = tmp.name
    
    try:
        # Run detection
        civic_detections, yolo_detections = simple_civic_detection(temp_path, confidence=0.2)
        
        print(f"\n🎯 Results:")
        print(f"YOLO detections: {len(yolo_detections)}")
        print(f"Civic issues found: {len(civic_detections)}")
        
        if civic_detections:
            print(f"\n📋 Civic Issues Detected:")
            for i, det in enumerate(civic_detections, 1):
                print(f"{i}. {det['type'].replace('_', ' ').title()}")
                print(f"   Confidence: {det['confidence']:.1%}")
                print(f"   {det['description']}")
        
        if yolo_detections:
            print(f"\n🔍 YOLO Objects Detected:")
            for det in yolo_detections:
                print(f"- {det['class_name']}: {det['confidence']:.1%}")
        
        if not civic_detections and not yolo_detections:
            print("❌ No detections found - try lowering confidence threshold")
        
    finally:
        os.unlink(temp_path)
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    main()