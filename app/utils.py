#!/usr/bin/env python3
"""
Utility functions for Civic Anomaly Detection
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    return image_b64

def decode_base64_to_image(image_b64: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    image_data = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def resize_image(image: np.ndarray, max_size: int = 640) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def draw_detection_boxes(image: np.ndarray, detections: List[Dict], 
                        class_colors: Optional[Dict] = None) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    
    if class_colors is None:
        class_colors = {
            0: (0, 0, 255),      # Red for potholes
            1: (0, 255, 0),      # Green for garbage
            2: (255, 0, 0),      # Blue for waterlogging
            3: (0, 255, 255),    # Yellow for streetlights
            4: (255, 0, 255),    # Magenta for sidewalks
            5: (255, 255, 0)     # Cyan for debris
        }
    
    result_image = image.copy()
    
    for det in detections:
        bbox = det['bbox']
        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Get color for this class
        color = class_colors.get(class_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Prepare label
        label = f"{class_name.replace('_', ' ').title()}: {confidence:.2f}"
        
        # Get label size
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(result_image, 
                     (bbox[0], bbox[1] - label_size[1] - 10),
                     (bbox[0] + label_size[0], bbox[1]), 
                     color, -1)
        
        # Draw label text
        cv2.putText(result_image, label, 
                   (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_image

def calculate_detection_stats(detections: List[Dict]) -> Dict:
    """Calculate statistics from detections"""
    if not detections:
        return {
            'total_count': 0,
            'class_counts': {},
            'avg_confidence': 0.0,
            'confidence_distribution': {}
        }
    
    # Count by class
    class_counts = {}
    confidences = []
    
    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidences.append(confidence)
    
    # Confidence distribution
    confidence_ranges = {
        'high (>0.8)': len([c for c in confidences if c > 0.8]),
        'medium (0.5-0.8)': len([c for c in confidences if 0.5 <= c <= 0.8]),
        'low (<0.5)': len([c for c in confidences if c < 0.5])
    }
    
    return {
        'total_count': len(detections),
        'class_counts': class_counts,
        'avg_confidence': np.mean(confidences) if confidences else 0.0,
        'confidence_distribution': confidence_ranges
    }

def save_detection_results(image_path: str, detections: List[Dict], 
                          output_dir: str = "results") -> str:
    """Save detection results to JSON file"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Prepare results data
    results = {
        'image_path': str(image_path),
        'timestamp': str(np.datetime64('now')),
        'total_detections': len(detections),
        'detections': detections,
        'statistics': calculate_detection_stats(detections)
    }
    
    # Generate output filename
    image_name = Path(image_path).stem
    output_file = output_path / f"{image_name}_results.json"
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return str(output_file)

def create_detection_report(detections: List[Dict], image_path: str) -> str:
    """Create a formatted text report of detections"""
    
    if not detections:
        return f"No civic anomalies detected in {Path(image_path).name}"
    
    stats = calculate_detection_stats(detections)
    
    report = f"""
CIVIC ANOMALY DETECTION REPORT
==============================

Image: {Path(image_path).name}
Total Anomalies Detected: {stats['total_count']}
Average Confidence: {stats['avg_confidence']:.2f}

DETECTED ANOMALIES:
"""
    
    for i, det in enumerate(detections, 1):
        report += f"""
{i}. {det['class_name'].replace('_', ' ').title()}
   Confidence: {det['confidence']:.2f}
   Location: ({det['bbox'][0]}, {det['bbox'][1]}) to ({det['bbox'][2]}, {det['bbox'][3]})
"""
    
    report += f"""
SUMMARY BY TYPE:
"""
    for class_name, count in stats['class_counts'].items():
        report += f"- {class_name.replace('_', ' ').title()}: {count}\n"
    
    report += f"""
CONFIDENCE DISTRIBUTION:
- High confidence (>0.8): {stats['confidence_distribution']['high (>0.8)']}
- Medium confidence (0.5-0.8): {stats['confidence_distribution']['medium (0.5-0.8)']}
- Low confidence (<0.5): {stats['confidence_distribution']['low (<0.5)']}
"""
    
    return report

def validate_image_file(file_path: str) -> bool:
    """Validate if file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_image_info(image_path: str) -> Dict:
    """Get basic information about an image"""
    try:
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'size_mb': Path(image_path).stat().st_size / (1024 * 1024)
            }
    except Exception as e:
        return {'error': str(e)}

class DetectionLogger:
    """Logger for detection results"""
    
    def __init__(self, log_file: str = "detection_log.json"):
        self.log_file = Path(log_file)
        self.log_data = self._load_log()
    
    def _load_log(self) -> List[Dict]:
        """Load existing log data"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def log_detection(self, image_path: str, detections: List[Dict], 
                     processing_time: float = 0.0):
        """Log a detection result"""
        entry = {
            'timestamp': str(np.datetime64('now')),
            'image_path': str(image_path),
            'detection_count': len(detections),
            'detections': detections,
            'processing_time': processing_time,
            'statistics': calculate_detection_stats(detections)
        }
        
        self.log_data.append(entry)
        self._save_log()
    
    def _save_log(self):
        """Save log data to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2)
        except Exception as e:
            print(f"Error saving log: {e}")
    
    def get_stats(self) -> Dict:
        """Get overall statistics from log"""
        if not self.log_data:
            return {}
        
        total_images = len(self.log_data)
        total_detections = sum(entry['detection_count'] for entry in self.log_data)
        avg_processing_time = np.mean([entry.get('processing_time', 0) for entry in self.log_data])
        
        # Most common anomaly
        all_detections = []
        for entry in self.log_data:
            all_detections.extend(entry['detections'])
        
        class_counts = {}
        for det in all_detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        most_common = max(class_counts.items(), key=lambda x: x[1]) if class_counts else ('none', 0)
        
        return {
            'total_images_processed': total_images,
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / total_images if total_images > 0 else 0,
            'average_processing_time': avg_processing_time,
            'most_common_anomaly': most_common[0],
            'class_distribution': class_counts
        }