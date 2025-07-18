#!/usr/bin/env python3
"""
Inference Script for Civic Anomaly Detection
"""

import numpy as np
from pathlib import Path
import argparse

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available. Some features may be limited.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO not available. Please install with: pip install ultralytics")

class CivicAnomalyDetector:
    def __init__(self, model_path="models/weights/civic_detector_final.pt"):
        """Initialize the detector with trained model"""
        self.model_path = model_path
        self.model = None
        self.class_names = {
            0: "pothole",
            1: "garbage_dump",
            2: "waterlogging", 
            3: "broken_streetlight",
            4: "damaged_sidewalk",
            5: "construction_debris"
        }
        self.colors = {
            0: (0, 0, 255),      # Red for potholes
            1: (0, 255, 0),      # Green for garbage
            2: (255, 0, 0),      # Blue for waterlogging
            3: (0, 255, 255),    # Yellow for streetlights
            4: (255, 0, 255),    # Magenta for sidewalks
            5: (255, 255, 0)     # Cyan for debris
        }
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLO model"""
        if not YOLO_AVAILABLE:
            print("❌ YOLO not available. Please install ultralytics.")
            self.model = None
            return
            
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                print(f"✅ Model loaded from: {self.model_path}")
            else:
                print(f"⚠️  Model not found at: {self.model_path}")
                print("🔄 Loading pretrained YOLOv8n as fallback...")
                self.model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def detect_anomalies(self, image_path, conf_threshold=0.5):
        """Detect civic anomalies in an image"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None, None
        
        if not CV2_AVAILABLE:
            print("❌ OpenCV not available!")
            return None, None
        
        try:
            # Run inference
            results = self.model(image_path, conf=conf_threshold)
            
            # Load original image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None, None
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Store detection info
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, 'unknown')
                        }
                        detections.append(detection)
                        
                        # Draw bounding box
                        color = self.colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        label = f"{self.class_names.get(class_id, 'unknown')}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                                    (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(image, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return image, detections
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return None, None
    
    def process_video(self, video_path, output_path=None, conf_threshold=0.5):
        """Process video for anomaly detection"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        print("🎥 Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Save frame temporarily for inference
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Run detection
            result_frame, detections = self.detect_anomalies(temp_path, conf_threshold)
            
            if result_frame is not None:
                if detections:
                    total_detections += len(detections)
                
                # Write frame to output video
                if out:
                    out.write(result_frame)
                
                # Display frame (optional)
                cv2.imshow('Civic Anomaly Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Remove temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        
        print(f"✅ Video processing complete!")
        print(f"📊 Processed {frame_count} frames")
        print(f"🎯 Total detections: {total_detections}")

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Civic Anomaly Detection Inference')
    parser.add_argument('--input', '-i', required=True, help='Input image or video path')
    parser.add_argument('--output', '-o', help='Output path for results')
    parser.add_argument('--model', '-m', default='models/weights/civic_detector_final.pt', 
                       help='Model path')
    parser.add_argument('--conf', '-c', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Show results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CivicAnomalyDetector(args.model)
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Check if input is image or video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    if input_path.suffix.lower() in image_extensions:
        # Process image
        print(f"🖼️  Processing image: {input_path}")
        result_image, detections = detector.detect_anomalies(input_path, args.conf)
        
        if result_image is not None:
            print(f"🎯 Found {len(detections)} anomalies:")
            for det in detections:
                print(f"   - {det['class_name']}: {det['confidence']:.2f}")
            
            # Save result
            if args.output:
                cv2.imwrite(args.output, result_image)
                print(f"💾 Result saved to: {args.output}")
            
            # Show result
            if args.show:
                cv2.imshow('Civic Anomaly Detection', result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    elif input_path.suffix.lower() in video_extensions:
        # Process video
        print(f"🎥 Processing video: {input_path}")
        detector.process_video(input_path, args.output, args.conf)
    
    else:
        print(f"❌ Unsupported file format: {input_path.suffix}")

if __name__ == "__main__":
    main()