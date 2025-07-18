#!/usr/bin/env python3
"""
Dataset Collection Script for Civic Anomaly Detection
This script helps collect and organize datasets from various sources
"""

import os
import requests
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml
from tqdm import tqdm
import random

class CivicDatasetCollector:
    def __init__(self, base_dir="scripts/data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories
        for split in ['train', 'val', 'test']:
            (self.processed_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Class mapping
        self.classes = {
            0: "pothole",
            1: "garbage_dump", 
            2: "waterlogging",
            3: "broken_streetlight",
            4: "damaged_sidewalk",
            5: "construction_debris"
        }
    
    def download_roboflow_dataset(self, api_key=None, workspace=None, project=None, version=1):
        """Download dataset from Roboflow"""
        if not api_key:
            print("⚠️  Roboflow API key required. Get one from https://roboflow.com")
            print("Set environment variable: export ROBOFLOW_API_KEY='your_key'")
            return False
        
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace).project(project)
            dataset = project.version(version).download("yolov8", location=str(self.raw_dir))
            
            print(f"✅ Downloaded dataset to {self.raw_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Roboflow download failed: {e}")
            return False
    
    def create_synthetic_dataset(self, num_images=100):
        """Create synthetic dataset for testing"""
        print(f"🎨 Creating {num_images} synthetic images...")
        
        synthetic_dir = self.raw_dir / "synthetic"
        synthetic_dir.mkdir(exist_ok=True)
        
        for i in tqdm(range(num_images), desc="Generating images"):
            # Create base road image
            width, height = 640, 480
            img = self.create_synthetic_road_image(width, height)
            
            # Add random anomalies
            annotations = []
            
            # Random chance for each anomaly type
            if random.random() < 0.4:  # 40% chance of pothole
                bbox, class_id = self.add_pothole(img, width, height)
                if bbox:
                    annotations.append((class_id, *bbox))
            
            if random.random() < 0.3:  # 30% chance of garbage
                bbox, class_id = self.add_garbage(img, width, height)
                if bbox:
                    annotations.append((class_id, *bbox))
            
            if random.random() < 0.2:  # 20% chance of water
                bbox, class_id = self.add_water(img, width, height)
                if bbox:
                    annotations.append((class_id, *bbox))
            
            # Save image
            img_path = synthetic_dir / f"synthetic_{i:04d}.jpg"
            img.save(img_path)
            
            # Save annotations in YOLO format
            if annotations:
                label_path = synthetic_dir / f"synthetic_{i:04d}.txt"
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        class_id, cx, cy, w, h = ann
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        print(f"✅ Created {num_images} synthetic images with annotations")
        return True
    
    def create_synthetic_road_image(self, width, height):
        """Create a realistic road background"""
        # Base asphalt color with variation
        base_color = (80 + random.randint(-20, 20), 
                     80 + random.randint(-20, 20), 
                     80 + random.randint(-20, 20))
        
        img = Image.new('RGB', (width, height), base_color)
        draw = ImageDraw.Draw(img)
        
        # Add road texture
        for _ in range(200):
            x = random.randint(0, width)
            y = random.randint(0, height)
            color = (base_color[0] + random.randint(-10, 10),
                    base_color[1] + random.randint(-10, 10),
                    base_color[2] + random.randint(-10, 10))
            draw.point((x, y), fill=color)
        
        # Add road markings (sometimes)
        if random.random() < 0.6:
            # Center line
            line_y = height // 2 + random.randint(-50, 50)
            draw.line([(0, line_y), (width, line_y)], fill=(255, 255, 255), width=3)
        
        if random.random() < 0.4:
            # Side lines
            draw.line([(width//4, 0), (width//4, height)], fill=(255, 255, 255), width=2)
            draw.line([(3*width//4, 0), (3*width//4, height)], fill=(255, 255, 255), width=2)
        
        return img
    
    def add_pothole(self, img, width, height):
        """Add a pothole to the image"""
        draw = ImageDraw.Draw(img)
        
        # Random pothole size and position
        pothole_w = random.randint(30, 100)
        pothole_h = random.randint(25, 80)
        x = random.randint(pothole_w//2, width - pothole_w//2)
        y = random.randint(height//2, height - pothole_h//2)  # Lower half of image
        
        # Dark irregular shape
        dark_color = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
        
        # Main pothole shape
        draw.ellipse([x - pothole_w//2, y - pothole_h//2, 
                     x + pothole_w//2, y + pothole_h//2], fill=dark_color)
        
        # Add some irregular edges
        for _ in range(5):
            offset_x = random.randint(-pothole_w//4, pothole_w//4)
            offset_y = random.randint(-pothole_h//4, pothole_h//4)
            small_w = random.randint(10, 20)
            small_h = random.randint(8, 15)
            
            draw.ellipse([x + offset_x - small_w//2, y + offset_y - small_h//2,
                         x + offset_x + small_w//2, y + offset_y + small_h//2], 
                        fill=dark_color)
        
        # Convert to YOLO format (normalized)
        cx = x / width
        cy = y / height
        w = pothole_w / width
        h = pothole_h / height
        
        return (cx, cy, w, h), 0  # Class 0 = pothole
    
    def add_garbage(self, img, width, height):
        """Add garbage/clutter to the image"""
        draw = ImageDraw.Draw(img)
        
        # Random garbage area
        garbage_w = random.randint(40, 120)
        garbage_h = random.randint(30, 80)
        x = random.randint(garbage_w//2, width - garbage_w//2)
        y = random.randint(garbage_h//2, height - garbage_h//2)
        
        # Multiple colorful objects
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 128, 128)]
        
        for _ in range(random.randint(3, 8)):
            obj_size = random.randint(5, 15)
            obj_x = x + random.randint(-garbage_w//2, garbage_w//2)
            obj_y = y + random.randint(-garbage_h//2, garbage_h//2)
            color = random.choice(colors)
            
            if random.random() < 0.5:
                # Rectangle
                draw.rectangle([obj_x - obj_size//2, obj_y - obj_size//2,
                               obj_x + obj_size//2, obj_y + obj_size//2], fill=color)
            else:
                # Circle
                draw.ellipse([obj_x - obj_size//2, obj_y - obj_size//2,
                             obj_x + obj_size//2, obj_y + obj_size//2], fill=color)
        
        # Convert to YOLO format
        cx = x / width
        cy = y / height
        w = garbage_w / width
        h = garbage_h / height
        
        return (cx, cy, w, h), 1  # Class 1 = garbage_dump
    
    def add_water(self, img, width, height):
        """Add water/flooding to the image"""
        draw = ImageDraw.Draw(img)
        
        # Random water area
        water_w = random.randint(60, 150)
        water_h = random.randint(30, 70)
        x = random.randint(water_w//2, width - water_w//2)
        y = random.randint(height//2, height - water_h//2)  # Lower half
        
        # Blue-ish water color with some transparency effect
        water_colors = [(50, 100, 200), (30, 80, 180), (70, 120, 220)]
        
        # Main water body
        main_color = random.choice(water_colors)
        draw.ellipse([x - water_w//2, y - water_h//2,
                     x + water_w//2, y + water_h//2], fill=main_color)
        
        # Add some reflective spots
        for _ in range(3):
            spot_size = random.randint(5, 15)
            spot_x = x + random.randint(-water_w//3, water_w//3)
            spot_y = y + random.randint(-water_h//3, water_h//3)
            bright_color = (min(255, main_color[0] + 50),
                           min(255, main_color[1] + 50),
                           min(255, main_color[2] + 50))
            
            draw.ellipse([spot_x - spot_size//2, spot_y - spot_size//2,
                         spot_x + spot_size//2, spot_y + spot_size//2], 
                        fill=bright_color)
        
        # Convert to YOLO format
        cx = x / width
        cy = y / height
        w = water_w / width
        h = water_h / height
        
        return (cx, cy, w, h), 2  # Class 2 = waterlogging
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train/val/test"""
        print("📊 Splitting dataset...")
        
        # Find all images in raw directory
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(self.raw_dir.rglob(f'*{ext}')))
            image_files.extend(list(self.raw_dir.rglob(f'*{ext.upper()}')))
        
        if not image_files:
            print("❌ No images found in raw directory")
            return False
        
        print(f"📸 Found {len(image_files)} images")
        
        # Shuffle and split
        random.shuffle(image_files)
        
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }
        
        # Copy files to respective directories
        for split_name, split_files in splits.items():
            print(f"📁 Processing {split_name}: {len(split_files)} images")
            
            for img_path in tqdm(split_files, desc=f"Copying {split_name}"):
                # Copy image
                dst_img = self.processed_dir / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Copy corresponding label if exists
                label_path = img_path.with_suffix('.txt')
                if label_path.exists():
                    dst_label = self.processed_dir / split_name / 'labels' / label_path.name
                    shutil.copy2(label_path, dst_label)
        
        print("✅ Dataset split completed!")
        return True
    
    def create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training"""
        dataset_config = {
            'path': str(self.processed_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.processed_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Dataset YAML created: {yaml_path}")
        return str(yaml_path)
    
    def validate_dataset(self):
        """Validate the prepared dataset"""
        print("🔍 Validating dataset...")
        
        issues = []
        stats = {'train': 0, 'val': 0, 'test': 0}
        class_counts = {i: 0 for i in range(len(self.classes))}
        
        for split in ['train', 'val', 'test']:
            img_dir = self.processed_dir / split / 'images'
            lbl_dir = self.processed_dir / split / 'labels'
            
            if not img_dir.exists():
                issues.append(f"Missing {split}/images directory")
                continue
            
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            stats[split] = len(images)
            
            for img_path in images:
                label_path = lbl_dir / f"{img_path.stem}.txt"
                
                if not label_path.exists():
                    issues.append(f"Missing label for {img_path.name}")
                    continue
                
                # Validate label format
                try:
                    with open(label_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) != 5:
                                issues.append(f"{label_path.name}:{line_num} - Invalid format")
                                continue
                            
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]
                            
                            if class_id < 0 or class_id >= len(self.classes):
                                issues.append(f"{label_path.name}:{line_num} - Invalid class: {class_id}")
                            else:
                                class_counts[class_id] += 1
                            
                            for coord in coords:
                                if coord < 0 or coord > 1:
                                    issues.append(f"{label_path.name}:{line_num} - Invalid coordinate: {coord}")
                
                except Exception as e:
                    issues.append(f"Error reading {label_path.name}: {e}")
        
        # Print results
        print(f"\n📊 Dataset Statistics:")
        for split, count in stats.items():
            print(f"  {split}: {count} images")
        
        print(f"\n📈 Class Distribution:")
        for class_id, count in class_counts.items():
            class_name = self.classes[class_id]
            print(f"  {class_name}: {count} instances")
        
        if issues:
            print(f"\n⚠️  Found {len(issues)} issues:")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("\n✅ Dataset validation passed!")
        
        return len(issues) == 0

def main():
    """Main dataset collection pipeline"""
    print("🏙️ Civic Anomaly Dataset Collector")
    print("=" * 40)
    
    collector = CivicDatasetCollector()
    
    # Check for existing data
    existing_images = list(collector.raw_dir.rglob('*.jpg')) + list(collector.raw_dir.rglob('*.png'))
    
    if existing_images:
        print(f"📸 Found {len(existing_images)} existing images")
        use_existing = input("Use existing images? (y/n): ").lower() == 'y'
    else:
        use_existing = False
    
    if not use_existing:
        # Try Roboflow first
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if api_key:
            print("🔄 Attempting Roboflow download...")
            success = collector.download_roboflow_dataset(
                api_key=api_key,
                workspace="your-workspace",  # Update these
                project="civic-anomalies",
                version=1
            )
            if not success:
                print("🎨 Creating synthetic dataset instead...")
                collector.create_synthetic_dataset(200)
        else:
            print("🎨 Creating synthetic dataset...")
            collector.create_synthetic_dataset(200)
    
    # Split dataset
    collector.split_dataset()
    
    # Create YAML config
    collector.create_dataset_yaml()
    
    # Validate
    collector.validate_dataset()
    
    print("\n✅ Dataset preparation completed!")
    print("🚀 Ready for training! Run: python3 scripts/train_model.py")

if __name__ == "__main__":
    main()