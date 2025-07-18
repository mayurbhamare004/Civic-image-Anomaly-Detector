#!/usr/bin/env python3
"""
Data Preparation Script for Civic Anomaly Detection
"""

import os
import shutil
import random
from pathlib import Path
from roboflow import Roboflow
import yaml

def download_sample_dataset():
    """Download sample civic anomaly dataset from Roboflow"""
    print("📥 Downloading sample dataset from Roboflow...")
    
    try:
        # Initialize Roboflow (you'll need to get your API key)
        # rf = Roboflow(api_key="YOUR_API_KEY")
        # project = rf.workspace("civic-anomalies").project("urban-issues")
        # dataset = project.version(1).download("yolov8")
        
        print("⚠️  To download datasets from Roboflow:")
        print("1. Sign up at https://roboflow.com")
        print("2. Get your API key")
        print("3. Search for civic/urban anomaly datasets")
        print("4. Update this script with your API key and project details")
        
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")

def create_sample_annotations():
    """Create sample annotation files for testing"""
    print("📝 Creating sample annotation structure...")
    
    # Sample class mapping
    classes = {
        0: "pothole",
        1: "garbage_dump", 
        2: "waterlogging",
        3: "broken_streetlight",
        4: "damaged_sidewalk",
        5: "construction_debris"
    }
    
    # Create sample annotation files
    for split in ['train', 'val', 'test']:
        labels_dir = Path(f'data/processed/{split}/labels')
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample annotation file
        sample_annotation = labels_dir / 'sample.txt'
        with open(sample_annotation, 'w') as f:
            # Sample YOLO format: class_id center_x center_y width height
            f.write("0 0.5 0.5 0.2 0.3\n")  # Sample pothole annotation
        
    print("✅ Sample annotation structure created!")

def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split dataset into train/val/test"""
    print(f"📊 Splitting dataset: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(source_path.glob(f'*{ext}'))
        images.extend(source_path.glob(f'*{ext.upper()}'))
    
    if not images:
        print("❌ No images found in source directory")
        return
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate split indices
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split images
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    # Copy files to respective directories
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, split_images in splits.items():
        img_dir = Path(f'data/processed/{split_name}/images')
        lbl_dir = Path(f'data/processed/{split_name}/labels')
        
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in split_images:
            # Copy image
            shutil.copy2(img_path, img_dir / img_path.name)
            
            # Copy corresponding label file if exists
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy2(label_path, lbl_dir / label_path.name)
        
        print(f"✅ {split_name}: {len(split_images)} images")

def validate_annotations(data_dir='data/processed'):
    """Validate YOLO format annotations"""
    print("🔍 Validating annotations...")
    
    issues = []
    total_annotations = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = Path(data_dir) / split / 'labels'
        images_dir = Path(data_dir) / split / 'images'
        
        if not labels_dir.exists():
            continue
            
        for label_file in labels_dir.glob('*.txt'):
            total_annotations += 1
            
            # Check if corresponding image exists
            img_name = label_file.stem
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            img_exists = any((images_dir / f"{img_name}{ext}").exists() for ext in img_extensions)
            
            if not img_exists:
                issues.append(f"Missing image for {label_file}")
                continue
            
            # Validate annotation format
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        issues.append(f"{label_file}:{line_num} - Invalid format (expected 5 values)")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        # Check class ID range
                        if class_id < 0 or class_id > 5:
                            issues.append(f"{label_file}:{line_num} - Invalid class ID: {class_id}")
                        
                        # Check coordinate ranges (should be 0-1)
                        for i, coord in enumerate(coords):
                            if coord < 0 or coord > 1:
                                issues.append(f"{label_file}:{line_num} - Invalid coordinate: {coord}")
                                
                    except ValueError:
                        issues.append(f"{label_file}:{line_num} - Invalid number format")
                        
            except Exception as e:
                issues.append(f"Error reading {label_file}: {e}")
    
    print(f"📊 Validation Results:")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Issues found: {len(issues)}")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more issues")
    else:
        print("✅ All annotations are valid!")

def main():
    """Main data preparation pipeline"""
    print("🏙️ Civic Anomaly Detector - Data Preparation")
    print("=" * 50)
    
    # Create directory structure
    print("📁 Setting up directory structure...")
    dirs = [
        "data/raw",
        "data/processed/train/images",
        "data/processed/train/labels",
        "data/processed/val/images",
        "data/processed/val/labels", 
        "data/processed/test/images",
        "data/processed/test/labels"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create sample annotations for testing
    create_sample_annotations()
    
    # Instructions for users
    print("\n📝 Next Steps:")
    print("1. Add your images to 'data/raw/' directory")
    print("2. Annotate images using:")
    print("   - Roboflow: https://roboflow.com")
    print("   - LabelImg: https://github.com/tzutalin/labelImg")
    print("   - CVAT: https://cvat.org")
    print("3. Export annotations in YOLO format")
    print("4. Run this script again to split and validate data")
    
    # Check if raw data exists
    raw_dir = Path('data/raw')
    if any(raw_dir.iterdir()):
        print(f"\n📊 Found data in {raw_dir}")
        response = input("Split dataset? (y/n): ")
        if response.lower() == 'y':
            split_dataset('data/raw')
            validate_annotations()
    
    print("\n✅ Data preparation setup complete!")

if __name__ == "__main__":
    main()