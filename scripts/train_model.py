#!/usr/bin/env python3
"""
Civic Anomaly Detector - Model Training Script
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def load_config(config_path="models/configs/config.yaml"):
    """Load training configuration"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "models/weights",
        "scripts/data/raw",
        "scripts/data/processed/train/images",
        "scripts/data/processed/train/labels",
        "scripts/data/processed/val/images", 
        "scripts/data/processed/val/labels",
        "scripts/data/processed/test/images",
        "scripts/data/processed/test/labels"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created successfully!")

def create_dataset_yaml(config):
    """Create dataset.yaml for YOLOv8"""
    dataset_config = {
        'path': str(Path.cwd() / 'scripts' / 'data' / 'processed'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': config['classes']['nc'],
        'names': config['classes']['names']
    }
    
    yaml_path = Path('scripts/data/processed/dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Dataset YAML created at: {yaml_path}")
    return str(yaml_path)

def train_model(config):
    """Train YOLOv8 model"""
    print("🚀 Starting model training...")
    
    # Initialize model
    model_name = config['model']['name']
    if config['model']['pretrained']:
        model = YOLO(f"{model_name}.pt")
        print(f"✅ Loaded pretrained {model_name}")
    else:
        model = YOLO(f"{model_name}.yaml")
        print(f"✅ Initialized {model_name} from scratch")
    
    # Training parameters
    train_params = {
        'data': config['data']['path'],
        'epochs': config['training']['epochs'],
        'batch': config['training']['batch_size'],
        'imgsz': config['training']['img_size'],
        'patience': config['training']['patience'],
        'save_period': config['training']['save_period'],
        'project': 'models/weights',
        'name': 'civic_anomaly_detector',
        'exist_ok': True,
        'pretrained': config['model']['pretrained'],
        'optimizer': 'SGD',
        'lr0': config['optimizer']['lr0'],
        'momentum': config['optimizer']['momentum'],
        'weight_decay': config['optimizer']['weight_decay'],
        'verbose': True
    }
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    try:
        # Start training
        results = model.train(**train_params)
        print("✅ Training completed successfully!")
        
        # Save final model
        model.save('models/weights/civic_detector_final.pt')
        print("💾 Model saved to models/weights/civic_detector_final.pt")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None

def validate_model(model_path="models/weights/civic_detector_final.pt"):
    """Validate trained model"""
    try:
        model = YOLO(model_path)
        results = model.val()
        print("✅ Model validation completed!")
        return results
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return None

def main():
    """Main training pipeline"""
    print("🏙️ Civic Image Anomaly Detector - Training Pipeline")
    print("=" * 50)
    
    # Setup
    setup_directories()
    config = load_config()
    
    # Create dataset configuration
    dataset_yaml = create_dataset_yaml(config)
    
    # Check if dataset exists
    train_images = Path('scripts/data/processed/train/images')
    if not train_images.exists() or not any(train_images.iterdir()):
        print("⚠️  No training images found!")
        print("📝 Please run the dataset collector first:")
        print("   python3 scripts/dataset_collector.py")
        print(f"\n📁 Expected location: {train_images}")
        return
    
    # Train model
    results = train_model(config)
    
    if results:
        print("\n🎉 Training Pipeline Completed!")
        print("📊 Check results in: models/weights/civic_anomaly_detector/")
        
        # Validate model
        print("\n🔍 Running model validation...")
        validate_model()

if __name__ == "__main__":
    main()