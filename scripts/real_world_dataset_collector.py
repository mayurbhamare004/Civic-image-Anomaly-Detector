#!/usr/bin/env python3
"""
Real-World Dataset Collector for Civic Anomaly Detection
This script downloads and integrates real-world datasets for better model performance
"""

import os
import requests
import json
import zipfile
import tarfile
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
import random
import urllib.request
from urllib.parse import urlparse
import kaggle

class RealWorldDatasetCollector:
    def __init__(self, base_dir="scripts/data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.real_world_dir = self.raw_dir / "real_world"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories
        self.real_world_dir.mkdir(parents=True, exist_ok=True)
        
        # Class mapping for civic anomalies
        self.classes = {
            0: "pothole",
            1: "garbage_dump", 
            2: "waterlogging",
            3: "broken_streetlight",
            4: "damaged_sidewalk",
            5: "construction_debris"
        }
        
        # Dataset sources configuration
        self.dataset_sources = {
            'potholes': {
                'kaggle_datasets': [
                    'andrewmvd/pothole-detection-dataset',
                    'chitholian/annotated-potholes-dataset',
                    'sachinpatel21/pothole-image-dataset'
                ],
                'roboflow_projects': [
                    'pothole-detection-system',
                    'road-damage-detection',
                    'street-damage-detection'
                ],
                'github_repos': [
                    'https://github.com/sekilab/RoadDamageDetector',
                    'https://github.com/anujdutt9/Pothole-Detection-System'
                ]
            },
            'garbage': {
                'kaggle_datasets': [
                    'asdasdasasdas/garbage-classification',
                    'mostafaabla/garbage-classification',
                    'sumn2u/garbage-classification-v2'
                ],
                'roboflow_projects': [
                    'waste-detection',
                    'garbage-detection-3',
                    'trash-detection'
                ]
            },
            'road_issues': {
                'kaggle_datasets': [
                    'chitholian/road-damage-detection-dataset',
                    'balraj98/road-crack-detection-dataset'
                ],
                'open_datasets': [
                    'https://data.mendeley.com/datasets/5y9wdsg2zt/2',  # Road crack dataset
                    'https://www.crcv.ucf.edu/data1/segtrack_v2/'      # Urban scene dataset
                ]
            }
        }
    
    def setup_kaggle_api(self):
        """Setup Kaggle API credentials"""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            print("⚠️  Kaggle API not configured!")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Save kaggle.json to ~/.kaggle/")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            return api
        except Exception as e:
            print(f"❌ Kaggle API setup failed: {e}")
            return False
    
    def download_kaggle_dataset(self, dataset_name, target_dir):
        """Download dataset from Kaggle"""
        print(f"📥 Downloading Kaggle dataset: {dataset_name}")
        
        api = self.setup_kaggle_api()
        if not api:
            return False
        
        try:
            # Download dataset
            download_path = target_dir / 'kaggle_temp'
            download_path.mkdir(exist_ok=True)
            
            api.dataset_download_files(
                dataset_name, 
                path=str(download_path), 
                unzip=True
            )
            
            print(f"✅ Downloaded: {dataset_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
            return False
    
    def download_roboflow_dataset(self, project_name, api_key=None):
        """Download dataset from Roboflow"""
        if not api_key:
            api_key = os.getenv('ROBOFLOW_API_KEY')
            
        if not api_key:
            print("⚠️  Roboflow API key required")
            print("Get one from: https://roboflow.com")
            return False
        
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=api_key)
            # Note: You'll need to adjust workspace/project names based on actual availability
            project = rf.workspace("public").project(project_name)
            dataset = project.version(1).download(
                "yolov8", 
                location=str(self.real_world_dir / f"roboflow_{project_name}")
            )
            
            print(f"✅ Downloaded Roboflow dataset: {project_name}")
            return True
            
        except Exception as e:
            print(f"❌ Roboflow download failed for {project_name}: {e}")
            return False
    
    def download_github_dataset(self, repo_url, target_dir):
        """Download dataset from GitHub repository"""
        print(f"📥 Downloading from GitHub: {repo_url}")
        
        try:
            # Extract repo info
            repo_name = repo_url.split('/')[-1]
            zip_url = f"{repo_url}/archive/main.zip"
            
            # Download zip file
            zip_path = target_dir / f"{repo_name}.zip"
            urllib.request.urlretrieve(zip_url, zip_path)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Clean up
            zip_path.unlink()
            
            print(f"✅ Downloaded GitHub repo: {repo_name}")
            return True
            
        except Exception as e:
            print(f"❌ GitHub download failed: {e}")
            return False
    
    def download_open_dataset(self, url, target_dir):
        """Download from open dataset URLs"""
        print(f"📥 Downloading open dataset: {url}")
        
        try:
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name or "dataset.zip"
            
            # Download file
            file_path = target_dir / filename
            urllib.request.urlretrieve(url, file_path)
            
            # Try to extract if it's an archive
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir / filename.stem)
            elif filename.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(target_dir / filename.stem)
            
            print(f"✅ Downloaded open dataset: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Open dataset download failed: {e}")
            return False
    
    def collect_pothole_datasets(self):
        """Collect real-world pothole datasets"""
        print("🕳️  Collecting pothole datasets...")
        
        pothole_dir = self.real_world_dir / "potholes"
        pothole_dir.mkdir(exist_ok=True)
        
        success_count = 0
        
        # Kaggle datasets
        for dataset in self.dataset_sources['potholes']['kaggle_datasets']:
            if self.download_kaggle_dataset(dataset, pothole_dir):
                success_count += 1
        
        # Roboflow datasets
        roboflow_key = os.getenv('ROBOFLOW_API_KEY')
        if roboflow_key:
            for project in self.dataset_sources['potholes']['roboflow_projects']:
                if self.download_roboflow_dataset(project, roboflow_key):
                    success_count += 1
        
        # GitHub repositories
        for repo_url in self.dataset_sources['potholes']['github_repos']:
            if self.download_github_dataset(repo_url, pothole_dir):
                success_count += 1
        
        print(f"✅ Collected {success_count} pothole datasets")
        return success_count > 0
    
    def collect_garbage_datasets(self):
        """Collect real-world garbage/waste datasets"""
        print("🗑️  Collecting garbage datasets...")
        
        garbage_dir = self.real_world_dir / "garbage"
        garbage_dir.mkdir(exist_ok=True)
        
        success_count = 0
        
        # Kaggle datasets
        for dataset in self.dataset_sources['garbage']['kaggle_datasets']:
            if self.download_kaggle_dataset(dataset, garbage_dir):
                success_count += 1
        
        # Roboflow datasets
        roboflow_key = os.getenv('ROBOFLOW_API_KEY')
        if roboflow_key:
            for project in self.dataset_sources['garbage']['roboflow_projects']:
                if self.download_roboflow_dataset(project, roboflow_key):
                    success_count += 1
        
        print(f"✅ Collected {success_count} garbage datasets")
        return success_count > 0
    
    def collect_road_damage_datasets(self):
        """Collect road damage and infrastructure datasets"""
        print("🛣️  Collecting road damage datasets...")
        
        road_dir = self.real_world_dir / "road_damage"
        road_dir.mkdir(exist_ok=True)
        
        success_count = 0
        
        # Kaggle datasets
        for dataset in self.dataset_sources['road_issues']['kaggle_datasets']:
            if self.download_kaggle_dataset(dataset, road_dir):
                success_count += 1
        
        # Open datasets
        for url in self.dataset_sources['road_issues']['open_datasets']:
            if self.download_open_dataset(url, road_dir):
                success_count += 1
        
        print(f"✅ Collected {success_count} road damage datasets")
        return success_count > 0
    
    def process_downloaded_datasets(self):
        """Process and standardize downloaded datasets"""
        print("🔄 Processing downloaded datasets...")
        
        processed_count = 0
        
        # Process each category
        for category in ['potholes', 'garbage', 'road_damage']:
            category_dir = self.real_world_dir / category
            if not category_dir.exists():
                continue
            
            print(f"📁 Processing {category} datasets...")
            
            # Find all images and annotations
            images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                images.extend(list(category_dir.rglob(f'*{ext}')))
                images.extend(list(category_dir.rglob(f'*{ext.upper()}')))
            
            if not images:
                print(f"⚠️  No images found in {category}")
                continue
            
            # Create standardized structure
            std_dir = self.real_world_dir / f"{category}_processed"
            std_dir.mkdir(exist_ok=True)
            
            for i, img_path in enumerate(tqdm(images, desc=f"Processing {category}")):
                try:
                    # Copy image with standardized name
                    new_name = f"{category}_{i:06d}.jpg"
                    new_img_path = std_dir / new_name
                    
                    # Convert to JPG if needed
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(new_img_path, 'JPEG', quality=95)
                    
                    # Look for corresponding annotation
                    possible_labels = [
                        img_path.with_suffix('.txt'),
                        img_path.with_suffix('.xml'),
                        img_path.parent / 'labels' / f"{img_path.stem}.txt"
                    ]
                    
                    for label_path in possible_labels:
                        if label_path.exists():
                            # Copy and potentially convert annotation
                            new_label_path = std_dir / f"{category}_{i:06d}.txt"
                            
                            if label_path.suffix == '.txt':
                                # Assume YOLO format, copy directly
                                shutil.copy2(label_path, new_label_path)
                            elif label_path.suffix == '.xml':
                                # Convert from XML (Pascal VOC) to YOLO format
                                self.convert_xml_to_yolo(label_path, new_label_path, img.size)
                            
                            break
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Error processing {img_path}: {e}")
                    continue
        
        print(f"✅ Processed {processed_count} real-world images")
        return processed_count > 0
    
    def convert_xml_to_yolo(self, xml_path, yolo_path, img_size):
        """Convert Pascal VOC XML annotation to YOLO format"""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            img_width, img_height = img_size
            
            with open(yolo_path, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text.lower()
                    
                    # Map class names to our class IDs
                    class_id = self.map_class_name_to_id(class_name)
                    if class_id is None:
                        continue
                    
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Convert to YOLO format (normalized center coordinates)
                    center_x = (xmin + xmax) / 2.0 / img_width
                    center_y = (ymin + ymax) / 2.0 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        except Exception as e:
            print(f"⚠️  XML conversion failed: {e}")
    
    def map_class_name_to_id(self, class_name):
        """Map various class names to our standardized class IDs"""
        class_name = class_name.lower().strip()
        
        # Pothole mappings
        if any(word in class_name for word in ['pothole', 'hole', 'crack', 'damage']):
            return 0
        
        # Garbage mappings
        if any(word in class_name for word in ['garbage', 'trash', 'waste', 'litter', 'debris']):
            return 1
        
        # Water/flooding mappings
        if any(word in class_name for word in ['water', 'flood', 'puddle']):
            return 2
        
        # Streetlight mappings
        if any(word in class_name for word in ['light', 'lamp', 'pole']):
            return 3
        
        # Sidewalk mappings
        if any(word in class_name for word in ['sidewalk', 'pavement', 'walkway']):
            return 4
        
        # Construction debris mappings
        if any(word in class_name for word in ['construction', 'debris', 'material']):
            return 5
        
        return None  # Unknown class
    
    def merge_with_existing_data(self):
        """Merge real-world data with existing synthetic data"""
        print("🔄 Merging real-world data with existing dataset...")
        
        # Find all processed real-world images
        real_world_images = []
        for category_dir in self.real_world_dir.glob("*_processed"):
            images = list(category_dir.glob("*.jpg"))
            real_world_images.extend(images)
        
        if not real_world_images:
            print("⚠️  No processed real-world images found")
            return False
        
        print(f"📸 Found {len(real_world_images)} real-world images")
        
        # Copy to raw directory for integration
        for img_path in tqdm(real_world_images, desc="Merging datasets"):
            # Copy image
            dst_img = self.raw_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Copy corresponding label if exists
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                dst_label = self.raw_dir / label_path.name
                shutil.copy2(label_path, dst_label)
        
        print("✅ Real-world data merged successfully!")
        return True
    
    def create_mixed_dataset_config(self):
        """Create dataset configuration including real-world data"""
        # Count images by source
        synthetic_count = len(list((self.raw_dir / "synthetic").glob("*.jpg"))) if (self.raw_dir / "synthetic").exists() else 0
        real_world_count = len([f for f in self.raw_dir.glob("*.jpg") if not f.name.startswith("synthetic_")])
        
        config = {
            'dataset_info': {
                'total_images': synthetic_count + real_world_count,
                'synthetic_images': synthetic_count,
                'real_world_images': real_world_count,
                'sources': {
                    'synthetic': 'Generated synthetic civic anomalies',
                    'kaggle': 'Real-world datasets from Kaggle',
                    'roboflow': 'Annotated datasets from Roboflow',
                    'github': 'Open-source datasets from GitHub',
                    'open_datasets': 'Public research datasets'
                }
            },
            'classes': self.classes,
            'training_recommendations': {
                'epochs': 50 if real_world_count > 100 else 30,
                'batch_size': 16 if real_world_count > 500 else 8,
                'augmentation': 'medium' if real_world_count > 200 else 'high'
            }
        }
        
        config_path = self.base_dir / "mixed_dataset_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Mixed dataset config saved: {config_path}")
        return config
    
    def collect_all_datasets(self):
        """Main method to collect all real-world datasets"""
        print("🌍 Starting real-world dataset collection...")
        print("=" * 50)
        
        success_flags = []
        
        # Collect different types of datasets
        success_flags.append(self.collect_pothole_datasets())
        success_flags.append(self.collect_garbage_datasets())
        success_flags.append(self.collect_road_damage_datasets())
        
        if not any(success_flags):
            print("❌ No datasets were successfully downloaded")
            print("\n💡 To download datasets, you need:")
            print("1. Kaggle API: pip install kaggle && setup ~/.kaggle/kaggle.json")
            print("2. Roboflow API: pip install roboflow && set ROBOFLOW_API_KEY")
            return False
        
        # Process downloaded datasets
        if self.process_downloaded_datasets():
            # Merge with existing data
            self.merge_with_existing_data()
            
            # Create configuration
            config = self.create_mixed_dataset_config()
            
            print("\n✅ Real-world dataset collection completed!")
            print(f"📊 Dataset summary:")
            print(f"   - Total images: {config['dataset_info']['total_images']}")
            print(f"   - Synthetic: {config['dataset_info']['synthetic_images']}")
            print(f"   - Real-world: {config['dataset_info']['real_world_images']}")
            
            return True
        
        return False

def main():
    """Main execution function"""
    print("🌍 Real-World Civic Anomaly Dataset Collector")
    print("=" * 50)
    
    collector = RealWorldDatasetCollector()
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    # Check Kaggle API
    try:
        import kaggle
        print("✅ Kaggle API available")
    except ImportError:
        print("⚠️  Kaggle API not installed. Run: pip install kaggle")
    
    # Check Roboflow API
    roboflow_key = os.getenv('ROBOFLOW_API_KEY')
    if roboflow_key:
        print("✅ Roboflow API key found")
    else:
        print("⚠️  Roboflow API key not set. Set ROBOFLOW_API_KEY environment variable")
    
    # Start collection
    if collector.collect_all_datasets():
        print("\n🚀 Ready for enhanced training with real-world data!")
        print("Run: python3 scripts/train_model.py")
    else:
        print("\n💡 Consider manually adding real-world images to scripts/data/raw/")

if __name__ == "__main__":
    main()