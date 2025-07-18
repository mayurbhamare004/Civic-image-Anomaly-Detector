#!/usr/bin/env python3
"""
Enhanced Dataset Collector for Civic Anomaly Detection
Integrates multiple real-world datasets for improved model performance
"""

import os
import sys
import json
import yaml
import shutil
import zipfile
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import random
import urllib.request
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

class EnhancedDatasetCollector:
    def __init__(self, config_path="scripts/dataset_config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()
        
        # Setup directories
        self.base_dir = Path("scripts/data")
        self.raw_dir = self.base_dir / "raw"
        self.real_world_dir = self.raw_dir / "real_world"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directory structure
        self.setup_directories()
        
        # Statistics tracking
        self.stats = {
            'downloaded': 0,
            'processed': 0,
            'errors': 0,
            'by_source': {},
            'by_class': {i: 0 for i in range(6)}
        }
    
    def load_config(self):
        """Load dataset configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ Loaded configuration from {self.config_path}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            sys.exit(1)
    
    def setup_directories(self):
        """Create necessary directory structure"""
        dirs = [
            self.raw_dir,
            self.real_world_dir,
            self.processed_dir / "train" / "images",
            self.processed_dir / "train" / "labels",
            self.processed_dir / "val" / "images", 
            self.processed_dir / "val" / "labels",
            self.processed_dir / "test" / "images",
            self.processed_dir / "test" / "labels"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def check_prerequisites(self):
        """Check if required APIs and tools are available"""
        print("🔍 Checking prerequisites...")
        
        issues = []
        
        # Check Kaggle API
        try:
            import kaggle
            kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
            if kaggle_config.exists():
                print("✅ Kaggle API configured")
            else:
                issues.append("Kaggle API not configured")
        except ImportError:
            issues.append("Kaggle package not installed (pip install kaggle)")
        
        # Check Roboflow API
        if os.getenv('ROBOFLOW_API_KEY'):
            try:
                import roboflow
                print("✅ Roboflow API available")
            except ImportError:
                issues.append("Roboflow package not installed (pip install roboflow)")
        else:
            print("⚠️  Roboflow API key not set (optional)")
        
        # Check other dependencies
        required_packages = ['opencv-python', 'pillow', 'tqdm', 'pyyaml']
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                issues.append(f"Missing package: {package}")
        
        if issues:
            print("❌ Prerequisites not met:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ All prerequisites met")
        return True
    
    def download_kaggle_dataset(self, dataset_info, category):
        """Download dataset from Kaggle"""
        dataset_name = dataset_info['name']
        print(f"📥 Downloading Kaggle dataset: {dataset_name}")
        
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            # Create download directory
            download_dir = self.real_world_dir / category / "kaggle" / dataset_name.replace('/', '_')
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Download and extract
            api.dataset_download_files(
                dataset_name,
                path=str(download_dir),
                unzip=True
            )
            
            self.stats['downloaded'] += 1
            self.stats['by_source'][f'kaggle_{category}'] = self.stats['by_source'].get(f'kaggle_{category}', 0) + 1
            
            print(f"✅ Downloaded: {dataset_name}")
            return download_dir
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
            self.stats['errors'] += 1
            return None
    
    def download_roboflow_dataset(self, dataset_info, category):
        """Download dataset from Roboflow"""
        project_name = dataset_info['name']
        workspace = dataset_info.get('workspace', 'public')
        
        print(f"📥 Downloading Roboflow dataset: {project_name}")
        
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            print("⚠️  Roboflow API key not found")
            return None
        
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace).project(project_name)
            
            download_dir = self.real_world_dir / category / "roboflow" / project_name
            download_dir.mkdir(parents=True, exist_ok=True)
            
            dataset = project.version(1).download(
                "yolov8",
                location=str(download_dir)
            )
            
            self.stats['downloaded'] += 1
            self.stats['by_source'][f'roboflow_{category}'] = self.stats['by_source'].get(f'roboflow_{category}', 0) + 1
            
            print(f"✅ Downloaded: {project_name}")
            return download_dir
            
        except Exception as e:
            print(f"❌ Failed to download {project_name}: {e}")
            self.stats['errors'] += 1
            return None
    
    def download_github_dataset(self, dataset_info, category):
        """Download dataset from GitHub"""
        repo_url = dataset_info['url']
        print(f"📥 Downloading GitHub dataset: {repo_url}")
        
        try:
            repo_name = repo_url.split('/')[-1]
            download_dir = self.real_world_dir / category / "github" / repo_name
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Try different archive formats
            for branch in ['main', 'master']:
                zip_url = f"{repo_url}/archive/{branch}.zip"
                zip_path = download_dir / f"{repo_name}.zip"
                
                try:
                    urllib.request.urlretrieve(zip_url, zip_path)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(download_dir)
                    
                    zip_path.unlink()  # Remove zip file
                    
                    self.stats['downloaded'] += 1
                    self.stats['by_source'][f'github_{category}'] = self.stats['by_source'].get(f'github_{category}', 0) + 1
                    
                    print(f"✅ Downloaded: {repo_name}")
                    return download_dir
                    
                except Exception:
                    continue
            
            raise Exception("Failed to download from any branch")
            
        except Exception as e:
            print(f"❌ Failed to download {repo_url}: {e}")
            self.stats['errors'] += 1
            return None
    
    def download_open_dataset(self, dataset_info, category):
        """Download from open dataset URLs"""
        url = dataset_info['url']
        name = dataset_info.get('name', 'open_dataset')
        
        print(f"📥 Downloading open dataset: {name}")
        
        try:
            download_dir = self.real_world_dir / category / "open" / name.replace(' ', '_')
            download_dir.mkdir(parents=True, exist_ok=True)
            
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name or "dataset.zip"
            file_path = download_dir / filename
            
            # Download file
            urllib.request.urlretrieve(url, file_path)
            
            # Extract if it's an archive
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
            elif filename.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(download_dir)
            
            self.stats['downloaded'] += 1
            self.stats['by_source'][f'open_{category}'] = self.stats['by_source'].get(f'open_{category}', 0) + 1
            
            print(f"✅ Downloaded: {name}")
            return download_dir
            
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
            self.stats['errors'] += 1
            return None
    
    def collect_datasets_by_category(self, category):
        """Collect all datasets for a specific category"""
        print(f"\n🔄 Collecting {category} datasets...")
        
        if category not in self.config['dataset_sources']:
            print(f"⚠️  Category {category} not found in config")
            return []
        
        category_config = self.config['dataset_sources'][category]
        downloaded_dirs = []
        
        # Download from Kaggle
        if 'kaggle' in category_config:
            for dataset_info in category_config['kaggle']:
                result = self.download_kaggle_dataset(dataset_info, category)
                if result:
                    downloaded_dirs.append(result)
        
        # Download from Roboflow
        if 'roboflow' in category_config:
            for dataset_info in category_config['roboflow']:
                result = self.download_roboflow_dataset(dataset_info, category)
                if result:
                    downloaded_dirs.append(result)
        
        # Download from GitHub
        if 'github' in category_config:
            for dataset_info in category_config['github']:
                result = self.download_github_dataset(dataset_info, category)
                if result:
                    downloaded_dirs.append(result)
        
        # Download from open datasets
        if 'open_datasets' in category_config:
            for dataset_info in category_config['open_datasets']:
                result = self.download_open_dataset(dataset_info, category)
                if result:
                    downloaded_dirs.append(result)
        
        print(f"✅ Collected {len(downloaded_dirs)} datasets for {category}")
        return downloaded_dirs
    
    def map_class_name_to_id(self, class_name):
        """Map external class names to our standard class IDs"""
        class_name = class_name.lower().strip()
        
        mappings = self.config['class_mapping']['external_mappings']
        
        for our_class, variants in mappings.items():
            class_id = variants[0]  # First element is the class ID
            variant_names = variants[1:]  # Rest are variant names
            
            if any(variant in class_name for variant in variant_names):
                return class_id
        
        return None  # Unknown class
    
    def convert_annotation_format(self, annotation_path, output_path, img_size, format_type):
        """Convert annotation from various formats to YOLO"""
        try:
            if format_type == 'yolo':
                # Already in YOLO format, just copy
                shutil.copy2(annotation_path, output_path)
                return True
            
            elif format_type == 'pascal_voc':
                return self.convert_xml_to_yolo(annotation_path, output_path, img_size)
            
            elif format_type == 'coco':
                return self.convert_coco_to_yolo(annotation_path, output_path, img_size)
            
            else:
                print(f"⚠️  Unsupported annotation format: {format_type}")
                return False
                
        except Exception as e:
            print(f"⚠️  Annotation conversion failed: {e}")
            return False
    
    def convert_xml_to_yolo(self, xml_path, yolo_path, img_size):
        """Convert Pascal VOC XML to YOLO format"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            img_width, img_height = img_size
            
            with open(yolo_path, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_id = self.map_class_name_to_id(class_name)
                    
                    if class_id is None:
                        continue
                    
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Convert to YOLO format
                    center_x = (xmin + xmax) / 2.0 / img_width
                    center_y = (ymin + ymax) / 2.0 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                    
                    self.stats['by_class'][class_id] += 1
            
            return True
            
        except Exception as e:
            print(f"⚠️  XML conversion failed: {e}")
            return False
    
    def process_downloaded_datasets(self):
        """Process all downloaded datasets into standardized format"""
        print("\n🔄 Processing downloaded datasets...")
        
        processed_images = []
        
        # Process each category directory
        for category_dir in self.real_world_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            print(f"📁 Processing {category_name} datasets...")
            
            # Find all images in this category
            images = []
            for ext in self.config['processing']['image_formats']:
                images.extend(list(category_dir.rglob(f'*{ext}')))
                images.extend(list(category_dir.rglob(f'*{ext.upper()}')))
            
            if not images:
                print(f"⚠️  No images found in {category_name}")
                continue
            
            # Process each image
            for i, img_path in enumerate(tqdm(images, desc=f"Processing {category_name}")):
                try:
                    # Generate standardized filename
                    new_name = f"{category_name}_{i:06d}"
                    
                    # Process image
                    processed_img_path = self.process_image(img_path, new_name)
                    if not processed_img_path:
                        continue
                    
                    # Process annotation
                    annotation_processed = self.process_annotation(img_path, new_name, processed_img_path)
                    
                    if processed_img_path:
                        processed_images.append(processed_img_path)
                        self.stats['processed'] += 1
                
                except Exception as e:
                    print(f"⚠️  Error processing {img_path}: {e}")
                    self.stats['errors'] += 1
                    continue
        
        print(f"✅ Processed {len(processed_images)} images")
        return processed_images
    
    def process_image(self, img_path, new_name):
        """Process and standardize a single image"""
        try:
            # Load image
            img = Image.open(img_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if specified in config
            target_size = self.config['processing'].get('target_size')
            if target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save with standardized name
            output_path = self.raw_dir / f"{new_name}.jpg"
            img.save(
                output_path, 
                'JPEG', 
                quality=self.config['processing']['quality']
            )
            
            return output_path
            
        except Exception as e:
            print(f"⚠️  Image processing failed for {img_path}: {e}")
            return None
    
    def process_annotation(self, img_path, new_name, processed_img_path):
        """Process and standardize annotation for an image"""
        try:
            # Get image dimensions
            img = Image.open(processed_img_path)
            img_size = img.size
            
            # Look for annotation files
            possible_annotations = [
                img_path.with_suffix('.txt'),
                img_path.with_suffix('.xml'),
                img_path.parent / 'labels' / f"{img_path.stem}.txt",
                img_path.parent / 'annotations' / f"{img_path.stem}.xml"
            ]
            
            for ann_path in possible_annotations:
                if ann_path.exists():
                    output_path = self.raw_dir / f"{new_name}.txt"
                    
                    # Determine format and convert
                    if ann_path.suffix == '.txt':
                        format_type = 'yolo'
                    elif ann_path.suffix == '.xml':
                        format_type = 'pascal_voc'
                    else:
                        continue
                    
                    success = self.convert_annotation_format(
                        ann_path, output_path, img_size, format_type
                    )
                    
                    if success:
                        return output_path
            
            return None
            
        except Exception as e:
            print(f"⚠️  Annotation processing failed: {e}")
            return None
    
    def validate_processed_data(self):
        """Validate the processed dataset"""
        print("\n🔍 Validating processed dataset...")
        
        # Find all processed images
        images = list(self.raw_dir.glob("*.jpg"))
        labels = list(self.raw_dir.glob("*.txt"))
        
        print(f"📊 Found {len(images)} images and {len(labels)} labels")
        
        # Validation checks
        issues = []
        
        for img_path in images:
            label_path = img_path.with_suffix('.txt')
            
            # Check if label exists
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
                        
                        # Validate class ID
                        if class_id < 0 or class_id >= 6:
                            issues.append(f"{label_path.name}:{line_num} - Invalid class: {class_id}")
                        
                        # Validate coordinates
                        for coord in coords:
                            if coord < 0 or coord > 1:
                                issues.append(f"{label_path.name}:{line_num} - Invalid coordinate: {coord}")
            
            except Exception as e:
                issues.append(f"Error reading {label_path.name}: {e}")
        
        # Print validation results
        if issues:
            print(f"⚠️  Found {len(issues)} validation issues:")
            for issue in issues[:10]:
                print(f"   - {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues) - 10} more")
        else:
            print("✅ All data validated successfully!")
        
        return len(issues) == 0
    
    def generate_dataset_report(self):
        """Generate comprehensive dataset report"""
        print("\n📊 Generating dataset report...")
        
        # Count final dataset
        total_images = len(list(self.raw_dir.glob("*.jpg")))
        total_labels = len(list(self.raw_dir.glob("*.txt")))
        
        # Count by class
        class_counts = {i: 0 for i in range(6)}
        for label_path in self.raw_dir.glob("*.txt"):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            if 0 <= class_id < 6:
                                class_counts[class_id] += 1
            except:
                continue
        
        # Create report
        report = {
            'dataset_summary': {
                'total_images': total_images,
                'total_annotations': total_labels,
                'total_objects': sum(class_counts.values())
            },
            'collection_stats': self.stats,
            'class_distribution': {
                self.config['class_mapping']['civic_classes'][i]: count 
                for i, count in class_counts.items()
            },
            'recommendations': self.get_training_recommendations(total_images)
        }
        
        # Save report
        report_path = self.base_dir / "dataset_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        # Print summary
        print(f"📈 Dataset Report:")
        print(f"   Total Images: {total_images}")
        print(f"   Total Labels: {total_labels}")
        print(f"   Total Objects: {sum(class_counts.values())}")
        print(f"\n📊 Class Distribution:")
        for class_name, count in report['class_distribution'].items():
            print(f"   {class_name}: {count}")
        
        print(f"\n✅ Full report saved to: {report_path}")
        return report
    
    def get_training_recommendations(self, total_images):
        """Get training recommendations based on dataset size"""
        if total_images < 500:
            return self.config['training_recommendations']['small_dataset']
        elif total_images < 2000:
            return self.config['training_recommendations']['medium_dataset']
        else:
            return self.config['training_recommendations']['large_dataset']
    
    def collect_all_real_world_data(self):
        """Main method to collect all real-world datasets"""
        print("🌍 Enhanced Real-World Dataset Collection")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("\n💡 Setup instructions:")
            for api, info in self.config['api_requirements'].items():
                if info['required']:
                    print(f"\n{api.upper()}:")
                    print(info['setup_instructions'])
            return False
        
        # Collect datasets by category
        categories = ['potholes', 'garbage', 'road_infrastructure', 'water_damage']
        
        for category in categories:
            if category in self.config['dataset_sources']:
                self.collect_datasets_by_category(category)
        
        # Process downloaded datasets
        if self.stats['downloaded'] > 0:
            self.process_downloaded_datasets()
            
            # Validate processed data
            self.validate_processed_data()
            
            # Generate report
            report = self.generate_dataset_report()
            
            print(f"\n✅ Real-world dataset collection completed!")
            print(f"📊 Summary: {self.stats['processed']} images processed, {self.stats['errors']} errors")
            print(f"🚀 Ready for training with enhanced real-world data!")
            
            return True
        else:
            print("❌ No datasets were successfully downloaded")
            return False

def main():
    """Main execution function"""
    collector = EnhancedDatasetCollector()
    
    if collector.collect_all_real_world_data():
        print("\n🎯 Next steps:")
        print("1. Review the dataset report: scripts/data/dataset_report.yaml")
        print("2. Run dataset splitting: python3 scripts/dataset_collector.py")
        print("3. Start training: python3 scripts/train_model.py")
    else:
        print("\n💡 Alternative: Add images manually to scripts/data/raw/")

if __name__ == "__main__":
    main()