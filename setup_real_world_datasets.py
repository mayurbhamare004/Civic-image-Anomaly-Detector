#!/usr/bin/env python3
"""
Setup script for real-world dataset collection
Installs dependencies and provides setup instructions
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package):
    """Check if a package is installed"""
    try:
        __import__(package.replace('-', '_'))
        return True
    except ImportError:
        return False

def setup_kaggle_api():
    """Setup Kaggle API"""
    print("\n🔧 Setting up Kaggle API...")
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✅ Kaggle API already configured")
        return True
    
    print("⚠️  Kaggle API not configured")
    print("\n📝 To setup Kaggle API:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This downloads kaggle.json")
    print("5. Move it to ~/.kaggle/kaggle.json")
    print("6. Run: chmod 600 ~/.kaggle/kaggle.json")
    
    setup_now = input("\nDo you want to setup Kaggle API now? (y/n): ").lower() == 'y'
    
    if setup_now:
        print("\n📁 Creating .kaggle directory...")
        kaggle_dir.mkdir(exist_ok=True)
        
        print("📋 Please paste your Kaggle API credentials:")
        username = input("Kaggle Username: ")
        key = input("Kaggle Key: ")
        
        kaggle_config = {
            "username": username,
            "key": key
        }
        
        import json
        with open(kaggle_json, 'w') as f:
            json.dump(kaggle_config, f)
        
        # Set proper permissions
        os.chmod(kaggle_json, 0o600)
        
        print("✅ Kaggle API configured!")
        return True
    
    return False

def setup_roboflow_api():
    """Setup Roboflow API"""
    print("\n🔧 Setting up Roboflow API...")
    
    if os.getenv('ROBOFLOW_API_KEY'):
        print("✅ Roboflow API key already set")
        return True
    
    print("⚠️  Roboflow API key not set")
    print("\n📝 To setup Roboflow API:")
    print("1. Go to https://roboflow.com")
    print("2. Sign up for a free account")
    print("3. Go to Account Settings")
    print("4. Copy your API key")
    print("5. Set environment variable: export ROBOFLOW_API_KEY='your_key'")
    
    setup_now = input("\nDo you want to set Roboflow API key now? (y/n): ").lower() == 'y'
    
    if setup_now:
        api_key = input("Enter your Roboflow API key: ")
        
        # Add to shell profile
        shell_profile = Path.home() / '.bashrc'
        if not shell_profile.exists():
            shell_profile = Path.home() / '.bash_profile'
        if not shell_profile.exists():
            shell_profile = Path.home() / '.zshrc'
        
        if shell_profile.exists():
            with open(shell_profile, 'a') as f:
                f.write(f"\n# Roboflow API Key\nexport ROBOFLOW_API_KEY='{api_key}'\n")
            
            print(f"✅ API key added to {shell_profile}")
            print("🔄 Please restart your terminal or run: source ~/.bashrc")
        else:
            print("⚠️  Could not find shell profile. Please set manually:")
            print(f"export ROBOFLOW_API_KEY='{api_key}'")
        
        # Set for current session
        os.environ['ROBOFLOW_API_KEY'] = api_key
        return True
    
    return False

def main():
    """Main setup function"""
    print("🌍 Real-World Dataset Collection Setup")
    print("=" * 40)
    
    # Required packages
    required_packages = [
        'kaggle',
        'roboflow', 
        'opencv-python',
        'pillow',
        'tqdm',
        'pyyaml',
        'requests'
    ]
    
    print("📦 Installing required packages...")
    
    failed_packages = []
    for package in required_packages:
        if not check_package(package):
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed")
            else:
                print(f"❌ Failed to install {package}")
                failed_packages.append(package)
        else:
            print(f"✅ {package} already installed")
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("Please install manually using: pip install <package_name>")
        return False
    
    print("\n✅ All packages installed successfully!")
    
    # Setup APIs
    kaggle_setup = setup_kaggle_api()
    roboflow_setup = setup_roboflow_api()
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    dirs = [
        "scripts/data/raw",
        "scripts/data/raw/real_world",
        "scripts/data/processed/train/images",
        "scripts/data/processed/train/labels",
        "scripts/data/processed/val/images",
        "scripts/data/processed/val/labels",
        "scripts/data/processed/test/images",
        "scripts/data/processed/test/labels"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created!")
    
    # Final instructions
    print("\n🎯 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Run the enhanced dataset collector:")
    print("   python3 scripts/enhanced_dataset_collector.py")
    print("\n2. Or collect specific datasets:")
    print("   python3 scripts/real_world_dataset_collector.py")
    print("\n3. After collection, split the dataset:")
    print("   python3 scripts/dataset_collector.py")
    print("\n4. Train the model:")
    print("   python3 scripts/train_model.py")
    
    if not kaggle_setup:
        print("\n⚠️  Note: Kaggle API not configured - some datasets won't be available")
    
    if not roboflow_setup:
        print("⚠️  Note: Roboflow API not configured - some datasets won't be available")
    
    print("\n📚 Available dataset sources:")
    print("- Kaggle: Pothole detection, garbage classification, road damage")
    print("- Roboflow: Community-annotated civic anomaly datasets")
    print("- GitHub: Open-source research datasets")
    print("- Open datasets: Academic and research datasets")

if __name__ == "__main__":
    main()