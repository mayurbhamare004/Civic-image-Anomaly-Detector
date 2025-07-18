# Civic Image Anomaly Detector 🏙️

A full-stack AI project for detecting urban civic issues using YOLOv8 object detection.

## 🎯 Detected Anomalies
- Potholes
- Garbage dumps
- Waterlogging
- Broken streetlights
- Damaged sidewalks
- Construction debris

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation
```bash
# Clone and setup
git clone <your-repo>
cd civic-anomaly-detector
pip install -r requirements.txt
```

### Usage

#### 1. Train Model
```bash
python scripts/train_model.py
```

#### 2. Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

#### 3. Run FastAPI Backend (Optional)
```bash
uvicorn app.api:app --reload
```

## 📁 Project Structure
```
civic-anomaly-detector/
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
├── models/
│   ├── weights/
│   └── configs/
├── scripts/
│   ├── train_model.py
│   ├── inference.py
│   └── data_preparation.py
├── app/
│   ├── streamlit_app.py
│   ├── api.py
│   └── utils.py
├── notebooks/
├── requirements.txt
└── README.md
```

## 🧠 Model Details
- **Architecture**: YOLOv8
- **Classes**: 6 civic anomaly types
- **Input Size**: 640x640
- **Framework**: Ultralytics

## 📊 Dataset
- Custom annotated civic images
- Roboflow integration for annotation
- YOLO format labels

## 🔧 Configuration
Edit `models/configs/config.yaml` for training parameters.