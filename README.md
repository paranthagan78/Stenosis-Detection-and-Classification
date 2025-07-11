# Coronary Artery Stenosis Detection and Classification

This AI-powered project focuses on the automatic detection and classification of coronary artery stenosis from angiographic X-ray images. By combining deep learning models for both Detection and classification of Stenosis, the system supports clinicians in accurate, early diagnosis of CAD (Coronary Artery Disease).

## 🧠 Features

- **Automated Stenosis Detection** using YOLOv8
- **Artery Classification** into LAD, LCX, and RCA using an ensemble of ResNet50, VGG16, and EfficientNetB0
- **Preprocessing**: CLAHE, denoising, and augmentation to improve sensitivity
- **XAI Support**: Explainable AI tools for better clinical transparency

## 📊 Results

- Ensemble Accuracy: **99.29%** (Classification of Stenosis)
- Improved vessel-specific precision and recall

## 📂 Dataset

- **Name**: ARCADE (Automatic Region-based Coronary Artery Disease diagnostics)
- **Format**: `.png` images with `.json` annotations for stenosis and vessel labels

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch, YOLOv8, EfficientNet, ResNet, VGG
- OpenCV, NumPy, Pandas, Scikit-learn
- Jupyter Notebook for development and experiments

## 🌍 Impact

- Supports **SDG 3: Good Health and Well-being**
- Reduces human error in diagnostics
- Enables **real-time, explainable, and accurate** detection of CAD

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. 

## Contributors

1. Paranthagan S
2. Nandana M
