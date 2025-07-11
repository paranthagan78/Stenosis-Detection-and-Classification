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

Follow these steps to set up and run the coronary artery stenosis detection and classification system:

### 1. **Clone the Repository**

```bash
git clone https://github.com/paranthagan78/Stenosis-Detection-and-Classification.git
cd Stenosis-Detection-and-Classification
```

---

### 2. **Set Up the Environment**

It’s recommended to use a virtual environment (optional but clean):

#### For `venv`:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On macOS/Linux
```

---

### 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### 4. **Prepare the Dataset**

* Download the ARCADE dataset from:
  🔗 [https://zenodo.org/records/10390295](https://zenodo.org/records/10390295)
* Place the dataset folders (`stenosis/` and `syntax/`) in the appropriate directories expected by your code.
* Ensure the structure includes:

  * `.png` images
  * `.json` annotation files (with bounding boxes and vessel labels)

---

### 5. **Run Detection and Classification Notebooks**

Open Jupyter Notebook or VS Code and run the following notebooks in order:

* `Detection_Stenosis/yolov8_train_detect.ipynb` – Train or infer stenosis detection
* `Classification_Stenosis/final_class_ensemble.ipynb` – Classify affected vessels using ensemble model
* Optionally run:

  * `Classification_Stenosis/auc_roc.ipynb`
  * `Classification_Stenosis/final_class_resnet.ipynb`
  * `Classification_Stenosis/final_class_vgg.ipynb`

---

### 6. **View Results**

* Check output directories or notebook visualizations for:

  * Detected stenotic regions
  * Predicted artery classes (LAD, LCX, RCA)
  * Confusion matrices and performance metrics
* Use included explainability tools (XAI) and batch analysis for further insights.

---

### 7. **To Run Site**

* Go to Web_Project Folder
```bash
cd Web_Project
```

* Run the Streamlit code
```bash
streamlit run final.py
```

---

## Contributors

1. Paranthagan S
2. Nandana M
