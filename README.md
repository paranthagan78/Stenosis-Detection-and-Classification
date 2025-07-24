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

### Model Architexture

<img width="1415" height="508" alt="image" src="https://github.com/user-attachments/assets/9c43a318-2352-4ff0-9c5d-e674395af819" />
<img width="1389" height="524" alt="image" src="https://github.com/user-attachments/assets/452b3481-e4e5-4d50-869d-01df6c9ad89d" />

---

### Screenshots of the Interface

<img width="1919" height="895" alt="image" src="https://github.com/user-attachments/assets/60198fa7-d215-4ca3-a202-b2b4fb4ca482" />
<img width="1912" height="906" alt="image" src="https://github.com/user-attachments/assets/8cc601f8-32a8-4caa-a7c8-dc2fa55576a8" />
<img width="1912" height="880" alt="image" src="https://github.com/user-attachments/assets/4bc09340-6dea-412c-ba50-f80277cf2bc2" />
<img width="1908" height="892" alt="image" src="https://github.com/user-attachments/assets/d854b838-912f-45fc-8d68-3d23bb4d8d59" />
<img width="1908" height="898" alt="image" src="https://github.com/user-attachments/assets/63b62643-1f65-4d6d-850e-f44acb8c3c57" />

---

## Contributors

1. Paranthagan S
2. Nandana M
