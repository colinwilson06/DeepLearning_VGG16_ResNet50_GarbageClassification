# ♻️ DeepLearning_VGG16_ResNet50_GarbageClassification
### End-to-End Deep Learning System for Automated Waste Classification (12 Classes)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_App-red?logo=streamlit)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Model_Registry-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/)

---

## 🧠 Executive Summary

This repository contains a complete **AI/ML Engineering pipeline** for multi-class image classification of **12 waste categories**, implemented using **Transfer Learning** with two high-performance CNN architectures:

- **VGG16** (baseline)
- **ResNet-50** (production candidate)

The project represents an **end-to-end MLOps approach**, integrating:

✔️ Data preprocessing & augmentation  
✔️ Transfer learning & fine-tuning  
✔️ Evaluation on a held-out test set  
✔️ Deployment through a **Streamlit inference service**  
✔️ External model hosting using **HuggingFace Hub**  
✔️ Reproducible environment via Dev Containers  

---

## 🎯 Key Results & Performance Benchmarks

The models were evaluated on an **independent 10% test split**.  
ResNet-50 achieved the strongest generalization performance, establishing itself as the **recommended production model**.

| Model | Transfer Learning | Test Accuracy | Weighted F1-Score | Deployment Status |
|------|------------------|--------------:|------------------:|-------------------|
| VGG16 | Fine-tuned (ImageNet) | 0.950 | 0.95 | Baseline |
| ResNet-50 | Fine-tuned (ImageNet) | 0.960 | 0.96 | Production Candidate |

🟩 **ResNet-50 demonstrates superior feature abstraction through residual connections**, enabling more stable deep-layer gradient propagation.

---

## 🔬 Technical Breakdown  
### **A. Transfer Learning Architecture & Optimization**

Both architectures follow a unified fine-tuning workflow:

| Component | Description |
|----------|-------------|
| **Backbone** | Pretrained ImageNet (frozen during initial training) |
| **Custom Head** | GAP → Dense(256, ReLU) → BatchNorm → Dropout(0.3) → Dense(12, Softmax) |
| **Loss Function** | Categorical Crossentropy |
| **Optimizer** | Adam + ReduceLROnPlateau |
| **Regularization** | BatchNorm, Dropout, EarlyStopping |
| **Augmentation** | Rotation, shift, zoom, flip, brightness (ImageDataGenerator) |

#### Why ResNet-50 Performed Better
- Residual identity blocks prevent **vanishing gradients**
- Deeper hierarchy captures more **semantic-level patterns**
- Outperforms VGG16 especially on **visually similar classes** (plastic/metal/paper/glass)

---

## 📈 Model Evaluation & Error Analysis

### **1. Training Curves (ResNet-50)**  
Smooth convergence and low generalization gap indicate proper regularization.

![ResNet Plot](assets/ResNet50 Plot.png)

---

### **2. Confusion Matrix (ResNet-50)**  
Shows strong diagonal dominance; remaining errors originate from **classes with overlapping visual features**.

![ResNet CM](assets/Confusion Matrix - ResNet50.png)

---

### **3. Breakdown of Difficult Classes**
| Class | Common Misclassification | Reason |
|-------|--------------------------|--------|
| Plastic | Metal / Paper | Similar reflectivity & texture |
| Metal | Plastic / Glass | Edge lighting & color similarity |
| Paper | Cardboard | Material similarity |

---

## ⚙️ Engineering & MLOps Highlights

### **✔ Dev Container Support**
`/.devcontainer/` ensures environment reproducibility (Docker-based).  
Recruiters love this — shows maturity in ML Engineering practices.

### **✔ Streamlit Inference Service**
`app.py` provides:

- Real-time classification  
- Model selector (VGG16 / ResNet50)  
- **Test-Time Augmentation (TTA)**  
- Letterbox preprocessing  
- Confidence gauge + interpretability  
- Top-5 prediction bar chart (Plotly)

### ✔ Model Hosting via HuggingFace Hub
Large `.keras` checkpoints are stored externally:  
- Faster deployments  
- Lightweight Git repo  
- Versioned model registry

---

## 📁 Repository Structure

```bash
📦 DeepLearning_VGG16_ResNet50_GarbageClassification
│
├── app.py                      # Streamlit inference application
├── main_model.ipynb            # Full training & evaluation notebook
├── requirements.txt            # Python dependencies
├── .gitattributes              # Git LFS Configuration
│
├── 📂 assets/                  # Confusion matrices, plots, metrics
│
├── 📂 .devcontainer/           # Dev environment configuration
│   └── devcontainer.json
│
└── 📂 .streamlit/              # Streamlit application configuration
    └── config.toml


🚀 Environment Setup & Execution (Bash-Ready)

**1. Setup Environment & Dependencies**
```bash
echo "[1/3] Setting up environment and pulling model files..."

# Pull large model files via Git LFS
git lfs pull

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "[✔] Environment setup complete."

```

**2. Launch Streamlit Application**
```bash
echo "[2/3] Launching Streamlit application..."
streamlit run app.py
```

The Streamlit service will automatically:
- Download the correct model from HuggingFace
- Preprocess input images
- Run TTA + model inference
- Display Top-5 confidence visualization

🌐 Live Demo
You can interact with the live deployed classification application here:
![Live Streamlit Demo Link](https://dlgarbageclassification.streamlit.app/)

🔭 Future Engineering Enhancements
- Add YOLO-based Object Detection to support multi-object scenes
- Create TensorFlow Lite mobile deployment
- Integrate attention blocks to strengthen class separability
- Collect real-world Indonesian waste images to improve domain adaptation

