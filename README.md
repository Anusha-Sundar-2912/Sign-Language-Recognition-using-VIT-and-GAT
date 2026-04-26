# 🤟 Sign Language Recognition using Vision Transformer and Graph Attention Networks

A deep learning-based system for **Sign Language Recognition (SLR)** that combines **Vision Transformers (ViT)** and **Graph Attention Networks (GAT)** to classify sign gestures from video sequences.

The model captures:
- Global visual context using Transformers  
- Spatial relationships using Graph Neural Networks  

---

## 🧭 System Architecture  

<p align="center">
  <img width="694" height="748" alt="Architecture_slr" src="https://github.com/user-attachments/assets/112bb6c2-8ba6-470f-b9a4-402108776781" />
</p>

<p align="center"><i>Hybrid ViT + GAT architecture for sign language recognition</i></p>

---

## ✨ Features

- 🎥 Frame extraction from video sequences  
- ✋ Hand keypoint extraction using MediaPipe  
- 🧠 Vision Transformer for visual feature learning  
- 🔗 Graph Attention Network for spatial modeling  
- ⚡ Feature fusion for improved classification  
- 📊 Streamlit-based interactive interface  

---

## 📂 Dataset

- **Dataset Used:** WLASL (Word-Level American Sign Language)  
- Contains labeled video samples of different sign gestures  

---

## 📊 Results

### 🔹 Ablation Study

| Model Configuration              | Accuracy (%) |
|---------------------------------|-------------|
| CNN Baseline                    | 68          |
| Vision Transformer Only         | 76          |
| Graph Attention Network Only    | 74          |
| **Proposed ViT + GAT Model**    | **81**      |

---

### 🔹 Performance Evaluation

| Metric     | Value  |
|------------|--------|
| Accuracy   | 81%    |
| Precision  | 80%    |
| Recall     | 79%    |
| F1 Score   | 79.5%  |

---

## 📊 Training Performance

<p align="center">
  <img width="644" height="526" alt="training performance" src="https://github.com/user-attachments/assets/1dd9aefb-ee7f-432e-a034-1aa2fe3a9c43" />
</p>

<p align="center"><i>Training accuracy curve showing steady convergence</i></p>

---

## 📊 Confusion Matrix

<p align="center">
  <img width="631" height="526" alt="confusion matrix" src="https://github.com/user-attachments/assets/f05a310b-1a9d-4698-acbe-16e2b7b3b17e" />
</p>
<p align="center"><i>Confusion matrix showing classification performance across classes</i></p>

---

## 🛠️ Tech Stack

### 🧠 AI / ML
- PyTorch  
- Vision Transformers (timm)  
- Graph Neural Networks (torch-geometric)  

### 🎥 Computer Vision
- OpenCV  
- MediaPipe  

### ⚙️ Libraries
- NumPy  
- Pandas  
- Scikit-learn  
- Albumentations  

### 🖥️ Frontend
- Streamlit  

---

## ▶️ How to Run

### 🔹 1. Extract Frames
```bash
python preprocessing/extract_frames.py
```
🔹 2. Extract Keypoints
```bash
python preprocessing/extract_keypoints.py
```
🔹 3. Cache Frames
```bash
python preprocessing/cache_frames.py
```
🔹 4. Train Model
```bash
python -m training.train
```
🔹 5. Run Application
```bash
streamlit run demo/streamlit_app.py
```
---
📦 Project Structure
Sign-Language-Recognition/
│
├── preprocessing/
├── models/
├── training/
├── inference/
├── dataset/
└── streamlit/
---
📂 Dataset
Dataset Used: WLASL (Word-Level American Sign Language)
A large-scale dataset containing labeled sign language video samples

🔗 Dataset Link: https://github.com/dxli94/WLASL
---
✔️ Highlights
Developed a hybrid deep learning architecture integrating Vision Transformers (ViT) and Graph Attention Networks (GAT) for robust sign language recognition
Effectively captures both global visual features and spatial relationships of hand keypoints, improving gesture understanding
Implements a complete end-to-end pipeline from raw video input to final classification output
Demonstrates performance improvement over individual models through ablation study
Evaluated using multiple metrics including Accuracy, Precision, Recall, and F1-Score, ensuring comprehensive performance analysis
---<img width="694" height="748" alt="Architecture_slr" src="https://github.com/user-attachments/assets/2fd54a04-4f18-4b9b-ac63-4fcd1e2952c1" />

📈 Future Work
Extend the system to support real-time sign language recognition using webcam input
Scale the model to larger datasets such as WLASL2000 for improved generalization
Incorporate advanced temporal modeling techniques (e.g., temporal transformers or sequence attention)
Deploy the solution as a web or mobile application for real-world accessibility
Optimize model performance for low-latency inference in edge or embedded environments<img width="644" height="526" alt="training performance" src="https://github.com/user-attachments/assets/b019fbaf-a4aa-436c-94d7-72b7ccc1b550" />
