# 🤟 Sign Language Recognition using Vision Transformer (ViT) and Graph Attention Networks (GAT)

## 📌 About

This project presents a hybrid deep learning framework for **Sign Language Recognition (SLR)** using video sequences. The system integrates **Vision Transformers (ViT)** for capturing global visual dependencies and **Graph Attention Networks (GAT)** for modeling spatial relationships between hand keypoints.

The model combines visual and structural information to improve gesture understanding and achieve high classification accuracy.

---

## 🧭 System Architecture  

<p align="center">
  <img src="https://github.com/user-attachments/assets/112bb6c2-8ba6-470f-b9a4-402108776781" width="500"/>
</p>

<p align="center"><i>Hybrid ViT + GAT architecture for sign language recognition</i></p>

---

## ✨ Key Features

- 🎥 Automated frame extraction from video sequences  
- ✋ Hand keypoint detection using MediaPipe  
- 🧠 Vision Transformer (ViT) for global feature learning  
- 🔗 Graph Attention Network (GAT) for spatial modeling  
- ⚡ Feature fusion for improved classification accuracy  
- 📊 Interactive Streamlit interface for real-time predictions  

---

## 📂 Dataset

- **Dataset Used:** WLASL (Word-Level American Sign Language)  
- Large-scale dataset containing labeled sign language video samples  
- 🔗 https://github.com/dxli94/WLASL  

---

## 📊 Results

The proposed hybrid model demonstrates significant improvement over standalone architectures.

### 🔹 Model Comparison

| Model                          | Accuracy (%) |
|--------------------------------|-------------|
| CNN Baseline                   | 68          |
| LSTM (Temporal Model)          | 72          |
| Vision Transformer (ViT)       | 82          |
| Graph Attention Network (GAT)  | 79          |
| **Proposed ViT + GAT (Ours)**  | **94.7**    |

---

### 🔹 Performance Metrics

| Metric     | Value  |
|------------|--------|
| Accuracy   | 94.7%  |
| Precision  | 93%    |
| Recall     | 92%    |
| F1 Score   | 92.5%  |

---

## 📊 Training Performance

<p align="center">
  <img width="1042" height="572" alt="accuracy curve slr" src="https://github.com/user-attachments/assets/8dc37b5d-e172-42c5-86de-7317c7d73d06" />
</p>

<p align="center"><i>Training accuracy curve showing convergence</i></p>

---

## 📊 Confusion Matrix

<p align="center">
  <img width="1018" height="803" alt="confusion_matrix_slr" src="https://github.com/user-attachments/assets/155a820f-e5f2-458a-8727-0385dae22531" />
</p>

<p align="center"><i>Confusion matrix showing classification performance</i></p>

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Sign-Language-Recognition.git
cd Sign-Language-Recognition

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```
---
## ▶️ How to Run
🔹 1. Extract Frames
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
Live Camera prediction:
python -m inference.predict_live
```
---
## 📦 Project Structure
```bash
Sign-Language-Recognition/
│
├── preprocessing/
├── models/
├── training/
├── inference/
├── dataset/
└── streamlit/
```
---
## ✔️ Highlights
- Hybrid architecture combining Transformer + Graph Neural Network
- Captures both global visual features and local spatial relationships
- End-to-end pipeline from raw video → prediction
- Achieves 94.7% accuracy, outperforming traditional models
- Designed for real-time and scalable applications  
---

## 📈 Future Work
- Real-time webcam-based recognition
- Scaling to WLASL2000 dataset
- Temporal attention / sequence transformers
- Mobile/web deployment
- Edge optimization for low-latency inference
---
