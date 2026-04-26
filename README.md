# 🤟 Sign Language Recognition using Vision Transformer and Graph Attention Networks

A deep learning-based **Sign Language Recognition (SLR)** system that combines **Vision Transformers (ViT)** and **Graph Attention Networks (GAT)** to classify sign gestures from video sequences.

The model leverages:
- **Global visual understanding** using Transformers  
- **Spatial hand relationships** using Graph Neural Networks  

This hybrid approach improves recognition of complex gestures in real-world scenarios.

---

## 🧭 System Architecture  

<p align="center">
  <img src="ADD_YOUR_IMAGE_LINK_HERE" alt="ViT + GAT Architecture" width="700"/>
</p>

<p align="center"><i>Hybrid ViT + GAT architecture for Sign Language Recognition</i></p>

### 🔁 Pipeline Overview
- Video → Frame Extraction  
- Frames → **ViT** (Visual Features)  
- Keypoints → **GAT** (Graph Features)  
- Feature Fusion → Classification Output  

---

## ✨ Features

### 🎥 Video Processing
- Extract frames from sign language videos  
- Fixed-length sequence modeling  
- Frame caching for faster training  

### ✋ Keypoint Extraction
- Hand landmark detection using **MediaPipe**  
- Graph construction from keypoints  
- Spatial feature learning using GAT  

### 🧠 Hybrid Model
- Vision Transformer for global context  
- Graph Attention Network for spatial relationships  
- Fusion of both modalities for accurate classification  

### ⚡ Inference
- Predict sign labels from videos  
- Real-time inference using Streamlit  

---

## 📂 Dataset

- **Dataset Used:** WLASL (Word-Level American Sign Language)  
- Contains labeled sign language videos for multiple classes  

### 📁 Expected Structure
