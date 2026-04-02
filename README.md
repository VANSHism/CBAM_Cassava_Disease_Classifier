# Multimodal Cassava Leaf Disease Classification 🌿

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

This repository contains a research-grade deep learning pipeline for classifying Cassava leaf diseases. Developed as part of the TEEP Internship Evaluation for the Smart Manufacturing Lab at National Central University (NCU), this project tackles severe class imbalance and background noise using a **Dual-Stream Late-Fusion Architecture** combined with mathematical texture analysis.

## 📋 Project Overview

The goal of this project is to accurately classify images of cassava leaves into 5 distinct categories (4 diseases and 1 healthy class). The dataset presents two major challenges:
1. **Severe Class Imbalance:** The majority of images belong to a single class (CMD), while minority classes (like CBB) are heavily underrepresented.
2. **Phenotypic Camouflage:** Background soil, harsh shadows, and overlapping disease symptoms confuse standard CNN architectures.

To solve this, we engineered a multimodal pipeline that extracts both spatial/color features (via a pre-trained backbone) and mathematical roughness (via Local Binary Patterns), fusing them through an attention mechanism.

---

## 🧠 Architecture: Dual-Stream Late-Fusion Net

Rather than relying on a standard out-of-the-box CNN, this project utilizes a custom PyTorch architecture:

* **Stream 1 (Visual Branch):** Uses **ConvNeXt-V2-Atto**, a lightweight, state-of-the-art purely convolutional model, to extract rich spatial and color features from the RGB images.
* **Stream 2 (Texture Branch):** A custom, lightweight Convolutional Neural Network built from scratch to process 1-channel **Local Binary Pattern (LBP)** maps, forcing the network to learn the high-frequency mathematical noise of mosaic diseases.
* **Attention Mechanism (CBAM):** The fused 448-channel tensor is passed through a **Convolutional Block Attention Module**. This computes both Channel and Spatial attention, effectively creating a localization heatmap that mutes background noise and amplifies disease lesions.

---

## ⚙️ Data Engineering & Pipeline

To ensure the neural network learns from high-quality data and avoids CPU bottlenecks during training, the pipeline includes an optimized caching script:

1. **HSV Color Thresholding:** Isolates the green/yellow/brown plant tissue and mathematically removes background soil and hands.
2. **Illumination Correction (CLAHE):** Applied strictly to the Lightness (L) channel in the LAB color space to flatten harsh shadows and sun glare without destroying the actual disease colors.
3. **Texture Extraction (LBP):** Generates a rotation-invariant grayscale map of the leaf's physical roughness.
4. **I/O Caching:** All preprocessing is executed once and saved to disk, dropping the PyTorch DataLoader iteration time from ~3.20s down to milliseconds.
5. **Synchronized Augmentations:** Using `Albumentations`, spatial warping (like GridDistortion and Rotation) is applied identically to both the RGB image and the LBP mask to maintain multimodal alignment.

---

## 📈 Training Methodology

* **Focal Loss Regularization:** To combat the extreme class imbalance, standard Cross-Entropy was replaced with Focal Loss. This heavily penalizes the model for ignoring minority classes (like CBB) and reduces the loss for easy, majority-class predictions.
* **Domain-Aware Fine-Tuning:** Implemented differential learning rates. The custom texture layers train at a standard speed, while the pre-trained ConvNeXt backbone trains at a microscopically slow rate to prevent catastrophic forgetting.
* **Stratified 5-Fold Cross-Validation:** Ensures rigorous, reproducible evaluation across the entire dataset, preventing validation-split bias.

---

## 📊 Results & Limitations
The architecture achieved a highly respectable average Macro F1-Score of ~0.77 over 5 folds after only 10 epochs and a Macro F1-Score of ~0.80 on the best fold. By utilizing Focal Loss and ROC-AUC evaluation, the metrics confirm the model successfully learned the minority disease classes rather than merely predicting the majority class.

**Current Limitations & Future Work:**

* **Rigid Masking:** The HSV mask can occasionally filter out severe necrotic spots that mathematically match the background soil.

* **Single-Label Constraint:** The model utilizes Softmax for single-class prediction, whereas real-world cassava plants often suffer from co-infections. Future iterations will pivot to BCEWithLogitsLoss for multi-label classification.

* **Self-Supervised Pre-training:** Given more computational time, integrating the 12,000 unlabeled extra images using a framework like MoCo would further stabilize the backbone before supervised fine-tuning.