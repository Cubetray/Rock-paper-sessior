README.md
# Rock Paper Scissors Classification using Transfer Learning

This project uses **Transfer Learning with MobileNetV2** to classify hand gestures representing **Rock, Paper, and Scissors** using TensorFlow and TensorFlow Datasets.

## 📌 Objective

Build an image classification model that:
- Uses a small dataset (`rock_paper_scissors` from TensorFlow Datasets)
- Applies transfer learning using a pre-trained MobileNetV2 model
- Classifies input images into three categories: **Rock**, **Paper**, or **Scissors**

## 🧠 Model Overview

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Added Layers**: Global Average Pooling + Dense (3-class softmax)
- **Loss**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Epochs**: 5

## 🔍 Dataset

- Loaded using `tensorflow_datasets`
- Split: 80% training, 20% validation
- Preprocessed with resizing and normalization

## 📊 Training Output

The model trains for 5 epochs and displays training/validation accuracy using Matplotlib.

## 📁 Requirements

Make sure you have the following Python libraries installed:

```bash
pip install tensorflow tensorflow_datasets matplotlib
