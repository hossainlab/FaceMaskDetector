# Face Mask Detection Using Convolutional Neural Network

This project implements a Convolutional Neural Network (CNN) model in Python to detect whether a person is wearing a face mask or not. The project showcases a complete pipeline, including data preprocessing, model training, evaluation, and visualization, aimed at addressing public health safety measures during pandemics.

---

## Table of Contents

- [Face Mask Detection Using Convolutional Neural Network](#face-mask-detection-using-convolutional-neural-network)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Methods](#methods)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Model Architecture](#2-model-architecture)
    - [3. Model Training](#3-model-training)
    - [4. Model Evaluation](#4-model-evaluation)
    - [5. Visualization and Interpretation](#5-visualization-and-interpretation)
  - [Installation and Usage](#installation-and-usage)
    - [Prerequisites](#prerequisites)

---

## Project Overview

This repository provides a deep learning solution to detect face masks using CNNs. The model is designed for real-time detection and is suitable for deployment in public health monitoring systems to ensure compliance with safety protocols.

---

## Dataset

The dataset contains images of individuals with two classes: **with mask** and **without mask**. These images were preprocessed to enhance training efficiency. The dataset was split into training, validation, and testing sets to ensure robust evaluation.

---

## Methods

### 1. Data Preprocessing

- **Data Augmentation**: Applied transformations like rotation, flipping, and scaling to increase data diversity.
- **Normalization**: Scaled pixel values to the [0, 1] range for better model performance.
- **Train-Test Split**: Divided data into 80% training and 20% testing subsets.

### 2. Model Architecture

The CNN architecture includes:
- **Input Layer**: Accepts resized 224x224 RGB images.
- **Convolutional Layers**: Extract spatial features using filters and ReLU activation.
- **Pooling Layers**: Downsample feature maps to reduce computational complexity.
- **Dense Layers**: Fully connected layers for classification.
- **Output Layer**: Softmax activation for binary classification (with mask vs. without mask).

### 3. Model Training

- **Loss Function**: Categorical cross-entropy for binary classification.
- **Optimizer**: Adam optimizer for efficient learning.
- **Batch Size**: 32 images per batch for balanced memory usage and performance.
- **Epochs**: Up to 25 epochs with early stopping based on validation loss.

### 4. Model Evaluation

- **Accuracy**: Measures the percentage of correctly classified images.
- **Confusion Matrix**: Evaluates true positives, true negatives, false positives, and false negatives.
- **Precision, Recall, F1-Score**: Assess classification performance, particularly for the minority class.

### 5. Visualization and Interpretation

- **Training Curves**: Plot accuracy and loss during training and validation.
- **Confusion Matrix**: Visualize class-wise prediction performance.
- **ROC Curve**: Demonstrates the trade-off between sensitivity and specificity.

---

## Installation and Usage

### Prerequisites

Ensure Python 3.8+ is installed along with the following libraries:
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`
- `opencv-python`

Install dependencies with:
```bash
pip install numpy pandas tensorflow matplotlib opencv-python
```