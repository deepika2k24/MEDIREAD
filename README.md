# MEDIREAD – AI System to Decode Doctors' Handwriting

**MEDIREAD** is an AI-powered system designed to recognize and extract text from doctors' handwritten prescriptions. The system leverages a Convolutional Neural Network (CNN) for handwriting recognition and integrates both local Tesseract OCR and the OCR.Space API for enhanced accuracy.

---

## Features

- **CNN-based handwriting recognition** for prescriptions.
- **Dual OCR processing system**:
  - Local Tesseract OCR (offline, customizable)
  - OCR.Space API (high accuracy, cloud-based)
- **Automatic preprocessing pipeline**:
  - Grayscale conversion
  - Resizing to 64×64 pixels
  - Pixel normalization
  - Noise removal and thresholding
- **Decision system** to compare OCR results and select the most reliable output.
- **JSON output** with extracted prescription text and metadata.
- **Visualization of uploaded prescriptions**.

---

## Dataset

- **Source**: Kaggle – Doctors Handwritten Prescription (BD Dataset)
- **Format**: Scanned prescription images (PNG, JPG, JPEG)
- **Preprocessing**:
  - Grayscale conversion
  - Resize to 64×64 pixels
  - Pixel normalization
  - Validation checks for corrupted/empty images
- **Structure**: Organized in folders by prescription types/handwriting styles

---

## Model Architecture

- **Framework**: TensorFlow 2.x / Keras
- **Type**: Convolutional Neural Network (CNN)
- **Input**: 64×64×1 grayscale images
- **Layers**:
  - Conv2D → BatchNorm → MaxPooling2D → Dropout (3 blocks)
  - Flatten → Dense(256, ReLU) → BatchNorm → Dropout(0.5)
  - Output Layer: Softmax (multi-class) or Sigmoid (binary)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Training Accuracy**: ~92% (sample dataset)
- **Trained Model**: `prescription_model.h5`
