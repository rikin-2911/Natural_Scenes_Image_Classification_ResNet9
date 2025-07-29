# 🌄 Natural Scenes Image Classification with Custom ResNet9 (PyTorch)

Welcome to the **Natural Scenes Image Classification** project — a deep learning-based image classifier built from scratch using a custom **ResNet9** architecture in **PyTorch**. This model was trained on Intel dataset of over **14,000+ natural scene images** and achieved an impressive **~91% accuracy**.

## 📌 Project Highlights

- 🔍 **Task**: Multi-class image classification of natural scenes (e.g., forests, deserts, mountains, etc.)
- 🧠 **Model**: Custom-built **ResNet9** architecture
- 📦 **Framework**: PyTorch (built from scratch, no pre-trained weights)
- 🔁 **Data Augmentation**: Applied transformations for robustness
- 🎯 **Accuracy Achieved**: ~91% on validation/test set
- 🗂️ **Dataset**: Intel Image dataset of 17,000+ labeled images across multiple natural categories

---
## 🧰 Tech Stack & Tools

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-FA0F00?style=for-the-badge&logo=Jupyter&logoColor=white)
![CNN](https://img.shields.io/badge/CNN-DeepLearning-8A2BE2?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)


---
## 📂 Dataset Overview

The dataset used contains 14,000+ high-quality labeled images of various natural scenes, such as:

- Forest
- Mountain
- Sea
- Glacier
- Buildings
- Streets
  

> The data is organized in folders by class and split into training and validation sets.

---

## 🧱 Model Architecture - ResNet9 (Custom)

Unlike standard deep CNNs, **ResNet9** is a compact and efficient architecture that stacks 9 layers with **residual connections**. It balances **computational efficiency** and **accuracy**, making it ideal for moderate-sized datasets.

### Key Components:

- Convolutional layers with BatchNorm and ReLU
- Skip connections for residual learning
- MaxPooling and AdaptiveAvgPooling
- Fully connected (Linear) classification head

📌 Built entirely from scratch using `torch.nn` modules.

---

## 🔄 Data Augmentation Techniques

To improve generalization and prevent overfitting, the following transformations were applied using `torchvision.transforms`:

- Random Horizontal Flip
- Random Rotation
- Random Crop / Resize
- Normalization with dataset-specific mean and std

---

## 📈 Training Details

- Optimizer: `Adam`
- Loss Function: `CrossEntropyLoss`
- Epochs: 20+
- Batch Size: 64
- Learning Rate: 1e-3 (with scheduling)
- Hardware: Trained on GPU (CUDA)

---

## 🚀 Results

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | ~91%      |
| Model Size    | Lightweight |
| Generalization | Excellent |
| Overfitting   | Minimal due to augmentation |

📉 Training and validation accuracy/loss plots included in the notebook.

## 🙋‍♂️ About Me
- 👨‍💻 Created with 💡 and PyTorch by Rikin Pithadia
- 🔬 Data Science & AI Enthusiast | Building AI that sees the world

