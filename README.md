# Lung Cancer Detection using Transfer Learning

A custom convolutional neural network (CNN) classifier augmented with InceptionV3 pretrained model layers and built with TensorFlow/Keras to classify lung histopathology images into three categories: adenocarcinoma, squamous cell carcinoma, and normal lung tissue.

---

## Overview

This project trains an image classification model on lung histology slides to distinguish between two malignant cancer subtypes and healthy tissue. The model achieves approximately **90% overall accuracy** on the validation set, with particularly strong performance on normal tissue (97% F1-score).

---

## Dataset

The notebook uses the [Lung Cancer dataset](https://www.kaggle.com/datasets/adnanabdulfetah/lung-cancer) from Kaggle, which contains histopathological images organized into three classes:

| Class | Description |
|---|---|
| `lung_aca` | Lung adenocarcinoma |
| `lung_scc` | Lung squamous cell carcinoma |
| `lung_n` | Normal lung tissue |

---

## Requirements

- Python 3.12+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- OpenCV (`cv2`)
- scikit-learn

Install dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib pillow opencv-python scikit-learn
```

---

## Usage

The notebook is designed to run on Kaggle with a GPU accelerator (NVIDIA Tesla T4). To run it locally or on another platform:

1. Download the dataset from the Kaggle link above and update the `path` variable to point to your local copy of `lung_image_sets/`.
2. Open the notebook and run all cells in order.
3. The final cells will output a confusion matrix and a full classification report.

---

## Results

Evaluated on a validation set of 3,000 images (1,000 per class):

```
              precision    recall  f1-score   support

    lung_aca       0.93      0.78      0.85      1021
    lung_scc       0.80      0.98      0.88       961
      lung_n       0.99      0.95      0.97      1018

    accuracy                           0.90      3000
   macro avg       0.91      0.90      0.90      3000
weighted avg       0.91      0.90      0.90      3000
```

**Confusion Matrix:**

```
[[795, 220,   6],
 [ 18, 943,   0],
 [ 38,   9, 971]]
```

The model performs best on normal tissue and lung squamous cell carcinoma. The primary source of misclassification is adenocarcinoma being predicted as squamous cell carcinoma, which is a known challenge in lung histopathology due to visual similarity between subtypes.

---
