# Pneumonia Detection from Chest X-Rays using Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/PROJECT1.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/status-completed-brightgreen.svg)

A binary image classification project that detects **pneumonia** from chest X-ray images. Two deep learning models are built, trained and compared on the same data pipeline:

1. A **Custom CNN** trained from scratch
2. A **DenseNet121 transfer learning** model with frozen ImageNet weights

Both models reach about **90% accuracy** and **94-95% recall on pneumonia** on the held-out test set.

> **Disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Architectures](#model-architectures)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Key Observations](#key-observations)
- [Tech Stack](#tech-stack)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Author](#author)

---

## Overview

Pneumonia is a lung infection that is commonly diagnosed with the help of chest X-rays. Reading these images takes trained radiologists, and an automated screening aid could help prioritise cases. This project explores how well deep learning can separate **NORMAL** chest X-rays from **PNEUMONIA** ones, and compares a lightweight custom CNN against a pretrained DenseNet121 backbone.

**Highlights**

- End-to-end pipeline in a single Colab notebook: data download, loading, augmentation, training, evaluation and inference
- Efficient `tf.data` input pipeline with `cache()` and `prefetch()`
- **Balanced class weights** to handle the class imbalance in the training set
- **Data augmentation** (random flip, rotation, zoom) built into the models
- **Early stopping** on validation loss with best-weight restoration
- Evaluation with classification reports, confusion matrices and AUC
- Single-image inference that shows the predicted label and confidence

---

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle (`paultimothymooney/chest-xray-pneumonia`, about 2.29 GB zipped)
- **Task:** Binary classification, `NORMAL` vs `PNEUMONIA`
- **Class labels:** `['NORMAL', 'PNEUMONIA']` (`NORMAL` = 0, `PNEUMONIA` = 1)

| Split | Images |
|-------|-------:|
| Train | 5,216 |
| Validation | 16 |
| Test | 624 (234 NORMAL, 390 PNEUMONIA) |

**Class imbalance:** The training set contains far more pneumonia images than normal ones. The notebook computes balanced class weights and applies them during training:

| Class | Weight |
|-------|-------:|
| NORMAL (0) | 1.945 |
| PNEUMONIA (1) | 0.673 |

Expected folder layout after download:

```
chest_xray_data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

> The dataset is **not** included in this repository. The notebook downloads it directly from Kaggle.

---

## Project Workflow

1. **Setup:** install the Kaggle CLI, upload `kaggle.json`, download and unzip the dataset into Colab storage
2. **Hardware check:** confirm that a GPU is available
3. **Configuration:** set paths and hyperparameters (image size, batch size, epochs)
4. **Data loading:** build `train_ds`, `val_ds` and `test_ds` with `tf.keras.utils.image_dataset_from_directory` (binary labels, 224 x 224 images)
5. **Pipeline optimisation:** `cache()` and `prefetch(AUTOTUNE)`
6. **Class weights:** compute balanced weights from the training labels
7. **Augmentation:** random horizontal flip, rotation and zoom
8. **Model building:** Custom CNN and DenseNet121 transfer model
9. **Training:** both models with early stopping and class weights
10. **Evaluation:** classification report and confusion matrix on the test set
11. **Inference:** predict a random test image and display the label with its confidence

---

## Model Architectures

### Model 1: Custom CNN

| Layer | Details |
|-------|---------|
| Input | 224 x 224 x 3 |
| Data augmentation | RandomFlip (horizontal), RandomRotation (0.08), RandomZoom (0.08) |
| Rescaling | 1/255 |
| Conv block 1 | Conv2D (32, 3x3, ReLU, same) → BatchNorm → MaxPool (2x2) |
| Conv block 2 | Conv2D (64, 3x3, ReLU, same) → BatchNorm → MaxPool (2x2) |
| Conv block 3 | Conv2D (128, 3x3, ReLU, same) → BatchNorm → MaxPool (2x2) |
| Head | Flatten → Dense (128, ReLU) → Dropout (0.5) → Dense (1, sigmoid) |

### Model 2: DenseNet121 Transfer Learning

| Layer | Details |
|-------|---------|
| Input | 224 x 224 x 3 |
| Data augmentation | RandomFlip (horizontal), RandomRotation (0.08), RandomZoom (0.08) |
| Preprocessing | `densenet.preprocess_input` |
| Backbone | DenseNet121, ImageNet weights, `include_top=False`, **frozen** |
| Head | GlobalAveragePooling2D → BatchNorm → Dense (128, ReLU) → Dropout (0.4) → Dense (1, sigmoid) |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224 x 224 |
| Batch size | 32 |
| Max epochs | 10 |
| Optimizer | Adam (learning rate = 1e-4) |
| Loss | Binary cross-entropy |
| Metrics | Accuracy, AUC |
| Class weights | Balanced (computed from training labels) |
| Callback | `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)` |
| Hardware | GPU on Google Colab |

**Training behaviour**

- **Custom CNN:** early stopping triggered after **7 epochs**, and the weights from the best validation-loss epoch (epoch 4) were restored. Training took about 20 seconds per epoch on GPU.
- **DenseNet121:** ran all **10 epochs** (about 24 seconds per epoch), reaching a final training accuracy of 95.71% and training AUC of 0.9930.

---

## Results

Evaluation on the **test set (624 images)** with a 0.5 decision threshold.

### Classification Report

**Custom CNN**

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| NORMAL | 0.91 | 0.82 | 0.86 | 234 |
| PNEUMONIA | 0.90 | 0.95 | 0.92 | 390 |
| **Accuracy** | | | **0.90** | 624 |
| Macro avg | 0.90 | 0.88 | 0.89 | 624 |
| Weighted avg | 0.90 | 0.90 | 0.90 | 624 |

**DenseNet121 (Transfer Learning)**

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| NORMAL | 0.89 | 0.83 | 0.86 | 234 |
| PNEUMONIA | 0.90 | 0.94 | 0.92 | 390 |
| **Accuracy** | | | **0.90** | 624 |
| Macro avg | 0.90 | 0.89 | 0.89 | 624 |
| Weighted avg | 0.90 | 0.90 | 0.90 | 624 |

### Confusion Matrices

| Custom CNN | DenseNet121 |
|:----------:|:-----------:|
| ![Custom CNN confusion matrix](images/custom_cnn_confusion_matrix.png) | ![DenseNet121 confusion matrix](images/densenet121_confusion_matrix.png) |

| Model | True Normal (correct) | False Pneumonia | False Normal | True Pneumonia (correct) |
|-------|----------------------:|----------------:|-------------:|-------------------------:|
| Custom CNN | 192 | 42 | 20 | 370 |
| DenseNet121 | 195 | 39 | 24 | 366 |

### Model Comparison

| Metric | Custom CNN | DenseNet121 |
|--------|-----------:|------------:|
| Test accuracy | ~90.1% | ~89.9% |
| Pneumonia recall | 0.95 | 0.94 |
| Pneumonia precision | 0.90 | 0.90 |
| Normal recall | 0.82 | 0.83 |
| Macro F1 | 0.89 | 0.89 |
| Epochs trained | 7 (early stopped) | 10 |

### Sample Inference

The notebook picks a random test image, runs it through the DenseNet121 model and shows the prediction with a confidence percentage. For the run included in the notebook:

```
Prediction: PNEUMONIA
Confidence: 95.69%
```

---

## Key Observations

- Both models perform almost identically on the test set (about 90% accuracy, macro F1 of 0.89). In this setup, the frozen DenseNet121 backbone did not beat the custom CNN.
- Recall on **PNEUMONIA is high (94-95%)**, which is desirable for a screening-style task where missing a positive case is costly. The trade-off is lower recall on **NORMAL (82-83%)**, so roughly 39-42 healthy X-rays were flagged as pneumonia.
- The Custom CNN was **unstable early in training**: its validation loss started very high (about 9 in the first epochs) before settling around epoch 4. Early stopping with best-weight restoration helped here.
- The DenseNet121 model trained **more smoothly**, with validation loss decreasing steadily from 0.35 to 0.07.
- The validation folder has only **16 images**, so validation metrics are noisy (for example, the DenseNet121 validation accuracy reached 100%). The **test set results are the more reliable measure** of performance.

---

## Tech Stack

- **Language:** Python 3
- **Deep learning:** TensorFlow / Keras
- **Data and metrics:** NumPy, scikit-learn
- **Visualisation:** Matplotlib, Seaborn
- **Dataset access:** Kaggle CLI
- **Environment:** Google Colab (GPU runtime)

---

## Installation and Setup

### Option 1: Google Colab (recommended)

1. Open `PROJECT1.ipynb` in Google Colab (use the badge at the top, after updating the link with your GitHub details)
2. Go to **Runtime → Change runtime type** and select a **GPU**
3. Run the cells in order

### Option 2: Run locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn kaggle jupyter
```

> **Note:** The first notebook cell uses `google.colab.files` to upload `kaggle.json`. When running locally, remove that part, place `kaggle.json` in `~/.kaggle/` yourself, and download the dataset with the Kaggle CLI.

### Kaggle API credentials

1. Log in to [Kaggle](https://www.kaggle.com/) and go to **Settings → API → Create New Token**
2. This downloads `kaggle.json`
3. Upload it when the notebook asks for it

> **Never commit `kaggle.json` to GitHub.** It contains your private API key.

---

## How to Run

Run the notebook cells in order:

1. **Download data:** install Kaggle CLI, upload `kaggle.json`, download and unzip the dataset
2. **Imports and GPU check**
3. **Paths and hyperparameters**
4. **Load datasets** and optimise the input pipeline
5. **Compute class weights**
6. **Define augmentation and both models**
7. **Train both models** (Custom CNN first, then DenseNet121)
8. **Evaluate** on the test set (classification report and confusion matrices)
9. **Run inference** on a random test image

To change the setup, edit the configuration cell:

```python
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 10
```

---

## Repository Structure

```
.
├── PROJECT1.ipynb                          # Complete notebook (data, training, evaluation, inference)
├── README.md                               # Project documentation
└── images/
    ├── custom_cnn_confusion_matrix.png     # Custom CNN test confusion matrix
    └── densenet121_confusion_matrix.png    # DenseNet121 test confusion matrix
```

---

## Limitations

- **Small validation set:** only 16 images, so it is a weak signal for early stopping and model selection
- **Single dataset:** the images come from one source (paediatric chest X-rays), so the models may not generalise to other hospitals, scanners or age groups
- **Binary only:** the models do not distinguish bacterial from viral pneumonia, and do not detect other conditions
- **No fixed random seed:** results can vary slightly between runs
- **Frozen backbone:** DenseNet121 was used only as a feature extractor, with no fine-tuning
- **Not clinically validated:** see the disclaimer above

---

## Future Improvements

- Fine-tune the top layers of DenseNet121 (unfreeze with a low learning rate)
- Create a larger, stratified validation split from the training data
- Tune the decision threshold to trade off precision and recall
- Add **Grad-CAM** heatmaps to show which lung regions drive each prediction
- Compare more backbones (ResNet50, EfficientNet, VGG16) and try ensembling
- Add ROC and precision-recall curves to the evaluation
- Set random seeds for reproducibility
- Deploy as a small web app (Streamlit or Gradio) for image upload and prediction

---

## Acknowledgements

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney on Kaggle. The images originate from the work of Kermany et al., *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"*, Cell, 2018.
- DenseNet121 architecture: Huang et al., *"Densely Connected Convolutional Networks"*, CVPR 2017.
- Pretrained ImageNet weights provided through `tf.keras.applications`.

---

## License

Add your preferred license here (for example, MIT) and include a `LICENSE` file in the repository. Please check the dataset's own terms on Kaggle before redistributing any of its images.

---

## Author

**<Your Name>**
GitHub: [@<your-username>](https://github.com/<your-username>)
LinkedIn: <your-linkedin-url>

If you found this project useful, consider giving it a star.
