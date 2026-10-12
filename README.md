# 🐱🐶 Dogs vs Cats Image Classification using CNN

A Deep Learning project that classifies images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

The project uses image preprocessing, Batch Normalization, Dropout, and model evaluation techniques to train an image classification model.

## 📌 Project Overview

Image classification is an important application of Computer Vision. In this project, a CNN learns visual features from labeled cat and dog images and predicts the class of an input image.

The model is trained on the Dogs vs Cats dataset and evaluated using accuracy and loss metrics.

## ✨ Features

- Dogs vs Cats image classification
- CNN-based deep learning architecture
- Image resizing to 256 × 256 pixels
- Pixel normalization to the range [0, 1]
- Batch Normalization for more stable training
- Dropout layers to help reduce overfitting
- Training and validation accuracy monitoring
- Training and validation loss visualization
- Prediction on individual images

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- KaggleHub
- Google Colab

## 📂 Dataset

**Dataset:** [Dogs vs Cats — Kaggle](https://www.kaggle.com/datasets/salader/dogsvscats)

The dataset is downloaded using KaggleHub directly from the notebook.

```python
import kagglehub

path = kagglehub.dataset_download("salader/dogsvscats")

print("Dataset path:", path)
```

The notebook loads the training and testing images from the dataset's `train` and `test` directories.

- Training images: 20,000
- Testing images: 5,000
- Image size: 256 × 256 pixels
- Classes: Cats and Dogs

## 🧠 CNN Architecture

The model uses the following layers:

1. **Conv2D (32 filters):** Extracts low-level image features.
2. **Batch Normalization:** Helps stabilize neural network training.
3. **MaxPooling2D:** Reduces spatial dimensions.
4. **Conv2D (64 filters):** Learns more complex features.
5. **Batch Normalization and MaxPooling2D**
6. **Conv2D (128 filters):** Learns higher-level visual features.
7. **Batch Normalization and MaxPooling2D**
8. **Flatten:** Converts feature maps into a one-dimensional vector.
9. **Dense (128 units):** Learns feature combinations.
10. **Dropout (0.3):** Helps reduce overfitting.
11. **Dense (64 units):** Learns additional representations.
12. **Dropout (0.3):** Provides additional regularization.
13. **Dense (1 unit, Sigmoid):** Produces a binary classification score.

### Model Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Batch Size | 32 |
| Epochs | 10 |
| Input Shape | 256 × 256 × 3 |
| Output | Binary classification |

## ⚙️ How to Run the Project

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab.
2. Run the import cells.
3. Download the dataset using KaggleHub.
4. Load the training and testing datasets.
5. Normalize image pixel values.
6. Build and compile the CNN model.
7. Train the model.
8. Visualize accuracy and loss graphs.
9. Test predictions on individual images.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Dogs-Vs-Cats-CNN.git
```

Navigate to the project directory:

```bash
cd Dogs-Vs-Cats-CNN
```

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib opencv-python kagglehub
```

Open the notebook in Jupyter Notebook or upload it to Google Colab.

**Note:** The notebook was developed for Google Colab. Local execution may require additional dataset configuration and image-path updates.

## 📊 Model Evaluation

The notebook plots training and validation accuracy and loss across epochs.

These visualizations help assess model learning and identify potential overfitting.

- **Accuracy:** Measures the proportion of correctly classified images.
- **Loss:** Measures the model's prediction error.
- **Training curves:** Show performance on the training dataset.
- **Validation curves:** Show performance on the dataset supplied as `validation_data` during training.

## 🔮 Image Prediction

The trained model can predict an individual image after resizing it to the required input dimensions.

```python
import cv2
import numpy as np

image = cv2.imread("/content/cat.jfif")
image = cv2.resize(image, (256, 256))

image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)

print("Prediction score:", prediction[0][0])
```

**Important:** The prediction code above assumes that the image has been loaded successfully and that the model was trained with normalized inputs. A sigmoid score must be mapped to the correct class using the dataset's class-label mapping. OpenCV loads images in BGR order, so convert to RGB if required by your preprocessing pipeline.

## 🚀 Future Improvements

- Add data augmentation.
- Use a separate validation dataset and reserve the test dataset for final evaluation.
- Evaluate precision, recall, F1-score, and a confusion matrix.
- Experiment with transfer learning using MobileNetV2 or EfficientNet.
- Save the trained model for reuse.
- Build a web interface for image uploads and predictions.

## 📁 Repository Structure

```text
Dogs-Vs-Cats-CNN/
│
├── dogs_vs_cats_cnn.ipynb
└── README.md
```

The notebook filename may differ depending on the name used when uploading it to GitHub.

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Computer Vision and image classification
- CNN architecture design
- Image preprocessing and normalization
- Batch Normalization and Dropout
- Model training and evaluation
- Visualization of learning curves
- Making predictions with a trained model

## 👨‍💻 Author

**Waheed Touheed**

GitHub: [WaheedTouheed](https://github.com/WaheedTouheed)

---

⭐ If you find this project useful, consider giving the repository a star.
