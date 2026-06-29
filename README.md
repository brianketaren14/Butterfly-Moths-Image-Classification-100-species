# Butterfly and Moth Classification 100 Species

## Project Overview
This project is a multi-class image classification task to identify butterfly and moth species. It uses a dataset containing 100 species classes and trains deep learning models to recognize each species from input images.

## Dataset
The dataset is organized into separate folders for training, validation, and testing. It also includes a CSV file named `butterflies and moths.csv` that describes image metadata and dataset splits.

## Approach
This project uses convolutional neural networks (CNNs) with transfer learning. Two pretrained models are used as feature extractors:
- EfficientNetB3
- ResNet50

These models are selected for their strong performance and efficient use of computational resources.

## Data Preparation
- Images are loaded from the `dataset/train`, `dataset/valid`, and `dataset/test` directories.
- Data augmentation and preprocessing are performed using `ImageDataGenerator`.
- Images are resized to `150x150` pixels and normalized before training.

## Model Architecture
Each model is built by:
1. Loading the pretrained base model with `include_top=False`.
2. Adding a `GlobalMaxPooling2D` layer.
3. Adding a `Dropout` layer for regularization.
4. Adding a dense `ReLU` layer with 512 units.
5. Adding a final dense softmax layer with 100 output units.

## Training
- Both models are compiled with the Adam optimizer and categorical crossentropy loss.
- Training is run for up to 10 epochs.
- A callback stops training early when both training and validation accuracy exceed 95%.

## Evaluation
The notebook evaluates both models using:
- Accuracy and loss curves
- Precision, recall, and f1-score with macro averaging
- Confusion matrices
- Sample prediction visualizations on test images

## Results
The notebook reports that EfficientNetB3 performs better than ResNet50 on this task, with EfficientNetB3 achieving around 97% accuracy and ResNet50 achieving around 92% accuracy.

<img width="1661" height="1597" alt="image" src="https://github.com/user-attachments/assets/cfe5e996-d3c0-4e56-b35f-84798edee939" />


## Output Files
The notebook saves:
- `best_butterflye3.h5` for the EfficientNetB3 model
- `resnet50.h5` for the ResNet50 model
- `class_names.pkl` for the class label mapping
- `trainHistoryDict` and `trainHistoryDictResnet` for training history data

## Notes
This project demonstrates how transfer learning can be applied to a fine-grained species classification problem using high-capacity CNN architectures and a structured image dataset.
