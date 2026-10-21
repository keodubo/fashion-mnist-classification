## 1. Task to solve

**Classifying clothing images using machine learning.**

This project will use supervised multiclass classification to predict the category of a clothing item from a grayscale image. Given an image, the model will output one of ten clothing, footwear, or accessory categories.

The objective is to compare two classical models with a convolutional neural network (CNN) and investigate whether learning spatial patterns improves classification. The analysis will identify which categories are most frequently confused and examine examples of incorrect predictions. Conclusions will apply to the benchmark images; performance on everyday phone photographs will remain untested.

## 2. Datasets to use

The project will use **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)**, published by Zalando Research.

- **Size:** 60,000 training images and 10,000 test images.
- **Inputs:** 28 × 28 grayscale images, with 784 pixel values per image.
- **Target classes:** T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

Preparation will include inspecting image quality, class counts, labels, and possible duplicates, then scaling pixel values from 0–255 to 0–1. A validation subset will be reserved from the training set while maintaining class proportions and keeping any identified duplicate groups together. The official test set will be reserved for final evaluation.

For the classical models, each image will be flattened into 784 features. The CNN will retain the two-dimensional image structure. All three models will use the same training, validation, and test observations.

## 3. Models to use

The project will follow the **two simple models + one complex model** option:

| Model | Type | Purpose in the comparison | Main limitation |
|---|---|---|---|
| k-nearest neighbours (kNN) classifier | Simple | Classify images using the labels of images with similar pixel values. | Prediction can be slow and sensitive to small image shifts. |
| Multiclass logistic regression | Simple | Provide a linear classification reference using pixel values. | Cannot directly learn nonlinear combinations of visual patterns. |
| Convolutional neural network (CNN) | Complex | Learn local patterns and combine them to distinguish categories. | Requires architecture and training choices and can overfit. |

The proposed CNN will use two convolutional layers with ReLU activations and max pooling, followed by a hidden dense layer and a ten-class output. It will be trained from scratch.

Model settings will be selected using validation results, including the number of neighbours, logistic-regression regularisation, and CNN training duration. Accuracy will be the main evaluation metric, supplemented by macro-F1, per-class recall, and a confusion matrix. Training and validation performance will be compared to assess overfitting, and misclassified images will be displayed to explain recurring errors.
