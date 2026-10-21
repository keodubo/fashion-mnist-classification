# Fashion-MNIST Classification

Course project comparing k-nearest neighbours (kNN), multiclass logistic regression, and a convolutional neural network trained from scratch on Fashion-MNIST.

## Task and scope

Predict one of ten clothing, footwear, or accessory categories from a 28 × 28 grayscale image. This is supervised multiclass classification. Conclusions will concern the benchmark images; performance on everyday phone photographs is untested.

Dataset: [Fashion-MNIST by Zalando Research](https://github.com/zalandoresearch/fashion-mnist).

## Status

Initial project structure only. Data has not been downloaded, models have not been trained, and no evaluation results are available.

- [Approved project proposal](docs/project-idea-ml%26ai.md)
- [Course project requirements](docs/project-requirements.txt)

## Setup

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

The requirements contain the initial exploration and classical-model tools. Versions are not locked yet; record a tested environment before running comparative experiments. The CNN framework will be added when its implementation is selected.

## Layout

```text
docs/               Original proposal and assignment requirements
notebooks/          Exploratory analysis and experiment notebooks
src/fashion_mnist/  Reusable project code
data/               Local datasets (ignored by Git)
results/            Local generated results and model files (ignored by Git)
```

## Work plan

1. Inspect image shapes, labels, pixel ranges, missing values, class balance, and duplicates; display sample images and discuss dataset limitations.
2. Reserve validation observations from the training set, preserving class proportions and keeping identified duplicate groups together. Keep the official test set for final evaluation and audit any overlap before interpreting results.
3. Normalize pixel values to 0–1. Flatten images for classical models and preserve spatial structure for the CNN. Use identical observations across models and record the split and random seeds.
4. Establish logistic-regression and kNN baselines, then train the CNN. Select settings using validation data only.
5. Evaluate accuracy, macro-F1, per-class recall, confusion matrices, and misclassified examples. Accuracy measures overall correctness; macro-F1 and per-class recall expose uneven performance across categories.
6. Compare training and validation performance, interpret limitations, and prepare the code submission and 15-minute presentation.

First milestone: a reproducible data-exploration notebook and documented evaluation split.
