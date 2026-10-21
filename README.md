# Fashion-MNIST Classification

Course project comparing k-nearest neighbours (kNN), multiclass logistic regression, and a convolutional neural network trained from scratch on Fashion-MNIST.

## Task and scope

Predict one of ten clothing, footwear, or accessory categories from a 28 × 28 grayscale image. This is supervised multiclass classification. Conclusions will concern the benchmark images; performance on everyday phone photographs is untested.

Dataset: [Fashion-MNIST by Zalando Research](https://github.com/zalandoresearch/fashion-mnist).

## Status

Initial project structure only. Data has not been downloaded, models have not been trained, and no evaluation results are available.

- [Approved project proposal](docs/project-idea-ml%26ai.md)
- [Course project requirements](docs/project-requirements.txt)
- [Work plan](docs/plans/work-plan.md)

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
```
