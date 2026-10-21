# Work plan

1. Inspect image shapes, labels, pixel ranges, missing values, class balance, and duplicates; display sample images and discuss dataset limitations.
2. Reserve validation observations from the training set, preserving class proportions and keeping identified duplicate groups together. Keep the official test set for final evaluation and audit any overlap before interpreting results.
3. Normalize pixel values to 0–1. Flatten images for classical models and preserve spatial structure for the CNN. Use identical observations across models and record the split and random seeds.
4. Establish logistic-regression and kNN baselines, then train the CNN. Select settings using validation data only.
5. Evaluate accuracy, macro-F1, per-class recall, confusion matrices, and misclassified examples. Accuracy measures overall correctness; macro-F1 and per-class recall expose uneven performance across categories.
6. Compare training and validation performance, interpret limitations, and prepare the code submission and 15-minute presentation.

First milestone: a reproducible data-exploration notebook and documented evaluation split.

## Local data and results

```text
data/               Local datasets (ignored by Git)
results/            Local generated results and model files (ignored by Git)
```
