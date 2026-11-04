# Final evaluation protocol

Frozen on 2026-10-28 before final test evaluation. No test data were loaded and no training was performed when recording this protocol. This supersedes the pending decisions in model-selection.md.

## Objective and models

Task: ten-class Fashion-MNIST classification. Primary metric: accuracy under equal costs for category errors. The wider CNN is the preselected primary model; logistic regression and kNN are predeclared comparison baselines, not candidates to select using test results.

- Logistic regression: C=0.1, L2, lbfgs, max_iter=2000, tol=0.0001; use the existing selected model.
- kNN: k=3, Euclidean distance, uniform votes, brute force; use the existing selected model.
- CNN: wider architecture, no weight decay, original seed 42, lowest-validation-loss checkpoint at epoch 12. Conv(1→32,3×3,padding=1), ReLU, max-pool(2); Conv(32→64,3×3,padding=1), ReLU, max-pool(2); flatten; dense 128, ReLU, dropout 0.25; dense 10 logits. 421,642 parameters.

The CNN was trained with Adam, initial learning rate 0.001, batches of 128, maximum 30 epochs, patience 5, and ReduceLROnPlateau (factor 0.5, patience 2, threshold 0, min_lr=0.00001). Retain the original checkpoint rather than selecting the best seed from the stability study. The weight-decay alternative remains a documented development result and is excluded from final test evaluation.

## Training policy

Evaluate the existing checkpoints trained on the shared 48,000-image training subset. Do not refit on all 60,000 images. The 12,000 validation observations were used for model and checkpoint selection. This choice preserves the exact reviewed models and their validation evidence; the trade-off is foregoing possible gains from training on additional observations. It is not a claim that refitting would perform worse.

## Preprocessing and execution

Use the official 10,000 test observations, unchanged and in original order. Verify publisher checksums and IDX structure before evaluation. Divide pixels by 255. Classical models receive flattened float64 vectors; CNN receives float32 N×1×28×28 tensors. No preprocessing statistics are fitted on test data.

Run inference on CPU with four computational threads and batches of 256, CNN in eval/inference mode. Check current checkpoint fingerprints against the table below before inference; if a file differs, stop and resolve provenance rather than silently evaluating a different model. Verify CPU predictions on the saved validation data against prior saved predictions before opening the test set. The existing notebook 08 already reports zero disagreements for these models under its CPU comparison.

## Reporting fixed before testing

Evaluate each of the three frozen models once in one final evaluation stage. Save image IDs, true/predicted classes, and class probabilities where available, plus source/model fingerprints and environment versions.

Report accuracy; macro precision, macro recall, and macro-F1; per-class precision, recall, F1, and support; count and row-normalized confusion matrices. Include shirt recall explicitly. Predict by maximum class probability / the existing model's classification rule. Do not introduce a confidence threshold. Use the known balanced-class constant-prediction reference of 10% accuracy.

Perform an exact-image overlap audit between the development pool and test set without modifying either. Primary metrics always cover all official test observations. If overlaps exist, additionally report all three models' metrics on the same overlap-free test subset, with its size and class counts, explicitly as a sensitivity analysis. Do not alter training or select a model based on this audit.

For error illustrations, sample up to 12 errors per model with seed 42 and retain source IDs. Do not treat sampled pictures as proof of causation or wrong labels. Compare final metrics with validation results while retaining their split labels.

No hyperparameter, architecture, epoch, seed, preprocessing, or winner changes based on test performance. A technical failure must be documented; retry only to evaluate the same frozen models correctly. A disappointing result is reported, not tuned away.

## Frozen checkpoint fingerprints

| Model | Local artifact | SHA-256 |
|---|---|---|
| Logistic regression | `results/logistic-regression-regularization/selected-model.joblib` | `a224e1ce1d4deb758910a9583cf1fcffa09d635040f2fe8ad1f2e21622533ff7` |
| kNN | `results/knn-comparison/selected-model.joblib` | `ea6ecc6f3b8adb599872c9278fba56f4093849afc81c96f5de1266bca741a78a` |
| CNN | `results/cnn-improvements/selected-model.pt` | `1a389bf05cfe4c5eacd30ec30bc04302bfc7b74de32cd5d1b6e9570841388c01` |

Model artifacts and data remain ignored by Git. Preserve these local files: a fresh retraining is not guaranteed to reproduce their exact fingerprints. This document records the protocol; it does not execute final evaluation.
