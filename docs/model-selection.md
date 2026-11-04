# Model-selection decisions

Recorded on 2026-10-21. These are development selections based on the fixed validation split. The final protocol was frozen on 2026-10-28 in [final-evaluation-protocol.md](final-evaluation-protocol.md). Final test evaluation was completed on 2026-11-04 without changing these choices; see [final evaluation results](final-evaluation-results.md).

## Shared protocol

- Use 48,000 training and 12,000 validation observations from the official 60,000-image training pool. Keep the saved source-index split identical across models.
- Split seed: 42. StratifiedGroupKFold with five folds; use fold 0 for validation and group exact duplicate images together.
- Pixel normalization: divide by 255. Classical inputs are 784 float64 features; CNN inputs are float32 tensors shaped N × 1 × 28 × 28.
- Primary model-selection metric: validation accuracy. Also inspect macro-F1, per-class recall, confusion matrices, and errors.
- Do not load test files during development. An initial historical integrity/overlap audit is distinct from model evaluation; no test model scores have been computed.

## Logistic regression

Selected C=0.1 from C in {0.01, 0.1, 1, 10}. Multinomial classification, L2 regularization (l1_ratio=0), lbfgs solver, tolerance 0.0001, maximum 2,000 iterations, seed 42. All candidates converged.

Selection rule: highest validation accuracy; exact ties prefer smaller C. C=0.1 achieved 85.40% accuracy and 0.8530 macro-F1. Its 0.083-point accuracy improvement over C=1 is only ten images, so it is a modest validation preference rather than a substantial performance gain.

Evidence: [notebook 03](../notebooks/03-logistic-regression-regularization.ipynb). Regenerated model: `results/logistic-regression-regularization/selected-model.joblib`.

## kNN

Selected k=3 from k in {3, 5, 9}. Euclidean distance, uniform voting, brute-force search, n_jobs=1 with computational thread limits applied during prediction.

Selection rule: highest validation accuracy; exact ties prefer larger k. k=3 achieved 85.38% accuracy and 0.8534 macro-F1. Its accuracy differs from selected logistic regression by only two images, while their errors differ substantially. Ordinary training-set accuracy was not used because each example can be its own neighbour.

Evidence: [notebook 04](../notebooks/04-knn-comparison.ipynb). Regenerated model: `results/knn-comparison/selected-model.joblib`.

## CNN

Current accuracy-first candidate: the wider architecture without weight decay.

Architecture: Conv2d(1→32, 3×3, padding 1), ReLU, 2×2 max pooling; Conv2d(32→64, 3×3, padding 1), ReLU, 2×2 max pooling; flatten; dense 128, ReLU, dropout 0.25; dense 10 logits. Total: 421,642 trainable parameters. Trained from scratch with cross-entropy loss and Adam.

Training settings: learning rate 0.001; batch size 128; maximum 30 epochs; early-stopping patience 5. ReduceLROnPlateau halves the learning rate after two stagnant validation-loss epochs (threshold 0, minimum learning rate 0.00001).

Within each run, restore the lowest-validation-loss checkpoint. Across candidates, prefer higher validation accuracy, then lower validation loss, then fewer parameters for exact ties. The reference checkpoint uses the original seed 42 and epoch 12; it was selected before the additional-seed study, not chosen as the best seed afterward.

Reference-checkpoint scores: 92.33% accuracy, 0.9225 macro-F1, 72.08% shirt recall. Evidence: [notebook 06](../notebooks/06-cnn-improvements.ipynb). Regenerated checkpoint: `results/cnn-improvements/selected-model.pt`. Reconstruct the architecture with `selected_config` from that directory's `run-metadata.json` before loading its state dictionary.

### Weight-decay alternative and stability

The alternative uses the identical wider architecture and schedule with Adam weight_decay=0.0001. Both configurations were evaluated with seeds 42, 7, and 123 on the same split.

| Configuration | Mean accuracy | Accuracy sample SD (percentage points) | Mean macro-F1 | Mean shirt recall |
|---|---:|---:|---:|---:|
| Wider, no weight decay | 92.400% | 0.347 | 0.9236 | 75.81% |
| Wider, weight decay 0.0001 | 92.339% | 0.059 | 0.9233 | 78.94% |

The accuracy-first rule retains the wider candidate, but its mean advantage is only 0.061 percentage points. Weight decay had lower observed variation, lower mean validation loss, and higher mean shirt recall. It remains a credible alternative if those trade-offs become the priority; it has not silently replaced the selected model.

The sample standard deviation describes variation across three training seeds, not a confidence interval or uncertainty from different data splits. Do not choose the best individual seed. Evidence: [notebook 07](../notebooks/07-cnn-seed-stability.ipynb).

## Comparison and limitations

The [validation comparison](../notebooks/08-validation-comparison.ipynb) uses the selected single checkpoints, including the original seed-42 wider CNN. It reports seed averages separately. Saved predictions are aligned and verified by source index.

On these observations, the CNN fixes 439 cases where both classical models fail, but misses 111 cases where both succeed. All three fail on 511 images, including 197 shirts. These observations do not prove labeling errors or a particular cause of model failure.

All results are validation results. Repeated selection on one validation set can overfit development decisions. The limited seed study does not establish a definitive accuracy ranking between the wider variants. Benchmark performance does not establish performance on ordinary photographs.

## Final protocol frozen on 2026-10-28

See [final-evaluation-protocol.md](final-evaluation-protocol.md) for the binding choices and checkpoint fingerprints. Retain C=0.1, k=3, and the wider CNN without weight decay (original seed 42, epoch 12). Evaluate existing models trained on 48,000 observations; do not refit on the full development pool. The CNN is the preselected primary model and the classical models are comparison baselines. Test reporting and the overlap-sensitivity policy are fixed in advance. No test evaluation has been performed as part of freezing this protocol.
