# Final evaluation results

Completed on 4 November 2026 using the protocol frozen on 28 October. No retraining or test-based model selection was performed.

| Model | Validation accuracy | Test accuracy | Test macro-F1 | Test shirt recall |
| --- | --- | --- | --- | --- |
| Logistic regression | 85.40% | 84.33% | 0.8424 | 56.20% |
| kNN | 85.38% | 85.04% | 0.8502 | 58.90% |
| CNN | 92.33% | 91.79% | 0.9167 | 67.60% |

## Interpretation

The preselected CNN achieved 9,179 correct predictions out of 10,000, an advantage of 6.75 percentage points over kNN and 7.46 over logistic regression. Its validation accuracy was 92.33%, so the test score is 0.54 percentage points lower. These comparisons are observed differences, without a claim of statistical significance.

## Per-class recall on the official test set

Recall is the fraction correctly classified within each true class, not one-versus-rest accuracy. Every test class has 1,000 observations.

| Class | Logistic regression | kNN | CNN |
| --- | --- | --- | --- |
| T-shirt/top | 80.7% | 85.1% | 89.2% |
| Trouser | 95.7% | 96.9% | 98.6% |
| Pullover | 73.6% | 80.6% | 90.1% |
| Dress | 86.9% | 84.6% | 91.5% |
| Coat | 75.6% | 73.7% | 89.6% |
| Sandal | 92.3% | 82.8% | 98.0% |
| Shirt | 56.2% | 58.9% | 67.6% |
| Sneaker | 94.0% | 95.2% | 97.3% |
| Bag | 94.5% | 95.3% | 99.0% |
| Ankle boot | 93.8% | 97.3% | 97.0% |

## Main errors

Of 1,000 actual shirts, the CNN correctly classified 676. It classified 127 as T-shirt/top, 89 as coat and 71 as pullover. Similar silhouettes and low resolution are plausible explanations, not proven causes. The model is not best in every class: ankle-boot recall was 97.0% for CNN and 97.3% for kNN.

## Generalisation and selection

- Selected logistic regression: training 87.42%, validation 85.40%; gap 2.02 points. C=10 increased training accuracy and reduced validation accuracy, consistent with overfitting as regularisation weakened. The linear feature representation may also limit fit; pure underfitting has not been established.
- Selected CNN: training 95.67%, validation 92.33%; gap 3.33 points. The lowest validation loss was at epoch 12; subsequent lower training loss without lower validation loss supports overfitting with further training. Early stopping restored epoch 12. Later higher validation accuracy does not override the predeclared lowest-loss checkpoint rule.
- kNN: ordinary training-set predictions include self-neighbours and are optimistic. A subsequent training-only check excluded each query index and correctly classified 40,931/48,000 images (85.27%), close to validation accuracy (85.38%). [Calculation](../presentation/editorial/score_knn.py) and [saved result](../presentation/editorial/knn-training.json). Validation selection among k=3,5,9 favoured 3.
- CNN experiments: two architecture widths; four initial runs plus four additional seed runs = eight CNN training runs. The two wider configurations each have three seeds; the original seed-42 checkpoints are reused in that summary. Three seeds describe training variation, not uncertainty across data splits.

## Integrity and reproducibility

- All three checkpoint hashes match the frozen protocol.
- All CPU validation predictions exactly match saved predictions before test scoring.
- Publisher MD5 checksums and IDX dimensions were verified.
- Exact development/test overlap: zero images. No sensitivity subset is needed because the overlap-free set equals the full test set. Near duplicates were not assessed.
- Class counts: 4,800 training, 1,200 validation and 1,000 test observations per class.
- CPU inference: four computational threads, batches of 256, CNN evaluation/inference mode.
- No model was refitted on the 60,000-image pool. No settings changed in response to test results.

Run code: `scripts/final_evaluation.py`. Saved evidence: `results/final-evaluation-v1/`, including predictions, probabilities, class reports, raw and row-normalised confusion matrices, seed-42 sampled errors and hashed metadata. The script preserves an existing evaluation directory and stops on a second invocation. Data, checkpoints and generated results remain Git-ignored; preserve them locally.

## Limitations

One fixed validation split was used, not full cross-validation. Repeated validation selection can overfit development choices. Balanced benchmark performance does not establish everyday-photo performance or fairness across demographic groups. Architecture capacity and training choices differ as well as use of spatial structure; the comparison does not isolate a causal effect of convolution. The official test set is now evaluated and must not be reused to tune or select a replacement model.

## Sources

- [Frozen protocol](final-evaluation-protocol.md)
- [Development decisions](model-selection.md)
- [Publisher dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Executable evaluation](../scripts/final_evaluation.py)
