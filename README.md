# Fashion-MNIST Classification

Course project comparing k-nearest neighbours (kNN), multiclass logistic regression, and a convolutional neural network trained from scratch on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

## Task and current status

Predict one of ten clothing, footwear, or accessory categories from a 28 × 28 grayscale image. Data exploration, model selection, CNN stability experiments, and validation error analysis are complete. Final evaluation was completed on 4 November 2026 using the frozen models, without refitting or test-based tuning. See [final evaluation results](docs/final-evaluation-results.md).

All development used the official training pool: 48,000 training images and 12,000 validation images. Notebooks 01–08 do not download or load test files. An earlier version of notebook 01 performed a file-integrity and exact-overlap audit without scoring models. The final evaluation script subsequently evaluated the frozen models on all 10,000 official test images. Exact development/test overlap was zero; every test class has 1,000 observations.

Results apply to benchmark images. Everyday phone photographs remain untested.

## Current model selections

| Model | Configuration | Validation accuracy | Macro-F1 | Shirt recall |
|---|---|---:|---:|---:|
| Logistic regression | L2, C=0.1 | 85.40% | 0.8530 | 59.42% |
| kNN | k=3, Euclidean distance, uniform voting | 85.38% | 0.8534 | 58.92% |
| CNN | Wider architecture, no weight decay; seed 42, epoch 12 | 92.33% | 0.9225 | 72.08% |

These are individual saved-checkpoint results on the same validation observations. Across seeds 42, 7, and 123, mean CNN accuracy was 92.40% without weight decay and 92.34% with weight decay. The weight-decay candidate showed lower observed variation and higher average shirt recall. Three seeds do not establish a decisive accuracy ranking.

## Final test results

| Frozen model | Test accuracy | Test macro-F1 | Test shirt recall |
|---|---:|---:|---:|
| Logistic regression | 84.33% | 0.8424 | 56.20% |
| kNN | 85.04% | 0.8502 | 58.90% |
| CNN | 91.79% | 0.9167 | 67.60% |

The CNN was preselected using validation data. All checkpoint fingerprints matched and CPU validation predictions had zero disagreements with saved predictions before test scoring. Run code: `scripts/final_evaluation.py`. Outputs: `results/final-evaluation-v1/` (ignored by Git). The script preserves existing outputs and stops if that directory already exists. Read the saved results instead of repeating evaluation for model selection. The test set must not now be used to tune models.

## Presentation mock

- [Interactive HTML presentation](presentation/output/fashion-mnist-presentation.html): scientific editorial redesign with real dataset exploration, large selectable CNN activation plates, per-class analysis and presenter notes. The visible wordmark, toolbar and shortcut strip are removed. Sixteen slides target 13 minutes, followed by four Q&A slides. See [presentation guide](presentation/editorial/README.md) and [design rules](presentation/DESIGN.md). Earlier [HTML v2](presentation/output/fashion-mnist-interactive-v2.html) and [HTML v1](presentation/output/fashion-mnist-interactive-v1.html) are preserved.
- [Editable presentation v1](presentation/output/fashion-mnist-mock-v1.pptx): 12 main slides, a target duration of 12 minutes 30 seconds, and four backup slides.
- [Slide-by-slide rehearsal guide](presentation/output/presentation-script.pdf) with [editable LaTeX source](presentation/output/presentation-script.tex), including saliency and occlusion, global and local SHAP plots, and the Q&A appendix.
- [Earlier PPTX speaker notes and practice questions](presentation/output/presentation-notes-v1.md).
- [Final evaluation report](docs/final-evaluation-results.md).

See [model-selection decisions](docs/model-selection.md) for settings, selection rules and alternatives.

## Reproduce the environment

The recorded environment is Python 3.12.0 on macOS ARM64. PyTorch uses Apple MPS when available and otherwise falls back to CPU. The comparison notebook measures prediction time on CPU for all models.

From a fresh checkout:

```bash
git clone https://github.com/keodubo/fashion-mnist-classification.git
cd fashion-mnist-classification
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check
python -m ipykernel install --sys-prefix --name fashion-mnist --display-name "Fashion-MNIST (.venv)"
python -m jupyter lab
```

`requirements.txt` pins direct dependencies; `requirements-lock.txt` records the complete installed package set. The lock was generated on macOS ARM64 and includes platform-specific packages. Other operating systems may need an adapted environment; installation and numerical equivalence there have not been verified.

The kernel registration is local to `.venv`. If the checkout is moved, recreate the virtual environment and kernel in the new location. No credentials or environment variables are required. Internet access is needed to install packages and download the two training-data files on the first run.

## Reproduce the experiments

Use the **Fashion-MNIST (.venv)** kernel. Run the notebooks in the order below, restarting the kernel and running all cells in each notebook. Each notebook loads its own inputs and relies on artifacts saved by earlier notebooks, not on another notebook's in-memory variables.

| Order | Notebook | Produces |
|---|---|---|
| 01 | [Data exploration](notebooks/01-data-exploration.ipynb) | Verified training files, saved split, data-quality outputs |
| 02 | [Logistic-regression baseline](notebooks/02-logistic-regression-baseline.ipynb) | C=1 baseline and error analysis |
| 03 | [Regularization comparison](notebooks/03-logistic-regression-regularization.ipynb) | Selected logistic-regression model and predictions |
| 04 | [kNN comparison](notebooks/04-knn-comparison.ipynb) | Selected kNN model and predictions |
| 05 | [CNN baseline](notebooks/05-cnn-baseline.ipynb) | Original CNN checkpoint and learning curves |
| 06 | [CNN improvements](notebooks/06-cnn-improvements.ipynb) | Wider-model experiments and selected checkpoint |
| 07 | [CNN seed stability](notebooks/07-cnn-seed-stability.ipynb) | Four additional runs and three-seed summary |
| 08 | [Validation comparison](notebooks/08-validation-comparison.ipynb) | Aligned errors, per-class metrics, and matched CPU timing |

Notebook 01 downloads training data from a fixed upstream Git commit and verifies publisher checksums. It creates a seed-42, group-aware stratified split; identical images stay together. The observed split has 4,800 training and 1,200 validation examples per class. Subsequent notebooks verify the saved split fingerprint and source checksums. They normalize pixels by dividing by 255; classical models flatten images and CNNs preserve spatial structure.

Datasets, checkpoints, generated tables/figures, and personal notes are excluded from Git. A fresh clone therefore contains the executed notebooks and their embedded outputs, but not the local artifacts required by later notebooks. Run the complete sequence to regenerate them. Their directories are described in the [work plan](docs/plans/work-plan.md).

Use a separate fresh checkout for a full reproduction if existing local results must be preserved: reruns write to the same artifact paths and replace their previous contents. An existing saved split is checked rather than silently replaced when its observations differ.

Seeds and versions are recorded, but exact retraining results can vary across hardware, numerical backends, and library versions. Notebook 08 deliberately checks that the recorded selections are C=0.1, k=3, and the wider CNN; if rerunning selects something else, inspect the difference instead of relabeling it as the original experiment. Final test evaluation is not part of this execution sequence.

## Documentation

- [Approved project proposal](docs/project-idea-ml%26ai.md)
- [Course project requirements](docs/project-requirements.txt)
- [Work plan](docs/plans/work-plan.md)
- [Model-selection decisions](docs/model-selection.md)
