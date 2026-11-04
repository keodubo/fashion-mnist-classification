"""Build presentation inputs and a readable record from saved experiment evidence."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from final_evaluation import read_idx, CLASSES

BUILD = ROOT / "presentation/.build"
ASSETS = BUILD / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
FINAL = ROOT / "results/final-evaluation-v1"
metrics = pd.read_csv(FINAL / "metrics.csv")
per_class = pd.read_csv(FINAL / "per-class-metrics.csv")
meta = json.loads((FINAL / "run-metadata.json").read_text())
images = read_idx(ROOT / "data/raw/train-images-idx3-ubyte.gz")
labels = read_idx(ROOT / "data/raw/train-labels-idx1-ubyte.gz")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "figure.facecolor": "#F7F5EF"})

# These are scientific displays of actual dataset observations, not illustrations.
fig, axes = plt.subplots(2, 5, figsize=(12, 4.5), facecolor="#F7F5EF")
for i, ax in enumerate(axes.flat):
    idx = np.flatnonzero(labels == i)[0]
    ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(CLASSES[i], fontsize=14)
    ax.axis("off")
fig.tight_layout(pad=1.1)
fig.savefig(ASSETS / "dataset-samples.png", dpi=170)
plt.close(fig)

test_images = read_idx(ROOT / "data/raw/t10k-images-idx3-ubyte.gz")
examples = pd.read_csv(FINAL / "sampled-errors.csv")
cnn_examples = examples[examples.model == "CNN"].head(6)
fig, axes = plt.subplots(2, 3, figsize=(10, 6), facecolor="#F7F5EF")
for (_, row), ax in zip(cnn_examples.iterrows(), axes.flat):
    ax.imshow(test_images[int(row.test_index)], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(f"True: {row.actual}\nPredicted: {row.predicted}\nTest image {row.test_index}", fontsize=13)
    ax.axis("off")
fig.tight_layout(pad=1.4)
fig.savefig(ASSETS / "cnn-test-errors.png", dpi=170)
plt.close(fig)

names = ["Logistic regression", "kNN", "CNN"]
history = pd.read_csv(ROOT / "results/cnn-improvements/wider-history.csv")
logistic = pd.read_csv(ROOT / "results/logistic-regression-regularization/comparison.csv")
knn = pd.read_csv(ROOT / "results/knn-comparison/comparison.csv")
cnn = pd.read_csv(ROOT / "results/cnn-improvements/comparison.csv")
confusions = pd.read_csv(FINAL / "confusions.csv")
matrix = pd.read_csv(FINAL / "cnn-confusion-counts.csv", index_col=0)

def value(model, split, metric):
    return float(metrics[(metrics.model == model) & (metrics.split == split)][metric].iloc[0])

data = {
    "classes": CLASSES, "models": names,
    "metrics": metrics.to_dict("records"), "per_class": per_class.to_dict("records"),
    "history": history.to_dict("records"), "logistic": logistic.to_dict("records"),
    "knn": knn.to_dict("records"), "cnn": cnn.to_dict("records"),
    "confusions": confusions.to_dict("records"), "cnn_matrix": matrix.values.tolist(),
    "class_counts": pd.read_csv(FINAL / "class-counts.csv").to_dict("records"),
    "timing": pd.read_csv(ROOT / "results/validation-comparison/timing-summary.csv").to_dict("records"),
    "stability": pd.read_csv(ROOT / "results/cnn-stability/summary.csv").to_dict("records"),
}

sources = {
    "protocol": "docs/final-evaluation-protocol.md",
    "selection": "docs/model-selection.md",
    "test": "results/final-evaluation-v1/metrics.csv; per-class-metrics.csv; run-metadata.json",
    "dataset": "https://github.com/zalandoresearch/fashion-mnist/tree/b2617bb6d3ffa2e429640350f613e3291e10b141",
}
slides = [
    {"title": "Clothing classification", "seconds": 60,
     "notes": "The question behind this project is whether a CNN improves clothing classification compared with two classical approaches using pixel values. This is a supervised multiclass classification task: each input is an image and each target is one of ten categories. A useful output is a correct category for an unseen benchmark image. I chose this comparison because it lets me investigate the value of a model designed for images while keeping the dataset and evaluation observations consistent. I also wanted to understand the errors, rather than stop at one overall score. This is a benchmark experiment; I have not built or tested a system for ordinary shopping photographs. I will explain the preparation and model choices, then use the final results and errors to answer the question.",
     "sources": [sources["dataset"], "docs/project-idea-ml&ai.md"]},
    {"title": "Fashion-MNIST", "seconds": 30,
     "notes": "Fashion-MNIST provides 28 by 28 grayscale images in ten labelled categories. There are 60,000 official development images and a separate 10,000-image test set. Every class has the same number of observations. The examples show that the task includes both footwear and visually similar upper-body garments. The small, standardised images keep the experiments manageable, but they also limit how much visual detail is available.",
     "sources": [sources["dataset"], "results/exploration/quality-summary.json", "results/final-evaluation-v1/class-counts.csv"]},
    {"title": "Preparation and dataset splitting", "seconds": 60,
     "notes": "I checked the development data for missing values, invalid labels, pixel ranges, constant images and exact duplicates. None of those checks required removing observations. I scaled pixels from zero to 255 into zero to one by dividing by 255. Logistic regression and kNN use a flattened vector of 784 pixel features; the CNN keeps the two-dimensional grid. From the official 60,000-image development pool I used 48,000 for training and 12,000 for validation, using the same observations for every model. The split was stratified and group-aware, so class proportions were preserved and any identical images would stay together. No exact duplicates were actually found. The official 10,000 test images were reserved for final scoring. This is one fixed hold-out split, not five-fold cross-validation, even though the splitting tool was StratifiedGroupKFold.",
     "sources": ["notebooks/01-data-exploration.ipynb", "data/splits/train-validation-v1.json", sources["protocol"]]},
    {"title": "Choosing the classical baselines", "seconds": 90,
     "notes": "I used logistic regression as a linear reference and kNN as a reference based on pixel similarity. They use the same flattened images, which makes the comparison straightforward. For logistic regression I compared C values of 0.01, 0.1, 1 and 10. C controls the inverse strength of L2 regularisation: a smaller C penalises large weights more strongly. C equals 0.1 had the highest validation accuracy, so it was selected. Its advantage over C equals 1 was only ten validation images, which is a modest preference. Increasing C to 10 improved training accuracy but reduced validation accuracy, consistent with weaker regularisation overfitting. For kNN I compared three, five and nine neighbours using Euclidean distance and equal voting. Three neighbours gave the highest validation accuracy. I did not exhaustively search every distance metric or preprocessing option. kNN and logistic regression were almost tied on validation accuracy, so I do not claim a decisive ranking between them from that split.",
     "sources": ["results/logistic-regression-regularization/comparison.csv", "results/knn-comparison/comparison.csv", sources["selection"]]},
    {"title": "The selected CNN", "seconds": 90,
     "notes": "The CNN keeps the image grid and learns local patterns using two convolutional layers. The selected model has 32 filters in the first layer and 64 in the second, each using a three by three kernel and padding of one. Each convolution is followed by ReLU and two by two max pooling. The feature maps shrink from 28 by 28 to 14 by 14 and then seven by seven. I flatten those maps, use a 128-unit hidden layer with ReLU, apply dropout of 0.25, and produce ten output scores. ReLU supplies nonlinearity with a simple computation; I did not compare it against Leaky ReLU experimentally. Pooling reduces spatial resolution and computation. Dropout was included to reduce overfitting. The final layer returns raw logits. CrossEntropyLoss handles log-softmax internally during training, so I do not add a separate softmax before that loss. For reporting probabilities I apply softmax to the logits. The largest score or probability determines the predicted class.",
     "sources": ["notebooks/06-cnn-improvements.ipynb", sources["protocol"]]},
    {"title": "CNN experiments and selection", "seconds": 90,
     "notes": "I evaluated two architecture widths. The original network used 16 and 32 convolutional filters and a 64-unit hidden layer. The wider network doubled those widths to 32, 64 and 128. There were four initial runs: the original baseline, an extended run of that architecture, the wider model, and the wider model with weight decay. The baseline reached 90.70 percent validation accuracy; the extended run reached 91.45; the wider model reached 92.33. I trained from scratch with Adam and a learning rate of 0.001. Within a run I restored the checkpoint with the lowest validation loss, and between configurations I used validation accuracy as the main selection metric. I then added four runs to compare the two wider configurations across three seeds each: eight CNN training runs in total. Their mean accuracy difference was only 0.061 percentage points. Weight decay gave lower observed variation and better average shirt recall. I retained the previously selected original seed-42 wider checkpoint rather than choosing the best seed afterward.",
     "sources": ["results/cnn-improvements/comparison.csv", "results/cnn-stability/comparison.csv", "results/cnn-stability/summary.csv", sources["selection"]]},
    {"title": "Learning curves and generalisation", "seconds": 60,
     "notes": "These curves explain why the final epoch is not automatically the best model. The selected CNN checkpoint is epoch 12, where validation loss reached its minimum of about 0.222. Training loss continued to decline after that while validation loss remained higher, which is evidence of overfitting with further training. Validation accuracy did improve slightly later, but our within-run selection rule was lowest validation loss, which also reflects confidence in the predictions. At the selected checkpoint training accuracy was 95.67 percent and validation accuracy was 92.33 percent, a gap of 3.33 percentage points. Early stopping ended the run at epoch 17 and restored epoch 12. Logistic regression had a smaller selected gap of 2.02 points. Its limited gains from weaker regularisation suggest limited representational capacity, but do not prove pure underfitting. For kNN ordinary training accuracy would include each image as its own neighbour, so I did not use that as evidence of generalisation.",
     "sources": ["results/cnn-improvements/wider-history.csv", "results/cnn-improvements/comparison.csv", "results/logistic-regression-regularization/comparison.csv", sources["selection"]]},
    {"title": "Final performance on unseen test images", "seconds": 60,
     "notes": "Before opening the test set for final scoring, I checked that the three model files matched the frozen fingerprints and reproduced all saved validation predictions on CPU with zero disagreements. I then evaluated those existing models without refitting. Logistic regression achieved 84.33 percent test accuracy, kNN 85.04, and the CNN 91.79. The CNN correctly classified 9,179 of the 10,000 images. Its test advantage was 6.75 percentage points over kNN and 7.46 over logistic regression. Its validation-to-test difference was about 0.54 points. The CNN had been selected as the primary model using validation evidence before these test results. Test scores were not used to change hyperparameters, epochs or seeds. Macro-F1 gives the same broad picture and per-class results explain what the overall scores hide.",
     "sources": [sources["test"], sources["protocol"], "scripts/final_evaluation.py"]},
    {"title": "Performance differs across categories", "seconds": 60,
     "notes": "Here I report recall for each actual category: of all the items that really belong to a class, how many did the model identify correctly? This is the useful meaning of accuracy within a true class. I avoid calling one-versus-rest accuracy a per-class score because the many negatives could make it misleading. Each test category has 1,000 images. The CNN identifies 99 percent of bags and 98.6 percent of trousers, but only 67.6 percent of shirts. It correctly identifies 676 shirts and misses 324. It improves shirt recall compared with both baselines, but that remains a major weakness. The CNN also does not win every individual category: kNN has slightly higher ankle-boot recall, 97.3 percent versus 97.0 percent. That is only three test images and should not be overstated.",
     "sources": ["results/final-evaluation-v1/per-class-metrics.csv", "results/final-evaluation-v1/class-counts.csv"]},
    {"title": "Where the CNN makes mistakes", "seconds": 60,
     "notes": "The largest CNN confusion is an actual shirt classified as a T-shirt or top: 127 test images. Another 89 shirts are classified as coats and 71 as pullovers. The confusion matrix in the appendix shows the complete pattern. The displayed errors are six examples from a seed-42 random sample saved under the frozen evaluation protocol, rather than examples chosen to tell a preferred story. Similar silhouettes and the limited detail of 28 by 28 grayscale images are plausible explanations for some mistakes. These examples do not establish the cause of every error or prove that labels are wrong. I can describe what is visible and relate it to the confusion counts, while keeping that interpretation separate from the measured result.",
     "sources": ["results/final-evaluation-v1/confusions.csv", "results/final-evaluation-v1/sampled-errors.csv", sources["dataset"]]},
    {"title": "Limits of the evidence", "seconds": 45,
     "notes": "The results apply to this benchmark, with balanced classes and equal error costs. They do not establish performance on phone photographs, different backgrounds or lighting. Repeated decisions on one validation set can overfit development choices, and three training seeds do not measure uncertainty across different data splits. The models differ in capacity and training as well as how they use spatial structure, so this is not a controlled causal isolation of convolution. The final audit found no exact images shared between the development pool and the test set, although that does not exclude near duplicates. I retained the checkpoints trained on 48,000 images so the final evaluation corresponds exactly to the reviewed models; I did not refit on all 60,000.",
     "sources": [sources["protocol"], sources["selection"], "results/final-evaluation-v1/run-metadata.json"]},
    {"title": "Conclusions", "seconds": 45,
     "notes": "The CNN is the strongest overall model among the approaches tested: 91.79 percent test accuracy, compared with about 84 to 85 percent for the classical baselines. This supports using a model designed for image patterns on this task, while not isolating the effect of spatial structure alone. The improvement does not remove the main weakness: shirts are still frequently confused with other upper-body garments. If very fast CPU prediction and a simple linear model are the priority, logistic regression remains a useful baseline; the existing matched validation timing shows it is much faster, although that timing is specific to our hardware and excludes loading and preprocessing. The main lesson is that data preparation, validation and error analysis are needed to make the headline accuracy meaningful. A future experiment could investigate augmentation or different image representations using fresh development evidence; I would not tune those changes on this test set.",
     "sources": [sources["test"], "results/validation-comparison/timing-summary.csv", sources["selection"]]},
]
assert sum(s["seconds"] for s in slides) == 750
data["slides"] = slides
(BUILD / "evidence.json").write_text(json.dumps(data, indent=2) + "\n")

def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])

report = "# Final evaluation results\n\nCompleted on 4 November 2026 using the protocol frozen on 28 October. No retraining or test-based model selection was performed.\n\n"
report += markdown_table(["Model", "Validation accuracy", "Test accuracy", "Test macro-F1", "Test shirt recall"], [[n, f"{value(n,'Validation','accuracy'):.2%}", f"{value(n,'Test','accuracy'):.2%}", f"{value(n,'Test','macro_f1-score'):.4f}", f"{value(n,'Test','shirt_recall'):.2%}"] for n in names])
report += "\n\n## Interpretation\n\nThe preselected CNN achieved 9,179 correct predictions out of 10,000, an advantage of 6.75 percentage points over kNN and 7.46 over logistic regression. Its validation accuracy was 92.33%, so the test score is 0.54 percentage points lower. These comparisons are observed differences, without a claim of statistical significance.\n\n"
report += "## Per-class recall on the official test set\n\nRecall is the fraction correctly classified within each true class, not one-versus-rest accuracy. Every test class has 1,000 observations.\n\n"
report += markdown_table(["Class"] + names, [[c] + [f"{per_class[(per_class.model==n)&(per_class.split=='Test')&(per_class['class']==c)].recall.iloc[0]:.1%}" for n in names] for c in CLASSES])
report += "\n\n## Main errors\n\nOf 1,000 actual shirts, the CNN correctly classified 676. It classified 127 as T-shirt/top, 89 as coat and 71 as pullover. Similar silhouettes and low resolution are plausible explanations, not proven causes. The model is not best in every class: ankle-boot recall was 97.0% for CNN and 97.3% for kNN.\n\n"
report += "## Generalisation and selection\n\n- Selected logistic regression: training 87.42%, validation 85.40%; gap 2.02 points. C=10 increased training accuracy and reduced validation accuracy, consistent with overfitting as regularisation weakened. The linear feature representation may also limit fit; pure underfitting has not been established.\n- Selected CNN: training 95.67%, validation 92.33%; gap 3.33 points. The lowest validation loss was at epoch 12; subsequent lower training loss without lower validation loss supports overfitting with further training. Early stopping restored epoch 12. Later higher validation accuracy does not override the predeclared lowest-loss checkpoint rule.\n- kNN: ordinary training-set predictions include self-neighbours and are optimistic. No leave-one-out training estimate was run, so a training-versus-validation diagnosis is unavailable. Validation selection among k=3,5,9 favoured 3.\n- CNN experiments: two architecture widths; four initial runs plus four additional seed runs = eight CNN training runs. The two wider configurations each have three seeds; the original seed-42 checkpoints are reused in that summary. Three seeds describe training variation, not uncertainty across data splits.\n\n"
report += "## Integrity and reproducibility\n\n- All three checkpoint hashes match the frozen protocol.\n- All CPU validation predictions exactly match saved predictions before test scoring.\n- Publisher MD5 checksums and IDX dimensions were verified.\n- Exact development/test overlap: zero images. No sensitivity subset is needed because the overlap-free set equals the full test set. Near duplicates were not assessed.\n- Class counts: 4,800 training, 1,200 validation and 1,000 test observations per class.\n- CPU inference: four computational threads, batches of 256, CNN evaluation/inference mode.\n- No model was refitted on the 60,000-image pool. No settings changed in response to test results.\n\nRun code: `scripts/final_evaluation.py`. Saved evidence: `results/final-evaluation-v1/`, including predictions, probabilities, class reports, raw and row-normalised confusion matrices, seed-42 sampled errors and hashed metadata. The script preserves an existing evaluation directory and stops on a second invocation. Data, checkpoints and generated results remain Git-ignored; preserve them locally.\n\n"
report += "## Limitations\n\nOne fixed validation split was used, not full cross-validation. Repeated validation selection can overfit development choices. Balanced benchmark performance does not establish everyday-photo performance or fairness across demographic groups. Architecture capacity and training choices differ as well as use of spatial structure; the comparison does not isolate a causal effect of convolution. The official test set is now evaluated and must not be reused to tune or select a replacement model.\n\n## Sources\n\n- [Frozen protocol](final-evaluation-protocol.md)\n- [Development decisions](model-selection.md)\n- [Publisher dataset](https://github.com/zalandoresearch/fashion-mnist)\n- [Executable evaluation](../scripts/final_evaluation.py)\n"
(ROOT / "docs/final-evaluation-results.md").write_text(report)

notes = "# Presentation rehearsal — v1\n\nMain talk: 12 slides, 12 minutes 30 seconds at the planned pacing. Backup slides are for questions and are not part of the timed talk. Rehearse aloud; the timings are targets, not measured delivery times.\n\n## Story\n\nCan a CNN improve clothing classification compared with classical models using pixel values? Explain the decisions, demonstrate the result, then show what still fails. The comparison supports a practical choice among these experiments, not a causal claim that spatial structure alone explains every gain.\n\n## Slide-by-slide speaker notes\n\n"
for i, s in enumerate(slides, 1):
    notes += f"### {i}. {s['title']} — {s['seconds']} seconds\n\n{s['notes']}\n\nEvidence: " + "; ".join(f"`{src}`" if not src.startswith('http') else src for src in s['sources']) + "\n\n"
notes += """## Questions to practise

**Why supervised multiclass classification?** Training examples have known category labels and the target is one of ten discrete categories.

**Why keep validation and test separate?** Validation guides choices such as C, k, architecture and checkpoint. Test data provide a final assessment after those choices are fixed. Tuning using test scores would make that assessment optimistic.

**Did you stratify? Did you cross-validate?** Yes, the training/validation split preserved class proportions. StratifiedGroupKFold produced the groups, but only fold zero was used as validation: one hold-out split, not five-fold cross-validation. Identical-image grouping was a precaution; none were found.

**Did you choose the 10,000 test images yourself?** No. They are the publisher's official test set. I split only the 60,000 development images.

**Did you use the test set before final evaluation?** A historical version performed an integrity and overlap audit, without computing model scores. Development notebooks then avoided loading the test set. Final scoring followed the frozen protocol after validation prediction parity checks.

**Does dividing by 255 standardise the features?** It scales pixel intensities into [0,1]. It does not subtract each feature's mean or divide by its standard deviation. Every raw pixel already has the same range; division by 255 is also a uniform scaling, so by itself it does not change Euclidean neighbour rankings.

**Does flattening destroy all spatial information?** Pixel positions still have fixed feature indices, but these baselines do not explicitly use local neighbourhoods and shared convolution filters. The CNN supplies those inductive biases.

**Why these models?** They satisfy the two-simple-plus-one-complex option and compare a linear reference, a similarity-based reference, and a model designed for image grids.

**Why C=0.1 and k=3?** They had the highest accuracy among the validation candidates tested. C=0.1 was only ten images better than C=1; k=3 was only seven images better than k=5. These are limited validation preferences, not universal optimal values.

**How many CNNs?** Two architecture widths. Four initial runs (baseline, extended original, wider, wider with weight decay) and four new seed runs: eight training runs. Each network has two convolutional layers. The stability table contains six runs, including two existing seed-42 runs, not six extra runs.

**Why ReLU rather than Leaky ReLU?** ReLU is a simple conventional way to introduce nonlinearity. I used it consistently and did not experimentally establish superiority over Leaky ReLU. Testing alternative activations would be a new development experiment.

**Why softmax?** It converts ten unrestricted logits into ten nonnegative probabilities summing to one. Predictions use the largest score. Training passes raw logits to CrossEntropyLoss, which includes log-softmax internally. A separate softmax before that loss is unnecessary. Softmax probabilities are not automatically calibrated confidence estimates.

**Why max pooling and dropout?** Pooling reduces spatial size and computation. Dropout randomly suppresses hidden activations during training to discourage reliance on particular units. Neither choice was isolated in an ablation, so I cannot quantify its individual contribution.

**Why Adam, batch size 128 and learning rate 0.001?** They were manageable conventional starting settings. The experiments did not establish that they are globally optimal. The scheduler reduced the rate when validation loss stalled.

**Why epoch 12 if epoch 16 had higher validation accuracy?** Within each run, the fixed rule was lowest validation loss, which accounts for the predicted probabilities as well as correct/incorrect labels. Epoch 12 had the minimum loss. Configuration selection then used validation accuracy at each restored checkpoint. The two rules answer different decisions and were fixed before testing.

**What is a seed?** A seed controls pseudo-random initialisation and training order. Changing it checks sensitivity to training randomness. Three seeds on the same split do not replace cross-validation or provide a confidence interval over datasets.

**What evidence supports overfitting?** For the CNN, training loss continues to decrease after epoch 12 while validation loss stays above its minimum. For logistic regression, weakening regularisation to C=10 raises training accuracy but lowers validation accuracy. For kNN, I did not compute a self-neighbour-free training estimate, so I should not invent a gap.

**Is logistic regression underfitting?** Its linear representation may limit performance, but lower accuracy than the CNN alone is insufficient evidence. Training accuracy is 87.42% versus validation 85.40%; the evidence suggests limited capacity plus some generalisation gap rather than a clean diagnosis of pure underfitting.

**Why report accuracy and macro-F1?** Accuracy is easy to interpret with balanced classes and assumed equal error costs. Macro-F1 weights the ten classes equally and reflects both precision and recall. Class recall and confusion matrices expose weaknesses hidden by overall accuracy.

**What does per-class accuracy mean here?** We report recall: correct predictions among the actual members of that class. It is explicitly labelled recall. Precision instead asks how many predictions of a category were correct.

**Which model is best?** CNN for the predeclared overall accuracy objective among these tested models. That does not make it best for every category or resource constraint. kNN has slightly higher ankle-boot recall; logistic regression predicts much faster in our matched CPU benchmark.

**Why are shirts difficult?** The measured pattern is confusion with T-shirts/tops, coats and pullovers. Similar silhouettes and low resolution are plausible explanations. They are not proven causes or evidence of wrong labels.

**Why not retrain on all 60,000 images?** The frozen protocol evaluates exactly the selected checkpoints trained on 48,000 images, preserving the reviewed evidence. This gives up possible benefits from more training data; it does not prove refitting would perform worse.

**Is the CNN improvement caused only by preserving spatial structure?** No. Models differ in capacity, architecture and training. These results support the tested CNN as a practical choice; a controlled ablation would be needed to isolate the contribution of one factor.

**What would you do next?** Finish presenting and defending this frozen experiment. Future work could investigate augmentation or alternative representations with fresh development evidence and an appropriate independent final evaluation. Do not tune on the now-observed test results.

## Code map

- Notebook 01: file checks, data quality, exact duplicates, stratified group-aware split, examples.
- Notebooks 02–03: linear baseline and C search; selected logistic regression.
- Notebook 04: k search; selected kNN.
- Notebook 05: original CNN.
- Notebook 06: extended training, wider models, weight decay, checkpoint selection.
- Notebook 07: additional seeds and variation.
- Notebook 08: aligned validation comparison, errors and matched CPU timing.
- `scripts/final_evaluation.py`: checkpoint verification → CPU validation parity → test checks → inference → metrics and provenance. It never calls fit or performs optimisation.

## Ask Álvaro to review

1. Does the balance of explanation, experiments and findings match the assignment?
2. Is the distinction between checkpoint selection by validation loss and configuration selection by validation accuracy clear?
3. Is the error analysis detailed enough, and is the interpretation sufficiently cautious?
4. Which technical details should move into or out of the backup slides?

## Rehearsal checklist

- [ ] Explain the opening question without reading.
- [ ] Draw the split and state what each subset is used for.
- [ ] Explain C, k, ReLU, logits, softmax and the checkpoint rule.
- [ ] State the test scores and explain the shirt weakness.
- [ ] Explain one limitation without being prompted.
- [ ] Time the 12-slide talk; keep backup slides outside the main talk.
- [ ] Show Álvaro the mock deck and record his corrections before preparing v2.
"""
(ROOT / "presentation/output/presentation-notes-v1.md").write_text(notes)
print("Saved evidence, dataset displays, evaluation report and rehearsal notes")
