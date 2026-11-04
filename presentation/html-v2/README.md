# The shape of learning — HTML presentation v2

Open `index.html` directly in Chrome, Edge, Firefox or Safari. It is a self-contained offline presentation: all CSS, JavaScript, dataset images, walkthrough outputs and result data are embedded. No installation or server is required. Fullscreen support depends on the browser.

The same standalone presentation is also saved as `../output/fashion-mnist-interactive-v2.html`, ready to share. Either HTML file works by itself. HTML v1 is preserved.

## Added in v2

- Scene 4: animated ten-class breakdowns for training, validation and official test. All classes are 10% of each subset. Fold 0 is validation and the other four folds are training; this is one saved 80/20 split, not five-fold cross-validation. Stratification, grouping, shared indices and the publisher-provided test set are distinguished.
- Scene 12: shirt recall, 676/1,000 = 67.6%; paired Shirt ↔ T-shirt/top, Coat and Pullover confusions; six genuine matching error examples from saved predictions. Observed results are separated from unproven explanations. The original sampled-error gallery remains optional.
- N opens concise defence notes for each addition. I contains supporting sources and questions.

The split implementation was inspected in notebook 01: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` and `next(splitter.split(...))`. Saved split fingerprints match all model metadata. Counts were reconciled against original labels; the CNN matrix was independently reconciled against saved predictions. Paired examples are the first matching saved index for each direction, not a random or representative sample.

## Controls

- Right/left arrows or Page Down/Page Up: next/previous scene.
- Space: next scene when the page body has focus.
- F: fullscreen; Escape: leave fullscreen or close an open dialog.
- N: current presenter notes.
- O: scene overview.
- I: evidence, technical backup and practice questions.
- Home/End: first/last scene.
- On a touch screen, swipe horizontally outside interactive controls.

There are 14 main scenes. Their declared times sum to 810 seconds: 13 minutes 30 seconds before optional interactions. Scene 4 targets 65 seconds and scene 12 targets 80 seconds. Rehearse aloud and keep exploration short to stay within 12–15 minutes. Overview, notes and evidence are outside the timed narrative. Reduced-motion system preferences are respected.

## What is real

- 80 dataset thumbnails are real validation observations.
- Ten model walkthroughs use the first saved validation observation of each class, not a selection based on correctness.
- kNN images are the actual three nearest training observations with Euclidean distances.
- Logistic regression shows actual class probabilities and signed pixel × weight contributions for its predicted class. The sum plus the intercept reproduces its logit. The heatmap is not a causal explanation.
- CNN images are actual post-ReLU, post-pooling activations from the frozen model. Eight channels per layer are displayed using one intensity scale across each activation tensor.
- All 30 walkthrough predictions were checked against the original saved validation predictions. The browser animates recorded outputs; it does not train models or run arbitrary inference.
- Test results and sampled errors come from the saved final evaluation. Building this presentation does not repeat test inference or change the models.

The animated convolution window illustrates a sliding local receptive field. It is not an exact time trace of the CPU computation, and the feature-map displays show saved tensor outputs rather than real-time calculations. The CNN walkthrough includes pooling stages; full architecture and training settings are available under I.

## Rebuild

From the `fashion-mnist-classification` root, with its existing environment, checkpoints and data:

```bash
.venv/bin/python presentation/html-v2/extend_evidence.py
.venv/bin/python presentation/html-v2/pack.py
```

The base walkthrough data are retained from v1 in `data.json`. Extending the evidence only reads saved results and dataset files; it does not load models, run inference, train or tune. For styling/content changes alone, edit `styles.css`, `app.js` or `template.html` and run only `pack.py`. Edit `scene-notes.json` for the added speaker notes, then run the two commands above. `index.html` remains the only file required to present or share. `evidence-manifest.json` records split verification, source fingerprints and the six selected error indices.

The earlier PPTX and rehearsal notes are preserved under `presentation/output/`.
