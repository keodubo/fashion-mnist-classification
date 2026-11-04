# The shape of learning — HTML presentation v1

Open `index.html` directly in Chrome, Edge, Firefox or Safari. It is a self-contained offline presentation: all CSS, JavaScript, dataset images, walkthrough outputs and result data are embedded. No installation or server is required. Fullscreen support depends on the browser.

The same standalone presentation is also saved as `../output/fashion-mnist-interactive-v1.html`, ready to share. Either HTML file works by itself.

## Controls

- Right/left arrows or Page Down/Page Up: next/previous scene.
- Space: next scene when the page body has focus.
- F: fullscreen; Escape: leave fullscreen or close an open dialog.
- N: current presenter notes.
- O: scene overview.
- I: evidence, technical backup and practice questions.
- Home/End: first/last scene.
- On a touch screen, swipe horizontally outside interactive controls.

There are 14 main scenes, targeting approximately 13 minutes 50 seconds before optional interactions. Rehearse aloud and keep exploration short to stay within 12–15 minutes. Overview, notes and evidence are outside the timed narrative. Reduced-motion system preferences are respected.

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
.venv/bin/python presentation/html-v1/build_data.py
.venv/bin/python presentation/html-v1/pack.py
```

For styling/content changes alone, edit `styles.css`, `app.js` or `template.html` and run only `pack.py`. `index.html` remains the only file required to present or share. `evidence-manifest.json` records the walkthrough provenance and frozen model fingerprints.

The earlier PPTX and rehearsal notes are preserved under `presentation/output/`.
