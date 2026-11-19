# Fashion-MNIST presentation

Open `../output/fashion-mnist-presentation.html` in a browser. It is one self-contained offline HTML file. The same content is available as `index.html` in this directory. Earlier presentations are preserved.

## Design

The presentation uses a scientific editorial direction: graphite and ivory, serif titles, a restrained mineral accent and large real observations. The opening contact sheet, large CNN activation plates and ivory results page establish the main compositions. See [design rules](../DESIGN.md).

The persistent wordmark, top toolbar, chapter footer, shortcut hints, progress strip, repeated coloured title fragments and decorative uppercase labels have been removed. Small previous/next controls remain. Presentation tools are available from the subtle bottom-left control or keyboard shortcuts.

Dataset statistics reflow on tablet and phone. Native channel selectors expose all eight recorded activation maps per tensor without crowding the scene. Supporting labels are larger, and numeric validation values accompany the test results.

## Controls

- Left/right arrows or Page Up/Page Down: change scene.
- Space: next scene when the page body has focus.
- N: current presenter notes.
- O: scene index.
- I: methods, evidence and practice questions.
- F: fullscreen.
- Escape: close a dialog or leave fullscreen.
- Home/End: first/last scene.
- The bottom-left presentation-tools control offers notes, index, fullscreen and evidence without a visible toolbar.
- On touch screens, swipe horizontally outside interactive controls or use the previous/next controls.

Input controls keep their normal keyboard behaviour. For example, arrows on the epoch slider adjust the epoch rather than changing scenes. The presentation respects reduced motion.

## Timing and evidence

The first 16 slides target 780 seconds (13 minutes). Saliency and occlusion are on slide 8; the SHAP beeswarm and waterfall are on slide 14. Stop on slide 16 for questions. Press O to jump to appendix slides 17–20.

The original metrics and examples in `data.json` remain unchanged: real observations, recorded model walkthroughs, exact class counts, the saved final results and genuine directional error examples. The new interpretation exporter performs validation-only forward and input-gradient passes through the frozen CNN. No training, test inference, test-based tuning or change to reported results was performed. Activations are recorded outputs; the scan is illustrative. The feature-map intensity scale is shared within each original layer tensor, not across layers.

## Rebuild

From the project root:

```bash
.venv/bin/python presentation/editorial/pack.py
```

Edit `template.html`, `styles.css` or `app.js`, then run the packer. Existing authoring output may be updated during requested edits; preserve unrelated original versions. The exported file includes all required images, data, CSS and JavaScript and makes no network request to present.

## Review

The original fourteen-scene presentation was rendered at 1440×900, 1280×720, 768×1024 and 390×844. That review covered all fourteen original scenes, representative individual screenshots, offline operation, keyboard/dialog/fullscreen behaviour, example propagation, channel selection, range controls, mobile statistics and reduced motion. See [review record](design-review.md).

## CNN interpretation (scene 8)

Choose Gradient saliency or Occlusion map; use the class selector to compare the predicted Coat logit and true Shirt logit for validation Shirt #32. All ten existing deterministic validation examples remain available. The moving patch uses stored scores; the HTML is offline and does not execute a model. Its slider supports keyboard/touch use. The patch scan starts and loops automatically on entry and continues after example or target changes; it pauses off-screen, while a dialog is open, or when the page is hidden. Returning resumes it automatically. Optional pause and position controls remain available. Reduced motion advances one patch per button click. The preprocessing slide also scans its pixels automatically, updating intensity, normalisation and the corresponding flattened-vector marker; reduced motion retains optional pointer inspection.

`interpretation.json` contains absolute input gradients of raw class logits, and signed occlusion changes for 169 full 5×5 patches (stride 2, final edge included; grey fill 0.5). The per-pixel occlusion map averages all signed changes from covering patches, then the display highlights positive means only. The patch readout retains negative changes. Each map has its own intensity scale; do not compare brightness across maps or infer causality/segmentation. Masking choices can affect the result or create unrealistic inputs. W10 CNN lecture pages 56–60 are the course reference.

To regenerate explanation data with the original local checkpoint and saved split:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_interpretation.py'
.venv/bin/python presentation/editorial/build_interpretation.py
.venv/bin/python presentation/editorial/pack.py
```

The exporter checks the frozen checkpoint fingerprint, deterministic validation membership, base probabilities and unchanged model state. A fresh clone needs the original local model/data artifacts to regenerate the maps; the already-bundled HTML and recorded JSON do not need them. No new dependencies were added.

The updated [rehearsal script](../output/presentation-script.pdf) and [LaTeX source](../output/presentation-script.tex) cover the 16 timed slides and four Q&A slides. Timing is a rehearsal target.

Both automatic scans advance every 200 milliseconds (five positions per second), slowing the displayed values enough to follow during narration. They do not need to complete a full cycle during the timed slide.

Regenerate the additional evidence with the project Python environment: run `presentation/editorial/score_knn.py` and `presentation/editorial/build_shap.py`, then run the packer. The kNN score excludes each training index from its neighbour set. The SHAP exporter evaluates every coalition of four image quadrants for the frozen CNN’s Coat probability, using 20 training reference images. All explanation targets are validation images.
