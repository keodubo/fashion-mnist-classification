# CNN interpretation implementation plan

**Goal:** Add the approved new CNN slide with real gradient saliency and occlusion, update presenter notes and LaTeX/PDF, and retain a 12–15-minute talk.

**Architecture:** Export explanations offline from the frozen CNN for the ten existing deterministic validation examples. Embed the output in the self-contained HTML; the browser displays stored maps and animates stored occlusion measurements. Model weights, test results and example-selection policy remain fixed.

**Technology:** Existing PyTorch/NumPy/Python environment, vanilla HTML/CSS/JS and pdflatex.

## Decisions and constraints
- New scene 8, after the CNN walkthrough. All original subsequent scenes shift by one.
- Fifteen scene targets: 45,30,40,60,65,65,85,60,60,55,60,55,75,50,35 seconds = 840 seconds (14 minutes).
- Gradient: absolute input derivative of a selected raw class logit, one grayscale channel.
- Occlusion: 5×5 patches, stride 2, include final edge position, replacement intensity 0.5 on [0,1]. Signed original-minus-occluded logit; per-pixel map averages covering patches. Display positive drops with a clear legend; retain signed measurements in the patch readout.
- Explain predicted class by default; also allow true-class target on the misclassified Shirt #32. Targets remain fixed when moving the patch.
- No training, test scoring, tuning, checkpoint modification or model-weight gradients accumulated. Input gradients require autograd, despite evaluation mode.
- Maps are local model diagnostics. They are not probabilities, causal explanations, segmentation masks or proof of shirt-error causes. Independent image normalisation precludes absolute comparisons between examples.
- Preserve existing visual style, offline operation, keyboard/navigation, reduced motion and earlier exported versions.

## Tasks
- [x] Compute maps with focused unit-level blackbox tests for gradient/occlusion numerical outputs and edge coverage; verify checkpoint hash and base predictions.
- [x] Add the interactive responsive scene; remap notes, scene numbers and result-scene styling; embed the new data.
- [x] Update editable script and compiled PDF, presenter notes, route, timings, Q&A and evidence documentation.
- [x] Inspect desktop/tablet/mobile renders and controls, inspect the PDF pages and verify 840-second timing and unchanged test metrics/checkpoint.

## Review focus
- Wrong target class or comparing logits to softmax probabilities.
- Disabling gradients along with dropout at evaluation time.
- Edge patches, overlap aggregation, constant or zero maps.
- Animation running off-screen or under reduced-motion preference.
- Shifted notes, results colours, scene index, script route or mobile overflow.
