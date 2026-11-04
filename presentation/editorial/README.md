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

Fourteen scenes retain their existing target times, totalling 810 seconds, or 13 minutes 30 seconds. Rehearse aloud to allow for interaction within the 12–15 minute limit. Supporting dialogs are outside the timed narrative.

The embedded `data.json` is unchanged from the verified prior presentation: real observations, recorded model walkthroughs, exact class counts, the saved final results and genuine directional error examples. No model training, inference or test-based tuning was performed for this redesign. Activations are recorded outputs; the scan is illustrative. The feature-map intensity scale is shared within each original layer tensor, not across layers.

## Rebuild

From the project root:

```bash
.venv/bin/python presentation/editorial/pack.py
```

Edit `template.html`, `styles.css` or `app.js`, then run the packer. Existing authoring output may be updated during requested edits; preserve unrelated original versions. The exported file includes all required images, data, CSS and JavaScript and makes no network request to present.

## Review

The presentation was rendered at 1440×900, 1280×720, 768×1024 and 390×844. The review covered all fourteen scenes, representative individual screenshots, offline operation, keyboard/dialog/fullscreen behaviour, example propagation, channel selection, range controls, mobile statistics and reduced motion. See [review record](design-review.md).
