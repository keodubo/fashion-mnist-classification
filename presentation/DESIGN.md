# Presentation design

## Intent

An interactive scientific exhibition about how three models classify the same clothing images. The data and model behaviour are the subject. The visual system gives specimens and results more space than interface controls.

## Direction considered

- Dark scientific exhibition: graphite, ivory, serif headlines, large specimens and restrained mineral colour. Selected because it retains the scientific atmosphere and image collage while changing the typography, compositions and evidence hierarchy.
- Research journal: paper, ink and oxide-red annotation. A viable alternative, but a greater departure from the requested dark environment.
- Cinematic projection: sequential full-canvas stages. Expressive, but less effective for comparison and a 12–15 minute talk.

## System

- Graphite background `#20231f`; ivory foreground `#eeeae1`; secondary text `#c0c3b9`; rules `#54594e`.
- Mineral citron `#c3ce9c` marks an active class, a selected model, or a data finding. Never colour an arbitrary headline word.
- Paper background `#eeeae1` and ink `#252923` for the final-results plate. Colour inversion is reserved for this narrative event.
- Georgia/Palatino-style editorial serif for titles and the opening; Avenir Next/clear sans serif for explanatory text, controls and data. Use tabular numerals for metrics.
- Desktop title 44–68px; opening 76–108px; body 18–22px; labels 16–18px; metadata at least 14px. At 1280×720, reduce headings or reflow rather than shrinking core evidence. Mobile body 16px and captions at least 14px.
- Spacing scale: 8, 12, 16, 24, 32, 48, 64, 80px. Use a twelve-column conceptual grid and consistent outer margins, with deliberately different compositions between scenes.
- No persistent wordmark, top toolbar, keyboard hints, chapter footer or progress strip. Preserve keyboard shortcuts. Small labelled-for-assistive-technology previous/next controls remain; a focus-revealed presenter-tools button provides a discoverable alternative to shortcuts.

## Canonical compositions

1. Opening: direct research title beside an asymmetric, slowly drifting contact sheet of real observations. No oversized condensed type or highlighted-word formula.
2. CNN: one large selected activation plate for each of the two pooled tensors, with a native selector exposing eight recorded channels and clearly labelled dimensions. A channel selector switches recorded evidence; it never implies live inference. Input and output remain visible. Native selectors prevent compact-screen overlap without reducing labels.
3. Interpretation: original validation specimen, one selected sensitivity map and a measured occlusion experiment. A labelled sequential mineral scale indicates within-map magnitude. Signed patch readouts distinguish score drops from increases; the positive-only heatmap is explicitly labelled. No suggestion that maps are segmentation masks or causal explanations.
4. Results: ivory canvas, a large CNN accuracy finding and a comparison chart with labelled test and validation scores. Never use an unlabeled truncated bar scale.

## Other scene families

- Dataset/contact sheet: larger real images and statistics that reflow on mobile, never disappear.
- Pixels and baseline models: image-led experiments with readable calculations and probabilities.
- Split and training: analytical diagrams with supporting facts in a restrained side column.
- Errors: directional comparisons and genuine paired specimens; observed findings remain separate from unproven hypotheses. Paired error bars use the same colour, left-hand zero and count scale; the written actual-to-predicted labels distinguish directions.
- Limitations: short editorial paragraphs with meaningful titles; no decorative numbered checklist.

## Motion and interaction

- Opening: one slow contact-sheet drift; no thumbnail breathing.
- CNN: short staged activation reveal and a crossfade when selecting channels. Stop the illustrative scan when the scene is inactive.
- Charts: a brief reveal or transition tied to a real change. Do not animate every text element or delay reading. Reserve space below the mobile content viewport for navigation, so its controls cannot cover data controls.
- Scene transitions: 200–350ms, outgoing content exits before the next title becomes prominent. No overlapping headlines.
- Respect reduced motion. All imagery and functionality work offline. Native focus states, Escape dismissal, input arrow keys, image inspection, dataset selection, notes and evidence dialogs must remain operable.
- Touch targets are at least 44×44 CSS pixels. Important content cannot depend on hover. Preserve keyboard and touch alternatives.

## Forbidden patterns

- Repeated coloured headline fragments, all-caps decorative eyebrows, ornamental ↗ arrows, neon dashboard styling, arbitrary gradients, tiny captions, repeated generic panel grids, hidden mobile statistics.
- Invented examples, measurements or causal claims; retraining or changing final results during presentation work.

## Validation

Render and inspect at 1440×900, 1280×720, 768×1024 and 390×844. Check every scene for clipping and missing content; inspect representative screenshots individually, not only a contact sheet. Verify keyboard and touch controls, dialogs, reduced motion and offline loading. Reconcile all numeric presentation inputs with the unchanged saved evidence. Record factual checks separately from aesthetic judgments.
