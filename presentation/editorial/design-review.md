# Design review

## Direction

The baseline used a large condensed heading, coloured title fragment, all-caps label and small visualization on most scenes. Its persistent header and footer consumed 150px at a 720px viewport. The redesign gives content the canvas and uses distinct specimen-led, model-led and results-led compositions.

Three directions were considered: dark scientific exhibition, paper research journal and cinematic sequential projection. The exhibition direction preserves the requested dark atmosphere and image collage while changing typography, colour use, layout and scale.

## Findings and corrections

| Severity | Finding | Root cause | Correction |
|---|---|---|---|
| High | Dataset statistics disappeared on tablet/phone | Entire statistics aside hidden below 900px | Reflow statistics before the contact sheet |
| High | Removing the toolbar could break navigation | Eager DOM references to removed header/footer elements | Remove footer update dependencies; put optional tools in a dialog; retain keyboard, focus and status behaviour |
| High | CNN evidence and labels were too small | Sixteen simultaneous map thumbnails and small captions | Two large selected activation plates with native channel selectors and larger text |
| High | First redesign overlapped CNN controls at 1280×720 | Intrinsic stage height exceeded its flex allocation | Compact plates, intrinsic stage minimum, native channel selectors instead of two button rows |
| Medium | Fixed mobile navigation covered CNN controls | Navigation floated above the scrollable content | Reserve a separate bottom area outside the mobile content viewport |
| Medium | Chart labels remained 10–11px | Inline SVG font sizes were independent of CSS | Responsive SVG viewBox, fewer phone ticks and 16/18px label values |
| Medium | Mobile scores nearly touched the right edge | Score text exceeded its fixed-width track | Reflow result rows with an auto-sized numeric column and proper inset |
| Medium | Validation reference was hard to distinguish | Pale marks on a lime background | Ivory results plate, darker marks and explicit validation values |
| Preference | Repeated highlighted-word headings felt formulaic | Same display treatment on almost every scene | Serif subject titles and varied scene compositions; accent reserved for data |

## Validation performed

- All 14 scenes rendered at 1440×900, 1280×720, 768×1024 and 390×844; 56 screenshots captured in an isolated Chromium session.
- Contact sheets inspected for narrative flow; representative scenes inspected individually at full size, including opening, dataset, pixels, splitting, baselines, CNN, results, learning curves, class recall, errors and limitations.
- Independent art-direction and usability reviews found the compact CNN overlap; it was repaired and rendered again.
- Dataset statistics remained present on tablet and phone. Portrait scenes scroll vertically where needed rather than removing evidence.
- No visible requested wordmark, toolbar or keyboard-hint strip; no visible broken images, horizontal foreground overflow or runtime errors in the final sweep.
- N/O/I/Escape, fullscreen, native channel selection and enlargement, shared validation examples, epoch-slider keyboard behaviour, gallery switching, resizing and offline loading checked through public UI behaviour.
- Reduced-motion mode checked. No general accessibility certification is implied by these focused checks.

## Scientific constraints

The source data and outputs are unchanged. Results remain final test results; walkthroughs remain recorded validation outputs. The stratification explanation, distinction from five-fold cross-validation, publisher-provided test set, shirt recall, all six directional counts and evidence-versus-hypothesis caveats are retained.

The presentation target remains 13 minutes 30 seconds. This is an authored pacing target, not a measured speaking time.
