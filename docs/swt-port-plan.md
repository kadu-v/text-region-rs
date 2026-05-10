# SWT Port Plan

Reference implementation:

- OpenCV contrib `modules/text/src/text_detector_swt.cpp`
- Public API documented as `cv::text::detectTextSWT`

Porting order:

1. Port the pure SWT operator from edge and gradient inputs.
   - Match `SWTFirstPass`: invalid value `-1`, normalized gradients, dark-on-light direction flip, 0.05 pixel ray step, opposite-edge gradient test.
   - Match `SWTSecondPass`: per-ray upper median and per-pixel minimum.
   - Verify with fixed numeric fixtures derived from OpenCV behavior.
2. Port normalization for debug/visual output.
   - Match `normalizeAndScale`: invalid pixels become 255, valid pixels scale between observed min and max.
3. Port connected components over SWT values.
   - Match OpenCV's 8-neighbor graph where stroke-width ratios are within 3.0.
   - Add fixtures for component membership and bounding boxes.
4. Port component filtering and chain construction.
   - Match variance, width, aspect ratio, containment, color-distance, and chain heuristics.
   - Compare final rectangles against `cv::text::detectTextSWT` on stable sample images.
5. Add optional OpenCV-backed golden regeneration.
   - Keep normal tests dependency-free and do not execute Python from Rust tests.
   - If golden values need to be regenerated manually, use the Python environment under `python/`.
6. Add the OpenCV-style public wrapper.
   - Convert BGR to grayscale.
   - Run Canny with OpenCV's SWT thresholds.
   - Build Scharr gradients with Gaussian blur.
   - Run SWT, component filtering, chain finding, normalization, and debug rendering.

Completed in this pass:

- Steps 1 through 6.
- TDD tests cover input validation, invalid pixels, matching/opposing edge gradients, first/second pass output, normalization output, fixed OpenCV-derived numeric reference values for synthetic rays, SWT connected components, component filtering, chain-region heuristics, BGR preprocessing, and the full BGR-to-detections pipeline.
