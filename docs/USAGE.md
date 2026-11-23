# Measurement guide

## Inputs

The engine recognizes JPG/JPEG, PNG, TIF/TIFF and BMP files. Images within a folder are naturally sorted (for example, `frame2` before `frame10`). Images are read as grayscale floating-point arrays; integer pixel values are normalized by their dtype range. Keep camera framing, orientation and image dimensions consistent.

Use a local `data/` or `private_data/` directory for research images; both are ignored by Git. Do not commit an application's saved setup JSON: it can contain full local paths and experiment-specific information.

## General desktop application (`app.py`)

1. In **הגדרת מדידה** (measurement setup), choose the side-A image folder. Enable side B if using a second specimen face.
2. Select a calibration image and enter a known physical distance. Click its endpoints. Calibrate each camera separately. Multiple calibration distances are combined by averaging pixels/mm and taking its reciprocal.
3. Mark the measurement origin first, then the approximate baseline crack tip. These determine the initial direction. The detector may refine the clicked tip; review the overlay and baseline length.
4. Inspect a later frame. Choose a registration region with stable surface texture away from the growing crack if automatic registration is unreliable.
5. Run batch measurement, or select **מעקב חי** (live monitoring) while a camera writes new images into the folder.
6. Review uncertain or failed rows and the per-side measurements. Correct a tip manually only when the image supports the correction. The UI can fit sensitivity to a corrected tip and remeasure the sequence.
7. Export CSV and, if useful, annotated images from **תוצאות וייצוא** (results and export).

The baseline image's growth is zero. Absolute lengths depend on the chosen origin and calibration. A second camera may have a different absolute origin, so review side-to-side growth differences separately from absolute-length differences.

## Two sides

Ordinary folders are paired by index after natural sorting. Missing or extra files can invalidate the physical pairing. Batch processing uses the smaller count and warns if the counts differ. Matching counts alone do not establish that the frames correspond to the same experimental state.

The GUI computes `a = mean(finite side lengths)`, then `delta_a = a - first_finite_combined_a`. A one-sided fallback can change the meaning of the combined series if the available side changes. Inspect `a_mm_sideA`, `a_mm_sideB`, file names and status fields. Failed values remain non-finite in the engine; do not treat them as zero.

## C(T) application (`ct_app.py`)

The additional **דגם ומידות** page accepts a specimen image and measured height/thickness. It provides automatic feature-detection helpers with manual correction, registration setup, projected lengths from the load line, a back-edge consistency check, a temporal growth guard, and point-selection/load-calculation helpers.

Dimension presets in this portfolio copy are illustrative. The laboratory drawing and load schedule are not distributed. The stage list starts empty; a newly added stage has zero loads until the user supplies reviewed values. The calculation helpers do not control a test machine.

The growth guard can revise recent rows after later frames arrive and withdraw unconfirmed growth at the end of a run. Export after the run has stopped and review its final status. Standards-related labels in the existing application are not a certification of compliance; verify settings against the applicable experiment and standard independently.
