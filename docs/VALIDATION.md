# Validation scope

The project owner reports that the system is used in a research laboratory and that outputs were checked against manual measurements while the solution was iterated and integrated into the experiment. This statement describes development practice, not a published independent benchmark.

## Reproducible checks

The included synthetic example and unit tests exercise the existing engine with artificial, known-length image sequences. Checks cover baseline-relative growth, registration of translated images, failure flags, manual correction, ordering, phone-message framing and selected C(T) calculations. Desktop verification is separate from numerical tests.

Synthetic tolerances are regression checks for the constructed examples. They must not be advertised as experimental measurement accuracy, robustness to real materials, a general error bound or a timing benchmark.

## Repository-preparation checks (2026-11-23)

The 14 included tests passed locally on Windows with Python 3.13.2 and the pinned dependency versions. Both desktop interfaces started in isolated offscreen processes, without connecting to hardware or reading the operator's saved settings. The headless example completed all thirty paired synthetic frames and produced CSV/annotated-image outputs.

The original laboratory folder was kept unchanged. The registration/tracing engine and temporal-growth module were copied without modifying their algorithms. GitHub Actions execution, live camera/phone capture and physical experiment operation are separate checks and are not implied by the local test results.

The real-image example also completed all 30 selected frames locally using the unchanged engine and owner-confirmed 1 mm ruler intervals. All selected frames returned `ok`, with registration correlations of approximately 0.856 to 1.000. These are operational checks for this subset and initialization, not independent accuracy measurements. The example retains a small decrease in one measured growth value rather than smoothing it away.

## Research datasets

Thirty explicitly authorized real-image inputs from one specimen face are distributed with `examples/lab_demo.py`; see [the real-image demonstration](REAL_IMAGE_DEMO.md). Measurement workbooks, other research images and earlier export tables are not included. Prior application-export CSVs are software outputs; reproducing their values is not validation against independent manual ground truth. Camera calibration, measurement-origin conventions and correct frame pairing must be established before comparisons are meaningful.

An experimental accuracy report would require an authorized dataset, independently verified manual measurements, an explicit calibration and pairing protocol, and a documented error analysis. No numeric laboratory accuracy claim is made here.
