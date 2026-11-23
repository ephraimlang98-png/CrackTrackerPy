# Architecture and limits

The repository keeps the original flat Python module layout so the existing entry points and imports remain recognizable. Legacy backups, frozen distributions and research data are not part of this checkout.

| Module | Responsibility |
|---|---|
| `app.py` | General PySide6 UI, calibration, batch/live workers, side aggregation, plots and exports. |
| `crack_core.py` | Image IO, translation registration, contrast/path tracing, per-frame results, annotations and manual correction. |
| `ct_app.py` | C(T) workflow and its integration with the general interface. |
| `ct_core.py` | Feature geometry, scale helpers, C(T) load calculations and measurement-point selection. |
| `ct_detect.py`, `ct_detect2.py`, `ct_detect3.py`, `ct_frame.py` | Specimen, notch and frame feature-detection helpers. Several revisions are retained because they import one another. |
| `ct_track.py` | Temporal growth confirmation and revision of provisional results. |
| `phone_b.py`, `fc_link.py`, `fc_proto.py`, `fc_autosetup.py` | Optional phone capture, folder triggers, USB/ADB link, message framing and camera setup. |
| `gcam_b.py` | Optional Google Camera UI automation for a specific phone layout. |
| `examples/synthetic_demo.py` | Reproducible artificial images and a headless run of the existing engine. |
| `examples/lab_demo.py` | Reproducible real-image run, with an English engine-output figure and animation. |

## Processing

The UI prepares a per-side dictionary containing a grayscale baseline, `mm_per_px`, reference/root and seed points, registration ROI and detector parameters. `prepare_side()` estimates the direction, baseline tip and threshold. `measure_frame()` registers a frame and returns a `FrameResult` containing length, growth, tip position, registration correlation and status.

Registration estimates translation, including subpixel refinement, rather than a general affine or projective transform. The crack path uses dynamic programming within local strips, local background evidence, bounded direction changes and a configurable search range. The general UI's worker handles sequence order and previous-frame priors. Its main window computes side aggregation and stores export rows.

The C(T) interface uses projected length rather than path arc length. Its worker wraps the core with `GrowthGuard`, which compares newly detected growth against reference images, requires temporal persistence and may revise earlier rows.

## Practical limits

- Calibration and origin selection directly affect reported lengths. The algorithm does not infer a universal physical scale.
- Rotation, perspective changes, occlusion, saturation, weak texture, reflections, markings and inconsistent framing can defeat registration or tip detection.
- Small image-size differences are padded/cropped, not geometrically rescaled. Such frames require review.
- The threshold perturbation check is a sensitivity diagnostic, not a statistical confidence interval or calibrated measurement uncertainty.
- Ordinary two-folder mode has no timestamp synchronization. Phone mode has its own name-based association and timeout handling.
- Live monitoring polls directories and waits for recent files; it provides no hard real-time or throughput guarantee.
- The growth guard intentionally delays confirmation and may revise prior values. It has not been revalidated on every material or camera.
- C(T) geometric heuristics include assumptions about specimen/fixture appearance. Replacing illustrative dimension presets does not make them universal.
- No automatic fatigue-machine control is implemented by this repository. Calculation tables and alerts require operator interpretation.

## Changes in the repository preparation

The tracing, registration, measurement, aggregation and growth-guard algorithms are retained. Repository preparation adds documentation, dependency files, synthetic examples and tests. Dataset-specific help references and a legacy sample-specific self-test are removed. Laboratory dimension/tolerance/load presets are replaced or cleared in the portfolio copy and labelled in the interface; the original laboratory folder is not modified.
