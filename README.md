# CrackTrackerPy

**Image-based crack-growth measurement for laboratory experiments.**

A Python desktop application for tracking visible crack tips, converting pixels to millimetres, and measuring growth on two specimen faces. Used in a research laboratory, it processes existing image sequences or monitors folders as new images arrive.

[Quick start](#quick-start) · [Measurement guide](docs/USAGE.md) · [How it works](docs/ARCHITECTURE.md) · [Documentation](docs/README.md)

![Crack-tip tracking and measured growth across thirty laboratory images](docs/demo/crack_progression.gif)

*The existing measurement engine applied to 30 authorized images from one specimen face. The animation shows computed results, rather than the desktop interface. [Calibration, inputs and reproduction details →](docs/REAL_IMAGE_DEMO.md)*

## Capabilities

| Area | Implemented functionality |
|---|---|
| Detection | Image registration, local intensity contrast and connected-path tracing. |
| Measurement | Per-side pixel-to-mm calibration, absolute crack length, baseline-relative growth and side A/B aggregation. |
| Processing | Batch sequences and live folder monitoring. |
| Review and export | Tip overlays, measurement status, sensitivity adjustment, manual correction, plots and CSV export. |
| Compact-tension workflow | A separate C(T) interface with geometry helpers, projected crack length, temporal growth confirmation and load calculations. |

## Quick start

Use **Windows and 64-bit Python 3.13**, with a virtual environment activated. Run commands from the repository root; `python` refers to the active environment.

```sh
git clone https://github.com/ephraimlang98-png/CrackTrackerPy.git
cd CrackTrackerPy
python -m pip install -r requirements.txt
python app.py
```

Choose the entry point for your workflow:

| Workflow | Command |
|---|---|
| General desktop application | `python app.py` |
| Compact-tension desktop application | `python ct_app.py` |
| Included real-image example | `python examples/lab_demo.py --output demo_output/lab` |

The desktop interface is primarily **Hebrew, right-to-left**; documentation and example figures are in English. See [installation](docs/INSTALLATION.md) for environment setup, the smaller headless dependency set and troubleshooting.

## Measurement workflow

1. Select the side-A image folder and, optionally, the matching side-B folder.
2. Calibrate each side, then mark the measurement origin and approximate baseline tip.
3. Check the baseline and a later frame; adjust sensitivity or the registration region when needed.
4. Run batch processing or start live folder monitoring.
5. Review tip overlays and per-side status, correct supported measurements, and export results.

**Two-side interpretation:** ordinary folders are paired by natural filename order, without timestamp synchronization. The combined length averages the available finite side values and can fall back to one side. Review pairing and per-side status before interpreting the mean. [Detailed measurement guide →](docs/USAGE.md)

## Project map

| Location | Purpose |
|---|---|
| [`app.py`](app.py), [`ct_app.py`](ct_app.py) | General and C(T) desktop entry points. |
| [`crack_core.py`](crack_core.py) | Registration, contrast-based tracing and per-frame measurement. |
| `ct_*.py` | C(T) geometry, feature detection and temporal tracking. |
| `phone_b.py`, `fc_*.py`, `gcam_b.py` | Optional phone and capture integrations. |
| [`examples/`](examples/) | Reproducible examples and the 30-frame laboratory subset. |
| [`tests/`](tests/) | Measurement, geometry, protocol and desktop checks. |
| [`docs/`](docs/README.md) | Installation, operation, architecture and validation notes. |

For the processing stages and individual module responsibilities, see [architecture and limits](docs/ARCHITECTURE.md).

## Verification

```sh
python -m unittest discover -s tests -v
```

Automated checks cover measurement growth, registration, failure handling, geometry, protocol framing and desktop startup. The real-image example exercises the engine on the included recording; it is not an independent accuracy benchmark. See [validation scope](docs/VALIDATION.md).

## Further documentation

- [Documentation index](docs/README.md) — choose a guide by task.
- [Real-image demonstration](docs/REAL_IMAGE_DEMO.md) — calibration, initialization, outputs and interpretation.
- [Optional hardware](docs/OPTIONAL_HARDWARE.md) — phone capture and integration requirements.
- [Data and distribution notes](docs/PUBLICATION_REVIEW.md) — included material, review scope and rights.

Only the authorized 30-frame laboratory subset is included. No open-source license has been granted.
