# Installation

## Supported setup

- Windows, 64-bit Python 3.13.
- A local checkout of this repository.
- No MATLAB or camera is required for folder-based processing.

The initial verification environment is Python 3.13.2 with the versions listed in the requirements files. This is not a claim of testing all Python or operating-system versions.

## Installation commands

Create and activate a virtual environment with Python 3.13 using your preferred environment manager. In the commands below, `python` refers to that environment's interpreter.

```sh
git clone https://github.com/ephraimlang98-png/CrackTrackerPy.git
cd CrackTrackerPy
python -m pip install -r requirements.txt
python app.py
```

Alternatively, download the repository ZIP from GitHub and run the installation commands after changing into its extracted directory.

The requirements are pinned from the original environment. If a package index or Python version cannot provide these exact versions, use Python 3.13 and the documented environment before changing the pins. A requirements file pins direct dependencies; it is not a complete transitive dependency lock.

## Headless use

```sh
python -m pip install -r requirements-core.txt
python examples/lab_demo.py --output demo_output/lab
```

## Common issues

| Symptom | What to check |
|---|---|
| `No module named PySide6` | Use the virtual environment's Python and install `requirements.txt`. |
| The active Python version is unsupported | Use a virtual environment with 64-bit Python 3.13; check it with `python --version`. |
| Qt cannot load its platform plugin | Use a clean virtual environment; avoid mixing Qt packages from other environments. |
| C(T) app fails on `winsound` | Use Windows. The C(T) desktop interface is Windows-specific. |
| Registration is low or a tip is missed | Verify calibration, image quality, crack markings and registration ROI; review overlays before accepting results. |
| Phone app / ADB is unavailable | Use folder-based processing, or follow the optional integration notes. Those binaries are not shipped here. |

Run all commands from the repository root with your virtual environment activated.
