# Installation

## Supported setup

- Windows, 64-bit Python 3.13.
- A local checkout of this repository.
- No MATLAB or camera is required for folder-based processing.

The initial verification environment is Python 3.13.2 with the versions listed in the requirements files. This is not a claim of testing all Python or operating-system versions.

## PowerShell

```powershell
git clone https://github.com/ephraimlang98-png/CrackTrackerPy.git
cd CrackTrackerPy
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

While the repository is private, cloning requires access to the owner's GitHub account or an explicit collaborator invitation. Alternatively, download the repository ZIP from GitHub and run the commands after changing into its extracted directory.

The requirements are pinned from the original environment. If a package index or Python version cannot provide these exact versions, use Python 3.13 and the documented environment before changing the pins. A requirements file pins direct dependencies; it is not a complete transitive dependency lock.

## Headless use

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe examples\synthetic_demo.py --output demo_output
```

## Common issues

| Symptom | What to check |
|---|---|
| `No module named PySide6` | Use the virtual environment's Python and install `requirements.txt`. |
| `py -3.13` is unavailable | Install 64-bit Python 3.13 from the official Python distribution, or use its explicit executable path. |
| Qt cannot load its platform plugin | Use a clean virtual environment; avoid mixing Qt packages from other environments. |
| C(T) app fails on `winsound` | Use Windows. The C(T) desktop interface is Windows-specific. |
| Registration is low or a tip is missed | Verify calibration, image quality, crack markings and registration ROI; review overlays before accepting results. |
| Phone app / ADB is unavailable | Use folder-based processing, or follow the optional integration notes. Those binaries are not shipped here. |

Run all commands from the repository root. No activation script or change to PowerShell execution policy is needed.
