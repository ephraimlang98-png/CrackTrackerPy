# Real-image example inputs

These thirty PNG files are an ordered subset from a single specimen face in a laboratory recording. The project owner explicitly authorized using these images to demonstrate the project on 2026-11-23. Every third acquisition from an initial 88-frame segment is retained (30 images). They are not a matched side-A/side-B dataset and are not consecutive acquisitions.

The source images were 4096 x 3000 grayscale BMPs. A fixed crop `(left=1850, top=500, right=2500, bottom=2650)` was applied to all thirty, followed by a 90-degree counterclockwise quarter-turn. The resulting images are 2150 x 650 pixels. Both operations preserve retained pixel intensities and spacing, with no interpolation, contrast enhancement or synthetic crack insertion. PNG encoding is lossless. Original filenames, timestamps and image metadata are not included.

The owner confirmed that successive numbered ruler marks are 1 mm apart. `examples/lab_demo.py` uses marks 1 and 31 for a 30 mm calibration distance, plus explicitly documented origin, approximate seed and registration region.

Run from the repository root:

```sh
python examples/lab_demo.py --output demo_output/lab
```

The script calls the unchanged engine, saves measurements, and creates an English result figure and animation. The animation presents computed engine outputs; it is not a screen recording of the desktop interface. No independent manual ground-truth measurements are supplied, so this example does not establish experimental accuracy. Permission to include these example images does not grant a general reuse license.
