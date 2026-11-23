"""Reproduce the real-image example with the unchanged measurement engine.

Inputs are a cropped, lossless, quarter-turned subset of one owner-authorized
laboratory recording. See docs/REAL_IMAGE_DEMO.md for setup and limitations.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import crack_core as cc

SAMPLES = ROOT / "examples" / "lab_sequence"
# Numbered ruler marks 1 and 31 in frame00.png. The owner confirmed that
# successive numbered marks are separated by 1 mm (30 intervals here).
CALIBRATION_POINTS = ((282.0, 220.0), (1443.0, 220.0))
CALIBRATION_DISTANCE_MM = 30.0
ORIGIN = (250.0, 432.0)
APPROXIMATE_TIP = (880.0, 432.0)
REGISTRATION_ROI = (1800, 350, 180, 140)


def prepare() -> tuple[list[str], dict]:
    files = cc.list_images(str(SAMPLES))
    if len(files) != 30:
        raise RuntimeError("Expected the thirty distributed lab_sequence PNG files")
    pixels = float(np.linalg.norm(np.subtract(*CALIBRATION_POINTS)))
    params = cc.default_params()
    params["max_shift_px"] = 80
    side = cc.prepare_side(dict(
        base=cc.read_gray(files[0]), base_file=files[0],
        mm_per_px=CALIBRATION_DISTANCE_MM / pixels,
        root=ORIGIN, seed=APPROXIMATE_TIP, roi=REGISTRATION_ROI, P=params,
    ))
    return files, side


def measure() -> tuple[dict, list[cc.FrameResult]]:
    files, side = prepare()
    results = []
    previous = None
    for path in files:
        result = cc.measure_frame(path, side, keep_image=True, prev_reach=previous)
        if not result.ok or not np.isfinite(result.a):
            raise RuntimeError(f"{Path(path).name}: {result.status}")
        previous = result.reach
        results.append(result)
        print(f"Measured {len(results)}/{len(files)}: {Path(path).name}, "
              f"growth={result.da:.3f} mm, status={result.status}", flush=True)
    return side, results


def figure(side: dict, results: list[cc.FrameResult], index: int) -> Image.Image:
    """An English presentation of actual core results, not a desktop screenshot."""
    result = results[index]
    navy, teal, orange = "#172b4d", "#007f86", "#e37b26"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#f7f9fc")
        grid = fig.add_gridspec(2, 3, left=.045, right=.965, top=.80, bottom=.10,
                               width_ratios=(1, 1, .92), wspace=.25, hspace=.30)
        raw = fig.add_subplot(grid[0, :2])
        detected = fig.add_subplot(grid[1, :2])
        zoom = fig.add_subplot(grid[0, 2])
        growth = fig.add_subplot(grid[1, 2])
        fig.text(.045, .94, "CrackTrackerPy", color=navy, fontsize=27, weight="bold")
        fig.text(.045, .895, "Real laboratory images · contrast-based crack tracking", color="#53647d", fontsize=14)
        fig.text(.965, .94, f"FRAME {index + 1} / {len(results)}", ha="right", color=teal, fontsize=13, weight="bold")
        fig.text(.965, .892, f"Growth from baseline  {result.da:.2f} mm", ha="right", color=navy, fontsize=16, weight="bold")
        fig.text(.965, .855, f"Registration correlation  {result.corr:.3f}   |   Status: {result.status}",
                 ha="right", color="#53647d", fontsize=10)
        for axis, title in [(raw, "INPUT  /  specimen face and visible ruler"),
                            (detected, "OUTPUT  /  traced crack and detected tip")]:
            axis.imshow(result.img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_xlim(200, 1750)
            axis.set_ylim(560, 95)
            axis.set_title(title, loc="left", pad=10, fontsize=10, color=navy, weight="bold")
            axis.set_axis_off()
        if result.meas_xy is not None:
            detected.plot(*result.meas_xy.T, color=orange, linewidth=1.6, label="Measured path")
        detected.scatter(*result.ref_xy, marker="+", s=90, c="#f4cd45", linewidths=1.7)
        detected.scatter(*result.tip_xy, s=220, facecolors="none", edgecolors="#ed4058", linewidths=1.8)
        detected.annotate(f"a = {result.a:.2f} mm", xy=result.tip_xy,
                          xytext=(result.tip_xy[0] + 115, result.tip_xy[1] - 80),
                          color="white", fontsize=11, weight="bold",
                          arrowprops={"arrowstyle": "-", "color": "#ed4058", "lw": 1.3},
                          bbox={"boxstyle": "round,pad=.4", "fc": navy, "ec": "none", "alpha": .9})
        zoom.imshow(cc.tip_zoom(result.img, result.tip_xy, side["mm_per_px"], size=420, half_mm=1.5))
        zoom.set_title("TIP DETAIL  /  raw image around the result", loc="left", pad=10,
                       fontsize=10, color=navy, weight="bold")
        zoom.set_axis_off()
        x = np.arange(1, len(results) + 1)
        values = [r.da for r in results]
        growth.plot(x, values, color="#cbd5e1", lw=1.6, marker="o", ms=4)
        growth.plot(x[:index + 1], values[:index + 1], color=teal, lw=2.5, marker="o", ms=5)
        growth.scatter(index + 1, result.da, color=orange, s=65, zorder=4)
        growth.set_title("BASELINE-RELATIVE GROWTH", loc="left", fontsize=10, color=navy, weight="bold", pad=12)
        growth.set_xlabel("Selected frame (ordered subset)", color="#53647d")
        growth.set_ylabel("Growth [mm]", color="#53647d")
        growth.set_xticks(np.linspace(1, len(results), 6).round().astype(int))
        growth.set_ylim(-.2, max(values) + .8)
        growth.set_facecolor("#f7f9fc")
        growth.grid(axis="y", color="#e2e8f0", lw=.8)
        growth.tick_params(colors="#53647d")
        for spine in growth.spines.values():
            spine.set_color("#cbd5e1")
        fig.text(.045, .045, "Single specimen face · existing engine output · not an independent accuracy benchmark",
                 color="#53647d", fontsize=10)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").copy()


def run(output: Path) -> list[cc.FrameResult]:
    output.mkdir(parents=True, exist_ok=True)
    side, results = measure()
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "a_mm", "growth_mm", "registration_correlation", "status"])
        for result in results:
            writer.writerow([Path(result.file).name, result.a, result.da, result.corr, result.status])
    frames = []
    for index in range(len(results)):
        frames.append(figure(side, results, index))
        print(f"Rendered {index + 1}/{len(results)}", flush=True)
    frames[-1].save(output / "real_measurement.png")
    # One shared palette avoids distracting palette changes between frames.
    palette = frames[-1].quantize(colors=128)
    animated = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    animated[0].save(output / "crack_progression.gif", save_all=True,
                     append_images=animated[1:], duration=[1100] + [140] * (len(frames) - 2) + [2200],
                     loop=0, optimize=True, disposal=2)
    for result in results:
        print(f"{Path(result.file).name}: a={result.a:.3f} mm, growth={result.da:.3f} mm, "
              f"correlation={result.corr:.3f}, status={result.status}")
    print(f"Output: {output.resolve()}")
    print("Demonstration results; no independent manual ground truth is distributed.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "demo_output" / "lab")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
