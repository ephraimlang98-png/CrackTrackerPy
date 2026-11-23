"""Run the existing core on artificial images, without lab data or a camera.

This example orchestrates two independent per-side core measurements. The desktop
application performs its own aggregation; its behavior is tested separately.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import crack_core as cc

MM_PER_PX = 0.05
BASE_TIP_X = 180
ROOT_X = 80
CRACK_Y = 160
REGISTRATION_ROI = (380, 35, 100, 80)
GROWTH_A_MM = (0.0, 0.5, 1.0, 1.5, 2.0)
GROWTH_B_MM = (0.0, 0.6, 1.1, 1.6, 2.1)


def make_images(folder: Path, growth_mm, seed: int = 7) -> list[Path]:
    """A textured surface with an extending horizontal dark line (not a specimen)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.normal(0.0, 0.08, (320, 640)), 1.0)
    background = np.clip(0.72 + texture, 0.1, 0.95)
    paths = []
    for index, growth in enumerate(growth_mm):
        image = background.copy()
        tip_x = BASE_TIP_X + int(round(float(growth) / MM_PER_PX))
        image[CRACK_Y - 1:CRACK_Y + 2, ROOT_X:tip_x + 1] = 0.08
        path = folder / f"frame{index:03d}.png"
        Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)
        paths.append(path)
    return paths


def prepare(paths: list[Path]) -> dict:
    """Use the same public functions as the UI, with documented example settings."""
    params = cc.default_params()
    params["max_shift_px"] = 12
    return cc.prepare_side(dict(
        base=cc.read_gray(str(paths[0])),
        base_file=str(paths[0]),
        mm_per_px=MM_PER_PX,
        root=(ROOT_X, CRACK_Y),
        seed=(BASE_TIP_X, CRACK_Y),
        roi=REGISTRATION_ROI,
        P=params,
    ))


def measure(paths: list[Path], side: dict) -> list[cc.FrameResult]:
    results = []
    previous = None
    for path in paths:
        result = cc.measure_frame(str(path), side, prev_reach=previous)
        if result.ok and np.isfinite(result.reach):
            previous = result.reach
        results.append(result)
    return results


def run(output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    paths_a = make_images(output / "side_a", GROWTH_A_MM, seed=7)
    paths_b = make_images(output / "side_b", GROWTH_B_MM, seed=19)
    side_a, side_b = prepare(paths_a), prepare(paths_b)
    results_a, results_b = measure(paths_a, side_a), measure(paths_b, side_b)
    results = []
    for index, (a, b) in enumerate(zip(results_a, results_b)):
        # Require two successful sides in this example. The GUI permits a
        # one-sided finite-value fallback; see docs/USAGE.md.
        if not (a.ok and b.ok and np.isfinite(a.a) and np.isfinite(b.a)):
            raise RuntimeError(f"Synthetic frame {index}: {a.status} / {b.status}")
        results.append(dict(
            frame=index,
            expected_growth_A_mm=GROWTH_A_MM[index],
            expected_growth_B_mm=GROWTH_B_MM[index],
            measured_growth_A_mm=a.da,
            measured_growth_B_mm=b.da,
            mean_length_mm=(a.a + b.a) / 2,
            mean_growth_mm=0.0,
            status_A=a.status,
            status_B=b.status,
        ))
    reference = results[0]["mean_length_mm"]
    for row in results:
        row["mean_growth_mm"] = row["mean_length_mm"] - reference
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for name, paths, side, frames in [
        ("A", paths_a, side_a, results_a), ("B", paths_b, side_b, results_b)
    ]:
        label = f"SYNTHETIC side {name} | growth {frames[-1].da:.3f} mm | not laboratory data"
        annotated = cc.render_annotated(cc.read_gray(str(paths[-1])), side, frames[-1], label)
        annotated.save(output / f"annotated_{name}.png")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "demo_output")
    args = parser.parse_args()
    results = run(args.output.resolve())
    print("Synthetic example only; not experimental accuracy validation.")
    print(f"Measured {len(results)} artificial paired frames.")
    print(f"Final mean growth: {results[-1]['mean_growth_mm']:.3f} mm")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
