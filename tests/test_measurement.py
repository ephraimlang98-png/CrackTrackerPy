"""Regression checks on artificial inputs, not laboratory accuracy estimates."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import shift

import crack_core as cc
from examples.synthetic_demo import MM_PER_PX, make_images, measure, prepare


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.paths = make_images(cls.folder / "images", (0.0, 0.5, 1.5))
        cls.side = prepare(cls.paths)
        cls.results = measure(cls.paths, cls.side)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_baseline_has_zero_growth(self):
        result = self.results[0]
        self.assertTrue(result.ok, result.status)
        self.assertEqual(result.da, 0.0)
        self.assertGreater(result.corr, 0.99)

    def test_known_synthetic_extensions(self):
        # 0.15 mm is a regression tolerance for this artificial construction.
        for result, expected in zip(self.results, (0.0, 0.5, 1.5)):
            with self.subTest(expected=expected):
                self.assertTrue(result.ok, result.status)
                self.assertAlmostEqual(result.da, expected, delta=0.15)

    def test_translation_does_not_become_growth(self):
        image = cc.read_gray(str(self.paths[0]))
        moved = shift(image, (-4, 7), order=0, mode="nearest")
        path = self.folder / "translated.png"
        Image.fromarray(np.rint(moved * 255).astype(np.uint8)).save(path)
        result = cc.measure_frame(str(path), self.side)
        self.assertTrue(result.ok, result.status)
        self.assertAlmostEqual(result.d[0], 7, delta=0.15)
        self.assertAlmostEqual(result.d[1], -4, delta=0.15)
        self.assertAlmostEqual(result.da, 0.0, delta=0.10)

    def test_unregistered_blank_frame_is_not_a_valid_measurement(self):
        path = self.folder / "blank.png"
        Image.fromarray(np.full((320, 640), 180, dtype=np.uint8)).save(path)
        result = cc.measure_frame(str(path), self.side)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "low registration")
        self.assertTrue(np.isnan(result.a))
        self.assertTrue(np.isnan(result.da))

    def test_projected_manual_correction_uses_translated_reference(self):
        side = dict(self.side, a_mode="proj")
        result = cc.FrameResult("synthetic.png", True, "ok", 0.0, 1.0, d=(7.0, -4.0))
        reference = np.asarray(side["ref"]) + result.d
        target = reference + side["u"] * 150 + side["v"] * 8
        corrected = cc.correct_tip(result, side, target)
        self.assertEqual(corrected.status, "manual")
        self.assertAlmostEqual(corrected.a, 150 * MM_PER_PX, places=6)
        self.assertAlmostEqual(corrected.da, corrected.a - side["a0_px"] * MM_PER_PX, places=6)


class ImageOrderingTests(unittest.TestCase):
    def test_natural_order_supported_files_and_missing_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("frame10.PNG", "frame2.bmp", "frame1.jpg", "notes.txt"):
                (folder / name).touch()
            (folder / "directory.png").mkdir()
            self.assertEqual([Path(p).name for p in cc.list_images(str(folder))],
                             ["frame1.jpg", "frame2.bmp", "frame10.PNG"])
            self.assertEqual(cc.list_images(str(folder / "absent")), [])


if __name__ == "__main__":
    unittest.main()
