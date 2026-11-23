"""Isolated offscreen startup and side-aggregation checks (no hardware access)."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "Install requirements.txt for desktop checks")
class DesktopTests(unittest.TestCase):
    def run_check(self, ct_mode: bool):
        script = textwrap.dedent('''
            import sys, tempfile
            from pathlib import Path
            from unittest.mock import patch
            import numpy as np
            from PySide6.QtCore import QSettings as RealSettings
            import app
            import crack_core as cc

            with tempfile.TemporaryDirectory() as directory:
                def settings(*args):
                    return RealSettings(str(Path(directory) / "settings.ini"), RealSettings.IniFormat)
                with patch.object(app, "QSettings", settings):
                    application = app.make_app([])
                    if sys.argv[1] == "ct":
                        import ct_app
                        window = ct_app.CTMainWindow()
                        assert window.loads["stages"] == []
                        assert window.ld_W0.value() == 50.0
                        assert window.ld_B0.value() == 10.0
                        window._add_stage()
                        assert window.loads["stages"][0][2:] == [0.0, 0.0]
                    else:
                        window = app.MainWindow()
                    application.processEvents()
                    assert window.worker is None
                    window.set_use_b(True)
                    a = cc.FrameResult("A.png", True, "ok", 0.0, 1.0, a=10.0)
                    b = cc.FrameResult("B.png", True, "ok", 0.0, 1.0, a=14.0)
                    window.R["res"] = [[a, b]]
                    assert window._row_values(0)[2] == 12.0
                    window.R["res"] = [[a, None]]
                    assert window._row_values(0)[2] == 10.0
                    window.R["res"] = [[None, None]]
                    assert np.isnan(window._row_values(0)[2])
                    window.ui_timer.stop()
                    window.close()
                    application.processEvents()
            print("offscreen desktop check passed")
        ''')
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", script, "ct" if ct_mode else "general"],
                                cwd=Path(__file__).resolve().parents[1], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("offscreen desktop check passed", result.stdout)

    def test_general_startup_and_side_aggregation(self):
        self.run_check(False)

    @unittest.skipUnless(sys.platform == "win32", "C(T) desktop uses Windows winsound")
    def test_ct_startup_and_side_aggregation(self):
        self.run_check(True)


if __name__ == "__main__":
    unittest.main()
