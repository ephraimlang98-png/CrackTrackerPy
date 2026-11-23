from __future__ import annotations

import io
import unittest

import numpy as np

import ct_core as ct
import fc_proto as protocol


class ProtocolTests(unittest.TestCase):
    class FragmentedInput:
        """A deterministic recv-compatible input; no network permissions needed."""
        def __init__(self, data):
            self.data = io.BytesIO(data)

        def recv(self, size):
            return self.data.read(min(size, 3))

    def test_fragmented_message_and_binary_payload(self):
        message = {"type": "capture", "name": "synthetic"}
        payload = b"\x00\x01\xffsynthetic-image-bytes"
        packed = protocol.pack(message, payload)
        self.assertEqual(protocol.recv(self.FragmentedInput(packed)), (message, payload))

    def test_oversized_header_is_rejected_before_reading_payload(self):
        header = protocol.HDR.pack(protocol.MAX_JSON + 1, 0)
        with self.assertRaisesRegex(ConnectionError, "bad frame header"):
            protocol.recv(self.FragmentedInput(header))

    def test_truncated_message_is_rejected(self):
        packed = protocol.pack({"type": "preview"}, b"image")
        with self.assertRaisesRegex(ConnectionError, "connection closed"):
            protocol.recv(self.FragmentedInput(packed[:-2]))


class CTCalculationTests(unittest.TestCase):
    def test_known_load_intensity_value_and_units(self):
        # Independent reference evaluation: alpha=0.5, f(alpha)~9.6591.
        # 1000 N, B=10 mm, W=50 mm -> K~4.319 MPa sqrt(m).
        self.assertAlmostEqual(float(ct.K_from_P(1000, 25, 10, 50)), 4.319, delta=0.005)
        self.assertAlmostEqual(float(ct.K_from_P(2000, 25, 10, 50)), 8.638, delta=0.01)

    def test_point_selection_preserves_order_and_reports_large_gap(self):
        selected, gaps = ct.select_points([np.nan, 15.0, 15.1, 15.3, 17.0], W=50.0, min_inc=0.25)
        self.assertEqual(np.flatnonzero(selected).tolist(), [1, 3, 4])
        self.assertTrue(any(start == 3 and end == 4 for start, end, _, _ in gaps))

    def test_laboratory_load_schedule_is_not_distributed(self):
        self.assertEqual(ct.DEFAULT_STAGES, [])


if __name__ == "__main__":
    unittest.main()
