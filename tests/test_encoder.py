from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from av1_spatial_temporal.encoder import (
    EncodeConfig, VideoInfo, allocate_bitrates, layering_mode, parse_fraction,
)
from av1_spatial_temporal.encoder_legacy import (
    EncoderError, source_bitrate_kbps, target_bitrate_kbps,
)


class EncoderConfigurationTests(unittest.TestCase):
    def test_l2t3_mode_and_default_bitrates(self) -> None:
        self.assertEqual(layering_mode(2, 3), 7)
        self.assertEqual(
            allocate_bitrates(3000, 2, 3),
            (300, 450, 600, 1200, 1800, 2400),
        )

    def test_auto_bitrate_and_speed_defaults(self) -> None:
        self.assertIsNone(EncodeConfig().bitrate_kbps)
        self.assertEqual(EncodeConfig().speed, 6)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "sample.mkv"
            source.write_bytes(b"x" * 6000)
            info = VideoInfo(16, 16, 30, 1, "yuv420p", 6.0, 180)
            self.assertEqual(source_bitrate_kbps(source, info), 8)
            self.assertEqual(target_bitrate_kbps(None, source, info), 9)
            self.assertEqual(target_bitrate_kbps(12, source, info), 12)
            missing_duration = VideoInfo(16, 16, 30, 1, "yuv420p", None, 180)
            with self.assertRaisesRegex(EncoderError, "pass --bitrate-kbps"):
                target_bitrate_kbps(None, source, missing_duration)

    def test_fraction_parser(self) -> None:
        self.assertEqual(str(parse_fraction("30000/1001")), "30000/1001")
        with self.assertRaises(ValueError):
            parse_fraction("0")

    def test_unsupported_l2t2_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported layer combination"):
            layering_mode(2, 2)


if __name__ == "__main__":
    unittest.main()

