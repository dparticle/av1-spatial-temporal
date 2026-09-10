from __future__ import annotations

import unittest

from av1_spatial_temporal.encoder import allocate_bitrates, layering_mode, parse_fraction


class EncoderConfigurationTests(unittest.TestCase):
    def test_l2t3_mode_and_default_bitrates(self) -> None:
        self.assertEqual(layering_mode(2, 3), 7)
        self.assertEqual(
            allocate_bitrates(3000, 2, 3),
            (150, 300, 600, 600, 1200, 2400),
        )

    def test_fraction_parser(self) -> None:
        self.assertEqual(str(parse_fraction("30000/1001")), "30000/1001")
        with self.assertRaises(ValueError):
            parse_fraction("0")

    def test_unsupported_l2t2_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported layer combination"):
            layering_mode(2, 2)


if __name__ == "__main__":
    unittest.main()

