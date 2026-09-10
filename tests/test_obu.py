from __future__ import annotations

import io
import unittest

from av1_spatial_temporal.obu import (
    OBU_FRAME,
    OBU_SEQUENCE_HEADER,
    OBU_TEMPORAL_DELIMITER,
    ObuParseError,
    encode_leb128,
    iter_obus,
    make_obu,
)


class ObuParserTests(unittest.TestCase):
    def test_leb128_boundaries(self) -> None:
        self.assertEqual(encode_leb128(0), b"\x00")
        self.assertEqual(encode_leb128(127), b"\x7f")
        self.assertEqual(encode_leb128(128), b"\x80\x01")
        self.assertEqual(encode_leb128(16384), b"\x80\x80\x01")

    def test_round_trip_and_layer_ids(self) -> None:
        raw = b"".join(
            (
                make_obu(OBU_TEMPORAL_DELIMITER),
                make_obu(OBU_SEQUENCE_HEADER, b"sequence"),
                make_obu(OBU_FRAME, b"base", spatial_id=0, temporal_id=0),
                make_obu(OBU_FRAME, b"enh", spatial_id=1, temporal_id=2),
                make_obu(OBU_TEMPORAL_DELIMITER),
                make_obu(OBU_FRAME, b"next", spatial_id=0, temporal_id=1),
            )
        )
        parsed = list(iter_obus(io.BytesIO(raw)))
        self.assertEqual(b"".join(obu.raw for obu in parsed), raw)
        self.assertEqual([obu.sequence_number for obu in parsed], list(range(6)))
        self.assertEqual([obu.temporal_unit_index for obu in parsed], [0, 0, 0, 0, 1, 1])
        self.assertEqual(parsed[2].layer_key, "s0_t0")
        self.assertEqual(parsed[3].layer_key, "s1_t2")
        self.assertEqual(parsed[1].layer_key, "global")

    def test_rejects_obu_without_size_field(self) -> None:
        header_without_size = bytes([OBU_FRAME << 3])
        with self.assertRaises(ObuParseError):
            list(iter_obus(io.BytesIO(header_without_size)))


if __name__ == "__main__":
    unittest.main()

