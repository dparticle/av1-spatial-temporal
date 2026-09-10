from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from av1_spatial_temporal.layer_stream import (
    LayerRecord,
    OrderedLayerMerger,
    merge_layer_streams,
    split_obu_stream,
)
from av1_spatial_temporal.obu import (
    OBU_FRAME,
    OBU_SEQUENCE_HEADER,
    OBU_TEMPORAL_DELIMITER,
    make_obu,
)
from av1_spatial_temporal.operations_v1 import sha256_file, unpack_layer_stream


def synthetic_stream() -> bytes:
    return b"".join(
        (
            make_obu(OBU_TEMPORAL_DELIMITER),
            make_obu(OBU_SEQUENCE_HEADER, b"header"),
            make_obu(OBU_FRAME, b"b0", spatial_id=0, temporal_id=0),
            make_obu(OBU_FRAME, b"e0", spatial_id=1, temporal_id=0),
            make_obu(OBU_TEMPORAL_DELIMITER),
            make_obu(OBU_FRAME, b"e1", spatial_id=0, temporal_id=2),
            make_obu(OBU_FRAME, b"e2", spatial_id=1, temporal_id=2),
            make_obu(OBU_TEMPORAL_DELIMITER),
            make_obu(OBU_FRAME, b"b1", spatial_id=0, temporal_id=0),
        )
    )


class LayerStreamTests(unittest.TestCase):
    def test_split_and_merge_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.obu"
            base = root / "base.a1ls"
            enhancement = root / "enhancement.a1ls"
            reconstructed = root / "reconstructed.obu"
            base_obu = root / "base.obu"
            source.write_bytes(synthetic_stream())

            split_report = split_obu_stream(source, base, enhancement)
            merge_report = merge_layer_streams((base, enhancement), reconstructed)
            unpack_report = unpack_layer_stream(base, base_obu)

            self.assertEqual(source.read_bytes(), reconstructed.read_bytes())
            self.assertEqual(sha256_file(source), sha256_file(reconstructed))
            self.assertEqual(split_report["base_records"], 6)
            self.assertEqual(split_report["enhancement_records"], 3)
            self.assertTrue(merge_report["byte_identical"])
            self.assertEqual(unpack_report["channel"], "base")
            self.assertLess(base_obu.stat().st_size, source.stat().st_size)

    def test_incremental_merger_releases_contiguous_records(self) -> None:
        record_zero = LayerRecord(0, 0, None, None, OBU_TEMPORAL_DELIMITER, b"zero")
        record_one = LayerRecord(1, 0, 0, 0, OBU_FRAME, b"one")
        record_two = LayerRecord(2, 0, 1, 2, OBU_FRAME, b"two")
        merger = OrderedLayerMerger(max_pending=4)
        self.assertEqual(merger.push(record_two), [])
        self.assertEqual(merger.push(record_zero), [record_zero])
        self.assertEqual(merger.push(record_one), [record_one, record_two])
        merger.finish(expected_count=3)


if __name__ == "__main__":
    unittest.main()
