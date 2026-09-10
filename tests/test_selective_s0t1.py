from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from av1_spatial_temporal.layer_stream import (
    LayerStreamError, LayerStreamWriter, merge_layer_streams, split_obu_stream,
)
from av1_spatial_temporal.obu import (
    OBU_FRAME, OBU_SEQUENCE_HEADER, OBU_TEMPORAL_DELIMITER, make_obu,
)
from av1_spatial_temporal.operations import extract_operating_point
from av1_spatial_temporal.selective_transport import (
    extract_enhancement_layers, inspect_layer_stream, merge_operating_point,
    plan_layer_change, read_layer_stream, required_layers,
)


class SelectiveLayerChangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.full = self.root / "full.obu"
        parts = []
        for frame, tid in enumerate((0, 2, 1, 2, 0, 2, 1, 2)):
            parts.append(make_obu(OBU_TEMPORAL_DELIMITER))
            if frame == 0:
                parts.append(make_obu(OBU_SEQUENCE_HEADER, b"sequence"))
            for sid in (0, 1):
                parts.append(make_obu(
                    OBU_FRAME, f"frame{frame}s{sid}".encode(),
                    spatial_id=sid, temporal_id=tid,
                ))
        self.full.write_bytes(b"".join(parts))
        self.base, self.enh = self.root / "base.a1ls", self.root / "enhancement.a1ls"
        split_obu_stream(self.full, self.base, self.enh)
        self.base_receipt = inspect_layer_stream(self.base)
        self.paths, self.receipts = {}, {}
        for layer in ((0, 1), (1, 0), (1, 1)):
            path = self.root / f"s{layer[0]}t{layer[1]}.a1ls"
            self.receipts[layer] = extract_enhancement_layers(self.enh, path, layers=[layer])
            self.paths[layer] = path

    def merge(self, target, layers, name="merged.obu"):
        output = self.root / name
        merge_operating_point(
            self.base, [self.paths[k] for k in layers], output,
            target=target, base_receipt=self.base_receipt,
            enhancement_receipts=[self.receipts[k] for k in layers],
        )
        return output

    def assert_reference(self, output, target):
        reference = self.root / f"reference_{output.stem}_{target}.obu"
        extract_operating_point(
            self.full, reference, max_spatial_id=int(target[1]), max_temporal_id=int(target[3])
        )
        self.assertEqual(output.read_bytes(), reference.read_bytes())

    def test_complete_cycle_reuses_cached_layers(self):
        current = "S0T0"
        active = set()
        additions = []
        drops = []
        for i, target in enumerate(("S0T1", "S1T1", "S1T0", "S0T1")):
            plan = plan_layer_change(current, target)
            additions.append(plan["add"])
            drops.append(plan["drop"])
            active.update(map(tuple, plan["add"]))
            active.difference_update(map(tuple, plan["drop"]))
            self.assert_reference(self.merge(target, sorted(active), f"step{i}.obu"), target)
            current = target
        self.assertEqual(additions, [[[0, 1]], [[1, 0], [1, 1]], [], [[0, 1]]])
        self.assertEqual(drops, [[], [], [[0, 1], [1, 1]], [[1, 0]]])
        self.assertEqual(active, {(0, 1)})
        self.assertEqual((self.root / "step0.obu").read_bytes(),
                         (self.root / "step3.obu").read_bytes())
        self.assertEqual(inspect_layer_stream(self.base), self.base_receipt)

    def test_exact_records_bytes_and_original_sequence_preserved(self):
        _, original = read_layer_stream(self.enh)
        for layer, path in self.paths.items():
            _, selected = read_layer_stream(path)
            self.assertEqual(selected, [r for r in original if (r.spatial_id, r.temporal_id) == layer])
            self.assertEqual(len(selected), 2)
            info = self.receipts[layer]
            self.assertEqual(info["record_header_bytes"], 19 * len(selected))
            self.assertEqual(info["file_bytes"],
                             info["file_header_bytes"] + info["record_header_bytes"] + info["obu_bytes"])

    def test_extract_multiple_layers_then_refilter_cached_bundle(self):
        bundle = self.root / "bundle.a1ls"
        receipt = extract_enhancement_layers(self.enh, bundle, layers=[(1, 0), (1, 1)])
        only_t0 = self.root / "t0_from_bundle.a1ls"
        extract_enhancement_layers(bundle, only_t0, layers=[(1, 0)])
        self.assertEqual(only_t0.read_bytes(), self.paths[(1, 0)].read_bytes())
        output = self.root / "bundled_s1t1.obu"
        merge_operating_point(
            self.base, [self.paths[(0, 1)], bundle], output, target="S1T1",
            base_receipt=self.base_receipt,
            enhancement_receipts=[self.receipts[(0, 1)], receipt],
        )
        self.assert_reference(output, "S1T1")

    def test_merge_filters_extra_cached_layers_for_downgrade(self):
        output = self.merge("S1T0", [(0, 1), (1, 0), (1, 1)])
        self.assert_reference(output, "S1T0")

    def test_all_l2t3_targets_from_full_enhancement(self):
        receipt = inspect_layer_stream(self.enh)
        for s in range(2):
            for t in range(3):
                target = f"S{s}T{t}"
                with self.subTest(target=target):
                    output = self.root / f"{target}.obu"
                    merge_operating_point(
                        self.base, [self.enh], output, target=target,
                        base_receipt=self.base_receipt, enhancement_receipts=[receipt],
                    )
                    self.assert_reference(output, target)

    def test_base_only_needs_no_enhancement(self):
        self.assert_reference(self.merge("S0T0", []), "S0T0")

    def test_missing_required_entire_layer_rejected(self):
        with self.assertRaisesRegex(LayerStreamError, "Missing target layers"):
            self.merge("S1T1", [(0, 1), (1, 1)])
        self.assertFalse((self.root / "merged.obu").exists())

    def test_missing_whole_record_rejected(self):
        path = self.paths[(0, 1)]
        metadata, records = read_layer_stream(path)
        missing = self.root / "missing.a1ls"
        with missing.open("xb") as stream:
            writer = LayerStreamWriter(stream, metadata)
            for record in records[:-1]:
                writer.write(record)
        self.paths[(0, 1)] = missing
        with self.assertRaisesRegex(LayerStreamError, "Incomplete file"):
            self.merge("S0T1", [(0, 1)])
        self.assertFalse((self.root / "merged.obu").exists())

    def test_corrupt_delta_and_base_rejected(self):
        for source, label in ((self.paths[(0, 1)], "delta"), (self.base, "base")):
            with self.subTest(label=label):
                data = bytearray(source.read_bytes())
                data[-1] ^= 1
                bad = self.root / f"bad_{label}.a1ls"
                bad.write_bytes(data)
                with self.assertRaisesRegex(LayerStreamError, "SHA-256"):
                    merge_operating_point(
                        bad if label == "base" else self.base,
                        [bad if label == "delta" else self.paths[(0, 1)]],
                        self.root / "bad.obu", target="S0T1",
                        base_receipt=self.base_receipt,
                        enhancement_receipts=[self.receipts[(0, 1)]],
                    )

    def test_different_encoded_chunk_rejected(self):
        other = self.root / "other.obu"
        other.write_bytes(self.full.read_bytes().replace(b"sequence", b"new_head"))
        other_base, other_enh = self.root / "other_base.a1ls", self.root / "other_enh.a1ls"
        split_obu_stream(other, other_base, other_enh)
        selected = self.root / "other_selected.a1ls"
        receipt = extract_enhancement_layers(other_enh, selected, layers=[(0, 1)])
        with self.assertRaisesRegex(LayerStreamError, "different encoded chunks"):
            merge_operating_point(
                self.base, [selected], self.root / "bad.obu", target="S0T1",
                base_receipt=self.base_receipt, enhancement_receipts=[receipt],
            )

    def test_duplicate_record_from_overlapping_inputs_rejected(self):
        with self.assertRaisesRegex(LayerStreamError, "Duplicate original sequence"):
            merge_operating_point(
                self.base, [self.enh, self.paths[(0, 1)]], self.root / "bad.obu",
                target="S0T1", base_receipt=self.base_receipt,
                enhancement_receipts=[inspect_layer_stream(self.enh), self.receipts[(0, 1)]],
            )

    def test_transport_layer_must_match_obu_layer(self):
        metadata, records = read_layer_stream(self.paths[(0, 1)])
        bad = self.root / "mismatched_layer.a1ls"
        with bad.open("xb") as stream:
            writer = LayerStreamWriter(stream, metadata)
            for r in records:
                writer.write(replace(r, spatial_id=1))
        with self.assertRaisesRegex(LayerStreamError, "does not match"):
            inspect_layer_stream(bad)

    def test_existing_output_not_overwritten(self):
        path = self.paths[(0, 1)]
        old = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.merge("S0T1", [(0, 1)], path.name)
        self.assertEqual(path.read_bytes(), old)

    def test_original_merger_still_requires_contiguous_full_stream(self):
        with self.assertRaisesRegex(LayerStreamError, "missing sequence"):
            merge_layer_streams((self.base, self.paths[(0, 1)]), self.root / "strict.obu")
        self.assertFalse((self.root / "strict.obu").exists())

    def test_invalid_and_unavailable_layer_requests(self):
        for layers in ([], [(0, 0)], [(4, 0)]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                extract_enhancement_layers(self.enh, self.root / "bad.a1ls", layers=layers)
        with self.assertRaisesRegex(LayerStreamError, "absent"):
            extract_enhancement_layers(self.enh, self.root / "bad.a1ls", layers=[(2, 0)])

    def test_plans_cover_all_l2t3_transitions_and_l3t3(self):
        points = [f"S{s}T{t}" for s in range(2) for t in range(3)]
        for current in points:
            for target in points:
                plan = plan_layer_change(current, target)
                have = required_layers(current)
                have.update(map(tuple, plan["add"]))
                have.difference_update(map(tuple, plan["drop"]))
                self.assertEqual(have, required_layers(target))
                self.assertNotIn([0, 0], plan["drop"])
        plan = plan_layer_change("S1T1", "S2T2", spatial_layers=3)
        self.assertEqual(plan["add"], [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]])
        with self.assertRaises(ValueError):
            plan_layer_change("S1T1", "S2T2")


if __name__ == "__main__":
    unittest.main()
