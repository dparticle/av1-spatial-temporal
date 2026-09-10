"""Reusable complete-chunk A1LS selection and merge, without codec dependencies.

Targets are cumulative rectangles (s <= S, t <= T), as in this project's
libaom SVC modes. Base is global OBUs plus S0T0. All inputs must be from the same
encoded chunk. Functions read a bounded chunk into memory; they do not implement
network packet reordering, arbitrary mid-stream switching, or reference recovery.
File receipts are trusted completeness metadata delivered with each transfer.
"""
from __future__ import annotations

import hashlib
import heapq
import io
import re
from pathlib import Path
from typing import Iterable

from .layer_stream import (
    LayerRecord, LayerStreamError, LayerStreamReader, LayerStreamWriter, RECORD_HEADER,
)
from .obu import iter_obus

Layer = tuple[int, int]
SOURCE_FIELDS = (
    "stream_id", "full_sha256", "full_size_bytes", "full_record_count",
    "base_max_spatial_id", "base_max_temporal_id",
)


def parse_point(point: str) -> Layer:
    match = re.fullmatch(r"S([0-3])T([0-7])", point.strip().upper())
    if match is None:
        raise ValueError(f"Invalid AV1 operating point: {point!r}")
    return tuple(map(int, match.groups()))


def _layers(values: Iterable[Layer]) -> set[Layer]:
    result = set()
    for value in values:
        if len(value) != 2:
            raise ValueError(f"Invalid exact layer: {value!r}")
        s, t = value
        if type(s) is not int or type(t) is not int or not (0 <= s <= 3 and 0 <= t <= 7):
            raise ValueError(f"Invalid exact layer: {value!r}")
        result.add((s, t))
    return result


def required_layers(point: str) -> set[Layer]:
    s_max, t_max = parse_point(point)
    return {(s, t) for s in range(s_max + 1) for t in range(t_max + 1)}


def plan_layer_change(
    current: str, target: str, *, spatial_layers: int = 2, temporal_layers: int = 3,
) -> dict:
    """Return exact layer IDs to keep/add/drop; current must be fully cached."""
    if not 1 <= spatial_layers <= 4 or not 1 <= temporal_layers <= 8:
        raise ValueError("Invalid layer configuration")
    for value in (current, target):
        s, t = parse_point(value)
        if s >= spatial_layers or t >= temporal_layers:
            raise ValueError(f"{value} exceeds L{spatial_layers}T{temporal_layers}")
    old, new = required_layers(current), required_layers(target)
    return {
        "current": current.strip().upper(), "target": target.strip().upper(),
        "keep": [list(v) for v in sorted(old & new)],
        "add": [list(v) for v in sorted(new - old)],
        "drop": [list(v) for v in sorted(old - new)],
    }


def read_layer_stream(path: Path) -> tuple[dict, list[LayerRecord]]:
    """Read and validate transport headers against the underlying OBU headers."""
    with Path(path).open("rb") as stream:
        reader = LayerStreamReader(stream, source=str(path))
        metadata, records = reader.metadata, list(reader)
    if any(metadata.get(key) is None for key in SOURCE_FIELDS):
        raise LayerStreamError("Missing source identity")
    if (metadata["base_max_spatial_id"], metadata["base_max_temporal_id"]) != (0, 0):
        raise LayerStreamError("This API requires a S0T0 base")
    previous_tu = -1
    for record in records:
        obus = list(iter_obus(io.BytesIO(record.raw)))
        if len(obus) != 1 or (
            obus[0].spatial_id, obus[0].temporal_id, obus[0].obu_type
        ) != (record.spatial_id, record.temporal_id, record.obu_type):
            raise LayerStreamError("A1LS layer/type does not match its OBU payload")
        if (record.sequence_number >= metadata["full_record_count"] or
                record.temporal_unit_index < previous_tu):
            raise LayerStreamError("Invalid original sequence or temporal-unit order")
        previous_tu = record.temporal_unit_index
    return metadata, records


def _sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_layer_stream(path: Path) -> dict:
    """Return a JSON-serializable receipt with exact file/OBU/envelope sizes."""
    path = Path(path)
    metadata, records = read_layer_stream(path)
    source = {key: metadata[key] for key in SOURCE_FIELDS}
    counts = {}
    for r in records:
        key = "global" if r.spatial_id is None else f"S{r.spatial_id}T{r.temporal_id}"
        entry = counts.setdefault(key, {"records": 0, "obu_bytes": 0})
        entry["records"] += 1
        entry["obu_bytes"] += len(r.raw)
    size = path.stat().st_size
    obu_bytes = sum(len(r.raw) for r in records)
    return {
        "format": "a1ls-file-receipt-v1", "source": source,
        "channel": metadata.get("channel"),
        "records": len(records), "file_bytes": size, "obu_bytes": obu_bytes,
        "record_header_bytes": len(records) * RECORD_HEADER.size,
        "file_header_bytes": size - obu_bytes - len(records) * RECORD_HEADER.size,
        "sha256": _sha256(path), "layers": counts,
    }


def extract_enhancement_layers(
    input_path: Path, output_path: Path, *, layers: Iterable[Layer],
) -> dict:
    """Copy only requested exact enhancement layers; preserve original records.

    Works on the original enhancement stream OR an earlier selected file.
    Requesting absent layers is an error (this is a complete-chunk API).
    Returns the receipt that accompanies the resulting A1LS file.
    """
    requested = _layers(layers)
    if not requested or (0, 0) in requested:
        raise ValueError("Select nonempty enhancement layers, excluding S0T0")
    metadata, records = read_layer_stream(input_path)
    if metadata.get("channel") not in ("enhancement", "enhancement_selection"):
        raise LayerStreamError("Expected an enhancement channel")
    if any(r.spatial_id is None or (r.spatial_id, r.temporal_id) == (0, 0) for r in records):
        raise LayerStreamError("Enhancement contains global or base records")
    present = {(r.spatial_id, r.temporal_id) for r in records}
    if not requested <= present:
        raise LayerStreamError(f"Requested enhancement layers are absent: {requested - present}")
    selected = [r for r in records if (r.spatial_id, r.temporal_id) in requested]
    out_metadata = {
        **metadata, "channel": "enhancement_selection",
        "selected_layers": [list(v) for v in sorted(requested)],
    }
    with Path(output_path).open("xb") as stream:
        writer = LayerStreamWriter(stream, out_metadata)
        for record in selected:
            writer.write(record)
    return inspect_layer_stream(output_path)


def _verified(path: Path, receipt: dict) -> tuple[dict, list[LayerRecord]]:
    if (receipt.get("format") != "a1ls-file-receipt-v1" or
            Path(path).stat().st_size != receipt.get("file_bytes") or
            _sha256(path) != receipt.get("sha256")):
        raise LayerStreamError(f"Incomplete file or SHA-256 mismatch: {path}")
    if inspect_layer_stream(path) != receipt:
        raise LayerStreamError(f"Receipt content does not match: {path}")
    return read_layer_stream(path)


def merge_operating_point(
    base_path: Path,
    enhancement_paths: Iterable[Path],
    output_path: Path,
    *,
    target: str,
    base_receipt: dict,
    enhancement_receipts: Iterable[dict],
) -> dict:
    """Validate completed files and output the requested cumulative OBU stream.

    Pass one receipt per enhancement file, in matching order. Cached files may
    contain extra layers: these are filtered from the output. Dropping cache
    files themselves is a caller decision. Missing target layers, duplicate
    records, changed files and wrong-chunk inputs are rejected before writing.
    """
    required = required_layers(target)
    paths, receipts = list(map(Path, enhancement_paths)), list(enhancement_receipts)
    if len(paths) != len(receipts):
        raise ValueError("One receipt per enhancement file is required")
    input_paths = [Path(base_path), *paths]
    if len({p.resolve() for p in input_paths}) != len(input_paths):
        raise LayerStreamError("Duplicate input files")
    base_meta, base = _verified(base_path, base_receipt)
    if base_meta.get("channel") != "base" or not base or any(
        (r.spatial_id, r.temporal_id) not in {(None, None), (0, 0)} for r in base
    ) or not any((r.spatial_id, r.temporal_id) == (0, 0) for r in base):
        raise LayerStreamError("Expected global OBUs plus S0T0 in base")
    streams = [base]
    available = {(0, 0)}
    for path, receipt in zip(paths, receipts):
        metadata, records = _verified(path, receipt)
        if any(metadata.get(key) != base_meta.get(key) for key in SOURCE_FIELDS):
            raise LayerStreamError("Inputs belong to different encoded chunks")
        if metadata.get("channel") not in ("enhancement", "enhancement_selection"):
            raise LayerStreamError("Expected enhancement input")
        if any(r.spatial_id is None or (r.spatial_id, r.temporal_id) == (0, 0) for r in records):
            raise LayerStreamError("Enhancement contains base/global records")
        available.update((r.spatial_id, r.temporal_id) for r in records)
        streams.append(records)
    if not required <= available:
        raise LayerStreamError(f"Missing target layers: {required - available}")
    ordered = list(heapq.merge(*streams, key=lambda r: r.sequence_number))
    if any(a.sequence_number >= b.sequence_number for a, b in zip(ordered, ordered[1:])):
        raise LayerStreamError("Duplicate original sequence numbers")
    selected = [
        r for r in ordered
        if r.spatial_id is None or (r.spatial_id, r.temporal_id) in required
    ]
    digest = hashlib.sha256()
    # Verify everything before creating output. Never overwrite inputs/results.
    with Path(output_path).open("xb") as stream:
        for record in selected:
            stream.write(record.raw)
            digest.update(record.raw)
    return {
        "target": target.strip().upper(), "records": len(selected),
        "obu_bytes": sum(len(r.raw) for r in selected), "sha256": digest.hexdigest(),
        "omitted_records": len(ordered) - len(selected),
    }
