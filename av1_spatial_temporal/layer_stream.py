from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

from .obu import Obu, analyze_obu_stream, iter_obus_path


MAGIC = b"A1LS\x01\r\n\x1a"
HEADER_LENGTH = struct.Struct("<I")
RECORD_HEADER = struct.Struct("<QIBBBI")
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024 * 1024
NONE_LAYER_ID = 0xFF


class LayerStreamError(ValueError):
    """Raised when an A1LS layer transport stream is invalid."""


@dataclass(frozen=True, slots=True)
class LayerRecord:
    sequence_number: int
    temporal_unit_index: int
    spatial_id: int | None
    temporal_id: int | None
    obu_type: int
    raw: bytes

    @classmethod
    def from_obu(cls, obu: Obu) -> "LayerRecord":
        return cls(
            sequence_number=obu.sequence_number,
            temporal_unit_index=obu.temporal_unit_index,
            spatial_id=obu.spatial_id,
            temporal_id=obu.temporal_id,
            obu_type=obu.obu_type,
            raw=obu.raw,
        )


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _prepare_output(path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}")


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class LayerStreamWriter:
    """Streaming writer for the small A1LS transport envelope."""

    def __init__(self, stream: BinaryIO, metadata: dict[str, object]):
        self._stream = stream
        self._last_sequence = -1
        self.record_count = 0
        self.payload_bytes = 0
        header = _canonical_json(metadata)
        if len(header) > MAX_HEADER_BYTES:
            raise LayerStreamError("A1LS JSON header is too large")
        stream.write(MAGIC)
        stream.write(HEADER_LENGTH.pack(len(header)))
        stream.write(header)

    def write(self, record: LayerRecord) -> None:
        if record.sequence_number <= self._last_sequence:
            raise LayerStreamError(
                "Records must be written in strictly increasing sequence order"
            )
        if not 0 <= record.obu_type <= 15:
            raise LayerStreamError(f"Invalid OBU type: {record.obu_type}")
        if len(record.raw) > MAX_RECORD_BYTES:
            raise LayerStreamError("A1LS record payload is too large")

        spatial_id = NONE_LAYER_ID if record.spatial_id is None else record.spatial_id
        temporal_id = NONE_LAYER_ID if record.temporal_id is None else record.temporal_id
        if spatial_id != NONE_LAYER_ID and not 0 <= spatial_id <= 3:
            raise LayerStreamError(f"Invalid spatial_id: {spatial_id}")
        if temporal_id != NONE_LAYER_ID and not 0 <= temporal_id <= 7:
            raise LayerStreamError(f"Invalid temporal_id: {temporal_id}")
        if (spatial_id == NONE_LAYER_ID) != (temporal_id == NONE_LAYER_ID):
            raise LayerStreamError("Spatial and temporal layer IDs must both be set")

        self._stream.write(
            RECORD_HEADER.pack(
                record.sequence_number,
                record.temporal_unit_index,
                spatial_id,
                temporal_id,
                record.obu_type,
                len(record.raw),
            )
        )
        self._stream.write(record.raw)
        self._last_sequence = record.sequence_number
        self.record_count += 1
        self.payload_bytes += len(record.raw)


class LayerStreamReader(Iterator[LayerRecord]):
    """Single-pass reader for an A1LS stream."""

    def __init__(self, stream: BinaryIO, *, source: str = "<stream>"):
        self._stream = stream
        self.source = source
        self._last_sequence = -1
        self.metadata = self._read_header()

    def _read_header(self) -> dict[str, object]:
        magic = self._stream.read(len(MAGIC))
        if magic != MAGIC:
            raise LayerStreamError(f"Not an A1LS v1 stream: {self.source}")
        raw_length = self._stream.read(HEADER_LENGTH.size)
        if len(raw_length) != HEADER_LENGTH.size:
            raise LayerStreamError(f"Truncated A1LS header: {self.source}")
        (length,) = HEADER_LENGTH.unpack(raw_length)
        if length > MAX_HEADER_BYTES:
            raise LayerStreamError(f"A1LS header is too large: {self.source}")
        raw_header = self._stream.read(length)
        if len(raw_header) != length:
            raise LayerStreamError(f"Truncated A1LS JSON header: {self.source}")
        try:
            metadata = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LayerStreamError(f"Invalid A1LS JSON header: {self.source}") from exc
        if not isinstance(metadata, dict):
            raise LayerStreamError(f"A1LS metadata must be an object: {self.source}")
        if metadata.get("format") != "a1ls" or metadata.get("version") != 1:
            raise LayerStreamError(f"Unsupported A1LS metadata version: {self.source}")
        return metadata

    def __iter__(self) -> "LayerStreamReader":
        return self

    def __next__(self) -> LayerRecord:
        raw_header = self._stream.read(RECORD_HEADER.size)
        if not raw_header:
            raise StopIteration
        if len(raw_header) != RECORD_HEADER.size:
            raise LayerStreamError(f"Truncated A1LS record header: {self.source}")
        sequence, temporal_unit, spatial_raw, temporal_raw, obu_type, size = (
            RECORD_HEADER.unpack(raw_header)
        )
        if sequence <= self._last_sequence:
            raise LayerStreamError(
                f"Non-increasing record sequence in {self.source}: {sequence}"
            )
        if size > MAX_RECORD_BYTES:
            raise LayerStreamError(f"A1LS record is too large in {self.source}")
        raw = self._stream.read(size)
        if len(raw) != size:
            raise LayerStreamError(f"Truncated A1LS record in {self.source}")

        if spatial_raw == NONE_LAYER_ID and temporal_raw == NONE_LAYER_ID:
            spatial_id = temporal_id = None
        elif spatial_raw == NONE_LAYER_ID or temporal_raw == NONE_LAYER_ID:
            raise LayerStreamError(f"Inconsistent layer IDs in {self.source}")
        else:
            if spatial_raw > 3 or temporal_raw > 7:
                raise LayerStreamError(f"Invalid layer IDs in {self.source}")
            spatial_id, temporal_id = spatial_raw, temporal_raw

        self._last_sequence = sequence
        return LayerRecord(
            sequence_number=sequence,
            temporal_unit_index=temporal_unit,
            spatial_id=spatial_id,
            temporal_id=temporal_id,
            obu_type=obu_type,
            raw=raw,
        )


def _is_base_obu(obu: Obu, max_spatial_id: int, max_temporal_id: int) -> bool:
    if obu.is_global:
        return True
    assert obu.spatial_id is not None and obu.temporal_id is not None
    return obu.spatial_id <= max_spatial_id and obu.temporal_id <= max_temporal_id


def split_obu_stream(
    input_path: Path,
    base_path: Path,
    enhancement_path: Path,
    *,
    max_base_spatial_id: int = 0,
    max_base_temporal_id: int = 0,
    force: bool = False,
) -> dict[str, object]:
    """Partition an OBU stream into ordered base and enhancement channels."""

    input_path = input_path.resolve()
    base_path = base_path.resolve()
    enhancement_path = enhancement_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input OBU stream does not exist: {input_path}")
    if base_path == enhancement_path or input_path in (base_path, enhancement_path):
        raise ValueError("Input, base, and enhancement paths must be distinct")
    if not 0 <= max_base_spatial_id <= 3:
        raise ValueError("max_base_spatial_id must be in [0, 3]")
    if not 0 <= max_base_temporal_id <= 7:
        raise ValueError("max_base_temporal_id must be in [0, 7]")

    _prepare_output(base_path, force=force)
    _prepare_output(enhancement_path, force=force)
    analysis = analyze_obu_stream(input_path)
    common_metadata: dict[str, object] = {
        "format": "a1ls",
        "version": 1,
        "source_name": input_path.name,
        "stream_id": analysis["sha256"],
        "full_sha256": analysis["sha256"],
        "full_size_bytes": analysis["size_bytes"],
        "full_record_count": analysis["obu_count"],
        "base_max_spatial_id": max_base_spatial_id,
        "base_max_temporal_id": max_base_temporal_id,
    }
    base_temp = _temporary_sibling(base_path)
    enhancement_temp = _temporary_sibling(enhancement_path)
    started = time.perf_counter()

    try:
        with base_temp.open("wb") as base_file, enhancement_temp.open("wb") as enh_file:
            base_writer = LayerStreamWriter(
                base_file, {**common_metadata, "channel": "base"}
            )
            enhancement_writer = LayerStreamWriter(
                enh_file, {**common_metadata, "channel": "enhancement"}
            )
            for obu in iter_obus_path(input_path):
                record = LayerRecord.from_obu(obu)
                if _is_base_obu(obu, max_base_spatial_id, max_base_temporal_id):
                    base_writer.write(record)
                else:
                    enhancement_writer.write(record)

        os.replace(base_temp, base_path)
        os.replace(enhancement_temp, enhancement_path)
    except Exception:
        _safe_unlink(base_temp)
        _safe_unlink(enhancement_temp)
        raise

    elapsed = time.perf_counter() - started
    return {
        "input": str(input_path),
        "base": str(base_path),
        "enhancement": str(enhancement_path),
        "base_max_spatial_id": max_base_spatial_id,
        "base_max_temporal_id": max_base_temporal_id,
        "base_records": base_writer.record_count,
        "enhancement_records": enhancement_writer.record_count,
        "base_payload_bytes": base_writer.payload_bytes,
        "enhancement_payload_bytes": enhancement_writer.payload_bytes,
        "full_sha256": analysis["sha256"],
        "elapsed_seconds": elapsed,
    }


class OrderedLayerMerger:
    """Incrementally reorder records arriving from independent channels."""

    def __init__(self, *, first_sequence: int = 0, max_pending: int = 4096):
        if first_sequence < 0:
            raise ValueError("first_sequence must be non-negative")
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.expected_sequence = first_sequence
        self.max_pending = max_pending
        self._pending: dict[int, LayerRecord] = {}

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def push(self, record: LayerRecord) -> list[LayerRecord]:
        sequence = record.sequence_number
        if sequence < self.expected_sequence or sequence in self._pending:
            raise LayerStreamError(f"Duplicate or late record: {sequence}")
        self._pending[sequence] = record
        if len(self._pending) > self.max_pending:
            self._pending.pop(sequence, None)
            raise LayerStreamError(
                f"Pending merge window exceeded {self.max_pending} records"
            )

        ready: list[LayerRecord] = []
        while self.expected_sequence in self._pending:
            ready.append(self._pending.pop(self.expected_sequence))
            self.expected_sequence += 1
        return ready

    def finish(self, *, expected_count: int | None = None) -> None:
        if self._pending:
            first_missing = self.expected_sequence
            raise LayerStreamError(
                f"Cannot finish merge: sequence {first_missing} is missing"
            )
        if expected_count is not None and self.expected_sequence != expected_count:
            raise LayerStreamError(
                f"Merged {self.expected_sequence} records; expected {expected_count}"
            )


def merge_layer_streams(
    input_paths: Iterable[Path],
    output_path: Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    """K-way merge A1LS channels and reconstruct the original OBU stream."""

    paths = [path.resolve() for path in input_paths]
    output_path = output_path.resolve()
    if not paths:
        raise ValueError("At least one A1LS input is required")
    if len(set(paths)) != len(paths):
        raise ValueError("A1LS input paths must be unique")
    if output_path in paths:
        raise ValueError("Output path must differ from every input path")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"A1LS input does not exist: {path}")
    _prepare_output(output_path, force=force)
    temp_path = _temporary_sibling(output_path)
    started = time.perf_counter()
    digest = hashlib.sha256()
    output_size = 0
    output_records = 0

    try:
        with ExitStack() as stack:
            files = [stack.enter_context(path.open("rb")) for path in paths]
            readers = [
                LayerStreamReader(stream, source=str(path))
                for stream, path in zip(files, paths, strict=True)
            ]
            stream_ids = {reader.metadata.get("stream_id") for reader in readers}
            if len(stream_ids) != 1 or None in stream_ids:
                raise LayerStreamError("A1LS inputs do not describe the same stream")
            expected_hashes = {
                reader.metadata.get("full_sha256") for reader in readers
            }
            expected_sizes = {
                reader.metadata.get("full_size_bytes") for reader in readers
            }
            expected_counts = {
                reader.metadata.get("full_record_count") for reader in readers
            }
            if len(expected_hashes) != 1 or len(expected_sizes) != 1 or len(expected_counts) != 1:
                raise LayerStreamError("A1LS inputs disagree about the source stream")

            import heapq

            heap: list[tuple[int, int, LayerRecord]] = []
            for reader_index, reader in enumerate(readers):
                try:
                    record = next(reader)
                except StopIteration:
                    continue
                heapq.heappush(
                    heap, (record.sequence_number, reader_index, record)
                )

            expected_sequence = 0
            with temp_path.open("wb") as output:
                while heap:
                    sequence, reader_index, record = heapq.heappop(heap)
                    if sequence != expected_sequence:
                        if sequence < expected_sequence:
                            problem = f"duplicate sequence {sequence}"
                        else:
                            problem = f"missing sequence {expected_sequence}"
                        raise LayerStreamError(f"Cannot merge A1LS inputs: {problem}")
                    output.write(record.raw)
                    digest.update(record.raw)
                    output_size += len(record.raw)
                    output_records += 1
                    expected_sequence += 1
                    try:
                        next_record = next(readers[reader_index])
                    except StopIteration:
                        continue
                    heapq.heappush(
                        heap,
                        (next_record.sequence_number, reader_index, next_record),
                    )

            expected_hash = expected_hashes.pop()
            expected_size = expected_sizes.pop()
            expected_count = expected_counts.pop()
            if output_records != expected_count:
                raise LayerStreamError(
                    f"Merged {output_records} records; expected {expected_count}"
                )
            if output_size != expected_size:
                raise LayerStreamError(
                    f"Merged {output_size} bytes; expected {expected_size}"
                )
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                raise LayerStreamError(
                    f"Merged SHA-256 {actual_hash} does not match {expected_hash}"
                )

        os.replace(temp_path, output_path)
    except Exception:
        _safe_unlink(temp_path)
        raise

    elapsed = time.perf_counter() - started
    mib_per_second = (
        output_size / (1024 * 1024) / elapsed if elapsed > 0 else float("inf")
    )
    return {
        "inputs": [str(path) for path in paths],
        "output": str(output_path),
        "record_count": output_records,
        "size_bytes": output_size,
        "sha256": digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "throughput_mib_per_second": mib_per_second,
        "byte_identical": True,
    }

