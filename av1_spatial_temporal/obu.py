from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


OBU_SEQUENCE_HEADER = 1
OBU_TEMPORAL_DELIMITER = 2
OBU_FRAME_HEADER = 3
OBU_TILE_GROUP = 4
OBU_METADATA = 5
OBU_FRAME = 6
OBU_REDUNDANT_FRAME_HEADER = 7
OBU_TILE_LIST = 8
OBU_PADDING = 15

OBU_TYPE_NAMES = {
    OBU_SEQUENCE_HEADER: "sequence_header",
    OBU_TEMPORAL_DELIMITER: "temporal_delimiter",
    OBU_FRAME_HEADER: "frame_header",
    OBU_TILE_GROUP: "tile_group",
    OBU_METADATA: "metadata",
    OBU_FRAME: "frame",
    OBU_REDUNDANT_FRAME_HEADER: "redundant_frame_header",
    OBU_TILE_LIST: "tile_list",
    OBU_PADDING: "padding",
}


class ObuParseError(ValueError):
    """Raised when a low-overhead AV1 OBU stream is malformed."""


@dataclass(frozen=True, slots=True)
class Obu:
    sequence_number: int
    temporal_unit_index: int
    offset: int
    obu_type: int
    extension_flag: bool
    has_size_field: bool
    temporal_id: int | None
    spatial_id: int | None
    payload_size: int
    raw: bytes

    @property
    def type_name(self) -> str:
        return OBU_TYPE_NAMES.get(self.obu_type, f"reserved_{self.obu_type}")

    @property
    def is_global(self) -> bool:
        return not self.extension_flag

    @property
    def layer_key(self) -> str:
        if self.spatial_id is None or self.temporal_id is None:
            return "global"
        return f"s{self.spatial_id}_t{self.temporal_id}"


def encode_leb128(value: int) -> bytes:
    if value < 0:
        raise ValueError("LEB128 value must be non-negative")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        output.append(byte)
        if not value:
            return bytes(output)


def _read_exact(stream: BinaryIO, size: int, *, what: str, offset: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ObuParseError(
            f"Unexpected EOF while reading {what} at byte {offset}: "
            f"wanted {size}, got {len(data)}"
        )
    return data


def _read_leb128(stream: BinaryIO, *, offset: int) -> tuple[int, bytes]:
    value = 0
    encoded = bytearray()
    for index in range(8):
        raw = stream.read(1)
        if not raw:
            raise ObuParseError(f"Unexpected EOF in OBU size field at byte {offset}")
        byte = raw[0]
        encoded.append(byte)
        value |= (byte & 0x7F) << (7 * index)
        if not byte & 0x80:
            return value, bytes(encoded)
    raise ObuParseError(f"OBU size LEB128 exceeds 8 bytes at byte {offset}")


def iter_obus(
    stream: BinaryIO,
    *,
    max_payload_size: int = 2 * 1024 * 1024 * 1024,
) -> Iterator[Obu]:
    """Iterate a low-overhead AV1 OBU stream without loading it all in memory."""

    offset = 0
    sequence_number = 0
    temporal_unit_index = 0
    saw_temporal_delimiter = False

    while True:
        header_bytes = stream.read(1)
        if not header_bytes:
            return

        obu_offset = offset
        header = header_bytes[0]
        offset += 1

        if header & 0x80:
            raise ObuParseError(f"obu_forbidden_bit is set at byte {obu_offset}")
        if header & 0x01:
            raise ObuParseError(f"obu_reserved_1bit is set at byte {obu_offset}")

        obu_type = (header >> 3) & 0x0F
        extension_flag = bool((header >> 2) & 0x01)
        has_size_field = bool((header >> 1) & 0x01)

        extension_bytes = b""
        temporal_id: int | None = None
        spatial_id: int | None = None
        if extension_flag:
            extension_bytes = _read_exact(
                stream, 1, what="OBU extension header", offset=offset
            )
            offset += 1
            extension = extension_bytes[0]
            temporal_id = (extension >> 5) & 0x07
            spatial_id = (extension >> 3) & 0x03
            if extension & 0x07:
                raise ObuParseError(
                    f"extension_header_reserved_3bits is non-zero at byte {offset - 1}"
                )

        if not has_size_field:
            raise ObuParseError(
                "OBU without an internal size field cannot be split from a raw "
                f"low-overhead stream (byte {obu_offset})"
            )

        payload_size, size_bytes = _read_leb128(stream, offset=offset)
        offset += len(size_bytes)
        if payload_size > max_payload_size:
            raise ObuParseError(
                f"OBU payload at byte {obu_offset} is too large: {payload_size} bytes"
            )
        payload = _read_exact(
            stream, payload_size, what="OBU payload", offset=offset
        )
        offset += payload_size

        if obu_type == OBU_TEMPORAL_DELIMITER:
            if saw_temporal_delimiter:
                temporal_unit_index += 1
            else:
                saw_temporal_delimiter = True

        raw = header_bytes + extension_bytes + size_bytes + payload
        yield Obu(
            sequence_number=sequence_number,
            temporal_unit_index=temporal_unit_index,
            offset=obu_offset,
            obu_type=obu_type,
            extension_flag=extension_flag,
            has_size_field=has_size_field,
            temporal_id=temporal_id,
            spatial_id=spatial_id,
            payload_size=payload_size,
            raw=raw,
        )
        sequence_number += 1


def iter_obus_path(path: Path) -> Iterator[Obu]:
    with path.open("rb") as stream:
        yield from iter_obus(stream)


def analyze_obu_stream(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    type_counts: Counter[str] = Counter()
    type_bytes: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    layer_bytes: Counter[str] = Counter()
    record_count = 0
    total_bytes = 0
    max_spatial_id = 0
    max_temporal_id = 0
    max_temporal_unit = 0

    for obu in iter_obus_path(path):
        digest.update(obu.raw)
        raw_size = len(obu.raw)
        record_count += 1
        total_bytes += raw_size
        type_counts[obu.type_name] += 1
        type_bytes[obu.type_name] += raw_size
        layer_counts[obu.layer_key] += 1
        layer_bytes[obu.layer_key] += raw_size
        max_temporal_unit = max(max_temporal_unit, obu.temporal_unit_index)
        if obu.spatial_id is not None:
            max_spatial_id = max(max_spatial_id, obu.spatial_id)
        if obu.temporal_id is not None:
            max_temporal_id = max(max_temporal_id, obu.temporal_id)

    if total_bytes != path.stat().st_size:
        raise ObuParseError(
            f"Parsed {total_bytes} bytes but file contains {path.stat().st_size} bytes"
        )

    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": total_bytes,
        "obu_count": record_count,
        "temporal_unit_count": (max_temporal_unit + 1) if record_count else 0,
        "max_spatial_id": max_spatial_id,
        "max_temporal_id": max_temporal_id,
        "obu_types": {
            name: {"count": type_counts[name], "bytes": type_bytes[name]}
            for name in sorted(type_counts)
        },
        "layers": {
            name: {"count": layer_counts[name], "bytes": layer_bytes[name]}
            for name in sorted(layer_counts)
        },
    }


def make_obu(
    obu_type: int,
    payload: bytes = b"",
    *,
    spatial_id: int | None = None,
    temporal_id: int | None = None,
) -> bytes:
    """Build a small low-overhead OBU, primarily for tests and examples."""

    if not 0 <= obu_type <= 15:
        raise ValueError("obu_type must be in [0, 15]")
    if (spatial_id is None) != (temporal_id is None):
        raise ValueError("spatial_id and temporal_id must be provided together")

    extension_flag = spatial_id is not None
    header = (obu_type << 3) | (int(extension_flag) << 2) | 0x02
    output = bytearray([header])
    if extension_flag:
        assert spatial_id is not None and temporal_id is not None
        if not 0 <= spatial_id <= 3:
            raise ValueError("spatial_id must be in [0, 3]")
        if not 0 <= temporal_id <= 7:
            raise ValueError("temporal_id must be in [0, 7]")
        output.append((temporal_id << 5) | (spatial_id << 3))
    output.extend(encode_leb128(len(payload)))
    output.extend(payload)
    return bytes(output)
