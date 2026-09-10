from __future__ import annotations

import hashlib
import os
import subprocess
import time
import uuid
from pathlib import Path

from .encoder import find_ffmpeg
from .layer_stream import LayerStreamReader
from .obu import iter_obus_path


class VerificationError(RuntimeError):
    """Raised when reconstruction or decoding verification fails."""


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _prepare_output(path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_operating_point(
    input_path: Path,
    output_path: Path,
    *,
    max_spatial_id: int,
    max_temporal_id: int,
    force: bool = False,
) -> dict[str, object]:
    """Copy OBUs for one cumulative spatial/temporal operating point."""

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input OBU stream does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must differ")
    if not 0 <= max_spatial_id <= 3:
        raise ValueError("max_spatial_id must be in [0, 3]")
    if not 0 <= max_temporal_id <= 7:
        raise ValueError("max_temporal_id must be in [0, 7]")
    _prepare_output(output_path, force=force)
    temp_path = _temporary_sibling(output_path)
    digest = hashlib.sha256()
    included_records = included_bytes = skipped_records = skipped_bytes = 0
    started = time.perf_counter()
    try:
        with temp_path.open("wb") as output:
            for obu in iter_obus_path(input_path):
                include = obu.is_global or (
                    obu.spatial_id is not None
                    and obu.temporal_id is not None
                    and obu.spatial_id <= max_spatial_id
                    and obu.temporal_id <= max_temporal_id
                )
                if include:
                    output.write(obu.raw)
                    digest.update(obu.raw)
                    included_records += 1
                    included_bytes += len(obu.raw)
                else:
                    skipped_records += 1
                    skipped_bytes += len(obu.raw)
        os.replace(temp_path, output_path)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "input": str(input_path),
        "output": str(output_path),
        "max_spatial_id": max_spatial_id,
        "max_temporal_id": max_temporal_id,
        "included_records": included_records,
        "included_bytes": included_bytes,
        "skipped_records": skipped_records,
        "skipped_bytes": skipped_bytes,
        "sha256": digest.hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def unpack_layer_stream(
    input_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Materialize the OBU payload carried by one A1LS channel."""

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"A1LS input does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must differ")
    _prepare_output(output_path, force=force)
    temp_path = _temporary_sibling(output_path)
    digest = hashlib.sha256()
    record_count = size_bytes = 0
    started = time.perf_counter()
    metadata: dict[str, object] = {}
    try:
        with input_path.open("rb") as source, temp_path.open("wb") as output:
            reader = LayerStreamReader(source, source=str(input_path))
            metadata = reader.metadata
            for record in reader:
                output.write(record.raw)
                digest.update(record.raw)
                record_count += 1
                size_bytes += len(record.raw)
        os.replace(temp_path, output_path)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "input": str(input_path),
        "output": str(output_path),
        "channel": metadata.get("channel"),
        "record_count": record_count,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def decode_obu_stream(
    path: Path,
    *,
    ffmpeg: str | Path | None = None,
) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OBU stream does not exist: {path}")
    ffmpeg_path = find_ffmpeg(ffmpeg)
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "obu",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise VerificationError(
            f"FFmpeg could not decode {path}:\n{completed.stderr.strip()}"
        )
    return {
        "path": str(path),
        "decoder": str(ffmpeg_path),
        "decodable": True,
        "elapsed_seconds": elapsed,
    }


def verify_reconstruction(
    original_path: Path,
    reconstructed_path: Path,
    *,
    decode: bool = True,
    ffmpeg: str | Path | None = None,
) -> dict[str, object]:
    original_path = original_path.resolve()
    reconstructed_path = reconstructed_path.resolve()
    for path in (original_path, reconstructed_path):
        if not path.is_file():
            raise FileNotFoundError(f"OBU stream does not exist: {path}")
    original_size = original_path.stat().st_size
    reconstructed_size = reconstructed_path.stat().st_size
    original_hash = sha256_file(original_path)
    reconstructed_hash = sha256_file(reconstructed_path)
    identical = (
        original_size == reconstructed_size and original_hash == reconstructed_hash
    )
    if not identical:
        raise VerificationError(
            "Reconstructed stream is not byte-identical: "
            f"original={original_size} bytes/{original_hash}, "
            f"reconstructed={reconstructed_size} bytes/{reconstructed_hash}"
        )
    report: dict[str, object] = {
        "original": str(original_path),
        "reconstructed": str(reconstructed_path),
        "size_bytes": original_size,
        "sha256": original_hash,
        "byte_identical": True,
    }
    if decode:
        report["decode"] = decode_obu_stream(reconstructed_path, ffmpeg=ffmpeg)
    return report

