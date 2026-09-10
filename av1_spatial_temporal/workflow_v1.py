from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .encoder import EncodeConfig, encode_av1_svc
from .layer_stream import merge_layer_streams, split_obu_stream
from .operations_v1 import (
    decode_obu_stream,
    unpack_layer_stream,
    verify_reconstruction,
)


@dataclass(frozen=True, slots=True)
class ToolPaths:
    encoder: str | Path | None = None
    ffmpeg: str | Path | None = None
    ffprobe: str | Path | None = None


def run_poc(
    input_path: Path,
    output_directory: Path,
    config: EncodeConfig,
    *,
    base_spatial_id: int = 0,
    base_temporal_id: int = 0,
    tools: ToolPaths = ToolPaths(),
    force: bool = False,
) -> dict[str, object]:
    """Run encode, layer split, fast merge, and decode verification."""

    output_directory = output_directory.resolve()
    encoded_directory = output_directory / "encoded"
    transport_directory = output_directory / "transport"
    base_transport = transport_directory / "base.a1ls"
    enhancement_transport = transport_directory / "enhancement.a1ls"
    base_obu = output_directory / "base.obu"
    reconstructed = output_directory / "reconstructed.obu"
    report_path = output_directory / "poc_report.json"
    if report_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {report_path}")
    output_directory.mkdir(parents=True, exist_ok=True)

    encode_report = encode_av1_svc(
        input_path,
        encoded_directory,
        config,
        encoder=tools.encoder,
        ffmpeg=tools.ffmpeg,
        ffprobe=tools.ffprobe,
        force=force,
    )
    full_obu = encoded_directory / "full.obu"
    split_report = split_obu_stream(
        full_obu,
        base_transport,
        enhancement_transport,
        max_base_spatial_id=base_spatial_id,
        max_base_temporal_id=base_temporal_id,
        force=force,
    )
    base_report = unpack_layer_stream(base_transport, base_obu, force=force)
    merge_report = merge_layer_streams(
        (base_transport, enhancement_transport), reconstructed, force=force
    )
    reconstruction_report = verify_reconstruction(
        full_obu, reconstructed, decode=True, ffmpeg=tools.ffmpeg
    )

    decode_reports: dict[str, object] = {
        "base.obu": decode_obu_stream(base_obu, ffmpeg=tools.ffmpeg)
    }
    for path in sorted(encoded_directory.glob("op_s*_t*.obu")):
        decode_reports[path.name] = decode_obu_stream(path, ffmpeg=tools.ffmpeg)

    report: dict[str, object] = {
        "encode": encode_report,
        "split": split_report,
        "base_materialization": base_report,
        "merge": merge_report,
        "reconstruction": reconstruction_report,
        "decode_checks": decode_reports,
        "result": {
            "byte_identical_reconstruction": True,
            "all_operating_points_decodable": True,
            "base_decodable": True,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

