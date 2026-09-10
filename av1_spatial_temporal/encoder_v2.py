from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from .encoder import (
    DEFAULT_SCALE_FACTORS,
    EncodeConfig,
    EncoderError,
    VideoInfo,
    allocate_bitrates,
    find_ffmpeg,
    find_ffprobe,
    find_svc_encoder,
    layering_mode,
    parse_fraction,
    probe_video,
)
from .obu import analyze_obu_stream


def _validate_config(config: EncodeConfig, width: int, height: int) -> None:
    layering_mode(config.spatial_layers, config.temporal_layers)
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("Output width and height must be positive even numbers")
    if not 0 <= config.speed <= 11:
        raise ValueError("speed must be in [0, 11]")
    if config.threads < 1:
        raise ValueError("threads must be positive")
    if config.frames is not None and config.frames < 1:
        raise ValueError("frames must be positive")
    if config.keyframe_distance < 1:
        raise ValueError("keyframe_distance must be positive")
    if not 0 <= config.min_q <= config.max_q <= 63:
        raise ValueError("quantizers must satisfy 0 <= min_q <= max_q <= 63")
    if config.bitrate_kbps < config.spatial_layers * config.temporal_layers:
        raise ValueError("bitrate_kbps is too small for the selected layer count")
    for num, den in DEFAULT_SCALE_FACTORS[config.spatial_layers]:
        layer_width = width * num // den
        layer_height = height * num // den
        if layer_width < 2 or layer_height < 2 or layer_width % 2 or layer_height % 2:
            raise ValueError(
                f"Scale factor {num}/{den} produces an invalid 4:2:0 layer "
                f"({layer_width}x{layer_height})"
            )


def _move_output(source: Path, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(f"Output already exists: {destination}")
    os.replace(source, destination)


def encode_av1_svc(
    input_path: Path,
    output_directory: Path,
    config: EncodeConfig,
    *,
    encoder: str | Path | None = None,
    ffmpeg: str | Path | None = None,
    ffprobe: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Encode low-overhead AV1 SVC; FFmpeg alone enforces the frame limit."""

    input_path = input_path.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    encoder_path = find_svc_encoder(encoder)
    ffmpeg_path = find_ffmpeg(ffmpeg)
    ffprobe_path = find_ffprobe(ffprobe)
    source_info = probe_video(input_path, ffprobe_path)
    width = config.width or source_info.width
    height = config.height or source_info.height
    fps = config.fps or Fraction(
        source_info.fps_numerator, source_info.fps_denominator
    )
    _validate_config(config, width, height)
    mode = layering_mode(config.spatial_layers, config.temporal_layers)
    scales = DEFAULT_SCALE_FACTORS[config.spatial_layers]
    bitrates = allocate_bitrates(
        config.bitrate_kbps,
        config.spatial_layers,
        config.temporal_layers,
        scales,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    final_full = output_directory / "full.obu"
    final_operating_points = [
        output_directory / f"op_s{spatial}_t{temporal}.obu"
        for spatial in range(config.spatial_layers)
        for temporal in range(config.temporal_layers)
    ]
    final_report = output_directory / "encode_report.json"
    for target in (final_full, *final_operating_points, final_report):
        if target.exists() and not force:
            raise FileExistsError(f"Output already exists: {target}")

    staging = output_directory / f".svc-staging-{uuid.uuid4().hex}"
    staging.mkdir()
    staged_full = staging / "full.obu"
    frame_filter = f"scale={width}:{height}:flags=lanczos,format=yuv420p"
    if config.fps is not None:
        frame_filter += f",fps={fps.numerator}/{fps.denominator}"
    ffmpeg_command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        frame_filter,
    ]
    if config.frames is not None:
        ffmpeg_command.extend(("-frames:v", str(config.frames)))
    ffmpeg_command.extend(("-strict", "-1", "-f", "yuv4mpegpipe", "-"))

    encoder_command = [
        str(encoder_path),
        f"--width={width}",
        f"--height={height}",
        f"--timebase={fps.denominator}/{fps.numerator}",
        f"--target-bitrate={config.bitrate_kbps}",
        f"--bitrates={','.join(map(str, bitrates))}",
        f"--spatial-layers={config.spatial_layers}",
        f"--temporal-layers={config.temporal_layers}",
        f"--layering-mode={mode}",
        "--scale-factors=" + ",".join(f"{n}/{d}" for n, d in scales),
        f"--speed={config.speed}",
        f"--threads={config.threads}",
        f"--kf-dist={config.keyframe_distance}",
        f"--min-q={config.min_q}",
        f"--max-q={config.max_q}",
        "--error-resilient=0",
        "--output-obu=1",
        f"--test-decode={int(config.test_decode)}",
        "-",
        "-o",
        str(staged_full),
    ]

    encoder_log = ""
    ffmpeg_log = ""
    try:
        with tempfile.TemporaryFile() as ffmpeg_log_file, tempfile.TemporaryFile() as encoder_log_file:
            ffmpeg_process = subprocess.Popen(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=ffmpeg_log_file,
            )
            assert ffmpeg_process.stdout is not None
            try:
                encoder_process = subprocess.Popen(
                    encoder_command,
                    stdin=ffmpeg_process.stdout,
                    stdout=encoder_log_file,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                ffmpeg_process.kill()
                ffmpeg_process.wait()
                raise
            finally:
                ffmpeg_process.stdout.close()

            encoder_returncode = encoder_process.wait()
            ffmpeg_returncode = ffmpeg_process.wait()
            encoder_log_file.seek(0)
            ffmpeg_log_file.seek(0)
            encoder_log = encoder_log_file.read().decode("utf-8", errors="replace")
            ffmpeg_log = ffmpeg_log_file.read().decode("utf-8", errors="replace")

        if ffmpeg_returncode != 0 or encoder_returncode != 0:
            raise EncoderError(
                "SVC encode pipeline failed\n"
                f"ffmpeg exit code: {ffmpeg_returncode}\n{ffmpeg_log.strip()}\n"
                f"svc_encoder_rtc exit code: {encoder_returncode}\n"
                f"{encoder_log.strip()}"
            )
        if not staged_full.is_file():
            raise EncoderError("svc_encoder_rtc did not create the full OBU stream")

        staged_operating_points: list[Path] = []
        for spatial in range(config.spatial_layers):
            for temporal in range(config.temporal_layers):
                index = spatial * config.temporal_layers + temporal
                generated = Path(f"{staged_full}_{index}.av1")
                if not generated.is_file():
                    raise EncoderError(
                        f"svc_encoder_rtc did not create operating point {index}"
                    )
                staged_operating_points.append(generated)

        _move_output(staged_full, final_full, force=force)
        for source, destination in zip(
            staged_operating_points, final_operating_points, strict=True
        ):
            _move_output(source, destination, force=force)

        output_analyses = {
            path.name: analyze_obu_stream(path)
            for path in (final_full, *final_operating_points)
        }
        serializable_config = asdict(config)
        serializable_config["fps"] = str(config.fps) if config.fps else None
        report: dict[str, object] = {
            "input": str(input_path),
            "output_directory": str(output_directory),
            "source": asdict(source_info),
            "effective": {
                "width": width,
                "height": height,
                "fps": f"{fps.numerator}/{fps.denominator}",
                "layering_mode": mode,
                "scale_factors": [f"{n}/{d}" for n, d in scales],
                "layer_bitrates_kbps": list(bitrates),
                "frame_limit_enforced_by": "ffmpeg",
            },
            "config": serializable_config,
            "tools": {
                "svc_encoder_rtc": str(encoder_path),
                "ffmpeg": str(ffmpeg_path),
                "ffprobe": str(ffprobe_path),
            },
            "outputs": output_analyses,
            "encoder_log": encoder_log,
            "ffmpeg_log": ffmpeg_log,
        }
        final_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "EncodeConfig",
    "EncoderError",
    "VideoInfo",
    "allocate_bitrates",
    "encode_av1_svc",
    "find_ffmpeg",
    "find_ffprobe",
    "find_svc_encoder",
    "layering_mode",
    "parse_fraction",
    "probe_video",
]

