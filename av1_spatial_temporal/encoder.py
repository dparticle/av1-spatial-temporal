"""Stable public encoder API backed by the verified libaom implementation."""

from __future__ import annotations

from .encoder_legacy import (
    DEFAULT_SCALE_FACTORS,
    LAYERING_MODES,
    TEMPORAL_CUMULATIVE_RATIOS,
    EncodeConfig,
    EncoderError,
    VideoInfo,
    allocate_bitrates as _allocate_bitrates,
    find_ffmpeg,
    find_ffprobe,
    find_svc_encoder,
    layering_mode,
    parse_fraction,
    probe_video,
)


def allocate_bitrates(
    total_kbps: int,
    spatial_layers: int,
    temporal_layers: int,
    scale_factors: tuple[tuple[int, int], ...] | None = None,
) -> tuple[int, ...]:
    if total_kbps < spatial_layers * temporal_layers:
        raise ValueError(
            "bitrate_kbps is too small to assign every spatial/temporal layer"
        )
    return _allocate_bitrates(
        total_kbps, spatial_layers, temporal_layers, scale_factors
    )


# Imported last: encoder_v2 intentionally consumes the stable definitions above.
from .encoder_v2 import encode_av1_svc  # noqa: E402


__all__ = [
    "DEFAULT_SCALE_FACTORS",
    "LAYERING_MODES",
    "TEMPORAL_CUMULATIVE_RATIOS",
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
