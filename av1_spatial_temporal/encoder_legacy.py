from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from .obu import analyze_obu_stream


LAYERING_MODES = {
    (1, 1): 0,
    (1, 2): 1,
    (1, 3): 2,
    (2, 1): 5,
    (3, 1): 6,
    (2, 3): 7,
    (3, 3): 9,
}

DEFAULT_SCALE_FACTORS = {
    1: ((1, 1),),
    2: ((1, 2), (1, 1)),
    3: ((1, 4), (1, 2), (1, 1)),
}

TEMPORAL_CUMULATIVE_RATIOS = {
    1: (1.0,),
    2: (0.5, 1.0),
    3: (0.25, 0.5, 1.0),
}


class EncoderError(RuntimeError):
    """Raised when probing, transcoding, or SVC encoding fails."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    pixel_format: str | None
    duration_seconds: float | None
    frame_count: int | None

    @property
    def fps(self) -> float:
        return self.fps_numerator / self.fps_denominator


@dataclass(frozen=True, slots=True)
class EncodeConfig:
    spatial_layers: int = 2
    temporal_layers: int = 3
    bitrate_kbps: int = 3000
    width: int | None = None
    height: int | None = None
    fps: Fraction | None = None
    frames: int | None = None
    speed: int = 9
    threads: int = 4
    keyframe_distance: int = 120
    min_q: int = 2
    max_q: int = 52
    test_decode: bool = True


def parse_fraction(value: str) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid fraction: {value}") from exc
    if result <= 0:
        raise ValueError("Fraction must be positive")
    return result


def _find_program(explicit: str | Path | None, names: tuple[str, ...]) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Executable does not exist: {path}")
        return path
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    raise FileNotFoundError(f"Could not find executable: {' or '.join(names)}")


def find_svc_encoder(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return _find_program(explicit, ())
    from_environment = os.environ.get("AOM_SVC_ENCODER")
    if from_environment:
        return _find_program(from_environment, ())
    found = shutil.which("svc_encoder_rtc") or shutil.which("svc_encoder_rtc.exe")
    if found:
        return Path(found).resolve()

    project_root = Path(__file__).resolve().parent.parent
    executable = "svc_encoder_rtc.exe" if os.name == "nt" else "svc_encoder_rtc"
    candidates = (
        project_root.parent / "aom" / "aom_build" / "examples" / executable,
        project_root.parent / "aom" / "build" / "examples" / executable,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find svc_encoder_rtc. Pass --encoder or set AOM_SVC_ENCODER."
    )


def find_ffmpeg(explicit: str | Path | None = None) -> Path:
    return _find_program(explicit, ("ffmpeg", "ffmpeg.exe"))


def find_ffprobe(explicit: str | Path | None = None) -> Path:
    return _find_program(explicit, ("ffprobe", "ffprobe.exe"))


def probe_video(path: Path, ffprobe: Path) -> VideoInfo:
    if not path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {path}")
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,pix_fmt,duration,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise EncoderError(f"ffprobe failed:\n{completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise EncoderError("ffprobe did not return a usable video stream") from exc

    rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not rate_text or rate_text == "0/0":
        rate_text = stream.get("r_frame_rate")
    try:
        rate = Fraction(rate_text)
    except (ValueError, ZeroDivisionError) as exc:
        raise EncoderError(f"Cannot determine input frame rate: {rate_text}") from exc
    if rate <= 0:
        raise EncoderError("Input frame rate must be positive")

    duration_text = stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(duration_text) if duration_text not in (None, "N/A") else None
    except ValueError:
        duration = None
    frame_text = stream.get("nb_frames")
    try:
        frame_count = int(frame_text) if frame_text not in (None, "N/A") else None
    except ValueError:
        frame_count = None
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps_numerator=rate.numerator,
        fps_denominator=rate.denominator,
        pixel_format=stream.get("pix_fmt"),
        duration_seconds=duration,
        frame_count=frame_count,
    )


def layering_mode(spatial_layers: int, temporal_layers: int) -> int:
    try:
        return LAYERING_MODES[(spatial_layers, temporal_layers)]
    except KeyError as exc:
        supported = ", ".join(f"L{s}T{t}" for s, t in LAYERING_MODES)
        raise ValueError(
            f"Unsupported layer combination L{spatial_layers}T{temporal_layers}; "
            f"supported combinations: {supported}"
        ) from exc


def allocate_bitrates(
    total_kbps: int,
    spatial_layers: int,
    temporal_layers: int,
    scale_factors: tuple[tuple[int, int], ...] | None = None,
) -> tuple[int, ...]:
    if total_kbps < 1:
        raise ValueError("bitrate_kbps must be positive")
    if scale_factors is None:
        scale_factors = DEFAULT_SCALE_FACTORS[spatial_layers]
    if len(scale_factors) != spatial_layers:
        raise ValueError("One scale factor is required per spatial layer")
    temporal_ratios = TEMPORAL_CUMULATIVE_RATIOS[temporal_layers]
    area_weights = [(num / den) ** 2 for num, den in scale_factors]
    weight_sum = sum(area_weights)

    spatial_tops: list[int] = []
    remaining = total_kbps
    for index, weight in enumerate(area_weights):
        if index == len(area_weights) - 1:
            top = remaining
        else:
            top = max(temporal_layers, round(total_kbps * weight / weight_sum))
            top = min(top, remaining - (len(area_weights) - index - 1))
        spatial_tops.append(top)
        remaining -= top

    bitrates: list[int] = []
    for top in spatial_tops:
        previous = 0
        for temporal_index, ratio in enumerate(temporal_ratios):
            if temporal_index == len(temporal_ratios) - 1:
                value = top
            else:
                value = max(previous + 1, round(top * ratio))
                value = min(value, top - (len(temporal_ratios) - temporal_index - 1))
            bitrates.append(value)
            previous = value
    return tuple(bitrates)


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
    """Encode one source into a low-overhead AV1 spatial/temporal SVC stream."""

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
    ]
    if config.frames is not None:
        encoder_command.append(f"--frames={config.frames}")
    encoder_command.extend(("-", "-o", str(staged_full)))

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

