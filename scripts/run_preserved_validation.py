from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from av1_spatial_temporal.encoder import (
    EncodeConfig,
    find_ffmpeg,
    find_ffprobe,
    find_svc_encoder,
)
from av1_spatial_temporal.layer_stream import merge_layer_streams, split_obu_stream
from av1_spatial_temporal.obu import OBU_FRAME, OBU_SEQUENCE_HEADER, make_obu
from av1_spatial_temporal.operations import (
    decode_obu_stream,
    extract_operating_point,
    sha256_file,
)
from av1_spatial_temporal.workflow import run_poc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_logged(
    logs: Path,
    name: str,
    command: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    rendered = " ".join(shlex.quote(part) for part in command)
    log = (
        f"command: {rendered}\n"
        f"cwd: {cwd}\n"
        f"exit_code: {completed.returncode}\n"
        f"elapsed_seconds: {elapsed:.6f}\n\n"
        f"[stdout]\n{completed.stdout}\n"
        f"[stderr]\n{completed.stderr}\n"
    )
    log_path = logs / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log, encoding="utf-8")
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}"
        )
    return completed


def prepare_validation_input(
    *,
    ffmpeg: Path,
    logs: Path,
    input_directory: Path,
    supplied_input: Path | None,
    width: int,
    height: int,
    fps: int,
    frames: int,
) -> tuple[Path, dict[str, object]]:
    """Create the fixed Y4M source used by a persistent validation run."""

    if supplied_input is None:
        source = (
            input_directory
            / f"testsrc_{width}x{height}_{fps}fps_{frames}frames.y4m"
        )
        run_logged(
            logs,
            "generate_source",
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={width}x{height}:rate={fps}",
                "-frames:v",
                str(frames),
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
        )
        return source, {
            "mode": "generated_testsrc2",
            "supplied_input": None,
            "preserved_input": None,
            "encoder_input_y4m": str(source),
            "normalization": {
                "width": width,
                "height": height,
                "fps": fps,
                "frames": frames,
                "pixel_format": "yuv420p",
            },
        }

    original_directory = input_directory / "original"
    original_directory.mkdir()
    preserved_input = original_directory / supplied_input.name
    shutil.copy2(supplied_input, preserved_input)

    source = (
        input_directory
        / f"normalized_{width}x{height}_{fps}fps_{frames}frames.y4m"
    )
    run_logged(
        logs,
        "normalize_supplied_input",
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(preserved_input),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"scale={width}:{height}:flags=lanczos,fps={fps},format=yuv420p",
            "-frames:v",
            str(frames),
            str(source),
        ],
    )
    return source, {
        "mode": "supplied_file",
        "supplied_input": str(supplied_input),
        "preserved_input": str(preserved_input),
        "encoder_input_y4m": str(source),
        "normalization": {
            "width": width,
            "height": height,
            "fps": fps,
            "maximum_frames": frames,
            "pixel_format": "yuv420p",
            "note": "If the supplied video has fewer frames, validation fails the fixed frame-count checks.",
        },
    }


def probe_obu(
    ffprobe: Path, path: Path, logs: Path, log_name: str
) -> dict[str, object]:
    completed = run_logged(
        logs,
        log_name,
        [
            str(ffprobe),
            "-v",
            "error",
            "-f",
            "obu",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,width,height,pix_fmt,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "codec_name": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pixel_format": stream.get("pix_fmt"),
        "decoded_frames": int(stream["nb_read_frames"]),
    }


def decoded_frame_hash(
    ffmpeg: Path, path: Path, logs: Path, log_name: str
) -> str:
    completed = run_logged(
        logs,
        log_name,
        [
            str(ffmpeg),
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
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
    )
    line = completed.stdout.strip()
    if not line.startswith("SHA256="):
        raise RuntimeError(f"Unexpected FFmpeg hash output for {path}: {line}")
    return line.split("=", 1)[1].lower()


def build_manifest(result_root: Path) -> None:
    excluded = {"MANIFEST.json", "SHA256SUMS.txt"}
    entries: list[dict[str, object]] = []
    for path in sorted(item for item in result_root.rglob("*") if item.is_file()):
        relative = path.relative_to(result_root).as_posix()
        if relative in excluded:
            continue
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    sums_path = result_root / "SHA256SUMS.txt"
    sums_path.write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries),
        encoding="utf-8",
    )
    entries.append(
        {
            "path": sums_path.name,
            "size_bytes": sums_path.stat().st_size,
            "sha256": sha256_file(sums_path),
        }
    )
    write_json(
        result_root / "MANIFEST.json",
        {
            "note": "MANIFEST.json excludes its own hash; every other result file is listed.",
            "file_count": len(entries),
            "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
            "files": entries,
        },
    )


def create_persistent_merge_benchmark(
    root: Path, size_mib: int
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    source = root / f"synthetic_{size_mib}mib.obu"
    base = root / "base.a1ls"
    enhancement = root / "enhancement.a1ls"
    reconstructed = root / "reconstructed.obu"
    payload = bytes(range(256)) * 128
    target_bytes = size_mib * 1024 * 1024
    with source.open("wb") as stream:
        stream.write(make_obu(OBU_SEQUENCE_HEADER, b"persistent-benchmark"))
        sequence = 0
        temporal_pattern = (0, 2, 1, 2)
        while stream.tell() < target_bytes:
            stream.write(
                make_obu(
                    OBU_FRAME,
                    payload,
                    spatial_id=sequence % 2,
                    temporal_id=temporal_pattern[sequence % 4],
                )
            )
            sequence += 1
    split_report = split_obu_stream(source, base, enhancement)
    merge_report = merge_layer_streams((base, enhancement), reconstructed)
    source_hash = sha256_file(source)
    reconstructed_hash = sha256_file(reconstructed)
    if source_hash != reconstructed_hash:
        raise AssertionError("Persistent benchmark reconstruction hash mismatch")
    return {
        "size_mib_requested": size_mib,
        "source": str(source),
        "base": str(base),
        "enhancement": str(enhancement),
        "reconstructed": str(reconstructed),
        "split": split_report,
        "merge": merge_report,
        "byte_identical": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and preserve the complete AV1 L2T3 validation set"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "test_results",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional MP4/Y4M or other FFmpeg-readable input; defaults to testsrc2",
    )
    parser.add_argument("--name", help="Result directory name")
    parser.add_argument("--benchmark-mib", type=int, default=16)
    args = parser.parse_args()

    supplied_input = args.input.expanduser().resolve() if args.input else None
    if supplied_input is not None and not supplied_input.is_file():
        parser.error(f"Input file does not exist: {supplied_input}")

    name = args.name or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_l2t3")
    result_root = (args.output_root / name).resolve()
    if result_root.exists():
        parser.error(f"Result directory already exists: {result_root}")
    result_root.mkdir(parents=True)
    logs = result_root / "logs"
    reports = result_root / "reports"
    input_directory = result_root / "input"
    input_directory.mkdir()
    logs.mkdir()
    reports.mkdir()

    summary: dict[str, object] = {
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "result_root": str(result_root),
        "checks": [],
    }
    checks: list[dict[str, object]] = summary["checks"]  # type: ignore[assignment]

    def record_check(name_: str, passed: bool, details: object) -> None:
        checks.append({"name": name_, "passed": passed, "details": details})

    try:
        ffmpeg = find_ffmpeg()
        ffprobe = find_ffprobe()
        encoder = find_svc_encoder()
        environment = {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "project_root": str(PROJECT_ROOT),
            "ffmpeg": str(ffmpeg),
            "ffprobe": str(ffprobe),
            "svc_encoder_rtc": str(encoder),
        }
        write_json(reports / "environment.json", environment)
        run_logged(logs, "ffmpeg_version", [str(ffmpeg), "-version"])
        run_logged(logs, "ffprobe_version", [str(ffprobe), "-version"])
        run_logged(logs, "svc_encoder_help", [str(encoder), "--help"], check=False)
        run_logged(logs, "cli_version", [sys.executable, "-m", "av1_spatial_temporal", "--version"])
        run_logged(logs, "cli_help", [sys.executable, "-m", "av1_spatial_temporal", "--help"])

        source, input_preparation = prepare_validation_input(
            ffmpeg=ffmpeg,
            logs=logs,
            input_directory=input_directory,
            supplied_input=supplied_input,
            width=320,
            height=180,
            fps=24,
            frames=48,
        )
        write_json(reports / "input_preparation.json", input_preparation)
        config = EncodeConfig(
            spatial_layers=2,
            temporal_layers=3,
            bitrate_kbps=800,
            frames=48,
            speed=10,
            threads=4,
            keyframe_distance=24,
            test_decode=True,
        )
        write_json(
            reports / "test_configuration.json",
            {
                "source": str(source),
                "source_width": 320,
                "source_height": 180,
                "source_fps": 24,
                "source_frames": 48,
                "spatial_layers": 2,
                "temporal_layers": 3,
                "bitrate_kbps": 800,
                "base_operating_point": "S0T0",
                "expected_operating_points": {
                    "S0T0": [160, 90, 12],
                    "S0T1": [160, 90, 24],
                    "S0T2": [160, 90, 48],
                    "S1T0": [320, 180, 12],
                    "S1T1": [320, 180, 24],
                    "S1T2": [320, 180, 48],
                },
            },
        )

        poc_root = result_root / "poc"
        poc_report = run_poc(source, poc_root, config)
        write_json(reports / "poc_report_copy.json", poc_report)

        full_obu = poc_root / "encoded" / "full.obu"
        reconstructed = poc_root / "reconstructed.obu"
        direct_s0t0 = result_root / "direct_s0t0" / "s0t0.obu"
        direct_extract_report = extract_operating_point(
            full_obu,
            direct_s0t0,
            max_spatial_id=0,
            max_temporal_id=0,
        )
        write_json(reports / "direct_s0t0_extract.json", direct_extract_report)

        targets: dict[str, Path] = {
            "full_s1t2": full_obu,
            "reconstructed_s1t2": reconstructed,
            "base_from_transport_s0t0": poc_root / "base.obu",
            "direct_extract_s0t0": direct_s0t0,
        }
        for spatial in range(2):
            for temporal in range(3):
                key = f"op_s{spatial}_t{temporal}"
                targets[key] = poc_root / "encoded" / f"{key}.obu"

        probes: dict[str, dict[str, object]] = {}
        decode_checks: dict[str, object] = {}
        for key, path in targets.items():
            probes[key] = probe_obu(ffprobe, path, logs, f"probe_{key}")
            decode_checks[key] = decode_obu_stream(path, ffmpeg=ffmpeg)
        write_json(reports / "obu_probes.json", probes)
        write_json(reports / "decode_checks.json", decode_checks)

        expected: dict[str, tuple[int, int, int]] = {
            "full_s1t2": (320, 180, 48),
            "reconstructed_s1t2": (320, 180, 48),
            "base_from_transport_s0t0": (160, 90, 12),
            "direct_extract_s0t0": (160, 90, 12),
            "op_s0_t0": (160, 90, 12),
            "op_s0_t1": (160, 90, 24),
            "op_s0_t2": (160, 90, 48),
            "op_s1_t0": (320, 180, 12),
            "op_s1_t1": (320, 180, 24),
            "op_s1_t2": (320, 180, 48),
        }
        for key, wanted in expected.items():
            actual = (
                int(probes[key]["width"]),
                int(probes[key]["height"]),
                int(probes[key]["decoded_frames"]),
            )
            record_check(f"{key}_dimensions_and_frames", actual == wanted, {"expected": wanted, "actual": actual})

        full_hash = sha256_file(full_obu)
        reconstructed_hash = sha256_file(reconstructed)
        record_check(
            "full_reconstruction_byte_identical",
            full_hash == reconstructed_hash,
            {"full_sha256": full_hash, "reconstructed_sha256": reconstructed_hash},
        )

        frame_hashes = {
            "encoder_generated_s0t0": decoded_frame_hash(
                ffmpeg,
                poc_root / "encoded" / "op_s0_t0.obu",
                logs,
                "frame_hash_encoder_generated_s0t0",
            ),
            "direct_extract_s0t0": decoded_frame_hash(
                ffmpeg, direct_s0t0, logs, "frame_hash_direct_extract_s0t0"
            ),
            "transport_base_s0t0": decoded_frame_hash(
                ffmpeg,
                poc_root / "base.obu",
                logs,
                "frame_hash_transport_base_s0t0",
            ),
        }
        write_json(reports / "decoded_frame_hashes.json", frame_hashes)
        record_check(
            "direct_s0t0_decoded_pixels_match_encoder_output",
            len(set(frame_hashes.values())) == 1,
            frame_hashes,
        )

        analyze_result = run_logged(
            logs,
            "cli_analyze_full",
            [sys.executable, "-m", "av1_spatial_temporal", "analyze", str(full_obu)],
        )
        write_json(reports / "cli_analyze_full.json", json.loads(analyze_result.stdout))
        run_logged(
            logs,
            "cli_verify_reconstruction",
            [
                sys.executable,
                "-m",
                "av1_spatial_temporal",
                "verify",
                str(full_obu),
                str(reconstructed),
            ],
        )

        test_environment = os.environ.copy()
        test_environment["RUN_AV1_INTEGRATION"] = "1"
        unit_test = run_logged(
            logs,
            "unittest_full",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            env=test_environment,
        )
        record_check(
            "complete_unittest_suite",
            unit_test.returncode == 0,
            "All unit and real integration tests passed; see logs/unittest_full.log",
        )

        benchmark_report = create_persistent_merge_benchmark(
            result_root / "benchmark", args.benchmark_mib
        )
        write_json(reports / "persistent_merge_benchmark.json", benchmark_report)
        record_check(
            "persistent_benchmark_byte_identical",
            bool(benchmark_report["byte_identical"]),
            {
                "throughput_mib_per_second": benchmark_report["merge"]["throughput_mib_per_second"],  # type: ignore[index]
                "size_bytes": benchmark_report["merge"]["size_bytes"],  # type: ignore[index]
            },
        )

        failed_checks = [check for check in checks if not check["passed"]]
        if failed_checks:
            raise AssertionError(f"{len(failed_checks)} validation checks failed")
        summary["status"] = "PASS"
        summary["completed_at"] = datetime.now().astimezone().isoformat()
        summary["key_outputs"] = {
            "full_s1t2": str(full_obu),
            "direct_s0t0": str(direct_s0t0),
            "encoder_s0t0": str(poc_root / "encoded" / "op_s0_t0.obu"),
            "reconstructed_s1t2": str(reconstructed),
            "poc_report": str(poc_root / "poc_report.json"),
            "benchmark_report": str(reports / "persistent_merge_benchmark.json"),
        }
    except Exception as exc:
        summary["status"] = "FAIL"
        summary["completed_at"] = datetime.now().astimezone().isoformat()
        summary["error"] = f"{type(exc).__name__}: {exc}"
        (logs / "failure_traceback.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
    finally:
        write_json(result_root / "SUMMARY.json", summary)
        build_manifest(result_root)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Preserved result directory: {result_root}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
