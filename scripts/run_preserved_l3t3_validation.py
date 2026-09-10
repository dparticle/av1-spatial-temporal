from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_preserved_validation import (
    build_manifest,
    decoded_frame_hash,
    prepare_validation_input,
    probe_obu,
    run_logged,
    write_json,
)

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


def create_l3t3_merge_benchmark(root: Path, size_mib: int) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    source = root / f"synthetic_l3t3_{size_mib}mib.obu"
    base = root / "base_s0t0.a1ls"
    enhancement = root / "enhancement_all.a1ls"
    reconstructed = root / "reconstructed_s2t2.obu"
    payload = bytes(range(256)) * 128
    target_bytes = size_mib * 1024 * 1024
    temporal_pattern = (0, 2, 1, 2)
    with source.open("wb") as stream:
        stream.write(make_obu(OBU_SEQUENCE_HEADER, b"persistent-l3t3-benchmark"))
        sequence = 0
        while stream.tell() < target_bytes:
            stream.write(
                make_obu(
                    OBU_FRAME,
                    payload,
                    spatial_id=sequence % 3,
                    temporal_id=temporal_pattern[sequence % 4],
                )
            )
            sequence += 1
    split_report = split_obu_stream(source, base, enhancement)
    merge_report = merge_layer_streams((base, enhancement), reconstructed)
    source_hash = sha256_file(source)
    reconstructed_hash = sha256_file(reconstructed)
    if source_hash != reconstructed_hash:
        raise AssertionError("L3T3 benchmark reconstruction hash mismatch")
    return {
        "size_mib_requested": size_mib,
        "logical_spatial_layers": 3,
        "logical_temporal_layers": 3,
        "source": str(source),
        "base": str(base),
        "enhancement": str(enhancement),
        "reconstructed": str(reconstructed),
        "split": split_report,
        "merge": merge_report,
        "source_sha256": source_hash,
        "reconstructed_sha256": reconstructed_hash,
        "byte_identical": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and preserve the complete AV1 L3T3 validation set"
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

    name = args.name or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_l3t3")
    result_root = (args.output_root / name).resolve()
    if result_root.exists():
        parser.error(f"Result directory already exists: {result_root}")
    result_root.mkdir(parents=True)
    logs = result_root / "logs"
    reports = result_root / "reports"
    inputs = result_root / "input"
    logs.mkdir()
    reports.mkdir()
    inputs.mkdir()

    summary: dict[str, object] = {
        "status": "RUNNING",
        "test_name": "AV1 L3T3 complete persistent validation",
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
        write_json(
            reports / "environment.json",
            {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "project_root": str(PROJECT_ROOT),
                "ffmpeg": str(ffmpeg),
                "ffprobe": str(ffprobe),
                "svc_encoder_rtc": str(encoder),
                "libaom_layering_mode": 9,
            },
        )
        run_logged(logs, "ffmpeg_version", [str(ffmpeg), "-version"])
        run_logged(logs, "ffprobe_version", [str(ffprobe), "-version"])
        run_logged(logs, "svc_encoder_help", [str(encoder), "--help"], check=False)
        run_logged(
            logs,
            "cli_version",
            [sys.executable, "-m", "av1_spatial_temporal", "--version"],
        )
        run_logged(
            logs,
            "cli_help",
            [sys.executable, "-m", "av1_spatial_temporal", "--help"],
        )

        source, input_preparation = prepare_validation_input(
            ffmpeg=ffmpeg,
            logs=logs,
            input_directory=inputs,
            supplied_input=supplied_input,
            width=384,
            height=216,
            fps=24,
            frames=48,
        )
        write_json(reports / "input_preparation.json", input_preparation)
        config = EncodeConfig(
            spatial_layers=3,
            temporal_layers=3,
            bitrate_kbps=1500,
            frames=48,
            speed=10,
            threads=4,
            keyframe_distance=24,
            test_decode=True,
        )
        expected_operations = {
            "S0T0": [96, 54, 12],
            "S0T1": [96, 54, 24],
            "S0T2": [96, 54, 48],
            "S1T0": [192, 108, 12],
            "S1T1": [192, 108, 24],
            "S1T2": [192, 108, 48],
            "S2T0": [384, 216, 12],
            "S2T1": [384, 216, 24],
            "S2T2": [384, 216, 48],
        }
        write_json(
            reports / "test_configuration.json",
            {
                "source": str(source),
                "source_width": 384,
                "source_height": 216,
                "source_fps": 24,
                "source_frames": 48,
                "spatial_layers": 3,
                "temporal_layers": 3,
                "scale_factors": ["1/4", "1/2", "1/1"],
                "temporal_factors": ["1/4", "1/2", "1/1"],
                "bitrate_kbps": 1500,
                "base_operating_point": "S0T0",
                "expected_operating_points": expected_operations,
            },
        )

        poc_root = result_root / "poc"
        started = time.perf_counter()
        poc_report = run_poc(source, poc_root, config)
        poc_elapsed = time.perf_counter() - started
        write_json(reports / "poc_report_copy.json", poc_report)
        write_json(
            reports / "poc_timing.json",
            {"elapsed_seconds": poc_elapsed},
        )

        full_obu = poc_root / "encoded" / "full.obu"
        reconstructed = poc_root / "reconstructed.obu"
        direct_s0t0 = result_root / "direct_extract" / "s0t0.obu"
        direct_s1t1 = result_root / "direct_extract" / "s1t1.obu"
        s0t0_extract_report = extract_operating_point(
            full_obu,
            direct_s0t0,
            max_spatial_id=0,
            max_temporal_id=0,
        )
        s1t1_extract_report = extract_operating_point(
            full_obu,
            direct_s1t1,
            max_spatial_id=1,
            max_temporal_id=1,
        )
        write_json(
            reports / "direct_extracts.json",
            {"S0T0": s0t0_extract_report, "S1T1": s1t1_extract_report},
        )

        targets: dict[str, Path] = {
            "full_s2t2": full_obu,
            "reconstructed_s2t2": reconstructed,
            "base_from_transport_s0t0": poc_root / "base.obu",
            "direct_extract_s0t0": direct_s0t0,
            "direct_extract_s1t1": direct_s1t1,
        }
        for spatial in range(3):
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
            "full_s2t2": (384, 216, 48),
            "reconstructed_s2t2": (384, 216, 48),
            "base_from_transport_s0t0": (96, 54, 12),
            "direct_extract_s0t0": (96, 54, 12),
            "direct_extract_s1t1": (192, 108, 24),
        }
        for spatial, dimensions in enumerate(((96, 54), (192, 108), (384, 216))):
            for temporal, frames in enumerate((12, 24, 48)):
                expected[f"op_s{spatial}_t{temporal}"] = (
                    dimensions[0],
                    dimensions[1],
                    frames,
                )
        for key, wanted in expected.items():
            actual = (
                int(probes[key]["width"]),
                int(probes[key]["height"]),
                int(probes[key]["decoded_frames"]),
            )
            record_check(
                f"{key}_dimensions_and_frames",
                actual == wanted,
                {"expected": wanted, "actual": actual},
            )

        full_hash = sha256_file(full_obu)
        reconstructed_hash = sha256_file(reconstructed)
        record_check(
            "full_s2t2_reconstruction_byte_identical",
            full_hash == reconstructed_hash,
            {
                "full_sha256": full_hash,
                "reconstructed_sha256": reconstructed_hash,
            },
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
            "encoder_generated_s1t1": decoded_frame_hash(
                ffmpeg,
                poc_root / "encoded" / "op_s1_t1.obu",
                logs,
                "frame_hash_encoder_generated_s1t1",
            ),
            "direct_extract_s1t1": decoded_frame_hash(
                ffmpeg, direct_s1t1, logs, "frame_hash_direct_extract_s1t1"
            ),
        }
        write_json(reports / "decoded_frame_hashes.json", frame_hashes)
        s0_hashes = {
            frame_hashes["encoder_generated_s0t0"],
            frame_hashes["direct_extract_s0t0"],
            frame_hashes["transport_base_s0t0"],
        }
        s1t1_hashes = {
            frame_hashes["encoder_generated_s1t1"],
            frame_hashes["direct_extract_s1t1"],
        }
        record_check(
            "direct_s0t0_decoded_pixels_match",
            len(s0_hashes) == 1,
            {key: value for key, value in frame_hashes.items() if "s0t0" in key},
        )
        record_check(
            "direct_s1t1_decoded_pixels_match",
            len(s1t1_hashes) == 1,
            {key: value for key, value in frame_hashes.items() if "s1t1" in key},
        )

        analyze_result = run_logged(
            logs,
            "cli_analyze_full",
            [
                sys.executable,
                "-m",
                "av1_spatial_temporal",
                "analyze",
                str(full_obu),
            ],
        )
        write_json(
            reports / "cli_analyze_full.json", json.loads(analyze_result.stdout)
        )
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
        complete_test = run_logged(
            logs,
            "unittest_full_with_l3t3",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ],
            env=test_environment,
        )
        record_check(
            "complete_unittest_suite_with_l2t3_and_l3t3",
            complete_test.returncode == 0,
            "All tests passed; see logs/unittest_full_with_l3t3.log",
        )

        benchmark_report = create_l3t3_merge_benchmark(
            result_root / "benchmark", args.benchmark_mib
        )
        write_json(
            reports / "persistent_l3t3_merge_benchmark.json", benchmark_report
        )
        merge_details: dict[str, Any] = benchmark_report["merge"]  # type: ignore[assignment]
        record_check(
            "persistent_l3t3_benchmark_byte_identical",
            bool(benchmark_report["byte_identical"]),
            {
                "throughput_mib_per_second": merge_details[
                    "throughput_mib_per_second"
                ],
                "size_bytes": merge_details["size_bytes"],
            },
        )

        failed_checks = [check for check in checks if not check["passed"]]
        if failed_checks:
            raise AssertionError(f"{len(failed_checks)} validation checks failed")
        summary["status"] = "PASS"
        summary["completed_at"] = datetime.now().astimezone().isoformat()
        summary["key_outputs"] = {
            "full_s2t2": str(full_obu),
            "direct_s0t0": str(direct_s0t0),
            "direct_s1t1": str(direct_s1t1),
            "encoder_s2t2": str(poc_root / "encoded" / "op_s2_t2.obu"),
            "reconstructed_s2t2": str(reconstructed),
            "poc_report": str(poc_root / "poc_report.json"),
            "benchmark_report": str(
                reports / "persistent_l3t3_merge_benchmark.json"
            ),
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
