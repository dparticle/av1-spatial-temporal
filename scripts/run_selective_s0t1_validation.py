"""Same-chunk test: cached S0T0 -> S0T1 -> S1T1 -> S1T0 -> S0T1.

No playback-time switching or re-encoding. Downgrade archives discarded files.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from av1_spatial_temporal.encoder import find_ffmpeg
from av1_spatial_temporal.operations import extract_operating_point, sha256_file
from av1_spatial_temporal.selective_transport import (
    extract_enhancement_layers, inspect_layer_stream, merge_operating_point,
    parse_point, plan_layer_change, required_layers,
)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def layer_name(layer):
    return f"enh_s{layer[0]}_t{layer[1]}.a1ls"


def file_stats(path, receipt, root):
    return {
        "file": Path(path).relative_to(root).as_posix(),
        **{k: receipt[k] for k in (
            "records", "file_bytes", "obu_bytes", "record_header_bytes",
            "file_header_bytes", "sha256", "layers",
        )},
    }


def run_logged(cmd, log_path, *, cwd=None):
    result = subprocess.run(
        list(map(str, cmd)), capture_output=True, text=True, errors="replace",
        timeout=60, cwd=cwd,
    )
    log_path.write_text(
        json.dumps(list(map(str, cmd)), ensure_ascii=False)
        + f"\nexit_code={result.returncode}\n" + result.stdout + result.stderr,
        encoding="utf-8",
    )
    result.check_returncode()


def decode_and_hash(ffmpeg, source, prefix, *, fps, oppoint, expected):
    raw, framehash = prefix.with_suffix(".yuv"), prefix.with_suffix(".framehash")
    run_logged([
        ffmpeg, "-hide_banner", "-loglevel", "info", "-nostdin", "-n", "-xerror",
        "-f", "obu", "-framerate", str(fps), "-c:v", "libdav1d",
        "-oppoint", str(oppoint), "-i", source,
        "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
        "-f", "rawvideo", raw,
        "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
        "-f", "framehash", "-hash", "sha256", framehash,
    ], prefix.with_suffix(".log"))
    text = framehash.read_text("utf-8")
    dimension = re.search(r"#dimensions\s+0:\s+(\d+)x(\d+)", text)
    if dimension is None:
        raise AssertionError("Decoded framehash has no dimensions")
    width, height = map(int, dimension.groups())
    rows = [line for line in text.splitlines() if line and not line.startswith("#")]
    hashes = [line.rsplit(",", 1)[-1].strip() for line in rows]
    if [width, height, len(hashes)] != expected:
        raise AssertionError(f"Unexpected decoded geometry/count: {width}, {height}, {len(hashes)}")
    if raw.stat().st_size != width * height * 3 // 2 * len(hashes):
        raise AssertionError("Unexpected raw YUV size")
    return {
        "width": width, "height": height, "fps": str(fps), "frames": len(hashes),
        "oppoint": oppoint, "raw_yuv_sha256": sha256_file(raw), "frame_sha256": hashes,
    }


def run_validation(source_run, output):
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    config = json.loads((source_run / "reports/test_configuration.json").read_text("utf-8"))
    if (config["spatial_layers"], config["temporal_layers"], config["base_operating_point"]) != (2, 3, "S0T0"):
        raise ValueError("Requires a preserved L2T3 run with default S0T0 base")
    ffmpeg = find_ffmpeg()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("server", "client", "client/active_cache", "client/receipts",
                 "reference", "stages", "discarded", "logs", "code"):
        (output / name).mkdir(exist_ok=True)
    summary = {"status": "RUNNING", "source_run": str(source_run),
               "scope": "one complete chunk; no playback-time switching; no re-encoding", "steps": []}
    try:
        for path in (
            Path(__file__), PROJECT_ROOT / "av1_spatial_temporal/selective_transport.py",
            PROJECT_ROOT / "tests/test_selective_s0t1.py",
        ):
            shutil.copy2(path, output / "code" / path.name)
        save_json(output / "source_configuration.json", config)
        original_base = source_run / "poc/transport/base.a1ls"
        original_enh = source_run / "poc/transport/enhancement.a1ls"
        originals = {str(p): sha256_file(p) for p in (original_base, original_enh)}
        server_base, server_enh = output / "server/base.a1ls", output / "server/enhancement.a1ls"
        shutil.copy2(original_base, server_base)
        shutil.copy2(original_enh, server_enh)
        client_base = output / "client/base.a1ls"
        shutil.copy2(server_base, client_base)  # simulate previously cached base
        base_receipt = inspect_layer_stream(client_base)
        save_json(output / "client/base_receipt.json", base_receipt)
        summary["base"] = base_receipt
        summary["original_enhancement"] = inspect_layer_stream(server_enh)
        full = output / "reference/full.obu"  # independent oracle; never client input
        shutil.copy2(source_run / "poc/encoded/full.obu", full)
        active_paths, active_receipts = {}, {}
        current = "S0T0"
        for index, target in enumerate(("S0T1", "S1T1", "S1T0", "S0T1"), 1):
            step_id = f"{index:02d}_{current.lower()}_to_{target.lower()}"
            stage = output / "stages" / step_id
            stage.mkdir()
            plan = plan_layer_change(current, target)
            step = {**plan, "sent": [], "discarded": [], "control_bytes": 0}
            if plan["add"]:
                transfer_dir = output / "server" / step_id
                transfer_dir.mkdir()
                packet = {"format": "a1ls-layer-transfer-v1", "current": current,
                          "target": target, "files": []}
                for values in plan["add"]:
                    layer = tuple(values)
                    path = transfer_dir / layer_name(layer)
                    receipt = extract_enhancement_layers(server_enh, path, layers=[layer])
                    packet["files"].append({"name": path.name, "receipt": receipt})
                    step["sent"].append(file_stats(path, receipt, output))
                manifest = transfer_dir / "transfer.json"
                save_json(manifest, packet)
                local_manifest = output / "client/receipts" / f"{step_id}.json"
                shutil.copy2(manifest, local_manifest)
                step["control_bytes"] = local_manifest.stat().st_size
                step["transfer_manifest"] = local_manifest.relative_to(output).as_posix()
                received = json.loads(local_manifest.read_text("utf-8"))
                for values, entry in zip(plan["add"], received["files"], strict=True):
                    layer = tuple(values)
                    local = output / "client/active_cache" / entry["name"]
                    shutil.copy2(transfer_dir / entry["name"], local)
                    active_paths[layer], active_receipts[layer] = local, entry["receipt"]
            for values in plan["drop"]:
                layer = tuple(values)
                path, receipt = active_paths.pop(layer), active_receipts.pop(layer)
                archived = output / "discarded" / f"{step_id}_{path.name}"
                if inspect_layer_stream(path) != receipt:
                    raise AssertionError("Cached file changed before discard")
                path.rename(archived)  # exact new test file; archive instead of deleting
                step["discarded"].append(file_stats(archived, receipt, output))
            if set(active_paths) != required_layers(target) - {(0, 0)}:
                raise AssertionError("Client active cache has wrong layers")
            keys = sorted(active_paths)
            merged = stage / "merged.obu"
            step["merged"] = merge_operating_point(
                client_base, [active_paths[k] for k in keys], merged,
                target=target, base_receipt=base_receipt,
                enhancement_receipts=[active_receipts[k] for k in keys],
            )
            step["active_cache"] = [file_stats(active_paths[k], active_receipts[k], output) for k in keys]
            s, t = parse_point(target)
            reference = output / "reference" / f"{step_id}.obu"
            extract_operating_point(full, reference, max_spatial_id=s, max_temporal_id=t)
            if merged.read_bytes() != reference.read_bytes():
                raise AssertionError(f"{target}: OBU differs from direct extraction")
            fps = Fraction(str(config["source_fps"])) / (2 ** (2 - t))
            oppoint = (1 - s) * 3 + (2 - t)  # ordering of the project's L2T3 encoder
            expected = config["expected_operating_points"][target]
            step["decode"] = {
                "merged": decode_and_hash(ffmpeg, merged, stage / "merged_decoded",
                                          fps=fps, oppoint=oppoint, expected=expected),
                "reference": decode_and_hash(ffmpeg, reference, stage / "reference_decoded",
                                             fps=fps, oppoint=oppoint, expected=expected),
            }
            if step["decode"]["merged"]["frame_sha256"] != step["decode"]["reference"]["frame_sha256"]:
                raise AssertionError(f"{target}: decoded frames differ")
            step.update(
                byte_identical_to_direct_extraction=True,
                all_decoded_frame_pixels_identical=True,
                transferred_a1ls_bytes=sum(x["file_bytes"] for x in step["sent"]),
                discarded_a1ls_bytes=sum(x["file_bytes"] for x in step["discarded"]),
                discarded_obu_bytes=sum(x["obu_bytes"] for x in step["discarded"]),
            )
            step["new_transferred_bytes"] = step["transferred_a1ls_bytes"] + step["control_bytes"]
            save_json(stage / "step.json", step)
            summary["steps"].append(step)
            print(f"{current} -> {target}: new={step['new_transferred_bytes']} B, "
                  f"discard={step['discarded_a1ls_bytes']} B, decoded={expected[2]} frames", flush=True)
            current = target

        if summary["steps"][2]["new_transferred_bytes"] != 0:
            raise AssertionError("Downgrade unexpectedly downloaded data")
        first, last = summary["steps"][0], summary["steps"][-1]
        if last["add"] != [[0, 1]] or last["drop"] != [[1, 0]]:
            raise AssertionError("Return to S0T1 has wrong transfer/drop plan")
        if len(last["sent"]) != 1 or last["sent"][0]["sha256"] != first["sent"][0]["sha256"]:
            raise AssertionError("Return must retransmit the discarded S0T1 enhancement")
        if last["merged"]["sha256"] != first["merged"]["sha256"]:
            raise AssertionError("Returning to S0T1 changed the reconstructed chunk")
        if inspect_layer_stream(client_base) != base_receipt:
            raise AssertionError("Cached base changed")
        if any(sha256_file(Path(path)) != digest for path, digest in originals.items()):
            raise AssertionError("Original test inputs changed")
        inventory = summary["original_enhancement"]
        # Source inventory counts each exact-layer file once; network totals below
        # count every transmission, including the S0T1 retransmission after eviction.
        unique_sent = {x["sha256"]: x for st in summary["steps"] for x in st["sent"]}
        selected_records = sum(x["records"] for x in unique_sent.values())
        selected_obu_bytes = sum(x["obu_bytes"] for x in unique_sent.values())
        omitted = {key: value for key, value in inventory["layers"].items()
                   if key not in ("S0T1", "S1T0", "S1T1")}
        if selected_records + sum(v["records"] for v in omitted.values()) != inventory["records"]:
            raise AssertionError("Record accounting mismatch")
        if selected_obu_bytes + sum(v["obu_bytes"] for v in omitted.values()) != inventory["obu_bytes"]:
            raise AssertionError("OBU accounting mismatch")
        summary.update(
            never_transferred_layers=omitted,
            total_new_transferred_bytes=sum(st["new_transferred_bytes"] for st in summary["steps"]),
            total_new_a1ls_bytes=sum(st["transferred_a1ls_bytes"] for st in summary["steps"]),
            final_active_layers=[list(k) for k in sorted(active_paths)],
            returned_s0t1_identical=True,
            original_inputs_unchanged=True,
        )
        run_logged(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
            output / "logs/unittest.log", cwd=PROJECT_ROOT,
        )
        summary["status"] = "PASS"
    except Exception as exc:
        summary.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(output / "SUMMARY.json", summary)
        paths = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
        (output / "SHA256SUMS.txt").write_text(
            "".join(f"{sha256_file(p)}  {p.relative_to(output).as_posix()}\n" for p in paths), encoding="utf-8",
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=PROJECT_ROOT / "test_results/20260909_input_option_l2t3_01")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "test_results" / datetime.now().strftime("%Y%m%d_%H%M%S_layer_changes"),
        help="New directory; existing directories are rejected",
    )
    args = parser.parse_args()
    report = run_validation(args.source_run, args.output)
    print(f"{report['status']}: {args.output.resolve() / 'SUMMARY.json'}")


if __name__ == "__main__":
    main()
