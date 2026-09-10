from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from av1_spatial_temporal.encoder import EncodeConfig, find_ffmpeg, find_ffprobe
from av1_spatial_temporal.workflow import run_poc


def main() -> None:
    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.y4m"
        subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=128x72:rate=12",
                "-frames:v",
                "12",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            check=True,
        )
        report = run_poc(
            source,
            root / "result",
            EncodeConfig(
                spatial_layers=2,
                temporal_layers=3,
                bitrate_kbps=300,
                frames=12,
                speed=10,
                threads=2,
                keyframe_distance=12,
            ),
        )
        probes: dict[str, object] = {}
        encoded = root / "result" / "encoded"
        for path in sorted(encoded.glob("op_s*_t*.obu")):
            completed = subprocess.run(
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
                    "stream=width,height,nb_read_frames",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            probes[path.name] = json.loads(completed.stdout)["streams"][0]
        print(
            json.dumps(
                {
                    "operating_points": probes,
                    "merge": report["merge"],
                    "result": report["result"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
