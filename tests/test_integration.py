from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from av1_spatial_temporal.encoder import (
    EncodeConfig,
    find_ffmpeg,
    find_ffprobe,
    find_svc_encoder,
)
from av1_spatial_temporal.workflow import run_poc


@unittest.skipUnless(
    os.environ.get("RUN_AV1_INTEGRATION") == "1",
    "set RUN_AV1_INTEGRATION=1 to run the real libaom/FFmpeg test",
)
class RealL2T3IntegrationTests(unittest.TestCase):
    def test_encode_split_merge_and_decode(self) -> None:
        ffmpeg = find_ffmpeg()
        ffprobe = find_ffprobe()
        find_svc_encoder()
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
            self.assertTrue(report["result"]["byte_identical_reconstruction"])
            self.assertTrue(report["result"]["all_operating_points_decodable"])
            self.assertGreater(report["merge"]["throughput_mib_per_second"], 0)

            expected = {
                "op_s0_t0.obu": (64, 36, 3),
                "op_s0_t1.obu": (64, 36, 6),
                "op_s0_t2.obu": (64, 36, 12),
                "op_s1_t0.obu": (128, 72, 3),
                "op_s1_t1.obu": (128, 72, 6),
                "op_s1_t2.obu": (128, 72, 12),
            }
            encoded = root / "result" / "encoded"
            for filename, dimensions_and_frames in expected.items():
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
                        str(encoded / filename),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                stream = json.loads(completed.stdout)["streams"][0]
                actual = (
                    int(stream["width"]),
                    int(stream["height"]),
                    int(stream["nb_read_frames"]),
                )
                self.assertEqual(actual, dimensions_and_frames, filename)


if __name__ == "__main__":
    unittest.main()

