from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from av1_spatial_temporal.layer_stream import merge_layer_streams, split_obu_stream
from av1_spatial_temporal.obu import OBU_FRAME, OBU_SEQUENCE_HEADER, make_obu


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark A1LS streaming merge")
    parser.add_argument("--size-mib", type=int, default=32)
    args = parser.parse_args()
    if args.size_mib < 1:
        parser.error("--size-mib must be positive")

    payload = bytes(range(256)) * 128
    target_bytes = args.size_mib * 1024 * 1024
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.obu"
        base = root / "base.a1ls"
        enhancement = root / "enhancement.a1ls"
        reconstructed = root / "reconstructed.obu"
        with source.open("wb") as stream:
            stream.write(make_obu(OBU_SEQUENCE_HEADER, b"benchmark"))
            sequence = 0
            while stream.tell() < target_bytes:
                spatial_id = sequence % 2
                temporal_id = (0, 2, 1, 2)[sequence % 4]
                stream.write(
                    make_obu(
                        OBU_FRAME,
                        payload,
                        spatial_id=spatial_id,
                        temporal_id=temporal_id,
                    )
                )
                sequence += 1
        split_obu_stream(source, base, enhancement)
        report = merge_layer_streams((base, enhancement), reconstructed)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
