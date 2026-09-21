from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .encoder import EncodeConfig, EncoderError, encode_av1_svc, parse_fraction
from .layer_stream import LayerStreamError, merge_layer_streams, split_obu_stream
from .obu import ObuParseError, analyze_obu_stream
from .operations_v1 import (
    VerificationError,
    decode_obu_stream,
    extract_operating_point,
    unpack_layer_stream,
    verify_reconstruction,
)
from .workflow_v1 import ToolPaths, run_poc


def _print_report(report: dict[str, object]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _add_encode_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="任意 FFmpeg 可读取的输入视频")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-layers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--temporal-layers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--bitrate-kbps", type=int,
        help="目标总码率（kbps）；省略时使用源文件平均码率的 1.1 倍",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=parse_fraction, help="例如 30 或 30000/1001")
    parser.add_argument("--frames", type=int, help="只编码前 N 帧")
    parser.add_argument("--speed", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keyframe-distance", type=int, default=120)
    parser.add_argument("--min-q", type=int, default=2)
    parser.add_argument("--max-q", type=int, default=52)
    parser.add_argument(
        "--no-test-decode",
        action="store_false",
        dest="test_decode",
        help="关闭 libaom 编码过程中的自检解码",
    )
    parser.set_defaults(test_decode=True)
    parser.add_argument("--encoder", help="svc_encoder_rtc 可执行文件")
    parser.add_argument("--ffmpeg", help="ffmpeg 可执行文件")
    parser.add_argument("--ffprobe", help="ffprobe 可执行文件")
    parser.add_argument("--force", action="store_true", help="覆盖已有输出")


def _config_from_args(args: argparse.Namespace) -> EncodeConfig:
    return EncodeConfig(
        spatial_layers=args.spatial_layers,
        temporal_layers=args.temporal_layers,
        bitrate_kbps=args.bitrate_kbps,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=args.frames,
        speed=args.speed,
        threads=args.threads,
        keyframe_distance=args.keyframe_distance,
        min_q=args.min_q,
        max_q=args.max_q,
        test_decode=args.test_decode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="av1-svc",
        description="AV1 分辨率 + 帧率二维可伸缩编码与快速分层合并工具",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="编码 AV1 SVC OBU 流")
    _add_encode_arguments(encode_parser)

    analyze_parser = subparsers.add_parser("analyze", help="分析低开销 AV1 OBU 流")
    analyze_parser.add_argument("input", type=Path)

    extract_parser = subparsers.add_parser(
        "extract", help="提取累计空间/时间操作点"
    )
    extract_parser.add_argument("input", type=Path)
    extract_parser.add_argument("output", type=Path)
    extract_parser.add_argument("--max-spatial-id", type=int, default=0)
    extract_parser.add_argument("--max-temporal-id", type=int, default=0)
    extract_parser.add_argument("--force", action="store_true")

    split_parser = subparsers.add_parser(
        "split", help="拆成带顺序信息的基层与增强层 A1LS 通道"
    )
    split_parser.add_argument("input", type=Path)
    split_parser.add_argument("--base", type=Path, required=True)
    split_parser.add_argument("--enhancement", type=Path, required=True)
    split_parser.add_argument("--base-spatial-id", type=int, default=0)
    split_parser.add_argument("--base-temporal-id", type=int, default=0)
    split_parser.add_argument("--force", action="store_true")

    merge_parser = subparsers.add_parser(
        "merge", help="流式归并 A1LS 通道，不重新编码"
    )
    merge_parser.add_argument("inputs", type=Path, nargs="+")
    merge_parser.add_argument("-o", "--output", type=Path, required=True)
    merge_parser.add_argument("--force", action="store_true")

    unpack_parser = subparsers.add_parser(
        "unpack", help="把单个 A1LS 通道还原为其携带的 OBU"
    )
    unpack_parser.add_argument("input", type=Path)
    unpack_parser.add_argument("output", type=Path)
    unpack_parser.add_argument("--force", action="store_true")

    decode_parser = subparsers.add_parser("decode-check", help="用 FFmpeg 验证解码")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("--ffmpeg")

    verify_parser = subparsers.add_parser(
        "verify", help="验证重组结果逐字节一致并可解码"
    )
    verify_parser.add_argument("original", type=Path)
    verify_parser.add_argument("reconstructed", type=Path)
    verify_parser.add_argument("--no-decode", action="store_true")
    verify_parser.add_argument("--ffmpeg")

    poc_parser = subparsers.add_parser(
        "poc", help="一键执行 L2T3 编码、拆层、快速合层与验证"
    )
    _add_encode_arguments(poc_parser)
    poc_parser.add_argument("--base-spatial-id", type=int, default=0)
    poc_parser.add_argument("--base-temporal-id", type=int, default=0)
    return parser


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "encode":
        return encode_av1_svc(
            args.input,
            args.output_dir,
            _config_from_args(args),
            encoder=args.encoder,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            force=args.force,
        )
    if args.command == "analyze":
        return analyze_obu_stream(args.input.resolve())
    if args.command == "extract":
        return extract_operating_point(
            args.input,
            args.output,
            max_spatial_id=args.max_spatial_id,
            max_temporal_id=args.max_temporal_id,
            force=args.force,
        )
    if args.command == "split":
        return split_obu_stream(
            args.input,
            args.base,
            args.enhancement,
            max_base_spatial_id=args.base_spatial_id,
            max_base_temporal_id=args.base_temporal_id,
            force=args.force,
        )
    if args.command == "merge":
        return merge_layer_streams(args.inputs, args.output, force=args.force)
    if args.command == "unpack":
        return unpack_layer_stream(args.input, args.output, force=args.force)
    if args.command == "decode-check":
        return decode_obu_stream(args.input, ffmpeg=args.ffmpeg)
    if args.command == "verify":
        return verify_reconstruction(
            args.original,
            args.reconstructed,
            decode=not args.no_decode,
            ffmpeg=args.ffmpeg,
        )
    if args.command == "poc":
        return run_poc(
            args.input,
            args.output_dir,
            _config_from_args(args),
            base_spatial_id=args.base_spatial_id,
            base_temporal_id=args.base_temporal_id,
            tools=ToolPaths(
                encoder=args.encoder, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
            ),
            force=args.force,
        )
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = _run(args)
    except (
        EncoderError,
        LayerStreamError,
        ObuParseError,
        VerificationError,
        FileNotFoundError,
        FileExistsError,
        ValueError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0
