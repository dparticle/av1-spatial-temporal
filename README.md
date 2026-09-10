# AV1 二维可伸缩编码与快速分层合并

本项目是一个独立的 AV1 SVC（Scalable Video Coding）验证工程，支持同时按分辨率和帧率伸缩。默认配置为 **L2T3**：2 个空间层、3 个时间层。编码由官方 libaom 示例程序 `svc_encoder_rtc` 完成；Python 代码负责输入转换、参数配置、OBU 分析、分层传输和无重编码合并。

本项目不依赖、也不会修改原来的 `h265_temporal` 项目。

## 能实现什么

以 1920×1080、30 fps 输入为例，默认 L2T3 可得到以下累计操作点：

| 操作点 | 典型分辨率 | 典型帧率 | 文件 |
|---|---:|---:|---|
| S0T0 | 960×540 | 7.5 fps | `op_s0_t0.obu` |
| S0T1 | 960×540 | 15 fps | `op_s0_t1.obu` |
| S0T2 | 960×540 | 30 fps | `op_s0_t2.obu` |
| S1T0 | 1920×1080 | 7.5 fps | `op_s1_t0.obu` |
| S1T1 | 1920×1080 | 15 fps | `op_s1_t1.obu` |
| S1T2 | 1920×1080 | 30 fps | `op_s1_t2.obu` |

实际帧率由输入帧率按 T0/T1/T2 约 1/4、1/2、1 倍累计得到。空间基层默认按宽高各 1/2 缩放。L2T3 使用 libaom layering mode 7，增强空间层引用基层，属于真正的 SVC，而不是两路互不相关的 simulcast。

`full.obu` 是完整二维可伸缩码流。项目还可以把它拆成：

- `base.a1ls`：全局 OBU + 指定基层（默认 S0T0）；
- `enhancement.a1ls`：其余空间和时间增强 OBU；
- `base.obu`：从基层通道直接取出的可解码低分辨率、低帧率码流；
- `reconstructed.obu`：基层和增强层快速归并后的完整码流，与 `full.obu` 逐字节相同。

合并过程不解码、不做运动估计、不重新编码，只按 64 位 OBU 序号顺序归并并拷贝原始字节，时间复杂度 O(n)，适合在代码运行时进行。

## 环境

- Python 3.11 或更新版本；
- FFmpeg 和 ffprobe；
- 使用 AV1 编码器构建的官方 libaom `svc_encoder_rtc`。

工具按以下顺序寻找编码器：

1. 命令行 `--encoder`；
2. 环境变量 `AOM_SVC_ENCODER`；
3. `PATH`；
4. 同级目录 `../aom/aom_build/examples/svc_encoder_rtc.exe` 或 `../aom/build/examples/svc_encoder_rtc`。

当前机器已存在第 4 种目录结构，因此无需复制 libaom 到本项目。

## 快速开始

在 `E:\graduate\av1_spatial_temporal` 中运行：

```powershell
python -m av1_spatial_temporal --help
```

对输入视频执行完整 L2T3 流程：

```powershell
python -m av1_spatial_temporal poc input.mp4 `
  -o demo_output `
  --spatial-layers 2 `
  --temporal-layers 3 `
  --bitrate-kbps 3000
```

该命令依次完成编码、拆层、基层物化、合层、SHA-256 逐字节校验，以及所有操作点的 FFmpeg 解码校验。最终结果写入 `demo_output/poc_report.json`。

也可以安装为命令：

```powershell
python -m pip install -e .
av1-svc --help
```

## 分步使用

只编码：

```powershell
python -m av1_spatial_temporal encode input.mp4 -o encoded \
  --spatial-layers 2 --temporal-layers 3 --bitrate-kbps 3000
```

把完整 OBU 拆成基层和增强层传输通道：

```powershell
python -m av1_spatial_temporal split encoded/full.obu \
  --base transport/base.a1ls \
  --enhancement transport/enhancement.a1ls
```

运行时快速合并：

```powershell
python -m av1_spatial_temporal merge \
  transport/base.a1ls transport/enhancement.a1ls \
  -o reconstructed.obu
```

验证逐字节一致和可解码：

```powershell
python -m av1_spatial_temporal verify \
  encoded/full.obu reconstructed.obu
```

提取或分析任意累计操作点：

```powershell
python -m av1_spatial_temporal extract encoded/full.obu s0t1.obu \
  --max-spatial-id 0 --max-temporal-id 1
python -m av1_spatial_temporal analyze encoded/full.obu
```

PowerShell 可把示例中的行尾 `\` 改为反引号，或直接写成一行。

## 在程序中在线合并

`OrderedLayerMerger` 接受来自两个网络通道、可能短暂乱序的 `LayerRecord`。只要下一个连续序号到达，它就立刻释放可写入解码器的 OBU：

```python
from av1_spatial_temporal.layer_stream import OrderedLayerMerger

merger = OrderedLayerMerger(max_pending=4096)

def on_record_from_either_channel(record, decoder_input):
    for ready in merger.push(record):
        decoder_input.write(ready.raw)
```

文件命令 `merge` 则对两个已排序通道执行常量级缓存的 k 路流式归并，并在结束时校验记录数、字节数和 SHA-256。

## A1LS 格式说明

`.a1ls` 是本项目用于分层传输的轻量封装，不是 AV1 标准容器。它给每个原始 OBU 增加序号、时间单元号、空间层 ID 和时间层 ID，使两个独立通道可以无歧义恢复原始交织顺序。记录中的 AV1 OBU 字节没有被修改。

基层通道可用 `unpack` 物化成 `.obu` 后直接解码；增强层通常依赖基层，不能单独解码。若系统使用 RTP/QUIC，可把 A1LS 记录字段映射到已有包头，合并算法保持不变。

## 支持的层组合

当前直接映射官方 `svc_encoder_rtc` 模式：L1T1、L1T2、L1T3、L2T1、L3T1、L2T3、L3T3。官方示例没有对应的 L2T2 真 SVC 模式，因此工具会明确拒绝 L2T2，避免悄悄改成 simulcast。

## 测试

运行纯 Python 单元测试：

```powershell
python -m unittest discover -s tests -v
```

运行真实 libaom/FFmpeg 端到端测试：

```powershell
$env:RUN_AV1_INTEGRATION = "1"
python -m unittest tests.test_integration -v
```

端到端测试生成 128×72、12 fps 的测试图，编码 L2T3，随后验证基层和六个累计操作点均可解码，并确认合并结果与原始完整码流逐字节一致。

## 持久化完整测试

以下命令会运行真实的 libaom/FFmpeg 完整测试，并把输入、编码码流、分层通道、合并结果、解码检查、日志、性能基准和 SHA-256 校验信息全部保存在 `test_results` 下。请在项目根目录 `E:\graduate\av1_spatial_temporal` 中运行。

运行 L2T3（2 档分辨率 × 3 档帧率）完整测试：

```powershell
python scripts\run_preserved_validation.py
```

运行 L3T3（3 档分辨率 × 3 档帧率）完整测试：

```powershell
python scripts\run_preserved_l3t3_validation.py
```

也可以使用 MP4、Y4M 或其他 FFmpeg 可读取的视频作为测试内容：

```powershell
python scripts\run_preserved_validation.py --input input.mp4
python scripts\run_preserved_l3t3_validation.py --input input.mp4
```

指定 `--input` 后，原始输入会复制到本次结果的 `input/original/`，然后由 FFmpeg 转换成测试使用的固定 Y4M：L2T3 为 320×180、24 fps、最多 48 帧，L3T3 为 384×216、24 fps、最多 48 帧。输入不足 48 帧时，固定帧数检查会失败；未指定 `--input` 时仍然自动生成 `testsrc2` 测试图。

默认使用时间戳作为结果目录名，并执行 16 MiB 的拆分/合并性能基准。也可以指定结果名称和基准大小：

```powershell
python scripts\run_preserved_validation.py `
  --name my_l2t3_run `
  --benchmark-mib 64

python scripts\run_preserved_l3t3_validation.py `
  --name my_l3t3_run `
  --benchmark-mib 64
```

可用参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--output-root` | `test_results` | 所有测试结果的父目录 |
| `--input` | 自动生成 `testsrc2` | 可选的 MP4、Y4M 或其他 FFmpeg 可读取的视频；原文件和规范化 Y4M 都会保留 |
| `--name` | 当前时间戳加 `l2t3` 或 `l3t3` | 本次测试的结果目录名；已有同名目录时拒绝覆盖 |
| `--benchmark-mib` | `16` | 用于测试拆分/合并吞吐率的合成码流大小（MiB）；不改变视频分辨率、帧率、码率或编码质量 |

每次测试的主要结果如下：

| 文件或目录 | 内容 |
|---|---|
| `SUMMARY.json` | 总体 `PASS`/`FAIL` 状态和每项检查结果 |
| `MANIFEST.json` | 所有保留文件的路径、大小和 SHA-256 |
| `SHA256SUMS.txt` | 可用于独立校验结果文件完整性的 SHA-256 列表 |
| `logs/` | FFmpeg、libaom、CLI 和完整测试套件的原始运行日志 |
| `reports/` | 环境、配置、解码探测、像素哈希和合并性能等结构化报告 |
| `poc/` | 完整 OBU、所有操作点、A1LS 基层/增强层、基层 OBU 和逐字节重建码流 |
| `direct_s0t0/` 或 `direct_extract/` | 从完整码流直接提取的 S0T0，以及 L3T3 测试中的 S1T1 |
| `benchmark/` | 指定大小的合成码流、拆分通道和合并后码流 |

即使某项测试失败，脚本也会尽量保留已生成的日志、`SUMMARY.json` 和清单，便于定位问题。
