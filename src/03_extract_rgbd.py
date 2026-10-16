"""RGB-D 提取：将 RealSense .db3 转为对齐的 PNG 数据集、内参与时间戳。"""

import argparse
import csv
import json
from pathlib import Path
import random

import cv2
import numpy as np
import pyrealsense2 as rs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/raw/chair_test_01.db3"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/extracted"


def save_pair(color_path, depth_path, color_image, depth_image):
    """两张图片都成功写入才算一对，失败时清理本对的文件。"""
    if color_path.exists() or depth_path.exists():
        raise FileExistsError(f"禁止覆盖已有 RGB-D 图片：{color_path.name}")
    try:
        if not cv2.imwrite(str(color_path), color_image):
            raise OSError(f"RGB PNG 写入失败：{color_path}")
        # 直接保存 uint16 深度值，不缩放、不归一化、不做伪彩色。
        if not cv2.imwrite(str(depth_path), depth_image):
            raise OSError(f"Depth PNG 写入失败：{depth_path}")
    except Exception:
        color_path.unlink(missing_ok=True)
        depth_path.unlink(missing_ok=True)
        raise


def check_saved_pairs(color_dir, depth_dir, csv_path, count):
    """核对两路 PNG 与 CSV 的连续编号，确保严格一一对应。"""
    expected = [f"{i:06d}" for i in range(count)]
    color_ids = sorted(path.stem for path in color_dir.glob("*.png"))
    depth_ids = sorted(path.stem for path in depth_dir.glob("*.png"))
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        timestamp_ids = [row["frame_id"] for row in csv.DictReader(csv_file)]
    if color_ids != expected or depth_ids != expected or timestamp_ids != expected:
        raise ValueError("RGB / Depth 文件或 CSV 编号不一致，数据集未通过一一对应检查。")
    print(f"RGB / Depth / timestamps.csv 一一对应检查通过：{count} 对。")


def main(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT):
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    csv_path = output_dir / "timestamps.csv"
    intrinsics_path = output_dir / "camera_intrinsics.json"
    pipeline = None
    playback = None
    resolved_profile = None
    active_profile = None
    started = False
    read_count = saved_count = skipped_count = 0
    color_size = depth_size = depth_dtype = depth_scale = intrinsics = None
    success = False

    print(f"输入文件：{input_path}")
    print(f"输出目录：{output_dir}")
    try:
        if not input_path.is_file() or input_path.suffix.lower() != ".db3":
            raise ValueError("输入必须是已存在的 RealSense .db3 文件。")
        # 不覆盖旧数据，避免图片、时间戳和内参来自不同提取批次。
        if output_dir.exists():
            if not output_dir.is_dir():
                raise FileExistsError(f"输出路径已存在且不是目录：{output_dir}")
            if any(output_dir.iterdir()):
                raise FileExistsError("输出目录非空，禁止覆盖已有数据；请用 --output 指定新目录或空目录。")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(str(input_path), repeat_playback=False)
        # 仅回放视频流；IMU 保留在原始录制文件中，图像提取不读取 IMU。
        config.enable_stream(rs.stream.color, 0, 0, rs.format.bgr8, 0)
        config.enable_stream(rs.stream.depth, 0, 0, rs.format.z16, 0)
        resolved_profile = config.resolve(rs.pipeline_wrapper(pipeline))
        playback = resolved_profile.get_device().as_playback()
        # 在启动前关闭实时节奏，避免磁盘写入较慢时按实时播放丢帧。
        playback.set_real_time(False)
        active_profile = pipeline.start(config)
        started = True
        depth_scale = active_profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)

        color_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame_id", "color_timestamp_ms", "depth_timestamp_ms"])
            while True:
                # 先取队列里的帧，再检查停止状态，避免遗漏文件末尾的已排队帧。
                try:
                    available, frames = pipeline.try_wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    if playback.current_status() == rs.playback_status.stopped:
                        print("已到达录制文件结尾。")
                        break
                    raise
                if not available:
                    if playback.current_status() == rs.playback_status.stopped:
                        print("已到达录制文件结尾。")
                        break
                    raise TimeoutError("回放等待 frame 超时，但尚未到达文件结尾。")

                read_count += 1
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    skipped_count += 1
                    continue

                # 保留原始两路传感器时间戳，单位为毫秒，不使用提取时的系统时间。
                color_timestamp = color_frame.get_timestamp()
                depth_timestamp = depth_frame.get_timestamp()
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    skipped_count += 1
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                if depth_image.dtype != np.uint16 or depth_image.ndim != 2:
                    raise ValueError(f"深度必须是二维 uint16 数组，实际为 {depth_image.dtype} {depth_image.shape}")
                if color_image.dtype != np.uint8 or color_image.ndim != 3 or color_image.shape[2] != 3:
                    raise ValueError("Color 必须是三通道 uint8 BGR 图像。")
                if color_image.shape[:2] != depth_image.shape:
                    raise ValueError("对齐后的 Color 与 Depth 分辨率不一致。")

                current = color_frame.profile.as_video_stream_profile().get_intrinsics()
                current_intrinsics = {
                    "width": current.width, "height": current.height,
                    "fx": current.fx, "fy": current.fy,
                    "cx": current.ppx, "cy": current.ppy,
                }
                if intrinsics is None:
                    # Depth 已对齐到 Color，因此后续 RGB-D 使用 Color 相机的内参。
                    intrinsics = current_intrinsics
                    color_size = (current.width, current.height)
                    depth_size = (depth_image.shape[1], depth_image.shape[0])
                    depth_dtype = str(depth_image.dtype)
                    metadata = {
                        **intrinsics,
                        "intrinsics_unit": "pixels",
                        "depth_scale": depth_scale,
                        "depth_scale_unit": "meters_per_depth_unit",
                        "depth_dtype": "uint16",
                        "depth_aligned_to": "color",
                        "depth_to_meters": "depth_uint16 * depth_scale",
                        "timestamp_unit": "milliseconds",
                        "timestamp_reference": "RealSense SDK frame timestamps, not necessarily Unix time",
                    }
                    with intrinsics_path.open("x", encoding="utf-8") as intrinsics_file:
                        intrinsics_file.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
                elif current_intrinsics != intrinsics:
                    raise ValueError("录制中的 Color 内参发生变化，无法用单个内参文件表示该数据集。")

                # 按成功保存的对数编号，缺失帧不会造成编号断档。
                frame_id = f"{saved_count:06d}"
                color_path = color_dir / f"{frame_id}.png"
                depth_path = depth_dir / f"{frame_id}.png"
                save_pair(color_path, depth_path, color_image, depth_image)
                try:
                    writer.writerow([frame_id, color_timestamp, depth_timestamp])
                except Exception:
                    color_path.unlink(missing_ok=True)
                    depth_path.unlink(missing_ok=True)
                    raise
                saved_count += 1
                if saved_count % 100 == 0:
                    print(f"已保存 {saved_count} 对 RGB-D，跳过 {skipped_count} 组。")

        if saved_count == 0:
            raise ValueError("没有提取到有效 RGB-D 对，请检查原始录制文件。")
        check_saved_pairs(color_dir, depth_dir, csv_path, saved_count)
        sample_path = depth_dir / f"{random.randrange(saved_count):06d}.png"
        # IMREAD_UNCHANGED 是必要的，否则 OpenCV 默认读取可能转换为 8-bit。
        sample = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
        if sample is None or sample.dtype != np.uint16 or sample.ndim != 2:
            raise ValueError(f"保存后的 Depth PNG 验证失败，必须为二维 uint16：{sample_path}")
        if sample.shape != (depth_size[1], depth_size[0]):
            raise ValueError(f"保存后的 Depth PNG 分辨率验证失败：{sample_path}")
        print(f"随机 Depth PNG 验证通过：{sample_path.name}，dtype=uint16。")
        success = True

    except KeyboardInterrupt:
        print("\n提取已中断，输出仅包含已写入的部分数据。")
    except Exception as exc:
        print(f"错误：提取失败（{type(exc).__name__}）：{exc}")
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                if started:
                    success = False
                    print(f"错误：停止回放失败：{exc}")
        playback = active_profile = resolved_profile = pipeline = None

    print(f"\n提取状态：{'完成' if success else '未完成'}")
    print(f"总读取 frameset 数：{read_count}")
    print(f"成功保存 RGB-D 对数量：{saved_count}")
    print(f"跳过数量：{skipped_count}")
    print(f"RGB 分辨率（width, height）：{color_size}")
    print(f"Depth 分辨率（width, height）：{depth_size}")
    print(f"Depth dtype：{depth_dtype}")
    print(f"Depth scale：{depth_scale} 米/深度单位")
    print(f"Color camera intrinsics：{intrinsics}")
    return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 RealSense .db3 提取为 RGB-D PNG 数据集")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 .db3 文件")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="空的输出目录")
    args = parser.parse_args()
    raise SystemExit(main(args.input, args.output))
