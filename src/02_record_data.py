"""RGB-D 录制：保存原始 Color、Depth 和可用的 IMU 数据到 .db3。"""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/raw/chair_test_01.db3"


def select_motion_profiles(device):
    """从实际支持的 profile 中选取 IMU 配置，不假定固定帧率。"""
    selected = []
    try:
        profiles = [p for s in device.query_sensors() for p in s.get_stream_profiles()]
        for stream in (rs.stream.accel, rs.stream.gyro):
            candidates = [
                p for p in profiles
                if p.stream_type() == stream and p.is_motion_stream_profile()
            ]
            if not candidates:
                print(f"提示：设备没有可用的 {stream} profile，仅录制 Color / Depth。")
                return []
            # 优先较低的有效帧率，减少 IMU 录制的数据量。
            profile = min(candidates, key=lambda p: p.fps())
            selected.append(profile)
            print(f"IMU 候选配置：{profile}")
    except RuntimeError as exc:
        print(f"警告：查询 IMU 配置失败，仅录制 Color / Depth：{exc}")
        return []
    return selected


def make_config(serial, preferred, motion_profiles):
    config = rs.config()
    config.enable_device(serial)
    if preferred:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    else:
        # 明确使用视频流重载，避免 format 枚举被当作 stream_index。
        config.enable_stream(rs.stream.color, 0, 0, rs.format.bgr8, 0)
        config.enable_stream(rs.stream.depth, 0, 0, rs.format.z16, 0)
    for profile in motion_profiles:
        config.enable_stream(
            profile.stream_type(), profile.stream_index(), profile.format(), profile.fps()
        )
    return config


def main(output_path=DEFAULT_OUTPUT):
    output_path = Path(output_path).resolve()
    pipeline = None
    active_profile = None
    started = False
    closed = False
    start_time = None
    duration = 0.0
    rgb_frames = 0
    depth_frames = 0
    imu_enabled = False
    print("提示：请先关闭 RealSense Viewer 和其他占用相机的程序。")
    print(f"文件保存位置：{output_path}")

    try:
        if output_path.suffix.lower() != ".db3":
            print("错误：录制仅支持 .db3 输出文件。")
            return 1
        # 防止再次运行时覆盖已经录好的原始数据。
        # 共享的 data/raw 可以存在；同名录制目录和 SQLite 附属文件不能复用。
        related_paths = [
            output_path, output_path.with_suffix(""),
            *[Path(str(output_path) + suffix) for suffix in ("-wal", "-shm", "-journal")],
        ]
        for path in related_paths:
            if path.exists():
                print(f"错误：输出文件或相关录制路径已存在：{path}")
                print("禁止覆盖已有数据，请用 --output 指定新的录制文件名。")
                return 1
        output_path.parent.mkdir(parents=True, exist_ok=True)

        context = rs.context()
        devices = context.query_devices()
        device = next(
            (d for d in devices if "D435I" in d.get_info(rs.camera_info.name).upper()),
            None,
        )
        if device is None:
            print("错误：未检测到 D435i，请检查 USB 连接和供电。")
            return 1
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"设备：{device.get_info(rs.camera_info.name)}，序列号：{serial}")

        pipeline = rs.pipeline(context)
        preferred = True
        config = make_config(serial, preferred, [])
        if not config.can_resolve(rs.pipeline_wrapper(pipeline)):
            preferred = False
            print("提示：首选视频配置不可用，尝试 SDK 自动选择 BGR8 / Z16 配置。")

        motion_profiles = select_motion_profiles(device)
        # IMU 配置或启动失败时，重新创建管线，仅启用 Color / Depth。
        while True:
            try:
                config = make_config(serial, preferred, motion_profiles)
                if not config.can_resolve(rs.pipeline_wrapper(pipeline)):
                    raise RuntimeError("设备无法解析所请求的流组合")
                # SDK 在设备层录制原始流，显示图像上的文字和伪彩色不会写入文件。
                config.enable_record_to_file(str(output_path))
                active_profile = pipeline.start(config)
                started = True
                start_time = time.perf_counter()
                break
            except RuntimeError as exc:
                if not motion_profiles:
                    print(f"错误：Color / Depth 录制启动失败：{exc}")
                    print("请检查 USB 连接、文件写入权限，并关闭 RealSense Viewer。")
                    return 1
                print(f"警告：包含 IMU 的录制启动失败，将回退到 Color / Depth：{exc}")
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass  # 启动未完成时，stop 可能提示管线尚未启动。
                pipeline = None
                # 这里只清理本次失败启动新建的文件；入口已拒绝已有文件。
                output_path.unlink(missing_ok=True)
                pipeline = rs.pipeline(context)
                motion_profiles = []

        streams = active_profile.get_streams()
        imu_enabled = all(
            any(p.stream_type() == stream for p in streams)
            for stream in (rs.stream.accel, rs.stream.gyro)
        )
        for profile in streams:
            print(f"已启用：{profile}")
        print(f"IMU：{'已启用加速度计和陀螺仪' if imu_enabled else '未启用'}")
        print("开始录制，请点击图像窗口后按 q 停止，也支持 Ctrl+C。")
        cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
        last_time = None
        last_color_number = None
        last_depth_number = None
        invalid_frames = 0

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as exc:
                print(f"错误：获取 frame 失败（超时或相机断开），停止录制：{exc}")
                return 1
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            # 混合视频和 IMU 流时，frameset 可能重复携带上一张视频帧。
            new_color = bool(color_frame) and color_frame.get_frame_number() != last_color_number
            if new_color:
                rgb_frames += 1
                last_color_number = color_frame.get_frame_number()
            if depth_frame and depth_frame.get_frame_number() != last_depth_number:
                depth_frames += 1
                last_depth_number = depth_frame.get_frame_number()
            if not color_frame or not depth_frame:
                invalid_frames += 1
                if invalid_frames == 1:
                    print("警告：未获得有效 RGB / Depth frame，正在重试。")
                if invalid_frames >= 30:
                    print("错误：连续 30 次缺少有效视频帧，停止录制。")
                    return 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            invalid_frames = 0
            if not new_color:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # 仅对显示副本添加文字，深度原始 Z16 数组不做原地修改。
            color_display = np.asanyarray(color_frame.get_data()).copy()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_display = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            now = time.perf_counter()
            fps = 0.0 if last_time is None else 1.0 / max(now - last_time, 1e-9)
            last_time = now
            text = f"RECORDING  {now - start_time:.1f}s  FPS: {fps:.1f}"
            for image in (color_display, depth_display):
                cv2.putText(
                    image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2,
                )
            cv2.imshow("RGB", color_display)
            cv2.imshow("Depth", depth_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止录制。")
    except Exception as exc:
        print(f"错误：录制异常（{type(exc).__name__}）：{exc}")
        return 1
    finally:
        try:
            if pipeline is not None:
                pipeline.stop()  # 停止所有流，让 SDK 完成录制文件写入。
                closed = started
        except Exception as exc:
            if started:
                print(f"错误：停止录制失败，文件可能不完整：{exc}")
        finally:
            if start_time is not None:
                duration = time.perf_counter() - start_time
            # 释放录制设备引用，确保文件句柄关闭后再输出结果。
            active_profile = None
            pipeline = None
            cv2.destroyAllWindows()

        print(f"\n文件保存位置：{output_path}")
        print(f"总录制时间：{duration:.2f} 秒")
        print(f"是否成功录制 RGB：{'是' if closed and rgb_frames else '否'}（收到 {rgb_frames} 帧）")
        print(f"是否成功录制 Depth：{'是' if closed and depth_frames else '否'}（收到 {depth_frames} 帧）")
        print(f"是否启用了 IMU：{'是' if imu_enabled else '否'}")
        if closed and output_path.is_file():
            print(f"录制已停止，文件已关闭，大小：{output_path.stat().st_size:,} 字节。")

    return 0 if closed and rgb_frames and depth_frames else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="录制 D435i 原始数据到 .db3")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="新的 .db3 文件路径，禁止覆盖已有录制")
    raise SystemExit(main(parser.parse_args().output))
