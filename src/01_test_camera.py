"""相机检查：检测 RealSense 并实时显示 RGB 和对齐后的 Depth。"""

import time

import cv2
import numpy as np
import pyrealsense2 as rs


def print_intrinsics(label, profile):
    """打印原始视频流内参；cx、cy 对应 SDK 的 ppx、ppy。"""
    intrinsics = profile.as_video_stream_profile().get_intrinsics()
    print(f"\n{label} intrinsic: {intrinsics}")
    print(f"  fx={intrinsics.fx}, fy={intrinsics.fy}")
    print(f"  cx={intrinsics.ppx}, cy={intrinsics.ppy}")
    print(f"  width={intrinsics.width}, height={intrinsics.height}")
    print(f"  FPS={profile.fps()}")


def main():
    pipeline = None
    started = False
    try:
        context = rs.context()
        pipeline = rs.pipeline(context)
        devices = context.query_devices()
        if len(devices) == 0:
            print("错误：未检测到 RealSense 相机，请检查 USB 连接和相机供电。")
            return

        # 使用第一台设备，并通过序列号确保启动的是同一台相机。
        device = devices[0]
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"设备名称：{device.get_info(rs.camera_info.name)}")
        print(f"设备序列号：{serial}")
        if device.supports(rs.camera_info.firmware_version):
            print(f"固件版本：{device.get_info(rs.camera_info.firmware_version)}")

        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 指定配置不支持时，让 SDK 为 BGR8 / Z16 选择兼容的分辨率和帧率。
        if not config.can_resolve(rs.pipeline_wrapper(pipeline)):
            print("提示：首选流配置不可用，尝试 SDK 自动选择兼容配置。")
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, rs.format.bgr8)
            config.enable_stream(rs.stream.depth, rs.format.z16)

        try:
            active_profile = pipeline.start(config)
            started = True
        except RuntimeError as exc:
            print(f"错误：Color / Depth 流启动失败：{exc}")
            print("请先关闭 RealSense Viewer 或其他占用相机的程序，并检查 USB 连接。")
            return

        # 这里打印的是对齐前两个原始流的内参。
        print_intrinsics("Color", active_profile.get_stream(rs.stream.color))
        print_intrinsics("Depth", active_profile.get_stream(rs.stream.depth))
        depth_scale = active_profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"\ndepth scale：{depth_scale} 米/深度单位")

        align = rs.align(rs.stream.color)
        cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
        print("\n实时显示已启动，请在图像窗口获得焦点时按 q 退出。")
        last_time = None
        invalid_frames = 0

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as exc:
                print(f"错误：等待有效 frame 失败（超时或设备断开）：{exc}")
                break

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                invalid_frames += 1
                if invalid_frames == 1:
                    print("警告：没有得到有效的 Color / Depth frame，正在重试。")
                if invalid_frames >= 30:
                    print("错误：连续 30 次未获得有效 frame，结束测试。")
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            invalid_frames = 0

            # RGB 使用副本添加文字；原始深度数组始终保持 Z16，不进行原地修改。
            color_image = np.asanyarray(color_frame.get_data()).copy()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_display = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            now = time.perf_counter()
            fps = 0.0 if last_time is None else 1.0 / max(now - last_time, 1e-9)
            last_time = now
            cv2.putText(
                color_image, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.imshow("RGB", color_image)
            cv2.imshow("Depth", depth_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，结束测试。")
    except Exception as exc:
        print(f"错误：相机测试异常（{type(exc).__name__}）：{exc}")
    finally:
        # 无设备或启动失败时 stop 可能报错，仍保证执行窗口清理。
        try:
            if pipeline is not None:
                pipeline.stop()
        except RuntimeError as exc:
            if started:
                print(f"警告：停止相机流时发生异常：{exc}")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
