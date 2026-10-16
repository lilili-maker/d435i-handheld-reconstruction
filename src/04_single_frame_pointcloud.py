"""阶段 3：使用一对已对齐的 RGB-D 图片生成米制彩色点云。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/extracted"
DEFAULT_OUTPUT = PROJECT_ROOT / "output/pointcloud/single_frame.ply"


def load_intrinsics(path):
    """按阶段 2B 实际生成的 JSON 结构读取并验证内参和深度单位。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到相机内参文件：{path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("相机内参 JSON 必须是对象。")
    required = ("width", "height", "fx", "fy", "cx", "cy", "depth_scale")
    for key in required:
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"内参字段 {key} 缺失或不是有限数值：{value!r}")
    for key in ("width", "height"):
        if metadata[key] <= 0 or int(metadata[key]) != metadata[key]:
            raise ValueError(f"{key} 必须是正整数。")
        metadata[key] = int(metadata[key])
    for key in ("fx", "fy", "depth_scale"):
        if metadata[key] <= 0:
            raise ValueError(f"{key} 必须大于零。")
    if not (0 <= metadata["cx"] < metadata["width"]
            and 0 <= metadata["cy"] < metadata["height"]):
        raise ValueError("主点 cx / cy 超出图片范围，请检查 Color 内参。")
    if metadata.get("intrinsics_unit") != "pixels":
        raise ValueError("内参单位必须为 pixels，请检查阶段 2B 的 JSON。")
    if metadata.get("depth_scale_unit") != "meters_per_depth_unit":
        raise ValueError("depth_scale 单位必须为 meters_per_depth_unit，不能猜测深度单位。")
    if metadata.get("depth_aligned_to") != "color":
        raise ValueError("JSON 未确认 Depth 对齐到 Color，请先检查阶段 2B 对齐结果。")
    if metadata.get("depth_dtype") != "uint16":
        raise ValueError("JSON 中的原始深度类型必须为 uint16。")
    return metadata


def create_pointcloud(dataset, frame_number, min_depth, max_depth):
    if frame_number < 0:
        raise ValueError("帧编号不能为负数。")
    if not (np.isfinite(min_depth) and np.isfinite(max_depth) and 0 <= min_depth < max_depth):
        raise ValueError("深度范围必须满足 0 <= min-depth < max-depth，单位为米。")

    frame_id = f"{frame_number:06d}"
    color_path = dataset / "color" / f"{frame_id}.png"
    depth_path = dataset / "depth" / f"{frame_id}.png"
    metadata = load_intrinsics(dataset / "camera_intrinsics.json")
    for path in (color_path, depth_path):
        if not path.is_file():
            raise FileNotFoundError(f"找不到当前帧图片：{path}")
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color_bgr is None or depth is None:
        raise ValueError("RGB 或 Depth PNG 无法读取，请检查文件是否损坏。")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Depth 必须是二维 uint16，实际为 {depth.dtype}，shape={depth.shape}。")
    if color_bgr.dtype != np.uint8 or color_bgr.shape[:2] != depth.shape:
        raise ValueError(f"RGB 与 Depth 分辨率或类型异常：RGB={color_bgr.shape}，Depth={depth.shape}。")
    expected_shape = (metadata["height"], metadata["width"])
    if depth.shape != expected_shape:
        raise ValueError(f"图片尺寸 {depth.shape} 与内参尺寸 {expected_shape} 不一致。")

    sdk_scale = float(metadata["depth_scale"])
    open3d_scale = 1.0 / sdk_scale
    if not np.isfinite(open3d_scale) or open3d_scale <= 0:
        raise ValueError("RealSense depth_scale 无法转换成有效的 Open3D 深度除数。")
    print(f"当前帧编号：{frame_id}")
    print(f"RGB 图像分辨率：{color_bgr.shape[1]} x {color_bgr.shape[0]}")
    print(f"Depth 图像分辨率：{depth.shape[1]} x {depth.shape[0]}")
    print("Depth 对齐信息：JSON 标记为 color，图片尺寸与 Color 内参一致。")
    print(f"Depth 数据类型：{depth.dtype}")
    print(f"Depth 原始最小值：{int(depth.min())}，最大值：{int(depth.max())}")
    print(f"非零深度像素比例：{np.count_nonzero(depth) / depth.size:.2%}")
    print(f"RealSense depth scale：{sdk_scale:.12g} 米/深度单位")
    print(f"实际使用的 Open3D depth_scale：{open3d_scale:.12g} 深度单位/米")
    print(f"fx={metadata['fx']}, fy={metadata['fy']}, cx={metadata['cx']}, cy={metadata['cy']}")

    # 距离过滤以相机光轴深度 Z（米）为准，0 表示无效深度。
    depth_meters = depth.astype(np.float64) * sdk_scale
    valid = (depth > 0) & (depth_meters >= min_depth) & (depth_meters <= max_depth)
    valid_count = int(np.count_nonzero(valid))
    print(f"深度过滤范围：{min_depth:g}～{max_depth:g} 米，有效像素：{valid_count}")
    if valid_count == 0:
        raise ValueError("深度范围内没有有效像素，无法生成点云；请检查深度图片、单位和过滤范围。")
    filtered_depth = depth.copy()
    filtered_depth[~valid] = 0  # 仅修改内存副本，保留磁盘上的原始 PNG。

    # OpenCV 读取 BGR，Open3D 彩色点云需要 RGB。
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_image = o3d.geometry.Image(np.ascontiguousarray(color_rgb))
    depth_image = o3d.geometry.Image(np.ascontiguousarray(filtered_depth))
    # 上界已在 uint16 副本中过滤；略放宽截断值，避免 float32 舍入排除边界点。
    truncation = float(np.nextafter(np.float32(max_depth), np.float32(np.inf)))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=open3d_scale,
        depth_trunc=truncation, convert_rgb_to_intensity=False,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        metadata["width"], metadata["height"],
        metadata["fx"], metadata["fy"], metadata["cx"], metadata["cy"],
    )
    pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)
    if len(points) == 0:
        raise ValueError("Open3D 生成了空点云，请检查深度单位和内参。")
    if not pointcloud.has_colors() or colors.shape != points.shape:
        raise ValueError("生成的点云缺少对应的 RGB 颜色。")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("点云坐标或颜色包含 NaN / Inf，请检查输入数据。")
    print(f"生成的点云数量：{len(points):,}")
    for axis, low, high in zip("XYZ", points.min(axis=0), points.max(axis=0)):
        print(f"有效点云 {axis} 范围：[{low:.6f}, {high:.6f}] 米")
    print("坐标系：相机 X 向右，Y 向下，Z 向前；PLY 坐标单位为米。")
    return pointcloud


def main(args):
    try:
        dataset = args.dataset.resolve()
        output = args.output.resolve()
        if output.suffix.lower() != ".ply":
            raise ValueError("点云输出文件必须使用 .ply 扩展名。")
        if output.exists():
            raise FileExistsError(f"禁止覆盖已有点云：{output}；请用 --output 指定新的 PLY 路径。")
        pointcloud = create_pointcloud(dataset, args.frame, args.min_depth, args.max_depth)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(output), pointcloud, write_ascii=False):
            raise OSError(f"PLY 保存失败：{output}")
        print(f"PLY 文件保存路径：{output}")
        if not args.no_view:
            print("打开三维窗口：左键拖动旋转，滚轮缩放，Ctrl+左键拖动平移；关闭窗口结束。")
            # 调整观察方向使画面正立，PLY 仍保留原始相机坐标，不翻转点云。
            o3d.visualization.draw_geometries(
                [pointcloud], window_name="Single Frame RGB-D Point Cloud",
                width=1024, height=768, lookat=pointcloud.get_center(),
                front=[0.0, 0.0, -1.0], up=[0.0, -1.0, 0.0], zoom=0.7,
            )
        return 0
    except KeyboardInterrupt:
        print("\n单帧点云程序已中断。")
        return 1
    except Exception as exc:
        print(f"错误：单帧点云生成失败（{type(exc).__name__}）：{exc}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从一帧已对齐 RGB-D 生成彩色点云，单位为米")
    parser.add_argument("--frame", type=int, default=0, help="帧编号，例如 0、42 或 000042")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="RGB-D 数据集目录")
    parser.add_argument("--min-depth", type=float, default=0.2, help="最小光轴深度，单位米")
    parser.add_argument("--max-depth", type=float, default=3.0, help="最大光轴深度，单位米")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="新的 PLY 路径，禁止覆盖已有文件")
    parser.add_argument("--no-view", action="store_true", help="仅保存 PLY，不打开三维窗口")
    raise SystemExit(main(parser.parse_args()))
