"""TSDF 重建：使用 Open3D CPU/CUDA VoxelBlockGrid 融合 RGB-D，提取彩色 Mesh。

统一输入是 T_world_camera；官方 tensor integrate 的外参是其逆 T_camera_world。
不优化位姿、不修补 Mesh、不采集相机或机器人数据。
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import open3d as o3d

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reconstruction.pose_io import load_poses, require_pose
    from reconstruction.pose_utils import invert_transform, validate_transform
    from reconstruction.merge_pointclouds import single_frame_module, selected_frames, read_timestamps, check_inputs
else:
    from .pose_io import load_poses, require_pose
    from .pose_utils import invert_transform, validate_transform
    from .merge_pointclouds import single_frame_module, selected_frames, read_timestamps, check_inputs


ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("tsdf_reconstruction")
from reconstruction.compute_device import select_device


def check_motion(poses, ids, max_translation, max_rotation):
    """在创建 Volume 前检查所有选中相邻帧；阈值是筛查策略，不是真实精度保证。"""
    world = require_pose(poses, ids[0]).coordinate_frame
    results = []
    for source, target in zip(ids, ids[1:]):
        a = require_pose(poses, source, coordinate_frame=world).T_world_camera
        b = require_pose(poses, target, coordinate_frame=world).T_world_camera
        # target 相机坐标到 source 相机坐标，计算相对运动大小。
        relative = validate_transform(invert_transform(a) @ b)
        distance = float(np.linalg.norm(relative[:3, 3]))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))))
        row = {"source": source, "target": target, "translation_m": distance, "rotation_deg": angle,
               "passed": distance <= max_translation and angle <= max_rotation}
        results.append(row)
    return world, results


def load_rgbd(dataset, i, metadata, min_depth, max_depth):
    """沿用 RGB-D 加载器的图像、内参与深度单位约定；TSDF 需要 RGBDImage，不能用点云代替。"""
    color_path, depth_path = [dataset / kind / f"{i:06d}.png" for kind in ("color", "depth")]
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color_bgr is None or depth is None:
        raise ValueError(f"帧 {i} 的 RGB / Depth 无法解码。")
    shape = (metadata["height"], metadata["width"])
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"帧 {i} Depth 必须是二维 uint16，实际 {depth.dtype} {depth.shape}。")
    if depth.shape != shape or color_bgr.shape != (*shape, 3) or color_bgr.dtype != np.uint8:
        raise ValueError(f"帧 {i} RGB / Depth 尺寸或类型与 Color 内参不匹配。")
    sdk_scale = float(metadata["depth_scale"])
    open3d_scale = 1.0 / sdk_scale
    if not np.isfinite(open3d_scale) or open3d_scale <= 0:
        raise ValueError("RealSense scale 无法转换为有效的 Open3D 深度除数。")
    meters = depth.astype(np.float64) * sdk_scale
    valid = (depth > 0) & (meters >= min_depth) & (meters <= max_depth)
    count = int(valid.sum())
    ratio = count / depth.size
    if count < 100 or ratio < .01:
        raise ValueError(f"帧 {i} 有效深度不足：{count} 像素、{ratio:.2%}；需要至少 100 像素且比例 ≥1%。")
    filtered = depth.copy()
    filtered[~valid] = 0  # TSDF integration 无 depth_min 参数，在副本中屏蔽无效深度。
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    # 官方优化内核使用 UInt16 depth + UInt8 RGB，深度只在 API 内除以 depth_scale。
    rgbd = {"depth": o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(filtered))),
            "color": o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(rgb))),
            "depth_scale": open3d_scale,
            "depth_max": float(np.nextafter(np.float32(max_depth), np.float32(np.inf)))}
    return rgbd, {"depth_dtype": str(depth.dtype), "width": shape[1], "height": shape[0],
                  "raw_depth_min": int(depth.min()), "raw_depth_max": int(depth.max()),
                  "valid_depth_pixels": count, "valid_depth_ratio": ratio,
                  "sdk_depth_scale_m_per_unit": sdk_scale, "open3d_depth_scale_units_per_m": open3d_scale}


class TensorTSDF:
    """薄封装官方 TSDF API；不实现 TSDF 算法，不改变实际世界坐标尺寸。

    Open3D 0.20.0 legacy ScalableTSDFVolume 在本环境复现空几何问题（#7592）。
    Tensor 实现使用原始 UInt16/UInt8 输入和 Float32 tsdf/weight/color 属性。
    """
    def __init__(self, voxel_size, sdf_trunc, device="CPU:0"):
        self.trunc_voxel_multiplier = sdf_trunc / voxel_size
        core = o3d.core
        self.device = core.Device(device)
        self.grid = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(core.float32, core.float32, core.float32), attr_channels=((1), (1), (3)),
            voxel_size=voxel_size, block_resolution=16, block_count=1000, device=self.device)

    def integrate(self, frame, intrinsic, extrinsic):
        k = o3d.core.Tensor(intrinsic.intrinsic_matrix, dtype=o3d.core.float64)
        e = o3d.core.Tensor(extrinsic, dtype=o3d.core.float64)
        depth, color = frame["depth"].to(self.device), frame["color"].to(self.device)
        scale, maximum = frame["depth_scale"], frame["depth_max"]
        blocks = self.grid.compute_unique_block_coordinates(depth, k, e, scale, maximum, self.trunc_voxel_multiplier)
        if blocks.shape[0] == 0:
            raise ValueError("当前 RGB-D 帧未激活任何 TSDF 体素块。")
        self.grid.integrate(blocks, depth, color, k, e, scale, maximum, self.trunc_voxel_multiplier)

    def extract_triangle_mesh(self):
        # 权重 >0.5 允许单次有效观察，类似 legacy；这不是重建质量保证。
        return self.grid.extract_triangle_mesh(weight_threshold=.5).cpu().to_legacy()


def make_volume(voxel_size, sdf_trunc, device="CPU:0"):
    return TensorTSDF(voxel_size, sdf_trunc, device)


def integrate_rgbd(volume, rgbd, intrinsic, T_world_camera):
    """显式传入 world-to-camera，独立于 pose_source。"""
    camera_to_world = validate_transform(T_world_camera)
    extrinsic = validate_transform(np.linalg.inv(camera_to_world))
    volume.integrate(rgbd, intrinsic, extrinsic)
    return extrinsic


def mesh_statistics(mesh):
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("TSDF 提取了空 Mesh（无顶点或无三角形），请检查深度、外参及融合帧。")
    if not np.isfinite(vertices).all() or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("Mesh 顶点或三角形索引非法。")
    if not mesh.has_vertex_colors() or not np.isfinite(np.asarray(mesh.vertex_colors)).all():
        raise ValueError("RGB8 TSDF Mesh 缺少有效顶点颜色。")
    colors = np.asarray(mesh.vertex_colors)
    # Tensor Marching Cubes 的 Float32 插值可产生约 1e-7 的边界舍入误差。
    # 只允许数值容差，不修改颜色或 Mesh；显著越界仍停止。
    if np.any(colors < -1e-6) or np.any(colors > 1 + 1e-6):
        raise ValueError(f"Mesh RGB 显著超出 [0,1]：min={colors.min()}，max={colors.max()}。")
    if not mesh.has_vertex_normals() or not np.isfinite(np.asarray(mesh.vertex_normals)).all():
        raise ValueError("Mesh 缺少有效顶点法向量。")
    return {"vertex_count": len(vertices), "triangle_count": len(triangles),
            "has_vertex_colors": True, "has_vertex_normals": True,
            "vertex_color_range": [float(colors.min()), float(colors.max())], "color_validation_tolerance": 1e-6,
            "xyz_bounds_m": {"min": mesh.get_min_bound().tolist(), "max": mesh.get_max_bound().tolist()}}


def write_mesh_checked(path, mesh):
    # 排他创建本次输出；不覆盖任何已有 PLY。
    with path.open("xb"):
        pass
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, write_vertex_normals=True,
                                     write_vertex_colors=True, write_triangle_uvs=False):
        raise OSError(f"Mesh PLY 写入失败：{path}")
    loaded = o3d.io.read_triangle_mesh(str(path))
    stats = mesh_statistics(loaded)
    if not np.array_equal(np.asarray(loaded.triangles), np.asarray(mesh.triangles)):
        raise ValueError("Mesh PLY 回读三角形与原始 Mesh 不一致。")
    if not np.allclose(np.asarray(loaded.vertices), np.asarray(mesh.vertices), atol=1e-10, rtol=0):
        raise ValueError("Mesh PLY 回读坐标不一致。")
    if not np.allclose(np.asarray(loaded.vertex_normals), np.asarray(mesh.vertex_normals), atol=1e-10, rtol=0):
        raise ValueError("Mesh PLY 回读法向量不一致。")
    if not np.allclose(np.asarray(loaded.vertex_colors), np.asarray(mesh.vertex_colors), atol=.5 / 255 + 1e-9, rtol=0):
        raise ValueError("Mesh PLY 回读 RGB 超过 8-bit 量化容差。")
    return {**stats, "path": str(path), "readback_verified": True}


def reconstruct(args):
    ids = selected_frames(args.start, args.end, args.step, getattr(args, "frames", None))
    output = args.output.resolve()
    report_path = output.with_name("tsdf_report_" + output.stem.removeprefix("scene_mesh_") + ".json")
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"禁止覆盖已有输出：{path}；请指定新的 --output。")
    if output.suffix.lower() != ".ply":
        raise ValueError("Mesh 输出必须使用 .ply 扩展名。")
    for protected in (args.dataset.resolve(), args.poses.resolve().parent, ROOT / "data", ROOT / "output/pointcloud"):
        if output.parent == protected or protected in output.parents:
            raise ValueError("Mesh 输出不能位于输入数据、位姿或已有点云目录中。")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"status": "failed", "implementation": "Open3D tensor VoxelBlockGrid TSDF (CPU:0)",
              "open3d_version": o3d.__version__, "input_frame_count": len(ids), "integrated_frame_count": 0,
              "frames": [{"frame_id": i, "status": "not_attempted"} for i in ids], "error": None,
              "input_pose_convention": "T_world_camera", "integration_extrinsic": "T_camera_world = inverse(T_world_camera)",
              "translation_unit": "m", "depth_range_m": [args.min_depth, args.max_depth],
              "voxel_size_m": args.voxel_size, "sdf_trunc_m": args.sdf_trunc,
              "block_resolution": 16, "initial_block_count": 1000, "mesh_weight_threshold": .5,
              "trunc_voxel_multiplier": args.sdf_trunc / args.voxel_size,
              "depth_input_dtype": "UInt16", "color_input_dtype": "UInt8", "attribute_dtype": "Float32",
              "backend_selection_reason": "Local Open3D 0.20.0 legacy ScalableTSDFVolume reproduces empty geometry issue #7592; tensor implementation validated independently",
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "quality_note": "Offline reconstruction from supplied poses; not calibrated robot accuracy. Inspect sparse 6-frame mesh before increasing frame count.",
              "automatic_ghosting_detection": False}
    try:
        metadata = single_frame_module().load_intrinsics(args.dataset / "camera_intrinsics.json")
        poses = load_poses(args.poses)
        timestamps = read_timestamps(args.dataset)
        issues, missing, invalid = check_inputs(args.dataset, ids, poses, timestamps)
        report.update(missing_frame_ids=missing, invalid_pose_frame_ids=invalid, camera_intrinsics=metadata)
        if issues:
            raise ValueError("输入检查失败，不使用单位矩阵补齐：\n" + "\n".join(issues))
        world, motion = check_motion(poses, ids, args.max_translation, args.max_rotation)
        report.update(coordinate_frame=world, pose_sources=sorted({poses[i].pose_source for i in ids}), motion_checks=motion)
        abnormal = [row for row in motion if not row["passed"]]
        if abnormal:
            raise ValueError(f"相邻选中帧运动超过筛查阈值，停止融合：{abnormal}")
        report["input_sha256"] = {}
        for path in (args.poses, args.dataset / "camera_intrinsics.json", args.dataset / "timestamps.csv"):
            with path.open("rb") as f:
                report["input_sha256"][str(path.resolve())] = hashlib.file_digest(f, "sha256").hexdigest()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(metadata["width"], metadata["height"],
                                                     metadata["fx"], metadata["fy"], metadata["cx"], metadata["cy"])
        device, device_report = select_device(getattr(args, "device", "CPU:0"))
        report.update(compute_device=device_report, implementation=f"Open3D tensor VoxelBlockGrid TSDF ({device})")
        volume = make_volume(args.voxel_size, args.sdf_trunc, device)
        LOG.info("输入 %d 帧，World=%s，来源=%s；voxel=%.3f 米，sdf_trunc=%.3f 米，深度 %.2f～%.2f 米。",
                 len(ids), world, report["pose_sources"], args.voxel_size, args.sdf_trunc, args.min_depth, args.max_depth)
        LOG.info("%s tensor TSDF integrate 使用 world-to-camera；RealSense scale=%.12g，Open3D 深度除数=%.12g。",
                 device, metadata["depth_scale"], 1 / metadata["depth_scale"])
        integration_start = time.perf_counter()
        for row in report["frames"]:
            i = row["frame_id"]
            try:
                frame_start = time.perf_counter()
                rgbd, stats = load_rgbd(args.dataset, i, metadata, args.min_depth, args.max_depth)
                pose = require_pose(poses, i, coordinate_frame=world)
                extrinsic = integrate_rgbd(volume, rgbd, intrinsic, pose.T_world_camera)
                row.update(status="integrated", integration_call_completed=True, depth_stats=stats,
                           T_world_camera=pose.T_world_camera.tolist(), T_camera_world=extrinsic.tolist(),
                           pose_source=pose.pose_source, pose_timestamp_s=pose.timestamp,
                           color_timestamp_ms=timestamps[i][0], depth_timestamp_ms=timestamps[i][1],
                           processing_seconds=time.perf_counter() - frame_start)
                report["integrated_frame_count"] += 1
                LOG.info("帧 %d：融合调用完成；有效深度 %d（%.2f%%）。", i, stats["valid_depth_pixels"], stats["valid_depth_ratio"] * 100)
                del rgbd
            except (Exception, KeyboardInterrupt) as exc:
                row.update(status="failed", error=str(exc) or "用户中断")
                raise
        report["integration_seconds"] = time.perf_counter() - integration_start
        extract_start = time.perf_counter()
        mesh = volume.extract_triangle_mesh()
        del volume
        mesh.compute_vertex_normals()
        report.update(mesh_statistics(mesh))
        report["mesh_extraction_seconds"] = time.perf_counter() - extract_start
        write_start = time.perf_counter()
        report["mesh_ply"] = write_mesh_checked(output, mesh)
        report["write_and_readback_seconds"] = time.perf_counter() - write_start
        report["status"] = "complete"
        report["runtime_seconds"] = time.perf_counter() - started
        LOG.info("成功融合 %d/%d 帧；Mesh 顶点 %d，三角形 %d；XYZ 范围（米）=%s；运行 %.3f 秒。",
                 report["integrated_frame_count"], len(ids), report["vertex_count"], report["triangle_count"], report["xyz_bounds_m"], report["runtime_seconds"])
        LOG.info("Mesh 保存并回读验证通过：%s", output)
        LOG.warning("位姿来源标签不代表真实精度；若发现严重重影，请停止扩大数据规模。6 帧只作功能验证。")
        if args.visualize:
            o3d.visualization.draw_geometries([mesh], window_name="Offline TSDF Mesh", mesh_show_back_face=True,
                                             lookat=mesh.get_center(), front=[0, 0, -1], up=[0, -1, 0], zoom=.7)
        return report
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", error=str(exc) or "用户中断")
        raise
    finally:
        report.setdefault("runtime_seconds", time.perf_counter() - started)
        report["total_wall_seconds_including_visualization"] = time.perf_counter() - started
        with report_path.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        LOG.info("TSDF 报告：%s", report_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="使用统一相机位姿进行 Open3D CPU tensor TSDF 彩色 Mesh 重建")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/extracted_scan_02")
    parser.add_argument("--poses", type=Path, default=ROOT / "output/poses/trajectory_500_550.json")
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--end", type=int, default=550)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--frames", nargs="+", type=int, help="显式已验帧列表，替代等间距采样；须位于 start/end 内")
    parser.add_argument("--voxel-size", type=float, default=.01)
    parser.add_argument("--device", choices=("auto", "CPU:0", "CUDA:0"), default="CPU:0")
    parser.add_argument("--sdf-trunc", type=float, default=.04)
    parser.add_argument("--min-depth", type=float, default=.2)
    parser.add_argument("--max-depth", type=float, default=3.)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-translation", type=float, default=.25, help="每对选中帧的运动筛查阈值，米")
    parser.add_argument("--max-rotation", type=float, default=30., help="每对选中帧的运动筛查阈值，度")
    args = parser.parse_args(argv)
    try:
        selected_frames(args.start, args.end, args.step, args.frames)
        if not np.isfinite([args.voxel_size, args.sdf_trunc, args.min_depth, args.max_depth, args.max_translation, args.max_rotation]).all():
            raise ValueError("所有数值参数必须有限，不能含 NaN / Inf。")
        if args.voxel_size <= 0 or args.sdf_trunc < args.voxel_size:
            raise ValueError("voxel-size 必须为正，sdf-trunc 至少为一个体素宽度（本程序的稳定性筛查策略）。")
        if not 0 <= args.min_depth < args.max_depth or args.max_translation <= 0 or not 0 < args.max_rotation <= 180:
            raise ValueError("深度范围或运动筛查阈值不合法。")
    except ValueError as exc:
        parser.error(str(exc))
    if args.sdf_trunc < 2 * args.voxel_size:
        LOG.warning("sdf-trunc 小于两个体素宽度，窄截断带可能使表面提取更敏感。")
    args.dataset, args.poses = args.dataset.resolve(), args.poses.resolve()
    if args.output is None:
        suffix = "" if args.step == 1 else f"_step{args.step}"
        args.output = ROOT / "output/mesh" / f"scene_mesh_{args.start}_{args.end}{suffix}.ply"
    args.output = args.output.resolve()
    return args


def main(argv=None):
    try:
        reconstruct(parse_args(argv))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        LOG.error("TSDF 重建停止（%s）：%s", type(exc).__name__, str(exc) or "用户中断")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
