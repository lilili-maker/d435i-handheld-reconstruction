"""点云融合：根据统一 T_world_camera 逐帧融合彩色点云，不改变输入位姿。"""

import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reconstruction.pose_io import load_poses, require_pose, normalize_frame_id
    from reconstruction.pose_utils import validate_transform, transform_points
else:
    from .pose_io import load_poses, require_pose, normalize_frame_id
    from .pose_utils import validate_transform, transform_points


ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("merge_pointclouds")
PALETTE = [[1, .15, .15], [.15, .8, .2], [.15, .35, 1], [1, .65, .1], [.8, .2, 1], [.1, .85, .85]]


def single_frame_module():
    """复用单帧点云加载函数，不复制 RGB-D 处理算法。"""
    spec = importlib.util.spec_from_file_location("single_frame_merge", ROOT / "src/04_single_frame_pointcloud.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorldVoxelAccumulator:
    """固定世界网格的增量体素降采样，保留每个原始点的等权重。

    voxel key=floor(P_world/voxel_size)，网格原点恒为 World 原点。
    单帧先压缩为 sums/counts，合并仍累计 sums/counts，最后生成全局均值。
    内存由已占用体素数决定，不保存所有帧的原始点。max_voxels 防止无限增长。
    """
    def __init__(self, voxel_size, max_voxels=4_000_000):
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("体素大小必须是有限正数，单位米。")
        if isinstance(max_voxels, bool) or not isinstance(max_voxels, int) or max_voxels <= 0:
            raise ValueError("max_voxels 必须是正整数。")
        self.voxel_size, self.max_voxels = float(voxel_size), max_voxels
        self.keys = np.empty((0, 3), dtype=np.int64)
        self.point_sums = np.empty((0, 3))
        self.color_sums = np.empty((0, 3))
        self.counts = np.empty(0, dtype=np.int64)
        self.raw_count = 0
        self.peak_voxels = 0

    @staticmethod
    def _group(keys, points, colors, weights=None):
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse, weights=weights, minlength=len(unique)).astype(np.int64)
        point_sums = np.column_stack([np.bincount(inverse, weights=points[:, axis], minlength=len(unique)) for axis in range(3)])
        color_sums = np.column_stack([np.bincount(inverse, weights=colors[:, axis], minlength=len(unique)) for axis in range(3)])
        return unique, point_sums, color_sums, counts

    def add(self, cloud):
        points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
        if len(points) == 0 or not cloud.has_colors() or colors.shape != points.shape:
            raise ValueError("融合输入必须是非空彩色点云。")
        if not np.isfinite(points).all() or not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 1):
            raise ValueError("点坐标或 RGB 数值非法。")
        scaled = np.floor(points / self.voxel_size)
        if not np.isfinite(scaled).all() or np.max(np.abs(scaled)) >= np.iinfo(np.int64).max:
            raise ValueError("体素索引溢出，请检查坐标单位和体素大小。")
        grouped = self._group(scaled.astype(np.int64), points, colors)
        frame_voxels = len(grouped[0])
        if frame_voxels > self.max_voxels:
            raise MemoryError("单帧体素数超过 --max-voxels；请缩小帧范围或合理增加体素尺寸。")
        if len(self.keys):
            # 不把已有体素当成单个等权点，否则重叠帧多时均值和颜色会改变。
            grouped = self._group(np.concatenate((self.keys, grouped[0])),
                                  np.concatenate((self.point_sums, grouped[1])),
                                  np.concatenate((self.color_sums, grouped[2])),
                                  np.concatenate((self.counts, grouped[3])))
        if len(grouped[0]) > self.max_voxels:
            raise MemoryError("全局体素数超过 --max-voxels，停止以避免继续增长；未保存部分融合点云。")
        self.keys, self.point_sums, self.color_sums, self.counts = grouped
        self.raw_count += len(points)
        self.peak_voxels = max(self.peak_voxels, len(self.keys))
        return frame_voxels

    def pointcloud(self):
        if len(self.keys) == 0:
            raise ValueError("没有成功融合的体素。")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.point_sums / self.counts[:, None])
        cloud.colors = o3d.utility.Vector3dVector(self.color_sums / self.counts[:, None])
        return cloud


def selected_frames(start, end, step, frame_ids=None):
    # 跳帧恢复后的轨迹不再等间隔；只接受显式、递增的真实已验帧号。
    if frame_ids is not None:
        if (not isinstance(frame_ids, list) or not frame_ids or
                any(type(i) is not int for i in frame_ids) or
                start < 0 or end < start or step <= 0 or
                frame_ids[0] < start or frame_ids[-1] > end or
                any(a >= b for a, b in zip(frame_ids, frame_ids[1:]))):
            raise ValueError("frame_ids 须为非空、严格递增且位于 start/end 内的非负整数列表。")
        return list(frame_ids)
    if start < 0 or end < start or step <= 0 or (end - start) % step:
        raise ValueError("须满足 0 <= start <= end、step > 0，且 step 能整除 end-start。")
    return list(range(start, end + 1, step))


def read_timestamps(dataset):
    path = dataset / "timestamps.csv"
    timestamps = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"frame_id", "color_timestamp_ms", "depth_timestamp_ms"}.issubset(reader.fieldnames or []):
            raise ValueError("timestamps.csv 缺少帧编号或两路时间戳。")
        for row in reader:
            i = normalize_frame_id(row["frame_id"])
            if i in timestamps:
                raise ValueError(f"CSV 帧编号重复：{i}")
            values = [float(row[key]) for key in ("color_timestamp_ms", "depth_timestamp_ms")]
            if not np.isfinite(values).all():
                raise ValueError(f"帧 {i} 的时间戳含 NaN / Inf。")
            timestamps[i] = values
    return timestamps


def check_inputs(dataset, ids, poses, timestamps):
    issues, missing, invalid = [], [], []
    for i in ids:
        absent = [str(dataset / kind / f"{i:06d}.png") for kind in ("color", "depth")
                  if not (dataset / kind / f"{i:06d}.png").is_file()]
        if i not in poses:
            absent.append("统一相机位姿")
        if i not in timestamps:
            absent.append("数据集时间戳")
        if absent:
            missing.append(i)
            issues.append(f"帧 {i} 缺少：{', '.join(absent)}")
            continue
        try:
            pose = require_pose(poses, i)
            # 只核对显式保留的相机时间戳。不同机器人时钟不能直接和相机时钟相减。
            old = pose.source_metadata
            for index, key in enumerate(("color_timestamp_ms", "depth_timestamp_ms")):
                if key in old and old[key] != timestamps[i][index]:
                    raise ValueError(f"{key} 与数据集不符，可能选择了错误扫描批次。")
        except (ValueError, KeyError) as exc:
            invalid.append(i)
            issues.append(f"帧 {i} 位姿不可用：{exc}")
    return issues, missing, invalid


def apply_world_pose(cloud, pose):
    """直接 camera-to-world，不求逆；抽样核对坐标和 RGB 未被意外更改。"""
    t = validate_transform(pose.T_world_camera)
    indices = np.linspace(0, len(cloud.points) - 1, min(32, len(cloud.points)), dtype=int)
    original = np.asarray(cloud.points)[indices].copy()
    rgb = np.asarray(cloud.colors)[indices].copy()
    expected = transform_points(t, original)
    cloud.transform(t)
    if not np.allclose(np.asarray(cloud.points)[indices], expected, atol=1e-10, rtol=0):
        raise ValueError("Open3D 点转换与 P_world=T_world_camera@P_camera 不一致。")
    if not np.array_equal(np.asarray(cloud.colors)[indices], rgb):
        raise ValueError("位姿转换意外改变了 RGB。")


def make_diagnostic_cloud(cloud, voxel_size, limit=8000):
    # 仅保存最多六个小型诊断点云，不保存完整原始帧列表。
    diagnostic = cloud.voxel_down_sample(max(.02, voxel_size * 4))
    if len(diagnostic.points) > limit:
        diagnostic = diagnostic.uniform_down_sample(int(np.ceil(len(diagnostic.points) / limit)))
    return diagnostic


def diagnose(clouds, distance_threshold):
    results = []
    ids = sorted(clouds)
    for source_id, target_id in zip(ids, ids[1:]):
        a, b = clouds[source_id], clouds[target_id]
        forward = np.asarray(a.compute_point_cloud_distance(b))
        backward = np.asarray(b.compute_point_cloud_distance(a))
        close_ratio = float(((forward <= distance_threshold).mean() + (backward <= distance_threshold).mean()) / 2)
        row = {"source": source_id, "target": target_id, "threshold_m": distance_threshold,
               "source_samples": len(forward), "target_samples": len(backward),
               "source_to_target_median_m": float(np.median(forward)),
               "source_to_target_p90_m": float(np.percentile(forward, 90)),
               "target_to_source_median_m": float(np.median(backward)),
               "target_to_source_p90_m": float(np.percentile(backward, 90)),
               "symmetric_close_ratio": close_ratio, "possible_poor_overlap": close_ratio < .5}
        results.append(row)
        LOG.info("诊断 %d↔%d：双向邻近比例 %.2f%%，阈值 %.3f 米。", source_id, target_id, close_ratio * 100, distance_threshold)
        if row["possible_poor_overlap"]:
            LOG.warning("帧 %d↔%d 重叠支持不足，可能存在 Odometry 漂移、遮挡或视角变化；需查看诊断点云。", source_id, target_id)
    LOG.info("最近邻距离仅用于采样表面重叠检查，不是 ground-truth 位姿误差；较好重叠也不能保证真实配准精度。")
    return results


def write_ply_checked(path, cloud):
    """排他创建占位后写入本次输出，并实际回读核对坐标和 RGB。"""
    with path.open("xb"):
        pass
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise OSError(f"点云写入失败：{path}")
    saved = o3d.io.read_point_cloud(str(path))
    if len(saved.points) != len(cloud.points) or not saved.has_colors():
        raise ValueError(f"PLY 回读点数或 RGB 验证失败：{path}")
    if not np.allclose(np.asarray(saved.points), np.asarray(cloud.points), atol=1e-10, rtol=0):
        raise ValueError(f"PLY 回读坐标验证失败：{path}")
    if not np.allclose(np.asarray(saved.colors), np.asarray(cloud.colors), atol=.5 / 255 + 1e-9, rtol=0):
        raise ValueError(f"PLY 回读颜色超过 8-bit 量化容差：{path}")
    return {"path": str(path), "point_count": len(saved.points), "readback_verified": True,
            "rgb_quantization_tolerance": .5 / 255 + 1e-9}


def merge(args):
    ids = selected_frames(args.start, args.end, args.step, getattr(args, "frames", None))
    output = args.output.resolve()
    report_path = output.with_name("merge_" + output.stem.removeprefix("merged_") + "_report.json")
    diagnostic_path = output.with_name(output.stem + "_diagnostic.ply")
    if output.suffix.lower() != ".ply":
        raise ValueError("输出必须是 PLY 文件。")
    for path in (output, diagnostic_path, report_path):
        if path.exists():
            raise FileExistsError(f"禁止覆盖已有输出：{path}；请指定新的 --output。")
    for protected in (args.dataset.resolve(), args.poses.resolve().parent, ROOT / "data"):
        if protected == output.parent or protected in output.parents:
            raise ValueError("融合输出不能写入数据集、原始数据目录或位姿目录。")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "input_frame_count": len(ids), "requested_frame_ids": ids,
              "processed_frame_count": 0, "missing_frame_count": 0, "missing_frame_ids": [],
              "invalid_pose_frame_ids": [], "points_before_fusion": 0, "points_after_fusion": 0,
              "pose_convention": "T_world_camera", "transform_direction": "camera_to_world", "translation_unit": "m",
              "voxel_size_m": args.voxel_size, "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "frames": [], "error": None,
              "fusion_method": "Incremental fixed-world-grid voxel means, weighted by original pixel-point counts; no XYZ scaling",
              "diagnostic_note": "Sampled nearest-neighbor surface distances; not ground-truth pose error or a guarantee of registration accuracy"}
    try:
        loader = single_frame_module()
        metadata = loader.load_intrinsics(args.dataset / "camera_intrinsics.json")
        poses = load_poses(args.poses)
        timestamps = read_timestamps(args.dataset)
        issues, missing, invalid = check_inputs(args.dataset, ids, poses, timestamps)
        report.update(missing_frame_ids=missing, missing_frame_count=len(missing), invalid_pose_frame_ids=invalid)
        if issues:
            raise ValueError("输入检查失败，不补齐、不跳过：\n" + "\n".join(issues))
        world = require_pose(poses, ids[0]).coordinate_frame
        report.update(coordinate_frame=world, pose_sources=sorted({poses[i].pose_source for i in ids}), camera_intrinsics=metadata)
        report["input_sha256"] = {}
        for path in (args.poses, args.dataset / "camera_intrinsics.json", args.dataset / "timestamps.csv"):
            with path.open("rb") as f:
                report["input_sha256"][str(path.resolve())] = hashlib.file_digest(f, "sha256").hexdigest()
        LOG.info("输入 %d 帧，World=%s，位姿来源=%s，体素=%.6f 米。", len(ids), world, report["pose_sources"], args.voxel_size)
        diagnostic_ids = [ids[index] for index in np.unique(np.rint(np.linspace(0, len(ids) - 1, min(6, len(ids)))).astype(int))]
        diagnostics = {}
        accumulator = WorldVoxelAccumulator(args.voxel_size, args.max_voxels)
        for i in ids:
            pose = require_pose(poses, i, coordinate_frame=world)
            # 原函数提供 RGB 转换、uint16 检查、米制深度过滤和 Color 内参。
            with redirect_stdout(io.StringIO()):
                cloud = loader.create_pointcloud(args.dataset, i, args.min_depth, args.max_depth)
            raw_count = len(cloud.points)
            apply_world_pose(cloud, pose)  # camera-to-world，绝不求逆。
            frame_voxels = accumulator.add(cloud)
            if i in diagnostic_ids:
                diagnostics[i] = make_diagnostic_cloud(cloud, args.voxel_size)
            report["frames"].append({"frame_id": i, "pose_source": pose.pose_source, "coordinate_frame": world,
                                     "pose_timestamp_s": pose.timestamp, "timestamp_reference": pose.timestamp_reference,
                                     "color_timestamp_ms": timestamps[i][0], "depth_timestamp_ms": timestamps[i][1],
                                     "T_world_camera": pose.T_world_camera.tolist(), "raw_point_count": raw_count,
                                     "frame_voxel_count": frame_voxels, "world_transform_and_rgb_verified": True})
            report["processed_frame_count"] += 1
            report["points_before_fusion"] = accumulator.raw_count
            LOG.info("帧 %d：原始彩色点 %d → 单帧体素 %d；当前全局体素 %d。", i, raw_count, frame_voxels, len(accumulator.keys))
            del cloud  # 及时释放单帧完整点云。
        merged = accumulator.pointcloud()  # 最终全局体素降采样结果，含等权 RGB 均值。
        report.update(points_after_fusion=len(merged.points), peak_occupied_voxels=accumulator.peak_voxels,
                      voxel_accumulator_storage_bytes=sum(a.nbytes for a in (accumulator.keys, accumulator.point_sums, accumulator.color_sums, accumulator.counts)),
                      memory_note="Storage grows with occupied voxels; capped by max_voxels. Temporary grouping arrays and current frame require additional memory.")
        del accumulator
        report["xyz_bounds_m"] = {"min": merged.get_min_bound().tolist(), "max": merged.get_max_bound().tolist()}
        report["quality_checks"] = diagnose(diagnostics, args.diagnostic_distance)
        diagnostic = o3d.geometry.PointCloud()
        report["diagnostic_palette"] = []
        for index, (i, cloud) in enumerate(sorted(diagnostics.items())):
            color = PALETTE[index]
            cloud.paint_uniform_color(color)  # 只修改独立的诊断点云，不改 RGB 融合结果。
            diagnostic += cloud
            report["diagnostic_palette"].append({"frame_id": i, "rgb": color, "points": len(cloud.points)})
        report["rgb_ply"] = write_ply_checked(output, merged)
        report["diagnostic_ply"] = write_ply_checked(diagnostic_path, diagnostic)
        report["status"] = "complete"
        LOG.info("成功处理 %d/%d 帧；缺失 %d；融合前 %d 点，融合后 %d 点。", report["processed_frame_count"], len(ids), len(missing), report["points_before_fusion"], len(merged.points))
        LOG.info("XYZ 范围（米）：%s；RGB PLY：%s；诊断 PLY：%s", report["xyz_bounds_m"], output, diagnostic_path)
        if args.visualize:
            LOG.info("先显示 RGB 融合点云；关闭窗口后显示独立伪彩色诊断点云。")
            o3d.visualization.draw_geometries([merged], window_name="Merged RGB World Point Cloud", lookat=merged.get_center(), front=[0, 0, -1], up=[0, -1, 0], zoom=.7)
            o3d.visualization.draw_geometries([diagnostic], window_name="Diagnostic Frame Overlay (Pseudo Color)", lookat=merged.get_center(), front=[0, 0, -1], up=[0, -1, 0], zoom=.7)
        return report
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = str(exc) or "用户中断"
        raise
    finally:
        with report_path.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        LOG.info("报告：%s", report_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="使用统一 T_world_camera 逐帧融合米制彩色点云")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/extracted_scan_02")
    parser.add_argument("--poses", type=Path, default=ROOT / "output/poses/trajectory_500_550.json")
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--end", type=int, default=550)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--frames", nargs="+", type=int, help="显式已验帧列表，替代等间距采样；须位于 start/end 内")
    parser.add_argument("--voxel-size", type=float, default=.005)
    parser.add_argument("--min-depth", type=float, default=.2)
    parser.add_argument("--max-depth", type=float, default=3.)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-voxels", type=int, default=4_000_000, help="全局体素数上限，超限停止以控制内存")
    parser.add_argument("--diagnostic-distance", type=float, default=.03, help="诊断最近邻距离阈值，米；不是位姿误差")
    args = parser.parse_args(argv)
    try:
        selected_frames(args.start, args.end, args.step, args.frames)
        if not np.isfinite([args.voxel_size, args.min_depth, args.max_depth, args.diagnostic_distance]).all():
            raise ValueError("参数不能含 NaN / Inf。")
        if args.voxel_size <= 0 or args.diagnostic_distance <= 0 or args.max_voxels <= 0 or not 0 <= args.min_depth < args.max_depth:
            raise ValueError("体素/诊断阈值/体素上限须为正；0 <= min-depth < max-depth。")
    except ValueError as exc:
        parser.error(str(exc))
    args.dataset, args.poses = args.dataset.resolve(), args.poses.resolve()
    if args.output is None:
        suffix = "" if args.step == 1 else f"_step{args.step}"
        args.output = ROOT / "output/pointcloud" / f"merged_{args.start}_{args.end}{suffix}.ply"
    args.output = args.output.resolve()
    return args


def main(argv=None):
    try:
        merge(parse_args(argv))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        LOG.error("点云融合停止（%s）：%s", type(exc).__name__, str(exc) or "用户中断")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
