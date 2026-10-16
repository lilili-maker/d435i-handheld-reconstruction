"""阶段 4A：Hybrid RGB-D Odometry；不融合点云、不做全局优化。

约定：T_source_to_target 将 source 相机坐标中的点变到 target 相机坐标。
C_i 是 camera-to-world，world 固定为起始相机坐标：C_t = C_s @ inv(T_s_to_t)。
"""

import argparse
import csv
import importlib.util
import json
import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # 轨迹图直接保存，无需桌面或实体相机。
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("rgbd_odometry")


def load_metadata(dataset):
    # 复用阶段 3 对项目真实 JSON 字段、深度单位和对齐标记的验证。
    path = Path(__file__).with_name("04_single_frame_pointcloud.py")
    spec = importlib.util.spec_from_file_location("single_frame_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_intrinsics(dataset / "camera_intrinsics.json")


def preflight(dataset, ids):
    paths = [dataset / "camera_intrinsics.json", dataset / "timestamps.csv"]
    paths += [dataset / kind / f"{i:06d}.png" for i in ids for kind in ("color", "depth")]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(missing[:20]))
    timestamps = {}
    with (dataset / "timestamps.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {"frame_id", "color_timestamp_ms", "depth_timestamp_ms"}.issubset(reader.fieldnames or []):
            raise ValueError("timestamps.csv 缺少帧编号或两路时间戳字段。")
        for row in reader:
            i = int(row["frame_id"])
            if i in timestamps:
                raise ValueError(f"CSV 中帧编号重复：{i}")
            values = [float(row[k]) for k in ("color_timestamp_ms", "depth_timestamp_ms")]
            if not np.isfinite(values).all():
                raise ValueError(f"帧 {i} 的时间戳含 NaN / Inf。")
            timestamps[i] = values
    for i in ids:
        if i not in timestamps:
            raise ValueError(f"CSV 中缺少帧 {i} 的时间戳。")
    return timestamps


def load_frame(dataset, i, metadata, timestamps, args):
    color = cv2.imread(str(dataset / "color" / f"{i:06d}.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(dataset / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise ValueError(f"帧 {i} 图片无法解码。")
    shape = (metadata["height"], metadata["width"])
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"帧 {i} Depth 必须是二维 uint16，实际 {depth.dtype} {depth.shape}。")
    if color.dtype != np.uint8 or color.shape != (*shape, 3) or depth.shape != shape:
        raise ValueError(f"帧 {i} RGB / Depth 尺寸与 Color 内参不匹配。")
    meters = depth.astype(np.float64) * metadata["depth_scale"]
    valid = (depth > 0) & (meters >= args.min_depth) & (meters <= args.max_depth)
    count = int(valid.sum())
    if count < 100 or count / depth.size < args.min_valid_ratio:
        raise ValueError(f"帧 {i} 范围内有效深度不足：{count} 像素，比例 {count / depth.size:.2%}。")
    filtered = depth.copy()
    filtered[~valid] = 0  # 保持 uint16；只修改副本，不写回原始 PNG。
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    rgbd = make_rgbd(rgb, filtered, metadata, args, intensity=True)
    stats = {
        "frame_id": i, "depth_dtype": str(depth.dtype), "valid_depth_pixels": count,
        "valid_depth_ratio": count / depth.size,
        "intensity_std": float(np.asarray(rgbd.color).std()),
        "color_timestamp_ms": timestamps[i][0], "depth_timestamp_ms": timestamps[i][1],
        "rgb_depth_timestamp_difference_ms": timestamps[i][0] - timestamps[i][1],
    }
    LOG.info("帧 %d：%dx%d，uint16，有效深度 %.2f%%，灰度标准差 %.4f，RGB-Depth 时间差 %.3f ms",
             i, shape[1], shape[0], 100 * stats["valid_depth_ratio"], stats["intensity_std"],
             stats["rgb_depth_timestamp_difference_ms"])
    if abs(stats["rgb_depth_timestamp_difference_ms"]) > 50:
        LOG.warning("帧 %d 两路时间戳差超过 50 ms；请确认时间戳时钟域、同步和重复深度帧。", i)
    if stats["intensity_std"] < 0.01:
        LOG.warning("帧 %d 灰度纹理变化很小，位姿可能不可靠。", i)
    return {"id": i, "rgb": rgb, "depth": filtered, "rgbd": rgbd, "stats": stats}


def make_rgbd(rgb, depth, metadata, args, intensity):
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0 / metadata["depth_scale"],
        depth_trunc=float(np.nextafter(np.float32(args.max_depth), np.float32(np.inf))),
        convert_rgb_to_intensity=intensity,
    )


def inspect_transform(transform, max_translation, max_rotation):
    """验证 SE(3) 和运动阈值；不修补坏矩阵，也不替换成单位矩阵。"""
    t = np.asarray(transform)
    reasons, metrics = [], {}
    if t.shape != (4, 4) or not np.isfinite(t).all():
        return ["4x4 变换矩阵形状错误或含 NaN / Inf"], metrics
    r = t[:3, :3]
    metrics = {
        "translation_m": float(np.linalg.norm(t[:3, 3])),
        "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))),
        "rotation_orthogonality_error": float(np.linalg.norm(r.T @ r - np.eye(3))),
        "rotation_determinant": float(np.linalg.det(r)),
    }
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        reasons.append("齐次矩阵最后一行异常")
    if metrics["rotation_orthogonality_error"] > 1e-3 or abs(metrics["rotation_determinant"] - 1) > 1e-3:
        reasons.append("旋转矩阵不满足正交性或 det(R)≈1")
    if metrics["translation_m"] > max_translation:
        reasons.append(f"相对平移超过阈值 {max_translation:g} 米")
    if metrics["rotation_deg"] > max_rotation:
        reasons.append(f"相对旋转超过阈值 {max_rotation:g} 度")
    return reasons, metrics


def compose_camera_to_world(previous, source_to_target):
    # p_t=T_s_to_t*p_s，p_w=C_s*p_s=C_s*inv(T_s_to_t)*p_t。
    return previous @ np.linalg.inv(source_to_target)


def depth_diagnostics(source, target, transform, metadata, threshold):
    """独立的重投影深度检查，不是 Open3D 优化器残差或真实位姿误差。

    每隔 4 像素采样 source，用估计矩阵投影到 target，比较目标有效深度。
    """
    sy, sx = np.mgrid[0:source["depth"].shape[0]:4, 0:source["depth"].shape[1]:4]
    z = source["depth"][sy, sx].astype(np.float64) * metadata["depth_scale"]
    valid = z > 0
    z, sx, sy = z[valid], sx[valid], sy[valid]
    points = np.column_stack(((sx - metadata["cx"]) * z / metadata["fx"],
                              (sy - metadata["cy"]) * z / metadata["fy"], z))
    points = points @ transform[:3, :3].T + transform[:3, 3]
    points = points[points[:, 2] > 0]
    x = np.rint(metadata["fx"] * points[:, 0] / points[:, 2] + metadata["cx"]).astype(np.int64)
    y = np.rint(metadata["fy"] * points[:, 1] / points[:, 2] + metadata["cy"]).astype(np.int64)
    inside = (x >= 0) & (x < metadata["width"]) & (y >= 0) & (y < metadata["height"])
    observed = target["depth"][y[inside], x[inside]].astype(np.float64) * metadata["depth_scale"]
    good = observed > 0
    errors = np.abs(observed[good] - points[inside, 2][good])
    return {
        "sampled_source_points": len(z), "projected_valid_target_samples": len(errors),
        "projective_depth_consistent_samples": int((errors <= threshold).sum()),
        "projective_depth_consistency_ratio": float((errors <= threshold).sum() / max(len(z), 1)),
        "projective_depth_mae_m": float(errors.mean()) if len(errors) else None,
        "diagnostic_note": "Independent sampled nearest-pixel depth check; not optimizer residual, fitness, or ground-truth error",
    }


def estimate_pair(source, target, intrinsic, metadata, args):
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_min, option.depth_max = args.min_depth, args.max_depth
    option.depth_diff_max = args.depth_diff_max
    LOG.info("估计 %d → %d（source-to-target，单位矩阵仅作为优化初值）", source["id"], target["id"])
    api_success, transform, information = o3d.pipelines.odometry.compute_rgbd_odometry(
        source["rgbd"], target["rgbd"], intrinsic, np.eye(4),
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option,
    )
    reasons, metrics = inspect_transform(transform, args.max_translation, args.max_rotation)
    if not api_success:
        reasons.append("Open3D API 返回失败")
    info = np.asarray(information)
    if info.shape == (6, 6) and np.isfinite(info).all():
        eigenvalues = np.linalg.eigvalsh((info + info.T) / 2)
        metrics["information_eigenvalues"] = eigenvalues.tolist()
        metrics["information_rank"] = int(np.linalg.matrix_rank(info))
        metrics["information_condition_number"] = float(np.linalg.cond(info))
        if metrics["information_rank"] < 6:
            reasons.append("信息矩阵秩不足，估计存在退化")
    else:
        reasons.append("信息矩阵异常或含 NaN / Inf")
    if np.asarray(transform).shape == (4, 4) and np.isfinite(transform).all() and not reasons:
        metrics.update(depth_diagnostics(source, target, transform, metadata, args.depth_diff_max))
        if metrics["projected_valid_target_samples"] < 50 or metrics["projective_depth_consistency_ratio"] < args.min_consistency:
            reasons.append("独立重投影深度检查的支持数量或比例不足")
    result = {
        "source": source["id"], "target": target["id"], "transform_direction": "source_to_target",
        "point_relation": "p_target = T_source_to_target @ p_source",
        "api_success": bool(api_success), "accepted": not reasons, "failure_reasons": reasons,
        "transformation": np.asarray(transform).tolist(), "information_matrix": info.tolist(),
        "metrics": metrics, "source_stats": source["stats"], "target_stats": target["stats"],
        "reliability_note": "API success and these checks do not establish true pose accuracy; no optimizer residual is provided by this API",
    }
    LOG.info("API 成功：%s；检查后接受：%s\n完整 4x4 source-to-target 矩阵：\n%s",
             api_success, result["accepted"], np.array2string(transform, precision=9, suppress_small=True))
    LOG.info("估计指标：%s", metrics)
    if "translation_m" in metrics:
        LOG.info("相对平移距离：%.6f 米；相对旋转角度：%.6f 度；det(R)=%.9f",
                 metrics["translation_m"], metrics["rotation_deg"], metrics["rotation_determinant"])
    if reasons:
        LOG.error("失败帧对 %d → %d：%s", source["id"], target["id"], "; ".join(reasons))
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None  # JSON 不写 NaN / Inf；失败原因仍保留。
    return value


def write_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(json_safe(data), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def save_trajectory(prefix, rows, pairs, args, metadata):
    pose_fields = [f"c2w_{i}{j}" for i in range(4) for j in range(4)]
    fields = ["frame_id", "color_timestamp_ms", "depth_timestamp_ms", "status", "reason", "pose_convention", "world_frame", "x_m", "y_m", "z_m"] + pose_fields
    with prefix.with_suffix(".csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            pose = row["camera_to_world"]
            data = {k: row[k] for k in ("frame_id", "color_timestamp_ms", "depth_timestamp_ms", "status", "reason")}
            data.update(pose_convention="camera_to_world", world_frame=args.start)
            if pose is not None:
                data.update(zip(pose_fields, np.asarray(pose).ravel()))
                data.update(zip(("x_m", "y_m", "z_m"), np.asarray(pose)[:3, 3]))
            writer.writerow(data)  # 失败/未估计行的位姿留空，绝不填单位矩阵。
    poses = [np.asarray(row["camera_to_world"]) for row in rows if row["camera_to_world"] is not None]
    positions = np.array([pose[:3, 3] for pose in poses]).reshape(-1, 3)
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    summary = {
        "pose_convention": "camera_to_world", "world_frame": args.start,
        "accumulation": "C_target = C_source @ inverse(T_source_to_target)",
        "successful_frames_including_reference": len(poses),
        "successful_pairs": sum(p["accepted"] for p in pairs),
        "failed_frames": sum(r["status"] == "failed" for r in rows),
        "not_estimated_frames": sum(r["status"] == "not_estimated" for r in rows),
        "total_translation_path_m": path_length,
        "complete": all(r["status"] in ("reference", "estimated") for r in rows),
        "settings": vars(args), "camera_intrinsics": metadata, "frames": rows, "pairs": pairs,
    }
    summary["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(prefix.with_suffix(".json"), summary)
    fig = plt.figure(figsize=(11, 5))
    axis = fig.add_subplot(121, projection="3d")
    if len(positions):
        axis.plot(*positions.T, marker=".")
        axis.scatter(*positions[0], color="green", label="Start")
        axis.scatter(*positions[-1], color="red", label="Last accepted")
        span = max(float(np.ptp(positions, axis=0).max()), 0.01)
        center = (positions.min(axis=0) + positions.max(axis=0)) / 2
        axis.legend()
    else:
        axis.text2D(0.1, 0.5, "No valid starting pose", transform=axis.transAxes)
        span, center = 0.01, np.zeros(3)
    axis.set_xlim(center[0] - span / 2, center[0] + span / 2)
    axis.set_ylim(center[1] - span / 2, center[1] + span / 2)
    axis.set_zlim(center[2] - span / 2, center[2] + span / 2)
    axis.set_box_aspect((1, 1, 1))
    axis.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    components = fig.add_subplot(122)
    frame_ids = [r["frame_id"] for r in rows if r["camera_to_world"] is not None]
    for i, label in enumerate("XYZ"):
        components.plot(frame_ids, positions[:, i], label=label)
    components.set(xlabel="Frame ID", ylabel="Camera position (m)")
    components.legend()
    components.grid(True)
    fig.suptitle(f"Camera-to-world: {args.start}-{args.end}; {'complete' if summary['complete'] else 'STOPPED on failure'}")
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)
    LOG.info("成功帧数（含起始参考帧）：%d；成功帧对：%d；失败帧数：%d；未估计：%d；总平移路径：%.6f 米",
             len(poses), summary["successful_pairs"], summary["failed_frames"], summary["not_estimated_frames"], path_length)
    return summary["complete"]


def run_trajectory(ids, timestamps, metadata, intrinsic, args):
    rows = [{"frame_id": i, "color_timestamp_ms": timestamps[i][0], "depth_timestamp_ms": timestamps[i][1],
             "status": "not_estimated", "reason": "", "camera_to_world": None} for i in ids]
    pairs = []
    previous = None
    camera_to_world = np.eye(4)
    for index, i in enumerate(ids):
        try:
            current = load_frame(args.dataset, i, metadata, timestamps, args)
            if previous is None:
                rows[index].update(status="reference", camera_to_world=camera_to_world.tolist())
            else:
                if timestamps[i][0] <= timestamps[previous["id"]][0] or timestamps[i][1] < timestamps[previous["id"]][1]:
                    raise ValueError("时间戳不按序递增，请检查 CSV、重复帧或时钟域。")
                if timestamps[i][1] == timestamps[previous["id"]][1]:
                    LOG.warning("帧 %d → %d 的 Depth 时间戳重复，运动估计需人工复核。", previous["id"], i)
                result = estimate_pair(previous, current, intrinsic, metadata, args)
                pairs.append(result)
                if not result["accepted"]:
                    raise ValueError("; ".join(result["failure_reasons"]))
                camera_to_world = compose_camera_to_world(camera_to_world, np.asarray(result["transformation"]))
                rows[index].update(status="estimated", camera_to_world=camera_to_world.tolist())
            previous = current
        except (Exception, KeyboardInterrupt) as exc:
            reason = str(exc) or "用户中断"
            rows[index].update(status="failed", reason=reason)
            LOG.error("轨迹停止：%s → %d，原因：%s；后续帧不再估计。", ids[index - 1] if index else "起始帧", i, reason)
            break
    # 起始帧无法读取时没有可绘制的位姿，仍保存失败记录。
    if not any(row["camera_to_world"] is not None for row in rows):
        rows[0]["reason"] += "（无有效起始位姿）"
    return save_trajectory(args.output / f"trajectory_{args.start}_{args.end}", rows, pairs, args, metadata)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="阶段 4A：两帧 RGB-D Odometry 或连续 camera-to-world 轨迹")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/extracted_scan_02")
    parser.add_argument("--output", type=Path, default=ROOT / "output/odometry", help="结果目录，已有同名结果不覆盖")
    parser.add_argument("--source", type=int)
    parser.add_argument("--target", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--depth-diff-max", type=float, default=0.03, help="深度对应阈值，米")
    parser.add_argument("--max-translation", type=float, default=0.25, help="每对帧的平移接受上限，米")
    parser.add_argument("--max-rotation", type=float, default=30.0, help="每对帧的旋转接受上限，度")
    parser.add_argument("--min-valid-ratio", type=float, default=0.01)
    parser.add_argument("--min-consistency", type=float, default=0.1, help="独立重投影深度一致性最低比例")
    parser.add_argument("--visualize", action="store_true", help="仅实验 A：显示 source 变换到 target 后的双色点云")
    args = parser.parse_args(argv)
    pair = args.source is not None and args.target is not None and args.start is None and args.end is None
    trajectory = args.start is not None and args.end is not None and args.source is None and args.target is None
    if not (pair or trajectory):
        parser.error("请选择 --source/--target 或 --start/--end，不能混用。")
    ids = [args.source, args.target] if pair else [args.start, args.end]
    if min(ids) < 0 or (pair and args.source == args.target):
        parser.error("帧编号必须非负；实验 A 的两个编号必须不同。")
    if trajectory and (args.end <= args.start or args.step <= 0 or (args.end - args.start) % args.step):
        parser.error("轨迹须满足 end > start、step > 0，且 step 能整除 end-start。")
    if trajectory and args.visualize:
        parser.error("--visualize 仅用于两帧实验。")
    values = [args.min_depth, args.max_depth, args.depth_diff_max, args.max_translation, args.max_rotation, args.min_valid_ratio, args.min_consistency]
    if not np.isfinite(values).all() or not 0 <= args.min_depth < args.max_depth:
        parser.error("深度范围或数值参数无效。")
    if min(args.depth_diff_max, args.max_translation, args.max_rotation) <= 0 or args.max_rotation > 180:
        parser.error("深度对应阈值及运动阈值必须为正，旋转上限不超过 180 度。")
    if not 0 < args.min_valid_ratio <= 1 or not 0 < args.min_consistency <= 1:
        parser.error("比例参数必须在 (0, 1]。")
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    return args


def main(args):
    handler = None
    try:
        ids = [args.source, args.target] if args.source is not None else list(range(args.start, args.end + 1, args.step))
        timestamps = preflight(args.dataset, ids)
        metadata = load_metadata(args.dataset)
        if args.output == args.dataset or args.dataset in args.output.parents:
            raise ValueError("结果目录不能位于输入数据集内。")
        name = f"pair_{args.source}_{args.target}" if args.source is not None else f"trajectory_{args.start}_{args.end}"
        suffixes = [".json"] if args.source is not None else [".csv", ".json", ".png"]
        for suffix in suffixes:
            if (args.output / (name + suffix)).exists():
                raise FileExistsError("已有同名结果，禁止覆盖；请用 --output 指定其他结果目录。")
        args.output.mkdir(parents=True, exist_ok=True)
        LOG.setLevel(logging.INFO)
        handler = logging.FileHandler(args.output / "odometry_log.txt", mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(handler)
        LOG.info("数据集：%s；结果：%s", args.dataset, args.output)
        LOG.info("RealSense scale=%.12g 米/单位；Open3D depth_scale=%.12g 单位/米；内参=%s",
                 metadata["depth_scale"], 1 / metadata["depth_scale"], metadata)
        LOG.info("变换 source→target；轨迹 camera-to-world，起点固定；API 成功不等于真实位姿准确。")
        LOG.info("本 API 不提供优化器真实残差；信息矩阵、纹理、时间戳和独立深度重投影仅作诊断。")
        intrinsic = o3d.camera.PinholeCameraIntrinsic(metadata["width"], metadata["height"], metadata["fx"], metadata["fy"], metadata["cx"], metadata["cy"])
        if args.source is None:
            complete = run_trajectory(ids, timestamps, metadata, intrinsic, args)
            return 0 if complete else 1
        source = load_frame(args.dataset, args.source, metadata, timestamps, args)
        target = load_frame(args.dataset, args.target, metadata, timestamps, args)
        result = estimate_pair(source, target, intrinsic, metadata, args)
        result["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        result["camera_intrinsics"] = metadata
        path = args.output / (name + ".json")
        write_json(path, result)
        LOG.info("两帧结果已保存：%s", path)
        if args.visualize and result["accepted"]:
            clouds = []
            for frame, color in ((source, [1, 0.3, 0.1]), (target, [0.1, 0.6, 1])):
                cloud = o3d.geometry.PointCloud.create_from_rgbd_image(make_rgbd(frame["rgb"], frame["depth"], metadata, args, False), intrinsic)
                cloud.paint_uniform_color(color)
                clouds.append(cloud)
            clouds[0].transform(np.asarray(result["transformation"]))
            o3d.visualization.draw_geometries(clouds, window_name="Source to Target Odometry Check")
        return 0 if result["accepted"] else 1
    except (Exception, KeyboardInterrupt) as exc:
        LOG.error("Odometry 运行失败：%s", str(exc) or "用户中断")
        return 1
    finally:
        if handler is not None:
            LOG.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main(parse_args()))
