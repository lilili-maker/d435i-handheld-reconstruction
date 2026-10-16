"""Tensor/legacy Hybrid Odometry：复用原加载与检查，双向扩展同一参考系。"""

from functools import lru_cache
import importlib.util
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d

from .pose_io import CameraPose, save_poses
from .pose_utils import validate_transform, invert_transform, compose_transforms
from .compute_device import select_device

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("handheld")


@lru_cache(maxsize=1)
def legacy():
    spec = importlib.util.spec_from_file_location("handheld_legacy_odometry", ROOT / "src/05_rgbd_odometry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LOG.setLevel(logging.WARNING)
    return module


def settings(dataset, **overrides):
    defaults = dict(dataset=Path(dataset), min_depth=.2, max_depth=3., depth_diff_max=.03,
                    max_translation=.15, max_rotation=15., min_valid_ratio=.1, min_consistency=.4,
                    min_geometry_fitness=.65, max_geometry_rmse=.02, max_cycle_translation=.02,
                    max_cycle_rotation=2., cloud_voxel=.03, max_photo_mae=.12, max_skew_ms=50.,
                    max_frame_gap_ms=100., odometry_backend="tensor", device="CPU:0", icp_refinement="off",
                    max_skip_frames=0, max_recovery_gap_ms=200.)
    return SimpleNamespace(**(defaults | overrides))


def read_frame(dataset, i, metadata, timestamps, args):
    frame = legacy().load_frame(Path(dataset), i, metadata, timestamps, args)
    if abs(frame["stats"]["rgb_depth_timestamp_difference_ms"]) > args.max_skew_ms:
        raise ValueError(f"帧 {i} RGB-Depth 时间差超过阈值")
    if frame["stats"]["intensity_std"] < .015:
        raise ValueError(f"帧 {i} 纹理过弱，拒绝可靠位姿声明")
    return frame


def camera_cloud(frame, intrinsic, metadata, args):
    # 同一帧会作为前一对的 target 和后一对的 source，避免重复生成/降采样。
    if "_check_cloud" in frame:
        return frame["_check_cloud"]
    rgbd = legacy().make_rgbd(frame["rgb"], frame["depth"], metadata, args, False)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic).voxel_down_sample(args.cloud_voxel)
    if len(cloud.points) < 100:
        raise ValueError("几何检查点数不足")
    frame["_check_cloud"] = cloud
    return cloud


def estimate_pair(source, target, intrinsic, metadata, args):
    if getattr(args, "odometry_backend", "legacy") == "legacy":
        return legacy().estimate_pair(source, target, intrinsic, metadata, args)
    device, _ = select_device(args.device)
    def tensor_rgbd(frame):
        key = "_tensor_rgbd_" + device
        if key not in frame:
            frame[key] = o3d.t.geometry.RGBDImage(
                o3d.t.geometry.Image(o3d.core.Tensor(frame["rgb"], device=o3d.core.Device(device))),
                o3d.t.geometry.Image(o3d.core.Tensor(frame["depth"], device=o3d.core.Device(device))))
        return frame[key]
    started = time.perf_counter()
    tensor_odom = o3d.t.pipelines.odometry
    result = tensor_odom.rgbd_odometry_multi_scale(
        tensor_rgbd(source), tensor_rgbd(target),
        o3d.core.Tensor(intrinsic.intrinsic_matrix, dtype=o3d.core.float64),
        depth_scale=1 / metadata["depth_scale"], depth_max=args.max_depth,
        method=tensor_odom.Method.Hybrid,
        params=tensor_odom.OdometryLossParams(depth_outlier_trunc=args.depth_diff_max,
                                             depth_huber_delta=min(.01, args.depth_diff_max)))
    transform = result.transformation.cpu().numpy()
    reasons, metrics = legacy().inspect_transform(transform, args.max_translation, args.max_rotation)
    fitness, rmse = float(result.fitness), float(result.inlier_rmse)
    metrics.update(tensor_fitness=fitness, tensor_inlier_rmse=rmse,
                   tensor_metric_note="Values reported by Open3D Hybrid API; not ground-truth pose error or legacy residual")
    if not np.isfinite([fitness, rmse]).all() or fitness <= 0:
        reasons.append("Tensor Odometry 没有有效解或有效对应")
    if not reasons:
        metrics.update(legacy().depth_diagnostics(source, target, transform, metadata, args.depth_diff_max))
        if metrics["projected_valid_target_samples"] < 50 or metrics["projective_depth_consistency_ratio"] < args.min_consistency:
            reasons.append("独立重投影深度检查的支持数量或比例不足")
    return {"source": source["id"], "target": target["id"], "transform_direction": "source_to_target",
            "point_relation": "p_target = T_source_to_target @ p_source",
            # Tensor API 没有 legacy 的 success 返回值，不能伪造同等成功标志。
            "api_success": None, "backend": "Open3D tensor Hybrid", "device": device,
            "solver_has_finite_correspondences": np.isfinite([fitness, rmse]).all().item() and fitness > 0,
            "accepted": not reasons, "failure_reasons": reasons, "transformation": transform.tolist(),
            "information_matrix": None, "metrics": metrics, "source_stats": source["stats"],
            "target_stats": target["stats"], "solver_seconds": time.perf_counter() - started}


def cycle_check(forward, reverse, max_translation=.02, max_rotation=2.):
    cycle = compose_transforms(reverse, forward)
    translation = float(np.linalg.norm(cycle[:3, 3]))
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(cycle[:3, :3])-1)/2, -1, 1))))
    return {"translation_m": translation, "rotation_deg": rotation,
            "passed": translation <= max_translation and rotation <= max_rotation}


def geometry_check(source, target, transform, distance=.03):
    forward = o3d.pipelines.registration.evaluate_registration(source, target, distance, transform)
    reverse = o3d.pipelines.registration.evaluate_registration(target, source, distance, invert_transform(transform))
    return {"source_to_target_fitness": float(forward.fitness), "target_to_source_fitness": float(reverse.fitness),
            "source_to_target_rmse_m": float(forward.inlier_rmse), "target_to_source_rmse_m": float(reverse.inlier_rmse),
            "distance_m": distance, "note": "Sampled point-cloud nearest-neighbor surface checks; not ground-truth pose error"}


def photometric_check(source, target, transform, metadata):
    yy, xx = np.mgrid[0:metadata["height"]:4, 0:metadata["width"]:4]
    z = source["depth"][yy, xx].astype(float)*metadata["depth_scale"]
    valid = z > 0
    points = np.column_stack(((xx[valid]-metadata["cx"])*z[valid]/metadata["fx"],
                              (yy[valid]-metadata["cy"])*z[valid]/metadata["fy"], z[valid]))
    points = points @ transform[:3,:3].T + transform[:3,3]
    positive = points[:,2] > 0
    tx = np.zeros(len(points), dtype=int); ty = tx.copy()
    tx[positive] = np.rint(metadata["fx"]*points[positive,0]/points[positive,2]+metadata["cx"]).astype(int)
    ty[positive] = np.rint(metadata["fy"]*points[positive,1]/points[positive,2]+metadata["cy"]).astype(int)
    inside = positive & (tx >= 0) & (tx < metadata["width"]) & (ty >= 0) & (ty < metadata["height"])
    indices = np.flatnonzero(inside)
    observed = target["depth"][ty[indices],tx[indices]].astype(float)*metadata["depth_scale"]
    indices = indices[(observed > 0) & (abs(observed-points[indices,2]) < .03)]
    if len(indices) < 50:
        return {"sample_count": len(indices), "mae": None}
    a = np.asarray(source["rgbd"].color)[yy[valid],xx[valid]][indices]
    b = np.asarray(target["rgbd"].color)[ty[indices],tx[indices]]
    return {"sample_count": len(indices), "mae": float(abs(a-b).mean()),
            "note": "Independent grayscale reprojection diagnostic, affected by exposure; not API residual"}


def information(source, target, transform, distance=.03):
    matrix = np.asarray(o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, distance, transform))
    if matrix.shape != (6,6) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 6:
        raise ValueError("几何信息矩阵非法或秩不足")
    return matrix


def checked_pair(source, target, intrinsic, metadata, args):
    result = estimate_pair(source, target, intrinsic, metadata, args)
    depth_gate = "独立重投影深度检查的支持数量或比例不足"
    result["hybrid_initial_accepted"] = result["accepted"]
    result["hybrid_initial_reasons"] = list(result["failure_reasons"])
    # 数值/API/运动失败不可使用；只有可用初值的深度支持门槛可在几何精化后重新验证。
    if not result["accepted"] and result["failure_reasons"] != [depth_gate]:
        return result
    transform = validate_transform(result["transformation"])
    reverse = estimate_pair(target, source, intrinsic, metadata, args)
    reasons = [r for r in result["failure_reasons"] if r != depth_gate]
    result["failure_reasons"] = reasons
    reverse_usable = reverse["accepted"] or reverse["failure_reasons"] == [depth_gate]
    if not reverse_usable:
        reasons.append("反向 Odometry 检查失败")
    source_cloud, target_cloud = camera_cloud(source, intrinsic, metadata, args), camera_cloud(target, intrinsic, metadata, args)
    result["hybrid_initial_transformation"] = transform.tolist()
    result["icp_refinement"] = getattr(args,"icp_refinement","off")
    if reverse_usable:
        result["hybrid_initial_cycle"] = cycle_check(transform,reverse["transformation"],args.max_cycle_translation,args.max_cycle_rotation)
        if not result["hybrid_initial_cycle"]["passed"]:
            reasons.append("原始 Hybrid 正反向不一致，禁止用几何 ICP 掩盖")
        reverse_transform=validate_transform(reverse["transformation"])
        # 默认保留 Hybrid 的光度+几何解；平面主导的几何 ICP 可引入切向漂移。
        # geometry 仅作为显式诊断选项，不能由初始循环失败变成接受状态。
        if result["icp_refinement"] == "geometry":
            for cloud in (source_cloud,target_cloud):
                if not cloud.has_normals():
                    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.09,max_nn=30))
            reg=o3d.pipelines.registration
            refined=reg.registration_icp(source_cloud,target_cloud,.05,transform,reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(max_iteration=60))
            refined_reverse=reg.registration_icp(target_cloud,source_cloud,.05,reverse_transform,reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(max_iteration=60))
            transform=validate_transform(refined.transformation)
            reverse_transform=validate_transform(refined_reverse.transformation)
        elif result["icp_refinement"] != "off":
            raise ValueError("icp_refinement 必须是 off 或 geometry")
        refinement_cycle=cycle_check(result["hybrid_initial_transformation"],invert_transform(transform),.08,5.)
        result["refinement_change"]=refinement_cycle
        if not refinement_cycle["passed"]:
            reasons.append("局部精化偏离原 Odometry 过大")
        cycle=cycle_check(transform,reverse_transform,args.max_cycle_translation,args.max_cycle_rotation)
        result["bidirectional_cycle"]=cycle
        if not cycle["passed"]:
            reasons.append("正反向精化变换不一致")
        motion_reasons,motion_metrics=legacy().inspect_transform(transform,args.max_translation,args.max_rotation)
        reasons.extend(motion_reasons);result["refined_motion"]=motion_metrics
        depth=legacy().depth_diagnostics(source,target,transform,metadata,args.depth_diff_max)
        result["refined_depth_diagnostics"]=depth
        reverse_depth=legacy().depth_diagnostics(target,source,invert_transform(transform),metadata,args.depth_diff_max)
        result["refined_reverse_depth_diagnostics"]=reverse_depth
        if min(depth["projective_depth_consistency_ratio"],reverse_depth["projective_depth_consistency_ratio"])<args.min_consistency or min(depth["projected_valid_target_samples"],reverse_depth["projected_valid_target_samples"])<50:
            reasons.append("精化后深度重投影支持不足")
        result["transformation"]=transform.tolist()
        result["selection"]="hybrid_without_geometric_override" if result["icp_refinement"]=="off" else "explicit_geometry_refinement"
    geometry = geometry_check(source_cloud, target_cloud, transform)
    result["geometry"] = geometry
    if min(geometry["source_to_target_fitness"], geometry["target_to_source_fitness"]) < args.min_geometry_fitness or max(geometry["source_to_target_rmse_m"],geometry["target_to_source_rmse_m"]) > args.max_geometry_rmse:
        reasons.append("双向几何重叠或距离检查不通过")
    photo = photometric_check(source, target, transform, metadata)
    result["photometric"] = photo
    if photo["mae"] is None or photo["mae"] > args.max_photo_mae:
        reasons.append("独立灰度重投影误差过大/支持不足")
    result["reverse_metrics"] = reverse["metrics"]
    result["graph_information_matrix"] = information(source_cloud, target_cloud, transform).tolist()
    result["accepted"] = not reasons
    return result


def track(dataset, ids, reference, output, args):
    """双向扩展；可有界尝试最后可靠帧→后续候选帧，始终实测重配准，不补位姿。"""
    dataset, output = Path(dataset), Path(output)
    max_skip = getattr(args, "max_skip_frames", 0)
    max_gap = getattr(args, "max_recovery_gap_ms", 200.)
    if type(max_skip) is not int or max_skip < 0 or not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("max_skip_frames 须为非负整数；max_recovery_gap_ms 须为正有限毫秒数")
    if reference not in ids or len(ids) < 2:
        raise ValueError("参考帧必须属于至少两帧的采样序列")
    timestamps = legacy().preflight(dataset, ids)
    metadata = legacy().load_metadata(dataset)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(metadata["width"],metadata["height"],metadata["fx"],metadata["fy"],metadata["cx"],metadata["cy"])
    output.mkdir(parents=True, exist_ok=False)
    reference_frame = read_frame(dataset, reference, metadata, timestamps, args)
    matrices = {reference: np.eye(4)}  # 已知参考系定义，不是失败帧补齐。
    states = {i: {"status": "not_estimated", "reason": "未跟踪；不能用于融合"} for i in ids}
    states[reference] = {"status": "reference", "reason": ""}
    pairs, failures, recovery_bridges = [], [], []
    interrupted = False
    center = ids.index(reference)
    for direction, sequence in (("backward", list(reversed(ids[:center]))), ("forward", ids[center+1:])):
        previous = reference_frame
        rejected_since_anchor = []
        for position, i in enumerate(sequence):
            entry = {"source": previous["id"], "target": i, "direction": direction}
            retry_allowed = False
            try:
                elapsed = np.array(timestamps[i]) - np.array(timestamps[previous["id"]])
                expected_sign = -1 if direction == "backward" else 1
                if np.any(elapsed*expected_sign <= 0) or np.any(abs(elapsed) > args.max_frame_gap_ms*abs(i-previous["id"])):
                    raise ValueError("相邻所选帧时间顺序/间隔异常")
                if rejected_since_anchor and np.any(abs(elapsed) > max_gap):
                    raise ValueError(f"恢复候选距最后可靠帧超过 {max_gap:g}ms；禁止跨越失跟踪区间")
                retry_allowed = True
                current = read_frame(dataset, i, metadata, timestamps, args)
                pair_started = time.perf_counter()
                result = checked_pair(previous, current, intrinsic, metadata, args)
                result["checked_pair_seconds"] = time.perf_counter() - pair_started
                entry.update(result)
                pairs.append(entry)
                if not result["accepted"]:
                    raise ValueError("; ".join(result["failure_reasons"]))
                matrices[i] = validate_transform(legacy().compose_camera_to_world(matrices[previous["id"]], np.asarray(result["transformation"])))
                states[i] = {"status": "estimated", "reason": ""}
                if rejected_since_anchor:
                    bridge = {"source": previous["id"], "target": i, "direction": direction,
                              "skipped_frame_ids": list(rejected_since_anchor), "elapsed_ms": abs(elapsed).tolist()}
                    entry["recovery_bridge"] = bridge
                    recovery_bridges.append(bridge)
                    LOG.info("重配准恢复 %d→%d；跳过 %s，所有原质量检查通过", previous["id"], i, rejected_since_anchor)
                    rejected_since_anchor.clear()
                previous = current
                LOG.info("跟踪 %d→%d 接受，累计有效帧 %d", entry["source"], i, len(matrices))
            except (Exception, KeyboardInterrupt) as exc:
                reason = str(exc) or "用户中断"
                states[i] = {"status": "failed", "reason": reason}
                entry.update(accepted=False, error=reason)
                if not pairs or pairs[-1] is not entry:
                    pairs.append(entry)
                failures.append({"source": entry["source"], "target": i, "direction": direction, "reason": reason})
                rejected_since_anchor.append(i)
                # 不扩大运动/残差/正反向阈值；未知程序错误、时间异常及中断均不可绕过。
                can_retry = (retry_allowed and isinstance(exc, (ValueError, RuntimeError)) and
                             len(rejected_since_anchor) <= max_skip and position + 1 < len(sequence))
                entry["retry_next_candidate"] = can_retry
                if can_retry:
                    LOG.warning("跳过异常帧 %d（%d→%d）：%s；继续从最后可靠帧 %d 尝试下一帧", i, entry["source"], i, reason, previous["id"])
                    continue
                LOG.error("%s 片段停止于 %d→%d：%s；未估计帧不融合", direction, entry["source"], i, reason)
                if isinstance(exc, KeyboardInterrupt):
                    interrupted = True
                break
        if interrupted:
            break
    poses = [CameraPose(frame_id=i, timestamp=timestamps[i][0]/1000, T_world_camera=matrices.get(i),
                       pose_source="rgbd_odometry", coordinate_frame=f"odometry_camera_{reference}",
                       timestamp_reference="RealSense Color frame timestamp; clock domain not asserted as Unix",
                       validity=i in matrices, status=states[i]["status"], reason=states[i]["reason"],
                       source_metadata={"color_timestamp_ms": timestamps[i][0], "depth_timestamp_ms": timestamps[i][1]}) for i in ids]
    save_poses(output / "poses_before.json", poses, metadata={"reference_frame": reference, "source": "validated_bidirectional_rgbd_odometry",
               "pose_convention": "T_world_camera", "reliability_note": "Checks are not ground-truth accuracy; failed/untracked frames have null matrices"})
    attempted = len(pairs)
    report = {"requested_frames": ids, "reference_frame": reference, "valid_frame_ids": sorted(matrices),
              "valid_frame_count": len(matrices), "attempted_pairs": attempted, "accepted_pairs": sum(p["accepted"] for p in pairs),
              "attempted_pair_success_ratio": sum(p["accepted"] for p in pairs)/max(attempted,1),
              "not_estimated_frames": [i for i in ids if states[i]["status"] == "not_estimated"],
              "failed_frame_ids": [i for i in ids if states[i]["status"] == "failed"],
              "skipped_frame_ids": sorted({i for b in recovery_bridges for i in b["skipped_frame_ids"]}),
              "recovery_bridges": recovery_bridges,
              "failures": failures, "pairs": pairs, "settings": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "interrupted": interrupted,
              "transform_direction": "source_to_target", "accumulation": "C_target=C_source@inverse(T_source_to_target)"}
    legacy().write_json(output / "registration_quality.json", report)
    if interrupted:
        # Ctrl+C 保存此前已验位姿后仍向入口传播中断，绝不擅自继续重建。
        raise KeyboardInterrupt("用户中断，已保存此前有效位姿与配准诊断")
    return poses, report, metadata, timestamps, intrinsic
