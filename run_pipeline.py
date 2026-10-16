"""统一离线入口：只编排已验证模块，不复制重建、裁剪或切片算法。"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import logging
import math
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from reconstruction import clean_mesh as cm
from reconstruction import merge_pointclouds as merge
from reconstruction import mesh_to_svg as slicing
from reconstruction import tsdf_reconstruction as tsdf
from reconstruction.pose_io import load_poses, require_pose

LOG = logging.getLogger("pipeline")
DEFAULTS = {
    "dataset_path": "data/extracted_scan_02", "camera_poses_path": "output/poses/trajectory_500_550.json",
    "start_frame": 500, "end_frame": 550, "frame_step": 1, "frame_ids": None,
    "min_depth": .2, "max_depth": 3., "pointcloud_voxel_size": .005,
    "tsdf_voxel_size": .01, "sdf_trunc": .04, "mesh_crop_config": None,
    "mesh_input_path": None, "slicing_axis": "Z", "slicing_spacing_mm": 10.,
    "slicing_start_mm": None, "slicing_end_mm": None,
    "output_directory": "output/pipeline_runs", "enable_input_check": True,
    "enable_pointcloud_merge": False, "enable_tsdf": True, "enable_mesh_crop": False,
    "enable_mesh_clean": False, "enable_slicing": True, "enable_report": True,
    "slicing_preview": False, "min_component_triangles": 0, "simplify_target_triangles": None,
    "max_translation": .25, "max_rotation": 30., "max_pointcloud_voxels": 4_000_000,
    "diagnostic_distance": .03,
    "compute_device": "CPU:0",
}
PATH_KEYS = ("dataset_path", "camera_poses_path", "mesh_crop_config", "mesh_input_path", "output_directory")
FLAGS = tuple(k for k, v in DEFAULTS.items() if type(v) is bool)
STAGE_NAMES = ("input_check", "pointcloud_merge", "tsdf", "mesh_processing", "slicing", "report")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_hash(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"JSON 出现重复字段：{key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"JSON 禁止非有限数值：{value}")

    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=pairs, parse_constant=reject)


def write_json(path: Path, data):
    # 报告、检查点、配置也排他创建；不重写旧运行的任何文件。
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_config(path: Path) -> dict:
    provided = read_json(path)
    if not isinstance(provided, dict) or set(provided) - set(DEFAULTS):
        raise ValueError(f"配置必须是对象且无未知字段：{set(provided) - set(DEFAULTS) if isinstance(provided, dict) else type(provided)}")
    cfg = {**DEFAULTS, **provided}
    if cfg["compute_device"] not in ("auto", "CPU:0", "CUDA:0"):
        raise ValueError("compute_device 必须是 auto、CPU:0 或 CUDA:0")
    for key in FLAGS:
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} 必须是 JSON 布尔值。")
    for key in ("start_frame", "end_frame", "frame_step", "min_component_triangles", "max_pointcloud_voxels"):
        if type(cfg[key]) is not int:
            raise ValueError(f"{key} 必须是整数。")
    merge.selected_frames(cfg["start_frame"], cfg["end_frame"], cfg["frame_step"], cfg["frame_ids"])
    numeric = ("min_depth", "max_depth", "pointcloud_voxel_size", "tsdf_voxel_size", "sdf_trunc",
               "slicing_spacing_mm", "max_translation", "max_rotation", "diagnostic_distance")
    for key in (*numeric, "slicing_start_mm", "slicing_end_mm"):
        value = cfg[key]
        if value is None and key in ("slicing_start_mm", "slicing_end_mm"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} 必须是有限数值。")
    if not 0 <= cfg["min_depth"] < cfg["max_depth"]:
        raise ValueError("深度必须满足 0 <= min_depth < max_depth（米）。")
    for key in ("pointcloud_voxel_size", "tsdf_voxel_size", "slicing_spacing_mm", "max_translation", "diagnostic_distance"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} 必须大于零。")
    if cfg["sdf_trunc"] < cfg["tsdf_voxel_size"] or not 0 < cfg["max_rotation"] <= 180:
        raise ValueError("sdf_trunc 至少一个体素宽度；旋转筛查阈值必须在 (0,180] 度。")
    if cfg["min_component_triangles"] < 0 or cfg["max_pointcloud_voxels"] <= 0:
        raise ValueError("组件阈值须非负，点云体素上限须为正。")
    target = cfg["simplify_target_triangles"]
    if target is not None and (type(target) is not int or target <= 0):
        raise ValueError("simplify_target_triangles 必须为正整数或 null。")
    if not isinstance(cfg["slicing_axis"], str) or cfg["slicing_axis"].upper() not in slicing.AXES:
        raise ValueError("slicing_axis 必须是 X/Y/Z。")
    cfg["slicing_axis"] = cfg["slicing_axis"].upper()
    if cfg["slicing_start_mm"] is not None and cfg["slicing_end_mm"] is not None:
        slicing.slice_positions(cfg["slicing_start_mm"] / 1000, cfg["slicing_end_mm"] / 1000, cfg["slicing_spacing_mm"])
    for key in PATH_KEYS:
        value = cfg[key]
        if value is None and key in ("mesh_crop_config", "mesh_input_path"):
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} 必须是非空路径字符串。")
        candidate = Path(value)
        cfg[key] = str((candidate if candidate.is_absolute() else ROOT / candidate).resolve())
    if cfg["enable_mesh_crop"] and cfg["mesh_crop_config"] is None:
        raise ValueError("启用裁剪必须明确提供 mesh_crop_config；可关闭裁剪或指定已有 mesh_input_path，禁止猜测范围。")
    if not cfg["enable_tsdf"] and not cfg["mesh_input_path"] and any(cfg[k] for k in ("enable_mesh_crop", "enable_mesh_clean", "enable_slicing")):
        raise ValueError("关闭 TSDF 后，Mesh 处理/切片需要显式指定 mesh_input_path。")
    destination = Path(cfg["output_directory"])
    protected = [ROOT / "data", Path(cfg["dataset_path"]), Path(cfg["camera_poses_path"]).parent]
    for key in ("mesh_crop_config", "mesh_input_path"):
        if cfg[key]:
            protected.append(Path(cfg[key]))
    if any(destination == p or p in destination.parents for p in protected):
        raise ValueError("输出运行目录不能位于受保护的输入数据或位姿目录中。")
    return cfg


def allocate_run(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name[4:]) for p in base.iterdir() if re.fullmatch(r"run_\d+", p.name)]
    number = max(numbers, default=0) + 1
    while True:
        candidate = base / f"run_{number:04d}"
        try:
            candidate.mkdir()  # 排他创建，多个进程也不能选中同一个运行目录。
            break
        except FileExistsError:
            number += 1
    for name in ("pointcloud", "mesh", "reports"):
        (candidate / name).mkdir()
    return candidate


def inspect_inputs(cfg: dict) -> tuple[dict, dict]:
    dataset, pose_path = Path(cfg["dataset_path"]), Path(cfg["camera_poses_path"])
    ids = merge.selected_frames(cfg["start_frame"], cfg["end_frame"], cfg["frame_step"], cfg.get("frame_ids"))
    metadata = merge.single_frame_module().load_intrinsics(dataset / "camera_intrinsics.json")
    poses, timestamps = load_poses(pose_path), merge.read_timestamps(dataset)
    issues, _, _ = merge.check_inputs(dataset, ids, poses, timestamps)
    if issues:
        raise ValueError("输入检查失败：\n" + "\n".join(issues))
    world, motion = tsdf.check_motion(poses, ids, cfg["max_translation"], cfg["max_rotation"])
    if any(not row["passed"] for row in motion):
        raise ValueError("相邻选中帧运动超过筛查阈值，禁止继续重建。")
    paths = [pose_path, dataset / "camera_intrinsics.json", dataset / "timestamps.csv"]
    paths += [dataset / kind / f"{i:06d}.png" for i in ids for kind in ("color", "depth")]
    manifest = {str(p.resolve()): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths}
    additional = {}
    if cfg["enable_mesh_crop"]:
        crop_path = Path(cfg["mesh_crop_config"])
        crop = cm.validate_crop_config(read_json(crop_path))
        additional["crop_config"] = {"path": str(crop_path), "sha256": sha256(crop_path), "config": crop}
    if cfg["mesh_input_path"]:
        mesh_path = Path(cfg["mesh_input_path"])
        cm.read_mesh(mesh_path)
        additional["external_mesh"] = {"path": str(mesh_path), "sha256": sha256(mesh_path)}
    frames = []
    for i in ids:
        # 重用原模块解码、uint16、分辨率、深度比例及有效像素检查，不复制规则。
        rgbd, stats = tsdf.load_rgbd(dataset, i, metadata, cfg["min_depth"], cfg["max_depth"])
        del rgbd
        pose = require_pose(poses, i, coordinate_frame=world)
        frames.append({"frame_id": i, "pose_source": pose.pose_source, "depth": stats})
    result = {"selected_frames": ids, "pose_sources": sorted({poses[i].pose_source for i in ids}),
              "coordinate_frame": world, "pose_convention": "T_world_camera", "translation_unit": "m",
              "camera_intrinsics": metadata, "frames": frames, "motion_checks": motion,
              "additional_inputs": additional}
    assert_input_hashes(manifest)
    return result, manifest


def assert_input_hashes(manifest: dict):
    for path, expected in manifest.items():
        if not Path(path).is_file() or sha256(Path(path)) != expected["sha256"]:
            raise ValueError(f"输入文件在运行中变化，停止并禁止复用当前处理结果：{path}")


def execute_algorithm(stage: int, cfg: dict, run: Path, mesh_input: Path | None) -> list[Path]:
    """参数适配层；算法全部调用现有模块。返回当前处理环节的新产物。"""
    common = ["--dataset", cfg["dataset_path"], "--poses", cfg["camera_poses_path"],
              "--start", str(cfg["start_frame"]), "--end", str(cfg["end_frame"]), "--step", str(cfg["frame_step"]),
              "--min-depth", str(cfg["min_depth"]), "--max-depth", str(cfg["max_depth"])]
    if cfg.get("frame_ids") is not None:
        common += ["--frames", *map(str, cfg["frame_ids"])]
    if stage == 2:
        output = run / "pointcloud/merged.ply"
        merge.merge(merge.parse_args(common + ["--output", str(output), "--voxel-size", str(cfg["pointcloud_voxel_size"]),
                    "--max-voxels", str(cfg["max_pointcloud_voxels"]), "--diagnostic-distance", str(cfg["diagnostic_distance"])]))
        return sorted((run / "pointcloud").iterdir())
    if stage == 3:
        output = run / "mesh/scene_mesh.ply"
        tsdf.reconstruct(tsdf.parse_args(common + ["--output", str(output), "--voxel-size", str(cfg["tsdf_voxel_size"]),
                  "--sdf-trunc", str(cfg["sdf_trunc"]), "--max-translation", str(cfg["max_translation"]),
                  "--max-rotation", str(cfg["max_rotation"]), "--device", cfg["compute_device"]]))
        return [output, output.with_name("tsdf_report_scene_mesh.json")]
    if stage == 4:
        output, report_path = run / "mesh/processed_mesh.ply", run / "reports/mesh_processing.json"
        if cfg["enable_mesh_crop"]:
            args = ["--input", str(mesh_input), "--output", str(output), "--crop-config", cfg["mesh_crop_config"], "--report", str(report_path)]
            if cfg["enable_mesh_clean"]:
                args += ["--clean", "--min-component-triangles", str(cfg["min_component_triangles"])]
                if cfg["simplify_target_triangles"] is not None:
                    args += ["--simplify-target-triangles", str(cfg["simplify_target_triangles"])]
            cm.run(cm.parse_args(args))
        else:
            # CLI 只允许带框裁剪；纯清理调用其已验证函数，不制造一个虚假的裁剪框。
            mesh = cm.read_mesh(mesh_input)
            original = cm.inspect_mesh(mesh)
            cleaned, operations = cm.clean_mesh(mesh, remove_degenerate=True, remove_duplicates=True,
                   remove_unreferenced=True, min_component_triangles=cfg["min_component_triangles"],
                   simplify_target_triangles=cfg["simplify_target_triangles"])
            output_info = cm.write_mesh_checked(output, cleaned)
            write_json(report_path, {"status": "complete", "mode": "clean_only", "input": str(mesh_input),
                       "input_sha256": sha256(mesh_input), "unit": "m", "original": original,
                       "cleaned": cm.inspect_mesh(cleaned), "cleaning_operations": operations, "output": output_info,
                       "target_identity": "Not verified; cleaning does not identify a chair"})
        return [output, report_path]
    if stage == 5:
        report = slicing.run_slicing(mesh_input, run / "slices", cfg["slicing_axis"], cfg["slicing_spacing_mm"],
                                    cfg["slicing_start_mm"], cfg["slicing_end_mm"], cfg["slicing_preview"])
        if report["failed_slice_count"] or report.get("preview_failure_reason"):
            raise ValueError("SVG/预览生成失败，具体原因见 slices/slicing_report.json。")
        return sorted((run / "slices").iterdir())
    raise ValueError(f"未知工作流操作：{stage}")


def mesh_stats(path: Path, *, require_colors: bool = False) -> dict:
    mesh = cm.read_mesh(path)
    if require_colors and not mesh.has_vertex_colors():
        raise ValueError(f"彩色 Mesh 丢失 RGB：{path}")
    return {"path": str(path), "vertex_count": len(mesh.vertices), "triangle_count": len(mesh.triangles),
            "has_vertex_colors": mesh.has_vertex_colors(),
            "xyz_bounds_m": {"min": mesh.get_min_bound().tolist(), "max": mesh.get_max_bound().tolist()}}


def validate_outputs(stage: int, cfg: dict, run: Path, mesh_input: Path | None) -> dict:
    """对本次真实文件回读；计数和毫米坐标必须与模块报告一致。"""
    expected_frames = len(merge.selected_frames(cfg["start_frame"], cfg["end_frame"], cfg["frame_step"], cfg.get("frame_ids")))
    if stage == 2:
        report = read_json(run / "pointcloud/merge_merged_report.json")
        cloud = o3d.io.read_point_cloud(str(run / "pointcloud/merged.ply"))
        if report["status"] != "complete" or report["processed_frame_count"] != expected_frames:
            raise ValueError("点云报告的完成状态/处理帧数不符。")
        if len(cloud.points) == 0 or not cloud.has_colors() or not np.isfinite(np.asarray(cloud.points)).all() or not np.isfinite(np.asarray(cloud.colors)).all():
            raise ValueError("融合 PLY 为空、坐标非法或没有 RGB。")
        if len(cloud.points) != report["points_after_fusion"]:
            raise ValueError("融合 PLY 点数与报告不符。")
        return {"path": str(run / "pointcloud/merged.ply"), "point_count": len(cloud.points), "has_colors": True,
                "pose_sources": report["pose_sources"]}
    if stage in (3, 4):
        path = run / ("mesh/scene_mesh.ply" if stage == 3 else "mesh/processed_mesh.ply")
        colors = stage == 3 or cm.read_mesh(mesh_input).has_vertex_colors()
        stats = mesh_stats(path, require_colors=colors)
        report = read_json(run / ("mesh/tsdf_report_scene_mesh.json" if stage == 3 else "reports/mesh_processing.json"))
        claimed = report if stage == 3 else report.get("cleaned", report.get("cropped"))
        if report["status"] != "complete" or any(stats[key] != claimed[key] for key in ("vertex_count", "triangle_count")):
            raise ValueError("Mesh 实际顶点/面数与模块报告不符。")
        if stage == 3 and report["integrated_frame_count"] != expected_frames:
            raise ValueError("TSDF 实际融合帧数与配置不符。")
        return stats
    if stage == 5:
        report = read_json(run / "slices/slicing_report.json")
        mesh = slicing.load_triangle_mesh(mesh_input)
        if report["input_sha256"] != sha256(mesh_input) or report["failed_slice_count"]:
            raise ValueError("切片输入 Mesh 校验或完成状态不符。")
        index = slicing.AXES[cfg["slicing_axis"]][0]
        start = mesh.bounds[0, index] if cfg["slicing_start_mm"] is None else cfg["slicing_start_mm"] / 1000
        end = mesh.bounds[1, index] if cfg["slicing_end_mm"] is None else cfg["slicing_end_mm"] / 1000
        positions = slicing.slice_positions(float(start), float(end), cfg["slicing_spacing_mm"])
        if len(positions) != len(report["layers"]):
            raise ValueError("切片层数与配置和 Mesh 范围不符。")
        actual = {"svg_count": 0, "open_path_count": 0, "closed_path_count": 0, "empty_slice_count": 0}
        for i, (layer, position) in enumerate(zip(report["layers"], positions)):
            path = run / "slices" / f"slice_{i:03d}.svg"
            if layer["index"] != i or not math.isclose(layer["position_m"], position, abs_tol=1e-12, rel_tol=0):
                raise ValueError("切片层编号或平面位置不符。")
            contours, _ = slicing.intersect_layer(mesh, cfg["slicing_axis"], float(position))
            slicing.verify_svg(path, contours, cfg["slicing_axis"])
            opened, closed = sum(not p["closed"] for p in contours), sum(p["closed"] for p in contours)
            if (layer["path_count"], layer["open_path_count"], layer["closed_path_count"]) != (len(contours), opened, closed):
                raise ValueError(f"层 {i} SVG 路径数量/闭合状态与报告不符。")
            actual["svg_count"] += 1
            actual["open_path_count"] += opened
            actual["closed_path_count"] += closed
            actual["empty_slice_count"] += not contours
        if len(list((run / "slices").glob("*.svg"))) != actual["svg_count"] or any(report[k] != value for k, value in actual.items()):
            raise ValueError("实际 SVG 数量/开放闭合统计与报告不符。")
        if actual["open_path_count"] + actual["closed_path_count"] == 0:
            raise ValueError("全部切片为空，没有生成有效矢量交线；请检查切片轴及范围。")
        return {**actual, "total_slice_count": len(positions), "valid_slice_count": len(positions) - actual["empty_slice_count"],
                "path": str(run / "slices"), "millimeter_units_and_actual_geometry_verified": True,
                "watertight": report["watertight"]}
    raise ValueError("未知输出检查操作。")


def fingerprint(stage: int, cfg: dict, inputs: dict, manifest: dict, mesh_input: Path | None) -> tuple[str, dict]:
    keys = {
        2: ("start_frame", "end_frame", "frame_step", "frame_ids", "min_depth", "max_depth", "pointcloud_voxel_size", "max_pointcloud_voxels", "diagnostic_distance"),
        3: ("start_frame", "end_frame", "frame_step", "frame_ids", "min_depth", "max_depth", "tsdf_voxel_size", "sdf_trunc", "max_translation", "max_rotation", "compute_device"),
        4: ("enable_mesh_crop", "enable_mesh_clean", "min_component_triangles", "simplify_target_triangles"),
        5: ("slicing_axis", "slicing_spacing_mm", "slicing_start_mm", "slicing_end_mm", "slicing_preview"),
    }
    files = {2: [merge.__file__, str(ROOT / "src/04_single_frame_pointcloud.py"), str(ROOT / "src/reconstruction/pose_io.py"), str(ROOT / "src/reconstruction/pose_utils.py")],
             3: [tsdf.__file__, merge.__file__, str(ROOT / "src/04_single_frame_pointcloud.py"), str(ROOT / "src/reconstruction/pose_io.py"), str(ROOT / "src/reconstruction/pose_utils.py")],
             4: [cm.__file__], 5: [slicing.__file__]}[stage] + [__file__]
    snapshot = {"parameters": {k: cfg[k] for k in keys[stage]},
                "implementation_sha256": {str(Path(p).resolve()): sha256(Path(p)) for p in files},
                "environment": {name: version(name) for name in ("numpy", "open3d", "opencv-python", "trimesh", "matplotlib")},
                "python_version": sys.version}
    if stage == 3:
        snapshot["cuda_available"] = o3d.core.cuda.is_available()
        snapshot["implementation_sha256"][str(ROOT / "src/reconstruction/compute_device.py")] = sha256(ROOT / "src/reconstruction/compute_device.py")
    if stage in (2, 3):
        snapshot["inputs"] = manifest
    else:
        snapshot["mesh_sha256"] = sha256(mesh_input)
        if stage == 4 and cfg["enable_mesh_crop"]:
            snapshot["crop_sha256"] = inputs["additional_inputs"]["crop_config"]["sha256"]
    return canonical_hash(snapshot), snapshot


def output_manifest(run: Path, paths: list[Path]) -> dict:
    result = {}
    for path in paths:
        path = path.resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file():
            raise ValueError(f"输出必须是本次运行目录中的实际文件：{path}")
        result[path.relative_to(run).as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return result


def resume_source(resume: str | Path | None, cfg: dict) -> Path | None:
    if resume is None:
        return None
    if str(resume) == "latest":
        base = Path(cfg["output_directory"])
        candidates = [p for p in base.glob("run_*") if re.fullmatch(r"run_\d+", p.name) and (p / "pipeline_config.json").is_file()]
        if not candidates:
            raise ValueError("没有可恢复的运行；先运行一次或指定 --resume 运行目录。")
        source = max(candidates, key=lambda p: int(p.name[4:]))
    else:
        source = Path(resume).resolve()
    if not (source / "pipeline_config.json").is_file():
        raise ValueError(f"恢复目录缺少 pipeline_config.json：{source}")
    # 配置本身必须合法；每个检查点还会核对该已保存配置的实际文件哈希。
    load_config(source / "pipeline_config.json")
    return source


def copy_cached_stage(source: Path, run: Path, checkpoint: dict):
    def relocate(data):
        if isinstance(data, dict):
            return {k: relocate(v) for k, v in data.items()}
        if isinstance(data, list):
            return [relocate(v) for v in data]
        if isinstance(data, str):
            for old, new in ((str(source), str(run)), (source.as_posix(), run.as_posix())):
                if data.startswith(old + "/") or data.startswith(old + "\\"):
                    return new + data[len(old):]
        return data

    for relative, expected in checkpoint["artifacts"].items():
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".json":
            payload = relocate(read_json(source / relative))
            if isinstance(payload, dict):
                payload["pipeline_reuse"] = {"source_run": str(source), "original_report_sha256": expected["sha256"]}
            write_json(target, payload)
        else:
            with (source / relative).open("rb") as original, target.open("xb") as saved:
                shutil.copyfileobj(original, saved)
            if sha256(target) != expected["sha256"]:
                raise ValueError(f"恢复复制期间输出内容变化：{source / relative}")


def cache_checkpoint(source: Path | None, stage: int, expected_fingerprint: str) -> tuple[dict | None, str]:
    if source is None:
        return None, "未请求恢复"
    try:
        checkpoint = read_json(source / "reports" / f"stage_{stage}_checkpoint.json")
        if checkpoint["status"] not in ("complete", "reused"):
            return None, "旧处理未成功完成"
        if checkpoint["configuration_sha256"] != sha256(source / "pipeline_config.json"):
            return None, "旧运行保存的配置文件已变化"
        if checkpoint["fingerprint"] != expected_fingerprint:
            return None, "处理参数、输入或代码/环境校验变化"
        if not checkpoint["artifacts"]:
            return None, "旧处理没有产物校验清单"
        for relative, expected in checkpoint["artifacts"].items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                return None, "旧产物路径不是规范的运行目录相对路径"
            file = (source / relative).resolve()
            if not file.is_relative_to(source.resolve()) or not file.is_file() or sha256(file) != expected["sha256"]:
                return None, f"旧产物缺失或内容变化：{relative}"
        return checkpoint, "处理依赖、配置与全部产物哈希匹配"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"旧检查点不可用：{exc}"


def perform_stage(stage: int, cfg: dict, run: Path, inputs: dict, manifest: dict,
                  mesh_input: Path | None, source: Path | None, row: dict):
    started = time.perf_counter()
    row["status"] = "running"
    digest, snapshot = fingerprint(stage, cfg, inputs, manifest, mesh_input)
    checkpoint, reason = cache_checkpoint(source, stage, digest)
    row["cache_check"] = reason
    paths = []
    if checkpoint is not None:
        # 哈希通过后还在旧文件上实际回读；不因文件存在或旧报告声称成功就跳过。
        try:
            validate_outputs(stage, cfg, source, mesh_input)
        except Exception as exc:
            checkpoint = None
            row["cache_check"] = f"旧产物语义/几何校验失败：{exc}"
    log_path = run / "reports" / f"stage_{stage}.log"
    with log_path.open("x", encoding="utf-8") as log:
        handler = logging.StreamHandler(log)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        try:
            LOG.info("工作流 %s：%s", STAGE_NAMES[stage - 1], row["cache_check"])
            with redirect_stdout(log), redirect_stderr(log):
                if checkpoint is not None:
                    copy_cached_stage(source, run, checkpoint)
                    paths = [run / p for p in checkpoint["artifacts"]]
                    row.update(status="reused", reused_from=str(source))
                else:
                    paths = execute_algorithm(stage, cfg, run, mesh_input)
                    row["status"] = "complete"
                row["statistics"] = validate_outputs(stage, cfg, run, mesh_input)
                assert_input_hashes(manifest)
                for extra in inputs["additional_inputs"].values():
                    if sha256(Path(extra["path"])) != extra["sha256"]:
                        raise ValueError(f"额外输入在运行中变化：{extra['path']}")
        finally:
            logging.getLogger().removeHandler(handler)
    row["runtime_seconds"] = time.perf_counter() - started
    row["output_paths"] = [str(p) for p in paths]
    record = {**row, "fingerprint": digest, "dependency_snapshot": snapshot,
              "configuration_sha256": sha256(run / "pipeline_config.json"), "artifacts": output_manifest(run, paths)}
    write_json(run / "reports" / f"stage_{stage}_checkpoint.json", record)
    LOG.info("工作流 %s 完成：%s，%.3f 秒。", STAGE_NAMES[stage - 1], row["status"], row["runtime_seconds"])


def run_pipeline(config_path: Path, resume: str | Path | None = None) -> dict:
    started = time.perf_counter()
    cfg = load_config(Path(config_path).resolve())
    source = resume_source(resume, cfg)  # 分配新目录前选择恢复来源，避免把自己当成旧运行。
    run = allocate_run(Path(cfg["output_directory"]))
    write_json(run / "pipeline_config.json", cfg)
    enabled = [cfg["enable_input_check"], cfg["enable_pointcloud_merge"], cfg["enable_tsdf"],
               cfg["enable_mesh_crop"] or cfg["enable_mesh_clean"], cfg["enable_slicing"], cfg["enable_report"]]
    report = {"schema_version": 1, "run_directory": str(run), "config_input": str(Path(config_path).resolve()),
              "configuration": cfg, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "resume_source": str(source) if source else None, "warnings": [], "error": None,
              "stages": [{"stage": i, "name": name, "enabled": flag, "status": "pending" if flag else "disabled",
                          "runtime_seconds": 0., "output_paths": []}
                         for i, (name, flag) in enumerate(zip(STAGE_NAMES, enabled), start=1)]}
    row = report["stages"][0]
    stage_started = time.perf_counter()
    LOG.info("新运行目录：%s", run)
    try:
        check_started = time.perf_counter()
        if enabled[0]:
            row["status"] = "running"
        # 关闭独立输入检查报告时，仍执行必要输入安全检查，不能绕过缺失位姿/深度单位。
        inputs, manifest = inspect_inputs(cfg)
        report.update(input_files=manifest, pose_sources=inputs["pose_sources"], coordinate_frame=inputs["coordinate_frame"],
                      pose_convention="T_world_camera", translation_unit="m", selected_frames=inputs["selected_frames"],
                      mandatory_input_checks_completed=True)
        write_json(run / "reports/input_manifest.json", {"files": manifest, "additional_inputs": inputs["additional_inputs"]})
        if enabled[0]:
            write_json(run / "reports/input_check.json", inputs)
            row.update(status="complete", output_paths=[str(run / "reports/input_check.json")])
        row["runtime_seconds"] = time.perf_counter() - check_started
        report["warnings"].append("位姿来源标签不证明真实精度；当前轨迹仅用于离线验证，不声称机器人标定精度或完整椅子。")
        if cfg["mesh_input_path"]:
            report["warnings"].append("后续 Mesh 输入显式来自已有文件，其位姿来源无法由本次 camera_poses_path 推断。")
        current_mesh = Path(cfg["mesh_input_path"]) if cfg["mesh_input_path"] else None
        report["mesh_origin"] = "external_mesh_source_not_inferred" if current_mesh else "tsdf_from_supplied_poses"
        for stage in range(2, 6):
            row = report["stages"][stage - 1]
            if not row["enabled"]:
                continue
            stage_started = time.perf_counter()
            perform_stage(stage, cfg, run, inputs, manifest, current_mesh, source, row)
            if stage == 3 and not cfg["mesh_input_path"]:
                current_mesh = run / "mesh/scene_mesh.ply"
            elif stage == 4:
                current_mesh = run / "mesh/processed_mesh.ply"
            if stage == 5:
                stats = row["statistics"]
                if not stats["watertight"] or stats["open_path_count"]:
                    report["warnings"].append(f"切片 Mesh 不封闭或存在开放轮廓；开放 {stats['open_path_count']}，闭合 {stats['closed_path_count']}；不是完整实体的制造截面。")
        if current_mesh is not None:
            report["final_mesh"] = mesh_stats(current_mesh)
        assert_input_hashes(manifest)
        report["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {str(exc) or '用户中断'}"
        row["status"] = report["status"]
        row["runtime_seconds"] = time.perf_counter() - stage_started
        row["error"] = report["error"]
        for pending in report["stages"][:5]:
            if pending["status"] == "pending":
                pending["status"] = "blocked"
        LOG.error("Pipeline 停止于 %s：%s", row["name"], report["error"])
    finally:
        # 最小审计报告始终写出；enable_report 控制汇总统计。
        final = report["stages"][5]
        if final["enabled"]:
            report["summary"] = {r["name"]: r.get("statistics", {}) for r in report["stages"][1:5]}
            final.update(status="complete", output_paths=[str(run / "pipeline_report.json")])
        report["runtime_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if not (run / "slices").exists():
            (run / "slices").mkdir()
        write_json(run / "pipeline_report.json", report)
        LOG.info("Pipeline %s；运行 %.3f 秒；完整报告：%s", report["status"], report["runtime_seconds"], run / "pipeline_report.json")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D435i RGB-D + T_world_camera 离线重建到 SVG，保留所有历史结果")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", nargs="?", const="latest", help="验证并复用指定旧运行；不带目录时选择最新运行；仍创建新运行目录")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return 0 if run_pipeline(args.config, args.resume)["status"] == "complete" else 1
    except (Exception, KeyboardInterrupt) as exc:
        LOG.error("无法启动 Pipeline（%s）：%s", type(exc).__name__, str(exc) or "用户中断")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
