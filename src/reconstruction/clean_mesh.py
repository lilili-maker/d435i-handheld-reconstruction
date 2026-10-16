"""阶段 5A：显式手动 AABB 三角形筛选与可选 Mesh 清理，不识别目标。"""

import argparse
import hashlib
import json
import logging
from numbers import Real
from pathlib import Path
import time

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("clean_mesh")
BOUNDS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def validate_mesh(mesh):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("输入 Mesh 没有顶点或三角形，不能继续。")
    bad = np.flatnonzero(~np.isfinite(vertices).all(axis=1))
    if len(bad):
        raise ValueError(f"Mesh 坐标包含 NaN / Inf：{len(bad)} 个顶点，示例索引 {bad[:20].tolist()}。")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("三角形索引非法或越界。")
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) and (colors.shape != vertices.shape or not np.isfinite(colors).all()):
        raise ValueError("顶点颜色数量不匹配或包含 NaN / Inf。")
    if len(colors) and (np.any(colors < -1e-6) or np.any(colors > 1 + 1e-6)):
        raise ValueError("顶点 RGB 颜色显著超出 [0,1]。")


def read_mesh(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入 Mesh：{path}")
    mesh = o3d.io.read_triangle_mesh(str(path))
    validate_mesh(mesh)
    return mesh


def degenerate_mask(mesh, area_epsilon=0.0):
    if not np.isfinite(area_epsilon) or area_epsilon < 0:
        raise ValueError("退化面积阈值必须为有限非负数，单位平方米。")
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 0] == faces[:, 2]) | (faces[:, 1] == faces[:, 2])
    area = .5 * np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                      vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    return repeated | (area <= area_epsilon), repeated, area


def component_stats(mesh):
    labels, counts, areas = mesh.cluster_connected_triangles()
    return np.asarray(labels), [{"component_id": i, "triangle_count": int(n), "surface_area_m2": float(areas[i])}
                               for i, n in enumerate(counts)]


def inspect_mesh(mesh, area_epsilon=0.0):
    """只读检查；组件 ID 只对当前检查阶段有效，完整列表保存在报告。"""
    validate_mesh(mesh)
    faces = np.asarray(mesh.triangles)
    degenerate, repeated, areas = degenerate_mask(mesh, area_epsilon)
    duplicate_check = o3d.geometry.TriangleMesh(mesh)
    duplicate_check.remove_duplicated_triangles()  # 在副本上调用真实 API，按相同朝向判重。
    duplicate_count = len(faces) - len(duplicate_check.triangles)
    del duplicate_check
    _, components = component_stats(mesh)
    return {"vertex_count": len(mesh.vertices), "triangle_count": len(faces),
            "xyz_bounds_m": {"min": mesh.get_min_bound().tolist(), "max": mesh.get_max_bound().tolist()},
            "nonfinite_vertex_count": 0, "has_vertex_colors": mesh.has_vertex_colors(),
            "has_vertex_normals": mesh.has_vertex_normals(),
            "repeated_index_triangles": int(repeated.sum()), "zero_area_triangles": int((areas == 0).sum()),
            "degenerate_triangles": int(degenerate.sum()), "area_epsilon_m2": area_epsilon,
            "duplicate_triangles_same_orientation": duplicate_count,
            "unreferenced_vertices": len(mesh.vertices) - len(np.unique(faces)),
            "connected_component_count": len(components), "components": components,
            "component_definition": "Triangles connected via shared edges; IDs refer only to this inspection stage"}


def validate_crop_config(config):
    if not isinstance(config, dict):
        raise ValueError("裁剪配置必须是 JSON 对象。")
    allowed = set(BOUNDS) | {"unit", "schema_version", "note"}
    if set(config) - allowed:
        raise ValueError(f"未知裁剪配置字段：{sorted(set(config) - allowed)}")
    if config.get("unit") != "m":
        raise ValueError("裁剪配置必须明确声明 unit='m'，不猜测单位。")
    if "schema_version" in config and (type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("裁剪配置 schema_version 必须为整数 1。")
    result = {"schema_version": 1, "unit": "m"}
    for key in BOUNDS:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"裁剪范围 {key} 必须是您明确提供的有限米制数值，实际 {value!r}。")
        result[key] = float(value)
    for axis in "xyz":
        if result[f"{axis}min"] >= result[f"{axis}max"]:
            raise ValueError(f"裁剪范围须满足 {axis}min < {axis}max。")
    if "note" in config:
        if not isinstance(config["note"], str):
            raise ValueError("裁剪配置 note 必须是字符串。")
        result["note"] = config["note"]
    return result


def make_box(config):
    config = validate_crop_config(config)
    box = o3d.geometry.AxisAlignedBoundingBox([config[f"{axis}min"] for axis in "xyz"],
                                             [config[f"{axis}max"] for axis in "xyz"])
    box.color = [1, .2, .1]
    return box


def crop_mesh(mesh, config):
    """只保留全部三个顶点位于 AABB（含边界）内的完整三角形。

    不生成交点、不切割跨框三角形；这是三角形筛选，不是几何精确裁切。
    仅清除因筛选而未被引用的顶点，不自动做其他 Mesh 清理。
    """
    validate_mesh(mesh)
    box = make_box(config)
    inside = np.zeros(len(mesh.vertices), dtype=bool)
    inside[np.asarray(box.get_point_indices_within_bounding_box(mesh.vertices), dtype=np.int64)] = True
    keep = np.all(inside[np.asarray(mesh.triangles)], axis=1)
    if not keep.any():
        raise ValueError("裁剪后没有完整三角形；请检查米制范围及边界，未生成输出 Mesh。")
    cropped = o3d.geometry.TriangleMesh(mesh)
    cropped.remove_triangles_by_mask(~keep)
    cropped.remove_unreferenced_vertices()
    cropped.compute_vertex_normals()
    validate_mesh(cropped)
    return cropped


def clean_mesh(mesh, *, remove_degenerate=False, remove_duplicates=False, remove_unreferenced=False,
               min_component_triangles=0, simplify_target_triangles=None, area_epsilon=0.0):
    """清理副本；不默认保留最大组件，不平滑、不补洞、不缩放坐标。"""
    validate_mesh(mesh)
    if isinstance(min_component_triangles, bool) or not isinstance(min_component_triangles, int) or min_component_triangles < 0:
        raise ValueError("组件面数阈值必须是非负整数。")
    if simplify_target_triangles is not None and (isinstance(simplify_target_triangles, bool) or not isinstance(simplify_target_triangles, int) or simplify_target_triangles <= 0):
        raise ValueError("简化目标必须是正整数。")
    result = o3d.geometry.TriangleMesh(mesh)
    operations = []
    had_colors = mesh.has_vertex_colors()
    if remove_degenerate:
        mask, _, _ = degenerate_mask(result, area_epsilon)
        result.remove_triangles_by_mask(mask)
        operations.append({"operation": "remove_degenerate_triangles", "removed_triangles": int(mask.sum()),
                           "criterion": "Repeated vertex indices or geometric area <= area_epsilon_m2", "area_epsilon_m2": area_epsilon})
    if remove_duplicates:
        before = len(result.triangles)
        result.remove_duplicated_triangles()
        operations.append({"operation": "remove_duplicate_triangles_same_orientation", "removed_triangles": before - len(result.triangles)})
    if len(result.triangles) == 0:
        raise ValueError("基础清理删除了所有三角形，停止而不输出假结果。")
    if min_component_triangles:
        labels, components = component_stats(result)
        removed = [item for item in components if item["triangle_count"] < min_component_triangles]
        remove_ids = [item["component_id"] for item in removed]
        removed_id_set = set(remove_ids)
        mask = np.isin(labels, remove_ids)
        if mask.all():
            raise ValueError(f"组件面数阈值 {min_component_triangles} 会删除所有组件，请降低阈值。")
        result.remove_triangles_by_mask(mask)
        before = len(result.vertices)
        result.remove_unreferenced_vertices()  # 避免被删除组件的孤立顶点继续影响边界或显示。
        operations.append({"operation": "filter_components_by_triangle_count", "threshold": min_component_triangles,
                           "component_id_scope": "after basic cleanup, before component filtering",
                           "removed_components": removed, "retained_components": [item for item in components if item["component_id"] not in removed_id_set],
                           "removed_triangles": int(mask.sum()), "removed_unreferenced_vertices": before - len(result.vertices)})
    if remove_unreferenced:
        before = len(result.vertices)
        result.remove_unreferenced_vertices()
        operations.append({"operation": "remove_unreferenced_vertices", "removed_vertices": before - len(result.vertices)})
    if simplify_target_triangles is not None:
        before = len(result.triangles)
        if simplify_target_triangles < before:
            result = result.simplify_quadric_decimation(simplify_target_triangles)
        operations.append({"operation": "quadric_decimation", "target_triangles": simplify_target_triangles,
                           "before_triangles": before, "actual_triangles": len(result.triangles),
                           "note": "Target may not be reached; vertices may move. No global XYZ scaling is applied."})
    result.compute_vertex_normals()
    validate_mesh(result)
    if had_colors and not result.has_vertex_colors():
        raise ValueError("清理/简化意外丢失顶点颜色，停止保存。")
    operations.append({"operation": "recompute_vertex_normals"})
    return result, operations


def write_mesh_checked(path, mesh):
    with path.open("xb"):
        pass
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=False):
        raise OSError(f"Mesh 写入失败：{path}")
    saved = read_mesh(path)
    if not np.array_equal(np.asarray(saved.triangles), np.asarray(mesh.triangles)) or not np.allclose(np.asarray(saved.vertices), np.asarray(mesh.vertices), atol=1e-10, rtol=0):
        raise ValueError("输出 PLY 的坐标/三角形回读验证失败。")
    if mesh.has_vertex_colors() and (not saved.has_vertex_colors() or not np.allclose(np.asarray(saved.vertex_colors), np.asarray(mesh.vertex_colors), atol=.5 / 255 + 1e-9, rtol=0)):
        raise ValueError("输出 PLY 的 RGB 回读验证失败。")
    return {"path": str(path), "readback_verified": True}


def log_inspection(label, data):
    LOG.info("%s：顶点 %d，三角形 %d，RGB=%s；XYZ 范围（米）=%s。", label, data["vertex_count"], data["triangle_count"], data["has_vertex_colors"], data["xyz_bounds_m"])
    LOG.info("%s：组件 %d，退化三角形 %d，重复三角形 %d，未引用顶点 %d；最大十个组件面数=%s。",
             label, data["connected_component_count"], data["degenerate_triangles"], data["duplicate_triangles_same_orientation"], data["unreferenced_vertices"],
             sorted([item["triangle_count"] for item in data["components"]], reverse=True)[:10])


def run(args):
    config = None
    if args.crop_config:
        config = validate_crop_config(json.loads(args.crop_config.read_text(encoding="utf-8")))
    elif args.xmin is not None:
        config = validate_crop_config({"unit": "m", **{key: getattr(args, key) for key in BOUNDS}})
    cleaning = args.clean or args.remove_degenerate or args.remove_duplicates or args.remove_unreferenced or args.min_component_triangles > 0 or args.simplify_target_triangles is not None
    mode = "inspection" if config is None else "preview_crop_box" if args.preview_only else "crop_and_clean" if cleaning else "crop"
    output = None
    if config is not None and not args.preview_only:
        output = (args.output or ROOT / "output/mesh" / ("chair_clean.ply" if cleaning else "chair_cropped.ply")).resolve()
        if output.suffix.lower() != ".ply":
            raise ValueError("输出 Mesh 必须为 .ply。")
    report_path = (args.report or (output.with_name(output.stem + "_report.json") if output else ROOT / "output/mesh" / (args.input.stem + "_inspection_report.json"))).resolve()
    destinations = [report_path] + ([output] if output else []) + ([args.save_crop_config] if args.save_crop_config else [])
    if len(set(destinations)) != len(destinations):
        raise ValueError("输出 Mesh、报告与配置的目标路径不能相同。")
    for path in destinations:
        if path == args.input or path == args.crop_config or path.exists():
            raise FileExistsError(f"禁止覆盖原始输入或已有文件：{path}；请选择新输出名称。")
        if ROOT / "data" in path.parents:
            raise ValueError("不能将输出写入原始 RGB-D 数据目录。")
        if path.suffix.lower() != (".ply" if path == output else ".json"):
            raise ValueError(f"报告和配置须为 JSON、Mesh 须为 PLY：{path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"status": "failed", "mode": mode, "input": str(args.input), "unit": "m",
              "coordinate_frame": "input_mesh_native_coordinates_no_transform", "crop_config": config,
              "crop_method": "Keep only triangles with all 3 vertices inside inclusive AABB; discard crossing triangles; no exact geometric clipping or hole capping",
              "target_identity": "Not verified; manual bounding-box selection is not semantic chair segmentation",
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "error": None}
    try:
        mesh = read_mesh(args.input)
        report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        report["original"] = inspect_mesh(mesh, args.degenerate_area_epsilon)
        log_inspection("原始 Mesh", report["original"])
        if args.save_crop_config:
            args.save_crop_config.parent.mkdir(parents=True, exist_ok=True)
            with args.save_crop_config.open("x", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.write("\n")
        if config is None:
            LOG.info("没有提供裁剪范围，本次仅检查%s，不生成裁剪或清理 PLY。", "并预览" if args.preview else "")
            if args.preview:
                show([mesh], "Original scene Mesh")
        elif args.preview_only:
            show([mesh, make_box(config)], "Manual AABB preview (no output Mesh)")
        else:
            cropped = crop_mesh(mesh, config)
            report["cropped"] = inspect_mesh(cropped, args.degenerate_area_epsilon)
            log_inspection("裁剪 Mesh", report["cropped"])
            result = cropped
            if cleaning:
                result, operations = clean_mesh(cropped, remove_degenerate=args.clean or args.remove_degenerate,
                                                remove_duplicates=args.clean or args.remove_duplicates,
                                                remove_unreferenced=args.clean or args.remove_unreferenced,
                                                min_component_triangles=args.min_component_triangles,
                                                simplify_target_triangles=args.simplify_target_triangles, area_epsilon=args.degenerate_area_epsilon)
                report["cleaning_operations"] = operations
                report["cleaned"] = inspect_mesh(result, args.degenerate_area_epsilon)
                log_inspection("清理 Mesh", report["cleaned"])
                for operation in operations:
                    if "removed_components" in operation:
                        LOG.info("已删除 %d 个小组件；具体组件 ID、面数及面积见报告 cleaning_operations。", len(operation["removed_components"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            report["output"] = write_mesh_checked(output, result)
            LOG.info("保存并回读验证通过：%s", output)
            if args.preview:
                show([mesh, make_box(config)], "Original scene + manual AABB")
                show([cropped], "Cropped triangles (not exact clipping)")
                if cleaning:
                    show([result], "Cleaned Mesh")
        report["status"] = "complete"
        return report
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = str(exc) or "用户中断"
        raise
    finally:
        report["runtime_seconds_including_preview"] = time.perf_counter() - started
        with report_path.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        LOG.info("检查/清理报告：%s；完整组件面数列表保存在 JSON。", report_path)


def show(geometries, title):
    mesh = geometries[0]
    o3d.visualization.draw_geometries(geometries, window_name=title, mesh_show_back_face=True,
                                     lookat=mesh.get_center(), front=[0, 0, -1], up=[0, -1, 0], zoom=.7)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="只在明确手动 AABB 范围下裁剪 Mesh；默认只检查，不猜椅子位置")
    parser.add_argument("--input", type=Path, default=ROOT / "output/mesh/scene_mesh_500_550.ply")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--crop-config", type=Path)
    parser.add_argument("--save-crop-config", type=Path, help="将显式提供的六个边界保存为新 JSON")
    parser.add_argument("--report", type=Path, help="新 JSON 报告路径，默认按输出或输入名称生成")
    for key in BOUNDS:
        parser.add_argument("--" + key, type=float, help="手动 AABB 边界，米；六个值须全部提供")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-only", action="store_true", help="仅显示原 Mesh 与裁剪框，不写裁剪 PLY")
    parser.add_argument("--clean", action="store_true", help="启用退化/重复三角形和未引用顶点的基础清理")
    parser.add_argument("--remove-degenerate", action="store_true")
    parser.add_argument("--remove-duplicates", action="store_true")
    parser.add_argument("--remove-unreferenced", action="store_true")
    parser.add_argument("--min-component-triangles", type=int, default=0, help="仅删除面数小于此阈值的组件；0 禁用")
    parser.add_argument("--simplify-target-triangles", type=int, help="可选 QEM 简化，目标面数不保证能达到")
    parser.add_argument("--degenerate-area-epsilon", type=float, default=0., help="几何退化面积阈值，平方米；默认只删除零面积")
    args = parser.parse_args(argv)
    supplied = [getattr(args, key) is not None for key in BOUNDS]
    if any(supplied) and (not all(supplied) or args.crop_config):
        parser.error("六个命令行边界须一起提供，不能与 --crop-config 混用。")
    has_crop = bool(args.crop_config) or all(supplied)
    if args.save_crop_config and not has_crop:
        parser.error("必须先提供明确裁剪范围，才能保存配置。")
    if not has_crop and (args.clean or args.remove_degenerate or args.remove_duplicates or args.remove_unreferenced or args.min_component_triangles or args.simplify_target_triangles is not None):
        parser.error("没有手动裁剪范围时只允许检查或预览；不清理整个场景来冒充椅子。")
    if args.min_component_triangles < 0 or (args.simplify_target_triangles is not None and args.simplify_target_triangles <= 0):
        parser.error("组件阈值须非负，简化目标须为正整数。")
    if not np.isfinite(args.degenerate_area_epsilon) or args.degenerate_area_epsilon < 0:
        parser.error("退化面积阈值必须为有限非负数。")
    if args.preview_only:
        args.preview = True
    for key in ("input", "output", "crop_config", "save_crop_config", "report"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    return args


def main(argv=None):
    try:
        run(parse_args(argv))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        LOG.error("Mesh 检查/处理停止（%s）：%s", type(exc).__name__, str(exc) or "用户中断")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
