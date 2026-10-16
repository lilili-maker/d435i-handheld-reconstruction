"""米制三角网格真实平面切片；SVG 路径坐标与物理尺寸均为毫米。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


AXES = {"X": (0, 1, 2), "Y": (1, 0, 2), "Z": (2, 0, 1)}
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
# 只合并数值上重合的端点（1 nm），不跨越扫描孔洞或补线。
ENDPOINT_TOLERANCE_M = 1e-9


def load_triangle_mesh(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"输入 Mesh 不存在：{path}")
    if path.suffix.lower() not in {".ply", ".obj", ".stl"}:
        raise ValueError("仅支持 PLY、OBJ、STL 三角网格。")
    # 禁止自动修补、焊接或平滑。Scene.to_mesh 应用文件中已有实例变换。
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry or any(not isinstance(g, trimesh.Trimesh)
                                    for g in mesh.geometry.values()):
            raise ValueError("文件包含非三角网格对象，不能当作 TriangleMesh。")
        mesh = mesh.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("输入没有三角形面，只有点的 Point Cloud 不能用于切片。")
    validate_mesh(mesh)
    return mesh


def validate_mesh(mesh: trimesh.Trimesh) -> None:
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Mesh 没有有效的三维顶点。")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh 顶点包含 NaN 或 Inf。")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Mesh 没有有效三角形面，不能使用点云。")
    if faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("三角形顶点索引非法。")
    if not np.any(mesh.area_faces > 0):
        raise ValueError("Mesh 所有三角形均退化，没有有效面。")


def slice_positions(start_m: float, end_m: float, spacing_mm: float) -> np.ndarray:
    if not all(math.isfinite(v) for v in (start_m, end_m, spacing_mm)):
        raise ValueError("切片位置和间距必须是有限数字。")
    if spacing_mm <= 0:
        raise ValueError("切片间距 spacing-mm 必须大于 0。")
    if end_m < start_m:
        raise ValueError("切片结束位置不能小于起始位置。")
    step = spacing_mm / 1000.0
    ratio = (end_m - start_m) / step
    if not math.isfinite(ratio) or ratio > 9999:
        raise ValueError("单次最多允许 10000 层，请增大间距或缩小范围。")
    count = math.floor(ratio + 1e-10) + 1
    # 使用整数索引，避免累加浮点误差；不追加不在等间距序列中的结束位置。
    return start_m + np.arange(count, dtype=np.float64) * step


def stitch_segments(segments: np.ndarray) -> tuple[list[dict], dict]:
    """连接真实交线；分叉处分开，绝不跨空隙连接两个开放端点。"""
    points: list[np.ndarray] = []
    buckets: dict[tuple, list[int]] = {}
    edges: set[tuple[int, int]] = set()
    duplicate_count = zero_count = 0
    tol = ENDPOINT_TOLERANCE_M
    offsets = [(a, b, c) for a in (-1, 0, 1)
               for b in (-1, 0, 1) for c in (-1, 0, 1)]

    def node(point: np.ndarray) -> int:
        key = tuple(math.floor(float(x) / tol) for x in point)
        for offset in offsets:
            neighbor = tuple(k + d for k, d in zip(key, offset))
            for index in buckets.get(neighbor, []):
                if np.linalg.norm(points[index] - point) <= tol:
                    return index
        index = len(points)
        points.append(point.copy())
        buckets.setdefault(key, []).append(index)
        return index

    for segment in segments:
        a, b = node(segment[0]), node(segment[1])
        if a == b:
            zero_count += 1
            continue
        edge = (min(a, b), max(a, b))
        if edge in edges:
            duplicate_count += 1
        edges.add(edge)
    adjacency = {i: set() for edge in edges for i in edge}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    unused = edges.copy()
    paths = []

    def walk(start: int, neighbor: int) -> None:
        indices = [start, neighbor]
        unused.remove((min(start, neighbor), max(start, neighbor)))
        previous, current = start, neighbor
        while current != start and len(adjacency[current]) == 2:
            nxt = next(i for i in adjacency[current] if i != previous)
            edge = (min(current, nxt), max(current, nxt))
            if edge not in unused:
                break
            unused.remove(edge)
            indices.append(nxt)
            previous, current = current, nxt
        paths.append({"points_m": np.asarray([points[i] for i in indices]),
                      "closed": indices[-1] == indices[0]})

    # 端点或分叉先处理；剩余 degree=2 的连通图才是闭合环。
    for start in sorted(adjacency):
        if len(adjacency[start]) != 2:
            for neighbor in sorted(adjacency[start]):
                if (min(start, neighbor), max(start, neighbor)) in unused:
                    walk(start, neighbor)
    while unused:
        start, neighbor = min(unused)
        walk(start, neighbor)
    return paths, {"intersection_segment_count": len(segments),
                   "unique_segment_count": len(edges),
                   "duplicate_segments_removed": duplicate_count,
                   "numerically_zero_segments_removed": zero_count,
                   "branch_node_count": sum(len(v) > 2 for v in adjacency.values())}


def intersect_layer(mesh: trimesh.Trimesh, axis: str, position_m: float):
    axis = axis.upper()
    index, _, _ = AXES[axis]
    normal, origin = np.zeros(3), np.zeros(3)
    normal[index], origin[index] = 1.0, position_m
    # 真正的三角形-平面求交，不用投影轮廓或模型包围盒冒充截面。
    segments = trimesh.intersections.mesh_plane(mesh, normal, origin)
    if not np.isfinite(segments).all():
        raise ValueError("平面求交返回 NaN/Inf。")
    paths, diagnostics = stitch_segments(segments)
    on_plane = np.abs(mesh.vertices[:, index] - position_m) <= trimesh.constants.tol.merge
    diagnostics.update({"vertices_on_plane": int(on_plane.sum()),
                        "coplanar_triangle_count": int(on_plane[mesh.faces].all(axis=1).sum())})
    return paths, diagnostics


def project_mm(points_m: np.ndarray, axis: str) -> np.ndarray:
    _, u, v = AXES[axis.upper()]
    # 固定平面基：(Y,Z)/(X,Z)/(X,Y)。SVG 的屏幕 Y 朝下，所以取 -v。
    return np.column_stack((points_m[:, u] * 1000.0, -points_m[:, v] * 1000.0))


def svg_canvas(mesh: trimesh.Trimesh, axis: str) -> list[float]:
    xy = project_mm(mesh.vertices, axis)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    margin_mm = 2.0
    return [*map(float, lo - margin_mm), *map(float, hi - lo + 2 * margin_mm)]


def write_svg(path: Path, paths: list[dict], axis: str, position_m: float,
              canvas: list[float]) -> None:
    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "version": "1.1", "width": f"{canvas[2]:.12g}mm",
        "height": f"{canvas[3]:.12g}mm",
        "viewBox": " ".join(f"{v:.12g}" for v in canvas)})
    ET.SubElement(root, f"{{{SVG_NS}}}metadata").text = json.dumps({
        "axis": axis, "position_m": position_m, "position_mm": position_m * 1000,
        "input_unit": "m", "path_unit": "mm", "svg_y": "negative_planar_v",
        "path_closed": [p["closed"] for p in paths]}, ensure_ascii=False)
    for index, contour in enumerate(paths):
        points = project_mm(contour["points_m"], axis)
        # 只有实际闭环才用 Z；开放路径没有补上末点到首点的线。
        commands = [f"M {points[0, 0]:.12g} {points[0, 1]:.12g}"]
        commands.extend(f"L {x:.12g} {y:.12g}" for x, y in points[1:])
        if contour["closed"]:
            commands.append("Z")
        ET.SubElement(root, f"{{{SVG_NS}}}path", {
            "id": f"contour_{index:03d}", "d": " ".join(commands),
            "fill": "none", "stroke": "#155a92", "stroke-width": "0.2",
            "stroke-linejoin": "round", "data-closed": str(contour["closed"]).lower()})
    with path.open("xb") as handle:
        ET.ElementTree(root).write(handle, encoding="utf-8", xml_declaration=True)


def read_svg_geometry(path: Path) -> tuple[list[dict], dict]:
    """从实际 SVG 路径重新读坐标，供导出验证和独立几何测试使用。"""
    root = ET.parse(path).getroot()
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError("输出不是标准 SVG。")
    viewbox = np.array([float(v) for v in root.attrib["viewBox"].split()])
    if any(not root.attrib[key].endswith("mm") for key in ("width", "height")):
        raise ValueError("SVG 物理宽高未使用 mm。")
    physical = np.array([float(root.attrib[key][:-2]) for key in ("width", "height")])
    if not np.allclose(viewbox[2:], physical, atol=1e-8, rtol=1e-10):
        raise ValueError("SVG viewBox 和物理尺寸不一致，1 路径单位不等于 1 mm。")
    contours = []
    for element in root.findall(f"{{{SVG_NS}}}path"):
        tokens = re.findall(r"[MLZ]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", element.attrib["d"])
        vertices, i, closed = [], 0, False
        while i < len(tokens):
            command = tokens[i]
            if command == "Z":
                closed = True
                i += 1
            elif command in {"M", "L"}:
                vertices.append([float(tokens[i + 1]), float(tokens[i + 2])])
                i += 3
            else:
                raise ValueError("SVG 路径命令无效。")
        xy = np.asarray(vertices)
        if not len(xy) or not np.isfinite(xy).all() or element.attrib.get("fill") != "none":
            raise ValueError("SVG 路径坐标或 fill 属性无效。")
        if closed and not np.allclose(xy[0], xy[-1], atol=1e-8, rtol=0):
            raise ValueError("SVG Z 尝试闭合原本未闭合的路径。")
        contours.append({"points_mm": xy, "closed": closed})
    return contours, {"physical_size_mm": physical.tolist(), "viewbox": viewbox.tolist()}


def verify_svg(path: Path, paths: list[dict], axis: str) -> dict:
    actual, canvas = read_svg_geometry(path)
    if len(actual) != len(paths):
        raise ValueError("SVG 回读路径数与真实交线不一致。")
    bounds = []
    for saved, original in zip(actual, paths):
        expected = project_mm(original["points_m"], axis)
        if saved["closed"] != original["closed"] or saved["points_mm"].shape != expected.shape:
            raise ValueError("SVG 回读开放/闭合状态或点数不一致。")
        if not np.allclose(saved["points_mm"], expected, atol=1e-8, rtol=0):
            raise ValueError("SVG 回读坐标与米到毫米转换后的真实交线不一致。")
        xy = saved["points_mm"]
        bounds.append({"min_mm": xy.min(axis=0).tolist(), "max_mm": xy.max(axis=0).tolist(),
                       "size_mm": np.ptp(xy, axis=0).tolist()})
    return {"verified": True, "path_bounds_mm": bounds, **canvas}


def write_preview(directory: Path, layers: list[dict]) -> list[int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [layer for layer in layers if layer["status"] == "valid"]
    selected = [valid[i] for i in np.unique(np.linspace(0, len(valid) - 1, min(12, len(valid)), dtype=int))] if valid else []
    if not selected:
        return []
    fig, axes = plt.subplots(math.ceil(len(selected) / 4), 4, figsize=(16, 4 * math.ceil(len(selected) / 4)), squeeze=False)
    for ax, layer in zip(axes.flat, selected):
        contours, _ = read_svg_geometry(Path(layer["svg_path"]))
        for contour in contours:
            xy = contour["points_mm"]
            ax.plot(xy[:, 0], xy[:, 1], color="tab:blue" if contour["closed"] else "tab:orange", linewidth=.8)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{layer['index']:03d}: {layer['position_mm']:.3f} mm; open={layer['open_path_count']}")
        ax.set_xlabel("SVG u (mm)")
        ax.set_ylabel("SVG -v (mm)")
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    fig.tight_layout()
    with (directory / "preview.png").open("xb") as handle:
        fig.savefig(handle, format="png", dpi=120)
    plt.close(fig)
    return [layer["index"] for layer in selected]


def run_slicing(input_path: Path, output_dir: Path, axis: str = "Z", spacing_mm: float = 10,
                start_mm: float | None = None, end_mm: float | None = None,
                preview: bool = False) -> dict:
    started = time.perf_counter()
    axis = axis.upper()
    if axis not in AXES:
        raise ValueError("axis 必须为 X、Y 或 Z。")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，禁止覆盖（即使为空）：{output_dir}")
    mesh = load_triangle_mesh(input_path)
    index, u, v = AXES[axis]
    start = float(mesh.bounds[0, index]) if start_mm is None else start_mm / 1000
    end = float(mesh.bounds[1, index]) if end_mm is None else end_mm / 1000
    positions = slice_positions(start, end, spacing_mm)
    canvas = svg_canvas(mesh, axis)
    report = {"input_mesh": str(input_path.resolve()),
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "trimesh_version": trimesh.__version__, "vertex_count": len(mesh.vertices),
              "triangle_count": len(mesh.faces), "watertight": bool(mesh.is_watertight),
              "mesh_bounds_m": mesh.bounds.tolist(),
              "degenerate_triangle_count": int((mesh.area_faces == 0).sum()),
              "axis": axis, "spacing_mm": spacing_mm, "start_mm": start * 1000,
              "end_mm": end * 1000, "slice_positions_mm": (positions * 1000).tolist(),
              "coordinate_conversion": {"input_unit": "m", "svg_unit": "mm",
                   "planar_basis": ["XYZ"[u], "XYZ"[v]],
                   "formula": "svg_x = planar_u_m * 1000; svg_y = -planar_v_m * 1000",
                   "translation": "none; viewBox retains absolute coordinates with 2 mm margin",
                   "physical_scale": "1 SVG user unit = 1 mm", "axis_normal": "+" + axis},
              "boundary_policy": "trimesh mesh_plane: coplanar faces omitted; only on-plane edges with positive-side third vertex included; planes never shifted",
              "intersection_tolerance_m": float(trimesh.constants.tol.merge),
              "endpoint_tolerance_m": ENDPOINT_TOLERANCE_M,
              "quality_warning": "Paths describe observed surface intersections, not complete solid sections or manufacturing-ready contours.",
              "layers": []}
    print(f"输入：{input_path.resolve()}；顶点 {len(mesh.vertices)}；三角形 {len(mesh.faces)}")
    if not mesh.is_watertight:
        print("警告：Mesh 不封闭；可以提取交线，但不能代表完整实体截面或直接制造轮廓。")
    # 所有参数先验证；排他创建目录并排他创建每个输出文件。
    output_dir.mkdir(parents=True, exist_ok=False)
    for number, position in enumerate(positions):
        layer = {"index": number, "position_m": float(position), "position_mm": float(position * 1000),
                 "path_count": 0, "open_path_count": 0, "closed_path_count": 0,
                 "svg_path": None, "status": "failed", "failure_reason": None}
        try:
            paths, diagnostics = intersect_layer(mesh, axis, float(position))
            svg_path = output_dir / f"slice_{number:03d}.svg"
            write_svg(svg_path, paths, axis, float(position), canvas)
            # 即使随后回读失败，也如实记录已经写出的文件，不把存在当成验证通过。
            layer["svg_path"] = str(svg_path.resolve())
            verification = verify_svg(svg_path, paths, axis)
            layer.update({"status": "valid" if paths else "empty", "svg_path": str(svg_path.resolve()),
                          "path_count": len(paths), "closed_path_count": sum(p["closed"] for p in paths),
                          "open_path_count": sum(not p["closed"] for p in paths),
                          "path_closed": [p["closed"] for p in paths],
                          "intersection_diagnostics": diagnostics, "svg_verification": verification})
            print(f"层 {number:03d}，{position * 1000:.3f} mm：闭合 {layer['closed_path_count']}，开放 {layer['open_path_count']}" + ("（空切片）" if not paths else ""))
        except Exception as exc:
            layer["failure_reason"] = f"{type(exc).__name__}: {exc}"
            print(f"错误：层 {number:03d} 求交或 SVG 导出验证失败：{exc}")
        report["layers"].append(layer)
    layers = report["layers"]
    report.update({"total_slice_count": len(layers),
                   "valid_slice_count": sum(l["status"] == "valid" for l in layers),
                   "empty_slice_count": sum(l["status"] == "empty" for l in layers),
                   "failed_slice_count": sum(l["status"] == "failed" for l in layers),
                   "svg_count": sum(l["svg_path"] is not None for l in layers),
                   "open_path_count": sum(l["open_path_count"] for l in layers),
                   "closed_path_count": sum(l["closed_path_count"] for l in layers)})
    if preview:
        try:
            report["preview_layer_indices"] = write_preview(output_dir, layers)
        except Exception as exc:
            report["preview_failure_reason"] = f"{type(exc).__name__}: {exc}"
            print(f"错误：PNG 预览生成失败：{exc}；SVG 检查结果仍保存到报告。")
    report["elapsed_seconds"] = time.perf_counter() - started
    with (output_dir / "slicing_report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"完成：SVG {report['svg_count']}；有效层 {report['valid_slice_count']}；空层 {report['empty_slice_count']}；失败层 {report['failed_slice_count']}\n报告：{(output_dir / 'slicing_report.json').resolve()}")
    return report


def default_input() -> Path:
    root = Path(__file__).resolve().parents[2]
    for name in ("chair_clean.ply", "chair_cropped.ply"):
        path = root / "output" / "mesh" / name
        if path.is_file():
            return path
    raise FileNotFoundError("没有 chair_clean.ply 或 chair_cropped.ply。不会自动使用完整场景；请先裁剪，或确认后显式指定 --input。")


def main() -> int:
    parser = argparse.ArgumentParser(description="三角网格平面求交导出 SVG；输入坐标 m，SVG mm；开放轮廓不补线。")
    parser.add_argument("--input", type=Path, help="PLY/OBJ/STL；默认优先 chair_clean.ply，其次 chair_cropped.ply")
    parser.add_argument("--output-dir", type=Path, required=True, help="新的输出目录；已存在即拒绝")
    parser.add_argument("--axis", type=str.upper, choices=AXES, default="Z")
    parser.add_argument("--spacing-mm", type=float, default=10, help="间距，毫米，默认 10")
    parser.add_argument("--start", type=float, help="沿切片轴的绝对起始位置，毫米；默认 Mesh 最小值")
    parser.add_argument("--end", type=float, help="沿切片轴的绝对结束位置，毫米；默认 Mesh 最大值")
    parser.add_argument("--preview", action="store_true", help="保存最多 12 个有效层的 PNG 概览，不替代 SVG")
    args = parser.parse_args()
    try:
        report = run_slicing(args.input if args.input is not None else default_input(), args.output_dir,
                             args.axis, args.spacing_mm, args.start, args.end, args.preview)
        return 1 if report["failed_slice_count"] or report.get("preview_failure_reason") else 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
