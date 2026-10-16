"""标准几何独立验证：检查实际 SVG 坐标，不用预期形状替代交线。"""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reconstruction import mesh_to_svg as ms


def cube():
    return trimesh.creation.box(extents=[.2, .2, .2])


def open_cube():
    original = cube()
    # 删除一个竖直侧面，中心 Z 截面必须是开放路径。
    return trimesh.Trimesh(vertices=original.vertices.copy(),
                           faces=original.faces[original.face_normals[:, 0] < .5], process=False)


class MeshToSvgTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.input = self.root / "cube.obj"
        cube().export(self.input)

    def run_slice(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return ms.run_slicing(self.input, self.root / "slices", **kwargs)

    def test_load_ply_obj_stl(self):
        for extension in ("ply", "obj", "stl"):
            with self.subTest(extension=extension):
                path = self.root / f"mesh.{extension}"
                cube().export(path)
                mesh = ms.load_triangle_mesh(path)
                self.assertEqual(len(mesh.faces), 12)
                paths, _ = ms.intersect_layer(mesh, "Z", 0)
                self.assertEqual(len(paths), 1)
                self.assertTrue(paths[0]["closed"])

    def test_point_cloud_rejected(self):
        path = self.root / "points.ply"
        trimesh.points.PointCloud(cube().vertices).export(path)
        with self.assertRaisesRegex(ValueError, "三角|Point Cloud"):
            ms.load_triangle_mesh(path)

    def test_invalid_mesh_rejected(self):
        mesh = cube()
        mesh.vertices[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            ms.validate_mesh(mesh)
        mesh = cube()
        mesh.faces[0, 0] = len(mesh.vertices)
        with self.assertRaisesRegex(ValueError, "索引"):
            ms.validate_mesh(mesh)
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                               faces=[[0, 1, 2]], process=False)
        with self.assertRaisesRegex(ValueError, "退化"):
            ms.validate_mesh(mesh)

    def test_invalid_spacing(self):
        for value in (0, -10, np.nan, np.inf, 1e-15):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ms.slice_positions(-.1, .1, value)

    def test_positions_absolute_mm_end_on_grid_only(self):
        np.testing.assert_allclose(ms.slice_positions(-.1, .1, 10), np.linspace(-.1, .1, 21))
        np.testing.assert_allclose(ms.slice_positions(0, .025, 10), [0, .01, .02])
        with self.assertRaises(ValueError):
            ms.slice_positions(1, 0, 10)

    def test_fixed_basis_and_svg_y_direction(self):
        point = np.array([[1., 2., 3.]])
        for axis, expected in {"X": [[2000, -3000]], "Y": [[1000, -3000]],
                               "Z": [[1000, -2000]]}.items():
            np.testing.assert_array_equal(ms.project_mm(point, axis), expected)

    def test_xyz_intersections_translated_geometry(self):
        mesh = cube()
        center = np.array([.7, -.4, 1.2])
        mesh.apply_translation(center)
        for axis, (index, u, v) in ms.AXES.items():
            with self.subTest(axis=axis):
                paths, _ = ms.intersect_layer(mesh, axis, center[index])
                self.assertEqual(len(paths), 1)
                self.assertTrue(paths[0]["closed"])
                xy = ms.project_mm(paths[0]["points_m"], axis)
                np.testing.assert_allclose(np.ptp(xy, axis=0), [200, 200], atol=1e-10)
                np.testing.assert_allclose(xy.min(axis=0), [(center[u] - .1)*1000, -(center[v] + .1)*1000])

    def test_exported_cube_actual_path_200mm(self):
        report = self.run_slice(start_mm=0, end_mm=0)
        layer = report["layers"][0]
        saved, physical = ms.read_svg_geometry(Path(layer["svg_path"]))
        self.assertEqual((report["valid_slice_count"], layer["closed_path_count"], layer["open_path_count"]), (1, 1, 0))
        np.testing.assert_allclose(np.ptp(saved[0]["points_mm"], axis=0), [200, 200], atol=1e-10)
        np.testing.assert_array_equal(saved[0]["points_mm"][0], saved[0]["points_mm"][-1])
        self.assertTrue(saved[0]["closed"])
        # Canvas 有每边 2 mm 留白；真实轮廓是 200，不把 204 mm 画布冒充截面。
        np.testing.assert_allclose(physical["physical_size_mm"], [204, 204])

    def test_two_disconnected_cubes(self):
        second = cube()
        second.apply_translation([.4, 0, 0])
        mesh = trimesh.util.concatenate([cube(), second])
        paths, _ = ms.intersect_layer(mesh, "Z", 0)
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(p["closed"] for p in paths))
        for p in paths:
            np.testing.assert_allclose(np.ptp(ms.project_mm(p["points_m"], "Z"), axis=0), [200, 200])

    def test_open_surface_stays_open_in_svg(self):
        open_cube().export(self.input)
        report = self.run_slice(start_mm=0, end_mm=0)
        layer = report["layers"][0]
        self.assertFalse(report["watertight"])
        self.assertEqual((layer["open_path_count"], layer["closed_path_count"]), (1, 0))
        saved, _ = ms.read_svg_geometry(Path(layer["svg_path"]))
        self.assertFalse(saved[0]["closed"])
        self.assertFalse(np.allclose(saved[0]["points_mm"][0], saved[0]["points_mm"][-1]))

    def test_empty_slice_report_and_empty_svg(self):
        report = self.run_slice(start_mm=500, end_mm=500)
        self.assertEqual((report["empty_slice_count"], report["failed_slice_count"]), (1, 0))
        paths, _ = ms.read_svg_geometry(Path(report["layers"][0]["svg_path"]))
        self.assertEqual(paths, [])

    def test_exact_boundary_coplanar_behavior(self):
        lower, lower_diag = ms.intersect_layer(cube(), "Z", -.1)
        upper, upper_diag = ms.intersect_layer(cube(), "Z", .1)
        # 实际安装版本只纳入第三顶点在法向正侧的共面边，不偏移平面。
        self.assertEqual((len(lower), len(upper)), (1, 0))
        self.assertTrue(lower[0]["closed"])
        self.assertEqual(lower_diag["coplanar_triangle_count"], 2)
        self.assertEqual(upper_diag["coplanar_triangle_count"], 2)
        self.assertEqual(lower_diag["vertices_on_plane"], 4)

    def test_no_overwrite_existing_directory(self):
        destination = self.root / "slices"
        destination.mkdir()
        sentinel = destination / "keep.svg"
        sentinel.write_text("untouched", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_slice()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")

    def test_input_file_is_unchanged(self):
        before = hashlib.sha256(self.input.read_bytes()).hexdigest()
        self.run_slice(start_mm=0, end_mm=0)
        self.assertEqual(hashlib.sha256(self.input.read_bytes()).hexdigest(), before)

    def test_branch_edges_not_arbitrarily_closed(self):
        segments = np.array([[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]],
                             [[0, 0, 0], [-1, 0, 0]]], dtype=float)
        paths, diagnostics = ms.stitch_segments(segments)
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(not p["closed"] for p in paths))
        self.assertEqual(diagnostics["branch_node_count"], 1)

    def test_numerical_endpoints_only_no_gap_repair(self):
        # 0.1 mm 的真实间隙必须保留，不能强行连成一条线。
        segments = np.array([[[0, 0, 0], [.1, 0, 0]], [[.1001, 0, 0], [.2, 0, 0]]])
        paths, _ = ms.stitch_segments(segments)
        self.assertEqual(len(paths), 2)
        duplicate = np.concatenate([segments[:1], segments[:1, ::-1]])
        paths, diagnostics = ms.stitch_segments(duplicate)
        self.assertEqual(len(paths), 1)
        self.assertEqual(diagnostics["duplicate_segments_removed"], 1)

    def test_failure_saved_with_actual_reason(self):
        with patch.object(ms.trimesh.intersections, "mesh_plane", side_effect=RuntimeError("intersection test failure")):
            report = self.run_slice(start_mm=0, end_mm=0)
        self.assertEqual(report["failed_slice_count"], 1)
        saved = json.loads((self.root / "slices/slicing_report.json").read_text(encoding="utf-8"))
        self.assertIn("intersection test failure", saved["layers"][0]["failure_reason"])
        self.assertEqual(saved["svg_count"], 0)

    def test_svg_geometry_tampering_detected(self):
        paths, _ = ms.intersect_layer(cube(), "Z", 0)
        output = self.root / "test.svg"
        ms.write_svg(output, paths, "Z", 0, ms.svg_canvas(cube(), "Z"))
        content = output.read_text(encoding="utf-8")
        # 只改物理单位，即使文件仍然存在也必须校验失败。
        output.write_text(content.replace('width="204mm"', 'width="204px"'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mm"):
            ms.verify_svg(output, paths, "Z")
        # 物理单位正确，但实际路径被改动时也必须失败。
        output.write_text(content.replace("L 100", "L 99", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "坐标"):
            ms.verify_svg(output, paths, "Z")

    def test_failed_readback_reports_written_svg(self):
        with patch.object(ms, "verify_svg", side_effect=ValueError("readback failure")):
            report = self.run_slice(start_mm=0, end_mm=0)
        self.assertEqual(report["failed_slice_count"], 1)
        self.assertEqual(report["valid_slice_count"], 0)
        self.assertEqual(report["svg_count"], 1)
        layer = report["layers"][0]
        self.assertTrue(Path(layer["svg_path"]).is_file())
        self.assertIn("readback failure", layer["failure_reason"])


if __name__ == "__main__":
    unittest.main()
