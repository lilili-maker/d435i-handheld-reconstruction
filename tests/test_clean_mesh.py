"""人工几何仅作单元测试，不替代真实椅子扫描。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reconstruction import clean_mesh as cm


def box_config(xmin=-.1, xmax=1.1, ymin=-.1, ymax=1.1, zmin=-.1, zmax=1.1):
    return {"unit": "m", "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax}


def cube():
    mesh = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
    mesh.paint_uniform_color([.2, .4, .8])
    mesh.compute_vertex_normals()
    return mesh


class CleanMeshTests(unittest.TestCase):
    def test_read_and_inspect_cube(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cube.ply"
            o3d.io.write_triangle_mesh(str(path), cube())
            mesh = cm.read_mesh(path)
            stats = cm.inspect_mesh(mesh)
            self.assertEqual(stats["vertex_count"], 8)
            self.assertEqual(stats["triangle_count"], 12)
            self.assertEqual(stats["connected_component_count"], 1)
            self.assertEqual(stats["components"][0]["triangle_count"], 12)
            self.assertTrue(stats["has_vertex_colors"])

    def test_aabb_triangle_filter_and_colors(self):
        original = cube()
        original_vertices = np.asarray(original.vertices).copy()
        result = cm.crop_mesh(original, box_config(xmax=.5))
        self.assertEqual(len(result.vertices), 4)
        self.assertEqual(len(result.triangles), 2)
        # 筛选不会在 x=.5 边界创建新顶点，保留的原始面是 x=0。
        np.testing.assert_array_equal(np.asarray(result.vertices)[:, 0], 0)
        self.assertTrue(result.has_vertex_colors())
        np.testing.assert_allclose(np.asarray(result.vertex_colors), np.tile([.2, .4, .8], (4, 1)))
        self.assertTrue(np.all(np.asarray(result.triangles) < len(result.vertices)))
        np.testing.assert_array_equal(original.vertices, original_vertices)

    def test_empty_crop_and_crossing_triangle_not_exact_clipped(self):
        with self.assertRaises(ValueError):
            cm.crop_mesh(cube(), box_config(xmin=5, xmax=6))
        triangle = o3d.geometry.TriangleMesh()
        triangle.vertices = o3d.utility.Vector3dVector([[-1, 0, 0], [1, 0, 0], [0, 1, 0]])
        triangle.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        with self.assertRaises(ValueError):
            cm.crop_mesh(triangle, box_config(-.1, .1, -.1, .1, -.1, .1))

    def test_multiple_components_retained_and_noise_listed(self):
        a = cube()
        b = cube().translate([2, 0, 0])
        sphere = o3d.geometry.TriangleMesh.create_sphere(.3, resolution=8).translate([4, .5, .5])
        sphere.paint_uniform_color([1, 0, 0])
        noise = o3d.geometry.TriangleMesh()
        noise.vertices = o3d.utility.Vector3dVector([[5, 0, 0], [5, .01, 0], [5, 0, .01]])
        noise.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        noise.paint_uniform_color([0, 1, 0])
        mesh = a + b + sphere + noise
        self.assertEqual(cm.inspect_mesh(mesh)["connected_component_count"], 4)
        result, operations = cm.clean_mesh(mesh, min_component_triangles=10)
        stats = cm.inspect_mesh(result)
        self.assertEqual(stats["connected_component_count"], 3)
        self.assertEqual(sorted(item["triangle_count"] for item in stats["components"])[:2], [12, 12])
        removed = operations[0]["removed_components"]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["triangle_count"], 1)
        self.assertEqual(stats["unreferenced_vertices"], 0)
        self.assertTrue(result.has_vertex_colors())

    def test_degenerate_duplicate_unreferenced_cleanup(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [9, 9, 9]])
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 0, 1], [0, 1, 3], [0, 1, 2]])
        mesh.paint_uniform_color([.5, .2, .8])
        stats = cm.inspect_mesh(mesh)
        self.assertEqual(stats["degenerate_triangles"], 2)
        self.assertEqual(stats["duplicate_triangles_same_orientation"], 1)
        self.assertEqual(stats["unreferenced_vertices"], 1)
        result, _ = cm.clean_mesh(mesh, remove_degenerate=True, remove_duplicates=True, remove_unreferenced=True)
        self.assertEqual(len(result.triangles), 1)
        self.assertEqual(len(result.vertices), 3)
        self.assertEqual(len(mesh.triangles), 4)
        self.assertTrue(result.has_vertex_normals())

    def test_cleaning_does_not_rescale_metric_coordinates(self):
        mesh = cube().translate([3.2, -1, 2.5])
        cleaned, _ = cm.clean_mesh(mesh, remove_unreferenced=True)
        np.testing.assert_array_equal(np.asarray(mesh.vertices), np.asarray(cleaned.vertices))
        np.testing.assert_allclose(cleaned.get_axis_aligned_bounding_box().get_extent(), [1, 1, 1])

    def test_optional_simplification_retains_rgb(self):
        mesh = o3d.geometry.TriangleMesh.create_sphere(1, resolution=15)
        mesh.paint_uniform_color([.3, .7, .1])
        result, operations = cm.clean_mesh(mesh, simplify_target_triangles=200)
        self.assertLess(len(result.triangles), len(mesh.triangles))
        self.assertTrue(result.has_vertex_colors())
        self.assertEqual(operations[0]["target_triangles"], 200)
        self.assertGreater(np.linalg.norm(result.get_max_bound() - result.get_min_bound()), 1)

    def test_illegal_matrix_data_and_bad_config(self):
        mesh = cube()
        np.asarray(mesh.vertices)[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            cm.inspect_mesh(mesh)
        for change in ({"unit": "mm"}, {"xmin": None}, {"xmin": True}, {"xmax": float("inf")}, {"xmin": 2}, {"unexpected": 0}):
            config = box_config()
            config.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                cm.validate_crop_config(config)

    def test_inspection_without_config_never_writes_mesh(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, report = root / "scene.ply", root / "should_not_exist.ply", root / "inspection.json"
            o3d.io.write_triangle_mesh(str(source), cube())
            original = source.read_bytes()
            result = cm.run(cm.parse_args(["--input", str(source), "--output", str(output), "--report", str(report)]))
            self.assertEqual(result["mode"], "inspection")
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), original)

    def test_crop_config_save_pipeline_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output, config, report = [root / name for name in ("scene.ply", "cropped.ply", "saved_config.json", "report.json")]
            o3d.io.write_triangle_mesh(str(source), cube())
            original = source.read_bytes()
            command = ["--input", str(source), "--output", str(output), "--report", str(report), "--save-crop-config", str(config)]
            for key, value in box_config(xmax=.5).items():
                if key != "unit":
                    command += ["--" + key, str(value)]
            result = cm.run(cm.parse_args(command))
            self.assertTrue(result["output"]["readback_verified"])
            self.assertEqual(cm.validate_crop_config(json.loads(config.read_text(encoding="utf-8")))["xmax"], .5)
            self.assertEqual(len(cm.read_mesh(output).triangles), 2)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                cm.run(cm.parse_args(command))
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                cm.run(cm.parse_args(["--input", str(source), "--output", str(source), "--crop-config", str(config), "--report", str(root / 'another.json')]))

    def test_empty_crop_records_failure_without_ply(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, config, output, report = [root / name for name in ("scene.ply", "box.json", "empty.ply", "failure.json")]
            o3d.io.write_triangle_mesh(str(source), cube())
            config.write_text(json.dumps(box_config(xmin=5, xmax=6)), encoding="utf-8")
            with self.assertRaises(ValueError):
                cm.run(cm.parse_args(["--input", str(source), "--crop-config", str(config), "--output", str(output), "--report", str(report)]))
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
