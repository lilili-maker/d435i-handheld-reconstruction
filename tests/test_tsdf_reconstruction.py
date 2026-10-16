"""阶段 4D：真实 Open3D TSDF 的离线小平面实验与失败保护。"""

from contextlib import redirect_stderr
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reconstruction.pose_io import CameraPose, save_poses
from reconstruction.pose_utils import transform_points
from reconstruction import tsdf_reconstruction as tsdf


def inputs(root, *, omit_second_pose=False):
    dataset = root / "dataset"
    (dataset / "color").mkdir(parents=True)
    (dataset / "depth").mkdir()
    metadata = dict(width=32, height=32, fx=40., fy=40., cx=15.5, cy=15.5, depth_scale=.001,
                    intrinsics_unit="pixels", depth_scale_unit="meters_per_depth_unit",
                    depth_dtype="uint16", depth_aligned_to="color")
    (dataset / "camera_intrinsics.json").write_text(json.dumps(metadata), encoding="utf-8")
    color = np.full((32, 32, 3), [10, 80, 240], np.uint8)  # BGR。
    depth = np.full((32, 32), 1000, np.uint16)  # 一米平面，不代表真实场景。
    records = []
    t = np.eye(4)
    t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    t[0, 3] = 1
    for i in (1, 2):
        cv2.imwrite(str(dataset / "color" / f"{i:06d}.png"), color)
        cv2.imwrite(str(dataset / "depth" / f"{i:06d}.png"), depth)
        if i != 2 or not omit_second_pose:
            records.append(CameraPose(frame_id=i, timestamp=i / 30, T_world_camera=t,
                                      pose_source="simulation", coordinate_frame="simulation_world",
                                      timestamp_reference="simulation clock", validity=True))
    with (dataset / "timestamps.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "color_timestamp_ms", "depth_timestamp_ms"])
        writer.writerows([[1, 33.3, 33.3], [2, 66.6, 66.6]])
    pose_path = root / "poses/camera.json"
    save_poses(pose_path, records)
    return dataset, pose_path, metadata, records


class TSDFTests(unittest.TestCase):
    def test_explicit_unavailable_cuda_stops_without_cpu_substitution(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,poses_path,_,_=inputs(root)
            output=root/"mesh/gpu.ply"
            args=tsdf.parse_args(["--dataset",str(dataset),"--poses",str(poses_path),"--start","1","--end","2","--output",str(output),"--device","CUDA:0"])
            with patch.object(o3d.core.cuda,"is_available",return_value=False),patch.object(tsdf,"make_volume") as volume:
                with self.assertRaisesRegex(ValueError,"CUDA"):
                    tsdf.reconstruct(args)
                volume.assert_not_called()
            self.assertFalse(output.exists())

    def test_integrate_passes_inverse_camera_pose(self):
        t = np.eye(4)
        t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        t[:3, 3] = [.3, .2, .1]
        volume = Mock()
        result = tsdf.integrate_rgbd(volume, "rgbd", "intrinsic", t)
        np.testing.assert_allclose(result, np.linalg.inv(t))
        np.testing.assert_allclose(volume.integrate.call_args.args[2], np.linalg.inv(t))
        self.assertFalse(np.allclose(result, t))

    def test_depth_scale_filter_uint16_and_rgb(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, _, metadata, _ = inputs(Path(folder))
            path = dataset / "depth/000001.png"
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            raw[0, :5] = [0, 100, 200, 3000, 4000]
            cv2.imwrite(str(path), raw)
            original = path.read_bytes()
            rgbd, stats = tsdf.load_rgbd(dataset, 1, metadata, .2, 3)
            np.testing.assert_array_equal(rgbd["depth"].as_tensor().numpy()[0, :5, 0], [0, 0, 200, 3000, 0])
            np.testing.assert_array_equal(rgbd["color"].as_tensor().numpy()[0, 0], [240, 80, 10])
            self.assertEqual(stats["open3d_depth_scale_units_per_m"], 1000)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(cv2.imread(str(path), cv2.IMREAD_UNCHANGED).dtype, np.uint16)

    def test_bad_depth_dtype_and_empty_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, _, metadata, _ = inputs(Path(folder))
            for raw in (np.ones((32, 32), np.uint8), np.zeros((32, 32), np.uint16), np.ones((16, 16), np.uint16)):
                cv2.imwrite(str(dataset / "depth/000001.png"), raw)
                with self.assertRaises(ValueError):
                    tsdf.load_rgbd(dataset, 1, metadata, .2, 3)

    def test_bad_parameters(self):
        for arguments in (["--voxel-size", "0"], ["--sdf-trunc", "0.001"], ["--max-depth", "0.1"],
                          ["--voxel-size", "nan"], ["--step", "0"], ["--max-rotation", "200"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tsdf.parse_args(arguments)

    def test_motion_jump_and_missing_pose_stop_before_volume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, poses_path, _, records = inputs(root)
            records[1].T_world_camera[0, 3] += 1
            save_poses(root / "poses/jump.json", records)
            for pose_path in (root / "poses/jump.json", root / "poses/missing.json"):
                if pose_path.name == "missing.json":
                    save_poses(pose_path, records[:1])
                output = root / "mesh" / (pose_path.stem + ".ply")
                args = tsdf.parse_args(["--dataset", str(dataset), "--poses", str(pose_path), "--start", "1", "--end", "2", "--output", str(output)])
                with patch.object(tsdf, "make_volume", side_effect=AssertionError("must not create volume")) as volume:
                    with self.assertRaises(ValueError):
                        tsdf.reconstruct(args)
                    volume.assert_not_called()
                self.assertFalse(output.exists())
                report = json.loads((root / "mesh" / ("tsdf_report_" + pose_path.stem + ".json")).read_text(encoding="utf-8"))
                self.assertEqual(report["integrated_frame_count"], 0)
                self.assertEqual(report["status"], "failed")

    def test_real_tsdf_plane_world_position_colors_and_mesh_readback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, poses_path, _, poses = inputs(root)
            before = {str(p): p.read_bytes() for p in dataset.rglob('*') if p.is_file()}
            before[str(poses_path)] = poses_path.read_bytes()
            output = root / "mesh/scene_mesh_test.ply"
            args = tsdf.parse_args(["--dataset", str(dataset), "--poses", str(poses_path), "--start", "1", "--end", "2", "--output", str(output)])
            report = tsdf.reconstruct(args)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["integrated_frame_count"], 2)
            self.assertTrue(report["mesh_ply"]["readback_verified"])
            mesh = o3d.io.read_triangle_mesh(str(output))
            self.assertGreater(len(mesh.vertices), 100)
            self.assertGreater(len(mesh.triangles), 100)
            vertices = np.asarray(mesh.vertices)
            self.assertAlmostEqual(float(vertices[:, 0].mean()), 1, delta=.03)
            camera_points = transform_points(np.linalg.inv(poses[0].T_world_camera), vertices)
            np.testing.assert_allclose(camera_points[:, 2], 1, atol=.015)
            np.testing.assert_allclose(np.asarray(mesh.vertex_colors), np.tile(np.array([240, 80, 10]) / 255., (len(vertices), 1)), atol=1/255)
            self.assertTrue(mesh.has_vertex_normals())
            for filename, content in before.items():
                self.assertEqual(Path(filename).read_bytes(), content)
            existing = output.read_bytes()
            with self.assertRaises(FileExistsError):
                tsdf.reconstruct(args)
            self.assertEqual(output.read_bytes(), existing)

    def test_empty_mesh_rejected(self):
        with self.assertRaisesRegex(ValueError, "空 Mesh"):
            tsdf.mesh_statistics(o3d.geometry.TriangleMesh())

    def test_float32_color_roundoff_allowed_but_large_error_rejected(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        mesh.vertex_colors = o3d.utility.Vector3dVector([[1.000000119, 0, 0]] * 3)
        mesh.compute_vertex_normals()
        self.assertEqual(tsdf.mesh_statistics(mesh)["vertex_count"], 3)
        mesh.vertex_colors = o3d.utility.Vector3dVector([[1.01, 0, 0]] * 3)
        with self.assertRaises(ValueError):
            tsdf.mesh_statistics(mesh)


if __name__ == "__main__":
    unittest.main()
