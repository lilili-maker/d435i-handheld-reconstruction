"""点云融合的离线测试：检查变换方向、增量均值、颜色、缺失输入及输出保护。"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reconstruction.pose_io import CameraPose, save_poses
from reconstruction.pose_utils import transform_points
from reconstruction.merge_pointclouds import WorldVoxelAccumulator, apply_world_pose, merge, parse_args, single_frame_module


def cloud(points, colors):
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    result.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return result


def make_pose(i, t, **kwargs):
    values = dict(frame_id=i, timestamp=i / 30, T_world_camera=t, pose_source="simulation",
                  coordinate_frame="simulation_world", timestamp_reference="simulation clock", validity=True)
    values.update(kwargs)
    return CameraPose(**values)


def synthetic_inputs(root, *, include_second_pose=True, invalid_second=False):
    dataset = root / "dataset"
    (dataset / "color").mkdir(parents=True)
    (dataset / "depth").mkdir()
    metadata = dict(width=8, height=8, fx=10., fy=10., cx=3.5, cy=3.5, depth_scale=.001,
                    intrinsics_unit="pixels", depth_scale_unit="meters_per_depth_unit",
                    depth_aligned_to="color", depth_dtype="uint16")
    (dataset / "camera_intrinsics.json").write_text(json.dumps(metadata), encoding="utf-8")
    color = np.zeros((8, 8, 3), np.uint8)
    color[:, :, 0] = np.arange(8, dtype=np.uint8)[None, :] * 30
    color[:, :, 1] = np.arange(8, dtype=np.uint8)[:, None] * 30
    color[:, :, 2] = 200
    depth = np.full((8, 8), 1000, np.uint16)
    depth[0, 0] = 0
    for i in (1, 2):
        cv2.imwrite(str(dataset / "color" / f"{i:06d}.png"), color)
        cv2.imwrite(str(dataset / "depth" / f"{i:06d}.png"), depth)
    with (dataset / "timestamps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "color_timestamp_ms", "depth_timestamp_ms"])
        writer.writerows([[1, 33.3, 33.3], [2, 66.6, 66.6]])
    a = np.eye(4)
    a[0, 3] = 1
    b = np.eye(4)
    b[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    b[:3, 3] = [1, .1, 0]
    poses = [make_pose(1, a)]
    if include_second_pose:
        poses.append(make_pose(2, None if invalid_second else b, validity=not invalid_second,
                               reason="unusable simulated pose" if invalid_second else ""))
    pose_path = root / "poses" / "camera.json"
    save_poses(pose_path, poses)
    return dataset, pose_path, poses


class MergeTests(unittest.TestCase):
    def test_camera_to_world_direct_and_colors_unchanged(self):
        t = np.eye(4)
        t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        t[:3, 3] = [1, .2, .3]
        raw = [[.1, .2, 1], [.4, .5, 2]]
        rgb = [[1, 0, 0], [0, 0, 1]]
        sample = cloud(raw, rgb)
        apply_world_pose(sample, make_pose(1, t))
        np.testing.assert_allclose(np.asarray(sample.points), transform_points(t, raw))
        self.assertFalse(np.allclose(np.asarray(sample.points), transform_points(np.linalg.inv(t), raw)))
        np.testing.assert_array_equal(np.asarray(sample.colors), rgb)

    def test_incremental_weights_equal_global_original_point_means(self):
        accumulator = WorldVoxelAccumulator(.1)
        first = cloud([[.01, 0, 0], [.02, 0, 0], [.03, 0, 0]], [[1, 0, 0]] * 3)
        second = cloud([[.09, 0, 0]], [[0, 0, 1]])
        accumulator.add(first)
        accumulator.add(second)
        merged = accumulator.pointcloud()
        np.testing.assert_allclose(np.asarray(merged.points), [[.0375, 0, 0]])
        np.testing.assert_allclose(np.asarray(merged.colors), [[.75, 0, .25]])
        self.assertEqual(accumulator.raw_count, 4)
        self.assertEqual(accumulator.counts.tolist(), [4])

    def test_fixed_world_grid_negative_coordinates_and_order(self):
        points = [[-.01, 0, 1], [.01, 0, 1], [.02, 0, 1], [.21, 0, 1]]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, .5]]
        a, b = WorldVoxelAccumulator(.1), WorldVoxelAccumulator(.1)
        a.add(cloud(points, colors))
        for i in reversed(range(4)):
            b.add(cloud([points[i]], [colors[i]]))
        np.testing.assert_array_equal(a.keys, b.keys)
        np.testing.assert_allclose(np.asarray(a.pointcloud().points), np.asarray(b.pointcloud().points))
        np.testing.assert_allclose(np.asarray(a.pointcloud().colors), np.asarray(b.pointcloud().colors))

    def test_memory_cap_and_bad_cloud_rejected(self):
        accumulator = WorldVoxelAccumulator(.1, max_voxels=1)
        with self.assertRaises(MemoryError):
            accumulator.add(cloud([[0, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]]))
        self.assertEqual(accumulator.raw_count, 0)
        with self.assertRaises(ValueError):
            accumulator.add(cloud([[float("nan"), 0, 0]], [[1, 0, 0]]))

    def test_offline_pipeline_pose_rgb_and_ply(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, poses_path, poses = synthetic_inputs(root)
            output = root / "out/merged_1_2.ply"
            args = parse_args(["--dataset", str(dataset), "--poses", str(poses_path), "--start", "1", "--end", "2", "--voxel-size", ".02", "--output", str(output)])
            before = {str(p): p.read_bytes() for p in dataset.rglob('*') if p.is_file()}
            before[str(poses_path)] = poses_path.read_bytes()
            report = merge(args)
            self.assertEqual(report["processed_frame_count"], 2)
            self.assertEqual(report["missing_frame_count"], 0)
            self.assertEqual(report["pose_sources"], ["simulation"])
            self.assertEqual(report["points_before_fusion"], 126)
            self.assertTrue(report["rgb_ply"]["readback_verified"])
            self.assertTrue(report["diagnostic_ply"]["readback_verified"])
            self.assertEqual([p["frame_id"] for p in report["diagnostic_palette"]], [1, 2])
            # 独立地把原始点投影到 World，再按同一固定网格计算全局均值。
            loader = single_frame_module()
            raw_points, raw_colors = [], []
            for pose in poses:
                camera = loader.create_pointcloud(dataset, pose.frame_id, .2, 3)
                raw_points.extend(transform_points(pose.T_world_camera, np.asarray(camera.points)))
                raw_colors.extend(np.asarray(camera.colors))
            raw_points, raw_colors = np.asarray(raw_points), np.asarray(raw_colors)
            keys = np.floor(raw_points / .02).astype(np.int64)
            unique = np.unique(keys, axis=0)
            expected_p, expected_c = [], []
            for key in unique:
                mask = np.all(keys == key, axis=1)
                expected_p.append(raw_points[mask].mean(axis=0))
                expected_c.append(raw_colors[mask].mean(axis=0))
            saved = o3d.io.read_point_cloud(str(output))
            np.testing.assert_allclose(np.asarray(saved.points), expected_p, atol=1e-10)
            np.testing.assert_allclose(np.asarray(saved.colors), expected_c, atol=.5/255+1e-9)
            for filename, contents in before.items():
                self.assertEqual(Path(filename).read_bytes(), contents)
            with self.assertRaises(FileExistsError):
                merge(args)

    def test_missing_and_invalid_poses_do_not_produce_partial_ply(self):
        for include, invalid in ((False, False), (True, True)):
            with self.subTest(include=include, invalid=invalid), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                dataset, poses, _ = synthetic_inputs(root, include_second_pose=include, invalid_second=invalid)
                output = root / "out/merged_1_2.ply"
                args = parse_args(["--dataset", str(dataset), "--poses", str(poses), "--start", "1", "--end", "2", "--output", str(output)])
                with self.assertRaises(ValueError):
                    merge(args)
                self.assertFalse(output.exists())
                report = json.loads((root / "out/merge_1_2_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["processed_frame_count"], 0)
                self.assertEqual(report["missing_frame_count"], 0 if include else 1)
                self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
