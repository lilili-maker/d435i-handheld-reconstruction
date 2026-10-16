"""不依赖相机：验证变换方向、累积顺序、失败中止和深度检查。"""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import open3d as o3d


SCRIPT = Path(__file__).resolve().parents[1] / "src/05_rgbd_odometry.py"
spec = importlib.util.spec_from_file_location("odometry", SCRIPT)
odometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(odometry)


class OdometryTests(unittest.TestCase):
    def test_composition_with_noncommuting_rotations(self):
        first = np.eye(4)
        first[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        first[:3, 3] = [0.1, 0, 0]
        second = np.eye(4)
        second[:3, 3] = [0, 0.2, 0]
        camera_pose = odometry.compose_camera_to_world(np.eye(4), first)
        camera_pose = odometry.compose_camera_to_world(camera_pose, second)
        # world-to-camera 必须按 second @ first；交换顺序会得到不同的相机位置。
        np.testing.assert_allclose(camera_pose, np.linalg.inv(second @ first))
        self.assertFalse(np.allclose(camera_pose, np.linalg.inv(first @ second)))
        point_world = np.array([0.5, 0.4, 1, 1])
        np.testing.assert_allclose(camera_pose @ second @ first @ point_world, point_world)

    def test_invalid_se3_and_motion_jump_are_rejected(self):
        for transform in (np.full((4, 4), np.nan), np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])):
            reasons, _ = odometry.inspect_transform(transform, 0.25, 30)
            self.assertTrue(reasons)
        jump = np.eye(4)
        jump[0, 3] = 1
        self.assertTrue(odometry.inspect_transform(jump, 0.25, 30)[0])

    def test_real_api_source_to_target_direction(self):
        # 同一平面纹理在 target 右移 2 像素，z=1m、fx=100，期望 T_x=+0.02m。
        rng = np.random.default_rng(3)
        image = cv2.GaussianBlur(rng.integers(0, 256, (160, 200), dtype=np.uint8), (3, 3), 0)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 0]]), (200, 160))
        depth = np.full((160, 200), 1000, np.uint16)
        target_depth = depth.copy()
        target_depth[:, :2] = 0
        def rgbd(gray, raw_depth):
            color = np.repeat(gray[:, :, None], 3, axis=2)
            return o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color), o3d.geometry.Image(raw_depth), depth_scale=1000, convert_rgb_to_intensity=True)
        success, transform, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd(image, depth), rgbd(shifted, target_depth),
            o3d.camera.PinholeCameraIntrinsic(200, 160, 100, 100, 99.5, 79.5),
            np.eye(4), o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            o3d.pipelines.odometry.OdometryOption(),
        )
        self.assertTrue(success)
        self.assertAlmostEqual(transform[0, 3], 0.02, delta=0.003)
        np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=0.005)

    def test_failure_stops_trajectory_without_identity_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            args = odometry.parse_args(["--start", "500", "--end", "503", "--output", folder])
            timestamps = {i: [i * 33., i * 33.] for i in range(500, 504)}
            transform = np.eye(4)
            transform[0, 3] = 0.02
            results = [
                {"accepted": True, "transformation": transform.tolist()},
                {"accepted": False, "failure_reasons": ["injected API failure"]},
            ]
            with patch.object(odometry, "load_frame", side_effect=lambda dataset, i, *rest: {"id": i}), patch.object(odometry, "estimate_pair", side_effect=results) as estimate:
                self.assertFalse(odometry.run_trajectory(list(timestamps), timestamps, {}, None, args))
            self.assertEqual(estimate.call_count, 2)
            prefix = Path(folder) / "trajectory_500_503"
            data = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual([r["status"] for r in data["frames"]], ["reference", "estimated", "failed", "not_estimated"])
            np.testing.assert_allclose(data["frames"][1]["camera_to_world"], np.linalg.inv(transform))
            self.assertIsNone(data["frames"][2]["camera_to_world"])
            self.assertIsNone(data["frames"][3]["camera_to_world"])
            self.assertAlmostEqual(data["total_translation_path_m"], 0.02)
            with prefix.with_suffix(".csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[2]["c2w_00"], "")
            self.assertEqual(rows[3]["x_m"], "")
            self.assertTrue(prefix.with_suffix(".png").is_file())

    def test_bad_start_still_saves_failure_records_and_plot(self):
        with tempfile.TemporaryDirectory() as folder:
            args = odometry.parse_args(["--start", "1", "--end", "2", "--output", folder])
            with patch.object(odometry, "load_frame", side_effect=ValueError("invalid depth")):
                self.assertFalse(odometry.run_trajectory([1, 2], {1: [1, 1], 2: [2, 2]}, {}, None, args))
            data = json.loads((Path(folder) / "trajectory_1_2.json").read_text(encoding="utf-8"))
            self.assertEqual(data["successful_frames_including_reference"], 0)
            self.assertTrue((Path(folder) / "trajectory_1_2.png").exists())

    def test_depth_dtype_empty_depth_and_dimension_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder)
            (dataset / "color").mkdir()
            (dataset / "depth").mkdir()
            args = odometry.parse_args(["--source", "1", "--target", "2"])
            metadata = {"width": 20, "height": 20, "depth_scale": .001}
            cv2.imwrite(str(dataset / "color/000001.png"), np.zeros((20, 20, 3), np.uint8))
            for depth in (np.ones((20, 20), np.uint8), np.zeros((20, 20), np.uint16), np.ones((10, 10), np.uint16)):
                cv2.imwrite(str(dataset / "depth/000001.png"), depth)
                with self.assertRaises(ValueError):
                    odometry.load_frame(dataset, 1, metadata, {1: [1, 1]}, args)

    def test_missing_file_and_duplicate_timestamp_id(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder)
            with self.assertRaises(FileNotFoundError):
                odometry.preflight(dataset, [1])
            for kind in ("color", "depth"):
                (dataset / kind).mkdir()
                (dataset / kind / "000001.png").write_bytes(b"placeholder")
            (dataset / "camera_intrinsics.json").write_text("{}")
            (dataset / "timestamps.csv").write_text("frame_id,color_timestamp_ms,depth_timestamp_ms\n1,1,1\n1,2,2\n")
            with self.assertRaises(ValueError):
                odometry.preflight(dataset, [1])

    def test_strict_json_replaces_nonfinite_diagnostics_with_null(self):
        self.assertEqual(odometry.json_safe({"failed_matrix": [float("nan"), float("inf")]}), {"failed_matrix": [None, None]})


if __name__ == "__main__":
    unittest.main()
