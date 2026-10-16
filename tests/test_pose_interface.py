"""统一位姿接口的纯数学和文件测试，不依赖相机、机器人或真实手眼标定。"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reconstruction.pose_utils import (
    validate_rotation_matrix, validate_transform, invert_transform,
    compose_transforms, transform_points, compute_base_camera,
)
from reconstruction.pose_io import CameraPose, load_poses, save_poses, require_pose
from reconstruction.convert_odometry_poses import CSV_FIELDS, MATRIX_FIELDS, convert_odometry_poses


def translation(x, y, z):
    t = np.eye(4)
    t[:3, 3] = [x, y, z]
    return t


def rotation_z90():
    t = np.eye(4)
    t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    return t


def pose(i=500, matrix=None, **kwargs):
    values = dict(frame_id=i, timestamp=1.5, T_world_camera=np.eye(4) if matrix is None else matrix,
                  pose_source="simulation", coordinate_frame="test_world",
                  timestamp_reference="simulation clock", validity=True)
    values.update(kwargs)
    return CameraPose(**values)


def legacy_row(i, matrix, status, *, world=500):
    row = dict(frame_id=str(i), color_timestamp_ms=str(i * 33. + 1),
               depth_timestamp_ms=str(i * 33.), status=status, reason="",
               pose_convention="camera_to_world", world_frame=str(world))
    if matrix is None:
        row.update({key: "" for key in MATRIX_FIELDS + ["x_m", "y_m", "z_m"]})
    else:
        row.update(zip(MATRIX_FIELDS, map(str, matrix.ravel())))
        row.update(zip(("x_m", "y_m", "z_m"), map(str, matrix[:3, 3])))
    return row


def write_legacy(path, rows, fields=CSV_FIELDS):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


class PoseMathTests(unittest.TestCase):
    def test_identity(self):
        np.testing.assert_array_equal(validate_transform(np.eye(4)), np.eye(4))
        np.testing.assert_array_equal(invert_transform(np.eye(4)), np.eye(4))
        np.testing.assert_array_equal(transform_points(np.eye(4), [1, 2, 3]), [1, 2, 3])

    def test_translation_and_point_cloud(self):
        points = np.array([[0., 0., 0.], [1., 2., 3.]])
        original = points.copy()
        np.testing.assert_array_equal(transform_points(translation(.1, .2, .3), points), points + [.1, .2, .3])
        np.testing.assert_array_equal(points, original)
        self.assertEqual(transform_points(np.eye(4), np.empty((0, 3))).shape, (0, 3))

    def test_rotation(self):
        r = rotation_z90()
        np.testing.assert_array_equal(validate_rotation_matrix(r[:3, :3]), r[:3, :3])
        np.testing.assert_array_equal(transform_points(r, [1, 0, 0]), [0, 1, 0])

    def test_inverse(self):
        t = compose_transforms(translation(.1, .2, .3), rotation_z90())
        inverse = invert_transform(t)
        np.testing.assert_allclose(t @ inverse, np.eye(4), atol=1e-15)
        np.testing.assert_allclose(inverse, np.linalg.inv(t), atol=1e-15)
        p = np.array([[.4, .5, .6]])
        np.testing.assert_allclose(transform_points(inverse, transform_points(t, p)), p)

    def test_chain_and_noncommutative_order(self):
        T_A_B = rotation_z90()
        T_B_C = translation(.2, 0, 0)
        T_C_D = translation(0, .3, 0)
        combined = compose_transforms(T_A_B, T_B_C, T_C_D)
        np.testing.assert_allclose(transform_points(combined, [0, 0, 0]), [-.3, .2, 0])
        np.testing.assert_allclose(combined, T_A_B @ T_B_C @ T_C_D)
        self.assertFalse(np.allclose(combined, T_C_D @ T_B_C @ T_A_B))

    def test_illegal_matrices_rejected_without_repair(self):
        bad_bottom = np.eye(4)
        bad_bottom[3, 0] = 1
        for invalid in (np.eye(3), np.full((4, 4), np.nan), np.full((4, 4), np.inf),
                        np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1]), bad_bottom,
                        np.eye(4).astype(str), np.eye(4).astype(bool), np.eye(4).astype(complex)):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                validate_transform(invalid)
        with self.assertRaises(ValueError):
            compose_transforms()
        with self.assertRaises(ValueError):
            transform_points(np.eye(4), [[1, 2]])

    def test_simulated_base_end_camera(self):
        # 纯人工矩阵，只验证数学关系；不代表真实手眼标定结果。
        T_base_end = compose_transforms(translation(.5, 0, .3), rotation_z90())
        T_end_camera = translation(.1, 0, .2)
        T_base_camera = compute_base_camera(T_base_end, T_end_camera,
                                           end_frame="flange", calibration_end_frame="flange")
        np.testing.assert_allclose(T_base_camera[:3, 3], [.5, .1, .5])
        np.testing.assert_allclose(transform_points(T_base_camera, [.1, .2, 1]),
                                   transform_points(T_base_end, transform_points(T_end_camera, [.1, .2, 1])))

    def test_tcp_not_assumed_equal_to_flange(self):
        with self.assertRaisesRegex(ValueError, "不匹配"):
            compute_base_camera(np.eye(4), np.eye(4), end_frame="tcp", calibration_end_frame="flange")
        # 使用已知（这里仍为人工模拟）的工具变换，先把 TCP 位姿转换到法兰。
        T_base_tcp = translation(.5, 0, .3)
        T_tcp_flange = translation(0, 0, -.1)
        T_flange_camera = translation(.2, 0, 0)
        computed = compute_base_camera(compose_transforms(T_base_tcp, T_tcp_flange), T_flange_camera,
                                       end_frame="flange", calibration_end_frame="flange")
        np.testing.assert_allclose(computed[:3, 3], [.7, 0, .2])


class PoseIOTests(unittest.TestCase):
    def test_roundtrip_sources_do_not_imply_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested/poses.json"
            original = [pose(i, translation(.1, .2, .3), pose_source=source)
                        for i, source in enumerate(("simulation", "robot", "rgbd_odometry"))]
            save_poses(path, original)
            loaded = load_poses(path)
            for item in original:
                actual = require_pose(loaded, f"{item.frame_id:06d}", coordinate_frame="test_world")
                np.testing.assert_array_equal(actual.T_world_camera, item.T_world_camera)
                self.assertEqual(actual.timestamp, item.timestamp)
                self.assertEqual(actual.pose_source, item.pose_source)
            self.assertIn("not a guarantee", json.loads(path.read_text(encoding="utf-8"))["validity_definition"])

    def test_missing_invalid_pose_and_world_mismatch(self):
        invalid = pose(501, T_world_camera=None, validity=False, reason="estimation failed")
        records = {500: pose(), 501: invalid}
        with self.assertRaises(KeyError):
            require_pose(records, 499)
        with self.assertRaises(ValueError):
            require_pose(records, 501)
        with self.assertRaises(ValueError):
            require_pose(records, 500, coordinate_frame="robot_base")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            save_poses(path, records)
            self.assertIsNone(load_poses(path)[501].T_world_camera)

    def test_bad_units_validity_and_source(self):
        for kwargs in (dict(translation_unit="mm"), dict(timestamp_unit="ms"), dict(pose_source="unknown"),
                       dict(timestamp=float("nan")), dict(validity="true"), dict(coordinate_frame=""),
                       dict(validity=False), dict(T_world_camera=None)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pose(**kwargs)

    def test_duplicate_mixed_worlds_and_mismatched_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            for records in ([pose(), pose()], [pose(), pose(501, coordinate_frame="robot_base")], {999: pose()}):
                with self.assertRaises(ValueError):
                    save_poses(Path(folder) / "bad.json", records)
            self.assertFalse((Path(folder) / "bad.json").exists())

    def test_no_overwrite_and_no_direction_guess(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poses.json"
            save_poses(path, [pose()])
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_poses(path, [pose(501)])
            self.assertEqual(path.read_bytes(), original)
            data = json.loads(original)
            data["transform_direction"] = "world_to_camera"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "transform_direction"):
                load_poses(path)


class LegacyConversionTests(unittest.TestCase):
    def test_c2w_preserved_and_invalid_rows_remain_null(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "legacy.csv", Path(folder) / "unified.json"
            t = compose_transforms(translation(.1, .2, .3), rotation_z90())
            write_legacy(source, [legacy_row(500, np.eye(4), "reference"), legacy_row(501, t, "estimated"),
                                  legacy_row(502, None, "failed"), legacy_row(503, None, "not_estimated")])
            original = source.read_bytes()
            records = convert_odometry_poses(source, output)
            np.testing.assert_array_equal(records[501].T_world_camera, t)
            self.assertFalse(np.array_equal(records[501].T_world_camera, invert_transform(t)))
            self.assertEqual(records[501].timestamp, (501 * 33 + 1) / 1000)
            self.assertEqual(records[501].coordinate_frame, "odometry_camera_500")
            self.assertEqual(records[501].pose_source, "rgbd_odometry")
            self.assertIsNone(records[502].T_world_camera)
            with self.assertRaises(ValueError):
                require_pose(records, 503)
            self.assertEqual(source.read_bytes(), original)

    def test_ambiguous_or_inconsistent_csv_stops_without_output(self):
        changes = [
            {"pose_convention": ""}, {"pose_convention": "world_to_camera"},
            {"world_frame": "501"}, {"x_m": "9"}, {"c2w_00": "nan"},
            {"c2w_33": "0"}, {"frame_id": "500"}, {"status": "unknown"},
            {"color_timestamp_ms": "nan"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "legacy.csv", Path(folder) / "bad.json"
            for update in changes:
                row = legacy_row(501, translation(.1, 0, 0), "estimated")
                row.update(update)
                write_legacy(source, [legacy_row(500, np.eye(4), "reference"), row])
                with self.subTest(update=update), self.assertRaises(ValueError):
                    convert_odometry_poses(source, output)
                self.assertFalse(output.exists())
            write_legacy(source, [], fields=[k for k in CSV_FIELDS if k != "pose_convention"])
            with self.assertRaises(ValueError):
                convert_odometry_poses(source, output)

    def test_missing_image_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "legacy.csv", root / "unified.json"
            write_legacy(source, [legacy_row(500, np.eye(4), "reference")])
            (root / "timestamps.csv").write_text("frame_id,color_timestamp_ms,depth_timestamp_ms\n500,16501,16500\n")
            with self.assertRaisesRegex(FileNotFoundError, "图像不存在"):
                convert_odometry_poses(source, output, dataset=root)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
