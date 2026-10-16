"""Pipeline 编排验证；小型 RGB-D 测试数据重用已有离线测试夹具。"""

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_pipeline as pipeline
from test_tsdf_reconstruction import inputs


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.dataset, self.poses, _, _ = inputs(self.root)
        self.config = self.root / "pipeline.json"
        self.cfg = {"dataset_path": str(self.dataset), "camera_poses_path": str(self.poses),
                    "output_directory": str(self.root / "runs"), "start_frame": 1, "end_frame": 2,
                    "enable_pointcloud_merge": False, "enable_tsdf": True,
                    "enable_mesh_crop": False, "enable_mesh_clean": False,
                    "enable_slicing": True, "slicing_axis": "X", "slicing_spacing_mm": 100}
        self.save_config()

    def save_config(self):
        self.config.write_text(json.dumps(self.cfg), encoding="utf-8")

    def run_pipeline(self, resume=None):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return pipeline.run_pipeline(self.config, resume)

    def expect_complete(self, result):
        self.assertEqual(result["status"], "complete", result.get("error"))
        return Path(result["run_directory"])

    def test_config_paths_units_and_strict_parameters(self):
        parsed = pipeline.load_config(self.config)
        self.assertEqual(parsed["dataset_path"], str(self.dataset.resolve()))
        self.assertEqual(parsed["slicing_axis"], "X")
        for key, value in [("frame_step", 0), ("min_depth", 5), ("slicing_spacing_mm", 0),
                           ("enable_tsdf", "true"), ("tsdf_voxel_size", True), ("typo", 1),
                           ("simplify_target_triangles", -1), ("compute_device", "GPU:99"), ("output_directory", str(self.dataset / "runs"))]:
            with self.subTest(key=key):
                self.save_config()
                data = {**self.cfg, key: value}
                self.config.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    pipeline.load_config(self.config)

    def test_compute_device_is_part_of_tsdf_resume_identity(self):
        cfg=pipeline.load_config(self.config)
        before,_=pipeline.fingerprint(3,cfg,{}, {},None)
        cfg["compute_device"]="auto"
        after,_=pipeline.fingerprint(3,cfg,{}, {},None)
        self.assertNotEqual(before,after)


    def test_missing_rgb_or_pose_stops_before_algorithms(self):
        (self.dataset / "color/000002.png").unlink()
        with patch.object(pipeline, "execute_algorithm") as algorithm:
            report = self.run_pipeline()
        self.assertEqual(report["status"], "failed")
        self.assertIn("000002", report["error"])
        algorithm.assert_not_called()
        self.assertEqual(report["stages"][2]["status"], "blocked")

    def test_missing_pose_record_not_replaced(self):
        data = json.loads(self.poses.read_text(encoding="utf-8"))
        data["poses"] = data["poses"][:1]
        self.poses.write_text(json.dumps(data), encoding="utf-8")
        report = self.run_pipeline()
        self.assertEqual(report["status"], "failed")
        self.assertIn("相机位姿", report["error"])

    def test_invalid_transform_rejected(self):
        data = json.loads(self.poses.read_text(encoding="utf-8"))
        data["poses"][0]["T_world_camera"][0][0] = 9
        self.poses.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.run_pipeline()["status"], "failed")

    def test_missing_depth_scale_and_bad_depth_dtype(self):
        path = self.dataset / "camera_intrinsics.json"
        original = path.read_text(encoding="utf-8")
        data = json.loads(original)
        del data["depth_scale"]
        path.write_text(json.dumps(data), encoding="utf-8")
        report = self.run_pipeline()
        self.assertEqual(report["status"], "failed")
        self.assertIn("depth_scale", report["error"])
        path.write_text(original, encoding="utf-8")
        cv2.imwrite(str(self.dataset / "depth/000001.png"), np.ones((32, 32), np.uint8))
        report = self.run_pipeline()
        self.assertEqual(report["status"], "failed")
        self.assertIn("uint16", report["error"])

    def test_stages_disabled_still_check_safety_and_report(self):
        for key in ("enable_tsdf", "enable_slicing", "enable_input_check", "enable_report"):
            self.cfg[key] = False
        self.save_config()
        report = self.run_pipeline()
        run = self.expect_complete(report)
        self.assertTrue(report["mandatory_input_checks_completed"])
        self.assertTrue(all(row["status"] == "disabled" for row in report["stages"]))
        self.assertTrue((run / "pipeline_report.json").is_file())
        self.assertNotIn("summary", report)

    def test_real_small_pipeline_outputs_colors_sources_and_svg(self):
        self.cfg["enable_pointcloud_merge"] = True
        self.save_config()
        report = self.run_pipeline()
        run = self.expect_complete(report)
        self.assertEqual(report["pose_sources"], ["simulation"])
        self.assertEqual(report["mesh_origin"], "tsdf_from_supplied_poses")
        self.assertTrue(report["final_mesh"]["has_vertex_colors"])
        self.assertGreater(report["final_mesh"]["triangle_count"], 0)
        self.assertTrue(report["stages"][1]["statistics"]["has_colors"])
        self.assertTrue(report["stages"][4]["statistics"]["millimeter_units_and_actual_geometry_verified"])
        count = len(list((run / "slices").glob("*.svg")))
        self.assertEqual(count, report["summary"]["slicing"]["svg_count"])
        saved = pipeline.read_json(run / "pipeline_report.json")
        self.assertEqual(saved["status"], report["status"])

    def test_existing_mesh_explicit_input_and_clean_only(self):
        mesh = self.root / "cube.obj"
        trimesh.creation.box(extents=[.2, .2, .2]).export(mesh)
        self.cfg.update(enable_tsdf=False, enable_mesh_clean=True, mesh_input_path=str(mesh), slicing_spacing_mm=10)
        self.save_config()
        report = self.run_pipeline()
        run = self.expect_complete(report)
        self.assertEqual(report["mesh_origin"], "external_mesh_source_not_inferred")
        processed = pipeline.read_json(run / "reports/mesh_processing.json")
        self.assertEqual(processed["mode"], "clean_only")
        # Windows临时目录可能使用8.3短路径；验证同一文件，而非路径拼写。
        self.assertTrue(Path(processed["input"]).samefile(mesh))
        self.assertEqual(report["stages"][2]["status"], "disabled")

    def test_crop_empty_stops_slicing(self):
        crop = self.root / "crop.json"
        crop.write_text(json.dumps({"unit": "m", "xmin": 10, "xmax": 11, "ymin": 10,
                                   "ymax": 11, "zmin": 10, "zmax": 11}), encoding="utf-8")
        self.cfg.update(enable_mesh_crop=True, mesh_crop_config=str(crop))
        self.save_config()
        report = self.run_pipeline()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stages"][3]["status"], "failed")
        self.assertEqual(report["stages"][4]["status"], "blocked")
        self.assertEqual(list((Path(report["run_directory"]) / "slices").glob("*.svg")), [])

    def test_stage_failure_and_interrupt_write_report(self):
        for error, expected in [(RuntimeError("injected failure"), "failed"), (KeyboardInterrupt(), "interrupted")]:
            with self.subTest(expected=expected), patch.object(pipeline, "execute_algorithm", side_effect=error):
                report = self.run_pipeline()
                self.assertEqual(report["status"], expected)
                self.assertEqual(report["stages"][4]["status"], "blocked")
                self.assertTrue((Path(report["run_directory"]) / "pipeline_report.json").is_file())

    def test_output_isolation_and_originals_not_overwritten(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.dataset.rglob('*') if p.is_file()}
        first = self.expect_complete(self.run_pipeline())
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in first.rglob('*') if p.is_file()}
        second = self.expect_complete(self.run_pipeline())
        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "run_0001")
        self.assertEqual(second.name, "run_0002")
        for p, digest in {**before, **hashes}.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)

    def test_resume_hash_and_geometry_checked_then_copied(self):
        first = self.expect_complete(self.run_pipeline())
        report = self.run_pipeline(first)
        second = self.expect_complete(report)
        self.assertEqual(report["stages"][2]["status"], "reused")
        self.assertEqual(report["stages"][4]["status"], "reused")
        self.assertEqual(pipeline.sha256(first / "mesh/scene_mesh.ply"), pipeline.sha256(second / "mesh/scene_mesh.ply"))
        slices = pipeline.read_json(second / "slices/slicing_report.json")
        self.assertTrue(all(Path(layer["svg_path"]).parent == second / "slices" for layer in slices["layers"]))
        self.assertEqual(slices["pipeline_reuse"]["source_run"], str(first))

    def test_resume_config_change_only_reexecutes_affected_stage(self):
        first = self.expect_complete(self.run_pipeline())
        self.cfg["slicing_spacing_mm"] = 50
        self.save_config()
        report = self.run_pipeline(first)
        self.expect_complete(report)
        self.assertEqual(report["stages"][2]["status"], "reused")
        self.assertEqual(report["stages"][4]["status"], "complete")
        self.assertIn("变化", report["stages"][4]["cache_check"])

    def test_resume_changed_rgb_input_reexecutes_tsdf(self):
        first = self.expect_complete(self.run_pipeline())
        path = self.dataset / "color/000001.png"
        rgb = cv2.imread(str(path))
        rgb[:, :, 0] = 120
        cv2.imwrite(str(path), rgb)
        report = self.run_pipeline(first)
        self.expect_complete(report)
        self.assertEqual(report["stages"][2]["status"], "complete")
        self.assertIn("变化", report["stages"][2]["cache_check"])

    def test_resume_changed_camera_pose_reexecutes_tsdf(self):
        first = self.expect_complete(self.run_pipeline())
        data = pipeline.read_json(self.poses)
        data["poses"][1]["T_world_camera"][0][3] += .01
        self.poses.write_text(json.dumps(data), encoding="utf-8")
        report = self.run_pipeline(first)
        self.expect_complete(report)
        self.assertEqual(report["stages"][2]["status"], "complete")
        self.assertIn("变化", report["stages"][2]["cache_check"])

    def test_resume_without_path_chooses_latest_previous_run(self):
        first = self.expect_complete(self.run_pipeline())
        report = self.run_pipeline("latest")
        second = self.expect_complete(report)
        self.assertEqual(report["resume_source"], str(first))
        self.assertNotEqual(first, second)
        self.assertEqual(report["stages"][2]["status"], "reused")

    def test_resume_modified_svg_recomputed_not_silently_skipped(self):
        first = self.expect_complete(self.run_pipeline())
        path = next((first / "slices").glob("*.svg"))
        path.write_text("invalid SVG", encoding="utf-8")
        report = self.run_pipeline(first)
        self.expect_complete(report)
        self.assertEqual(report["stages"][2]["status"], "reused")
        self.assertEqual(report["stages"][4]["status"], "complete")
        self.assertIn("内容变化", report["stages"][4]["cache_check"])

    def test_svg_count_mismatch_validation_fails(self):
        run = self.expect_complete(self.run_pipeline())
        path = run / "slices/slicing_report.json"
        data = pipeline.read_json(path)
        data["svg_count"] += 1
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "数量"):
            pipeline.validate_outputs(5, pipeline.load_config(self.config), run, run / "mesh/scene_mesh.ply")

    def test_resume_incomplete_run_reuses_completed_stages(self):
        original = pipeline.execute_algorithm
        def fail_slice(stage, *args):
            if stage == 5:
                raise RuntimeError("slice failure")
            return original(stage, *args)
        with patch.object(pipeline, "execute_algorithm", side_effect=fail_slice):
            stopped = self.run_pipeline()
        self.assertEqual(stopped["status"], "failed")
        report = self.run_pipeline(Path(stopped["run_directory"]))
        self.expect_complete(report)
        self.assertEqual(report["stages"][2]["status"], "reused")
        self.assertEqual(report["stages"][4]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
