"""执行实际两种 Odometry 和 TSDF，验证非零旋转/平移方向、单位及重影。"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import open3d as o3d

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from reconstruction import handheld_odometry as h
from reconstruction import tsdf_reconstruction as tsdf
from reconstruction.pose_io import load_poses
from reconstruction.pose_utils import invert_transform,transform_points
from rgbd_synthetic_scene import write_dataset,surface_errors


class DirectionFusionTests(unittest.TestCase):
    def test_inconsistent_initial_cycle_cannot_be_hidden_by_symmetric_icp(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,_,_,_=write_dataset(root)
            metadata=h.legacy().load_metadata(dataset);times=h.legacy().preflight(dataset,[0])
            intrinsic=o3d.camera.PinholeCameraIntrinsic(metadata["width"],metadata["height"],metadata["fx"],metadata["fy"],metadata["cx"],metadata["cy"])
            args=h.settings(dataset,icp_refinement="geometry")
            source=h.read_frame(dataset,0,metadata,times,args);target=h.read_frame(dataset,0,metadata,times,args)
            reverse=np.eye(4);reverse[0,3]=.04
            def fake(transform):return dict(accepted=True,failure_reasons=[],transformation=transform.tolist(),metrics={})
            with patch.object(h,"estimate_pair",side_effect=[fake(np.eye(4)),fake(reverse)]),patch.object(o3d.pipelines.registration,"registration_icp",return_value=SimpleNamespace(transformation=np.eye(4))):
                result=h.checked_pair(source,target,intrinsic,metadata,args)
            self.assertTrue(result["bidirectional_cycle"]["passed"])
            self.assertFalse(result["hybrid_initial_cycle"]["passed"])
            self.assertFalse(result["accepted"])
            self.assertIn("原始 Hybrid 正反向不一致，禁止用几何 ICP 掩盖",result["failure_reasons"])

    def test_default_checked_pair_keeps_hybrid_solution_without_icp_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,_,local,_=write_dataset(root)
            metadata=h.legacy().load_metadata(dataset);times=h.legacy().preflight(dataset,[0,1])
            intrinsic=o3d.camera.PinholeCameraIntrinsic(metadata["width"],metadata["height"],metadata["fx"],metadata["fy"],metadata["cx"],metadata["cy"])
            args=h.settings(dataset,min_consistency=.6)
            source=h.read_frame(dataset,0,metadata,times,args);target=h.read_frame(dataset,1,metadata,times,args)
            with patch.object(o3d.pipelines.registration,"registration_icp",side_effect=AssertionError("default must not override Hybrid")):
                result=h.checked_pair(source,target,intrinsic,metadata,args)
            self.assertTrue(result["accepted"],result["failure_reasons"])
            np.testing.assert_allclose(result["transformation"],invert_transform(local[1]),atol=.005)
            np.testing.assert_allclose(result["transformation"],result["hybrid_initial_transformation"],atol=1e-12)

    def test_actual_backends_rotation_translation_direction_and_accumulation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,posefile,local,gauge=write_dataset(root)
            metadata=h.legacy().load_metadata(dataset);times=h.legacy().preflight(dataset,[0,1])
            intrinsic=o3d.camera.PinholeCameraIntrinsic(metadata["width"],metadata["height"],metadata["fx"],metadata["fy"],metadata["cx"],metadata["cy"])
            poses=load_poses(posefile)
            truth=invert_transform(local[1])@local[0]
            for backend in ("tensor","legacy"):
                args=h.settings(dataset,odometry_backend=backend,min_consistency=.6)
                source=h.read_frame(dataset,0,metadata,times,args);target=h.read_frame(dataset,1,metadata,times,args)
                result=h.estimate_pair(source,target,intrinsic,metadata,args)
                self.assertTrue(result["accepted"],result["failure_reasons"])
                estimate=np.asarray(result["transformation"])
                np.testing.assert_allclose(estimate,truth,atol=.005)
                # 非交换旋转+平移，反向矩阵或错误乘法顺序会失败。
                target_world=h.legacy().compose_camera_to_world(poses[0].T_world_camera,estimate)
                np.testing.assert_allclose(target_world,poses[1].T_world_camera,atol=.007)
                sample=np.array([.1,-.1,1.4])
                np.testing.assert_allclose(transform_points(target_world,transform_points(estimate,sample)),transform_points(poses[0].T_world_camera,sample),atol=1e-10)
                self.assertGreater(np.linalg.norm(estimate[:3,3]-invert_transform(truth)[:3,3]),.03)

    def test_actual_single_and_multiple_static_plane_tsdf(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,posefile,_,gauge=write_dataset(root,corner=False)
            for end in (0,2):
                output=root/"meshes"/f"plane_{end}.ply"
                args=tsdf.parse_args(["--dataset",str(dataset),"--poses",str(posefile),"--start","0","--end",str(end),"--output",str(output)])
                report=tsdf.reconstruct(args);mesh=o3d.io.read_triangle_mesh(str(output))
                self.assertTrue(report["mesh_ply"]["readback_verified"])
                errors=surface_errors(mesh,gauge,corner=False)
                self.assertLess(np.percentile(errors,99),.012)
                self.assertLess(np.std(((np.asarray(mesh.vertices)-gauge[:3,3])@gauge[:3,:3])[:,2]),.004)
                self.assertTrue(mesh.has_vertex_colors())
                poses=load_poses(posefile)
                for row in report["frames"]:
                    np.testing.assert_allclose(row["T_camera_world"],invert_transform(poses[row["frame_id"]].T_world_camera),atol=1e-12)

    def test_wrong_tsdf_direction_is_detectable_by_known_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,posefile,_,gauge=write_dataset(root,corner=False)
            metadata=h.legacy().load_metadata(dataset);poses=load_poses(posefile)
            intrinsic=o3d.camera.PinholeCameraIntrinsic(metadata["width"],metadata["height"],metadata["fx"],metadata["fy"],metadata["cx"],metadata["cy"])
            volume=tsdf.make_volume(.01,.04)
            frame,_=tsdf.load_rgbd(dataset,0,metadata,.2,3)
            # 故意输入错误方向作负对照，只生成测试对象，不改变真实重建逻辑。
            tsdf.integrate_rgbd(volume,frame,intrinsic,invert_transform(poses[0].T_world_camera))
            mesh=volume.extract_triangle_mesh()
            self.assertGreater(np.median(surface_errors(mesh,gauge,corner=False)),.1)


if __name__=="__main__":unittest.main()
