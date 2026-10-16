"""已知合成几何验证方向、丢失跟踪、回环拒绝和官方全局优化；不代表实测精度。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import open3d as o3d
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT))
from reconstruction import handheld_odometry as h
from reconstruction import handheld_pose_graph as pg
from reconstruction.pose_io import load_poses
from reconstruction.analyze_handheld_dataset import analyze_dataset
from reconstruction.compute_device import select_device
from test_tsdf_reconstruction import inputs
import run_handheld_reconstruction as runner


def translation(x):
    t=np.eye(4);t[0,3]=x;return t


def chain(d=.1):
    graph=o3d.pipelines.registration.PoseGraph()
    for x in (0,d,2*d):
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(translation(x)))
    for i in (0,1):
        graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i,i+1,translation(-d),np.eye(6)*100,uncertain=False))
    return graph


class HandheldTests(unittest.TestCase):
    def test_40_percent_default_and_strict_cli(self):
        args=runner.parse_args(["--output","unused","--no-crop"])
        self.assertEqual(args.min_consistency,.4)
        self.assertEqual(args.odometry_backend,"tensor")
        self.assertEqual(args.icp_refinement,"off")
        self.assertEqual(h.settings(Path('.')).min_consistency,.4)
        for value in ("0","1.1","nan"):
            with self.assertRaises(ValueError):
                runner.parse_args(["--output","unused","--no-crop","--min-consistency",value])

    def test_auto_cpu_fallback_and_explicit_cuda_rejection(self):
        with patch.object(o3d.core.cuda,"is_available",return_value=False):
            device,report=select_device("auto")
            self.assertEqual(device,"CPU:0")
            self.assertFalse(report["cuda_available"])
            with self.assertRaises(ValueError):select_device("CUDA:0")
        with patch.object(o3d.core.cuda,"is_available",return_value=True):
            self.assertEqual(select_device("auto")[0],"CUDA:0")
            self.assertEqual(select_device("CPU:0")[0],"CPU:0")

    def test_fusion_sampling_uses_only_real_valid_frames(self):
        ids=list(range(12,63))
        chosen=runner.fusion_frames(ids,3)
        self.assertEqual(chosen,list(range(12,61,3)))
        self.assertEqual(len(chosen),17)
        self.assertTrue(set(chosen)<=set(ids))
        with self.assertRaises(ValueError):runner.fusion_frames(ids,0)

    def test_vectorized_candidates_equal_original_transform_rule(self):
        poses=[]
        for i in range(10):
            matrix=translation(i*.05)
            angle=np.deg2rad(i*5)
            matrix[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
            poses.append(SimpleNamespace(frame_id=i*10,T_world_camera=matrix))
        expected=[]
        for i,a in enumerate(poses):
            for j in range(i+1,len(poses)):
                b=poses[j]
                if b.frame_id-a.frame_id<30:continue
                rel=pg.relative(a.T_world_camera,b.T_world_camera)
                angle=np.degrees(np.arccos(np.clip((np.trace(rel[:3,:3])-1)/2,-1,1)))
                distance=np.linalg.norm(rel[:3,3])
                if distance<.4 and angle<40:expected.append((distance,i,j))
        actual=pg.loop_candidates(poses,30)
        self.assertEqual([(i,j) for _,i,j in actual],[(i,j) for _,i,j in expected])
        np.testing.assert_allclose([d for d,_,_ in actual],[d for d,_,_ in expected],atol=1e-12)

    def test_tensor_known_textured_plane_direction_scale_and_failure(self):
        # 真值：平面距离1m，fx=150，目标相机向右移动0.02m => 图像左移3像素。
        rng=np.random.default_rng(3)
        rgb=cv2.GaussianBlur(rng.integers(20,235,(128,128,3),dtype=np.uint8),(5,5),0)
        target_rgb=np.zeros_like(rgb);target_rgb[:,:-3]=rgb[:,3:]
        depth=np.full((128,128),1000,np.uint16);target_depth=depth.copy();target_depth[:,-3:]=0
        metadata=dict(width=128,height=128,fx=150.,fy=150.,cx=63.5,cy=63.5,depth_scale=.001)
        source=dict(id=0,rgb=rgb,depth=depth,stats={})
        target=dict(id=1,rgb=target_rgb,depth=target_depth,stats={})
        intrinsic=o3d.camera.PinholeCameraIntrinsic(128,128,150,150,63.5,63.5)
        args=h.settings(Path('.'),odometry_backend="tensor",device="CPU:0")
        result=h.estimate_pair(source,target,intrinsic,metadata,args)
        self.assertTrue(result["accepted"],result["failure_reasons"])
        self.assertIsNone(result["api_success"])
        np.testing.assert_allclose(result["transformation"],translation(-.02),atol=.003)
        self.assertEqual(depth.dtype,np.uint16)
        failed=SimpleNamespace(transformation=o3d.core.Tensor(np.eye(4)),fitness=0.,inlier_rmse=float('inf'))
        with patch.object(o3d.t.pipelines.odometry,"rgbd_odometry_multi_scale",return_value=failed):
            result=h.estimate_pair(source,target,intrinsic,metadata,args)
        self.assertFalse(result["accepted"])

    def test_source_target_direction_from_camera_to_world(self):
        source=translation(.2);source[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        target=translation(.5)
        t=pg.relative(source,target)
        p=np.array([.1,.2,1,1])
        np.testing.assert_allclose(target@t@p,source@p)
        np.testing.assert_allclose(h.legacy().compose_camera_to_world(source,t),target,atol=1e-12)

    def test_bidirectional_cycle_rejects_inconsistent_motion(self):
        self.assertTrue(h.cycle_check(translation(.1),translation(-.1))["passed"])
        self.assertFalse(h.cycle_check(translation(.1),translation(-.04))["passed"])

    def test_api_failure_cannot_be_rescued_by_icp(self):
        failure={"accepted":False,"failure_reasons":["Odometry API 返回失败"],"transformation":np.eye(4).tolist()}
        with patch.object(h.legacy(),"estimate_pair",return_value=failure),patch.object(h,"camera_cloud") as cloud:
            result=h.checked_pair({}, {}, None, {}, None)
        self.assertFalse(result["accepted"])
        cloud.assert_not_called()

    def test_depth_gate_does_not_hide_invalid_motion(self):
        failure={"accepted":False,"failure_reasons":["独立重投影深度检查的支持数量或比例不足","平移异常"],"transformation":np.eye(4).tolist()}
        with patch.object(h.legacy(),"estimate_pair",return_value=failure),patch.object(h,"camera_cloud") as cloud:
            result=h.checked_pair({}, {}, None, {}, None)
        self.assertFalse(result["accepted"])
        cloud.assert_not_called()

    def test_information_degeneracy_is_rejected(self):
        with patch.object(o3d.pipelines.registration,"get_information_matrix_from_point_clouds",return_value=np.zeros((6,6))):
            with self.assertRaises(ValueError):
                h.information(None,None,np.eye(4))

    def test_no_crop_does_not_assume_old_world_roi(self):
        args=runner.parse_args(["--output","unused","--no-crop","--reference","100","--start","100","--end","150"])
        self.assertIsNone(args.crop_config)

    def test_analysis_refuses_ambiguous_number_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,_,_,_=inputs(root)
            (dataset/"color/1.png").write_bytes((dataset/"color/000001.png").read_bytes())
            with self.assertRaises(ValueError):
                analyze_dataset(dataset,root/"analysis")

    def test_official_graph_chain_without_loops_preserves_poses(self):
        graph=chain()
        matrices,report=pg.optimize_graph(graph,1)
        for i,m in enumerate(matrices):
            np.testing.assert_allclose(m,translation(.1*i),atol=1e-8)
        self.assertEqual(report["loop_edges_retained"],0)

    def test_known_loop_reduces_synthetic_trajectory_error(self):
        # 合成真实位置为 0,.1,.2 m；相邻里程计有已知 10% 偏差；回环来自真值。
        graph=chain(.11)
        graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(0,2,translation(-.2),np.eye(6)*10000,uncertain=True))
        matrices,report=pg.optimize_graph(graph,0)
        self.assertLess(abs(matrices[2][0,3]-.2),.02)
        np.testing.assert_allclose(matrices[0],np.eye(4),atol=1e-8)
        self.assertGreaterEqual(report["loop_edges_retained"],1)

    def test_wrong_feature_correspondences_rejected_before_graph(self):
        rng=np.random.default_rng(42)
        p=rng.uniform(-1,1,(40,3));q=rng.uniform(-1,1,(40,3));uv=rng.uniform(0,400,(40,2))
        with patch.object(pg,"orb_depth_matches",return_value=(p,q,uv,uv)):
            result=pg.validate_loop({"id":0},{"id":200},o3d.geometry.PointCloud(),o3d.geometry.PointCloud(),np.eye(4),
                                    {"width":640,"height":480},None)
        self.assertFalse(result["accepted"])
        self.assertIsNotNone(result["reason"])

    def test_failed_tracking_keeps_null_pose_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,_,_,_=inputs(root)
            output=root/"track"
            with patch.object(h,"read_frame",side_effect=lambda dataset,i,*a:{"id":i}),patch.object(h,"checked_pair",return_value={"accepted":False,"failure_reasons":["injected mismatch"]}):
                _,report,*_=h.track(dataset,[1,2],1,output,h.settings(dataset))
            poses=load_poses(output/"poses_before.json")
            self.assertTrue(poses[1].validity)
            self.assertFalse(poses[2].validity)
            self.assertIsNone(poses[2].T_world_camera)
            self.assertEqual(report["valid_frame_count"],1)
            self.assertEqual(report["accepted_pairs"],0)

    def test_interruption_saves_partial_poses_without_starting_fusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,_,_,_=inputs(root)
            with patch.object(h,"read_frame",side_effect=lambda dataset,i,*a:{"id":i}),patch.object(h,"checked_pair",side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    h.track(dataset,[1,2],1,root/"track",h.settings(dataset))
            poses=load_poses(root/"track/poses_before.json")
            self.assertTrue(poses[1].validity)
            self.assertFalse(poses[2].validity)
            self.assertIsNone(poses[2].T_world_camera)
            report=json.loads((root/"track/registration_quality.json").read_text(encoding="utf-8"))
            self.assertTrue(report["interrupted"])
            self.assertEqual(report["accepted_pairs"],0)

    def test_reference_bidirectional_accumulation_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,_,_,_=inputs(root)
            for kind in ("color","depth"):
                (dataset/kind/"000003.png").write_bytes((dataset/kind/"000002.png").read_bytes())
            with (dataset/"timestamps.csv").open("a",encoding="utf-8") as f:f.write("3,99.9,99.9\n")
            def pair(s,t,*a):
                return {"accepted":True,"failure_reasons":[],"transformation":translation(.1 if t["id"]==1 else -.1).tolist(),"graph_information_matrix":np.eye(6).tolist()}
            with patch.object(h,"read_frame",side_effect=lambda dataset,i,*a:{"id":i}),patch.object(h,"checked_pair",side_effect=pair):
                poses,report,*_=h.track(dataset,[1,2,3],2,root/"track",h.settings(dataset))
            np.testing.assert_allclose(poses[0].T_world_camera,translation(-.1))
            np.testing.assert_allclose(poses[1].T_world_camera,np.eye(4))
            np.testing.assert_allclose(poses[2].T_world_camera,translation(.1))
            self.assertEqual(report["attempted_pair_success_ratio"],1)

    def test_input_analysis_missing_pair_and_repeated_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,_,_,_=inputs(root)
            rows=(dataset/"timestamps.csv").read_text(encoding="utf-8").replace("2,66.6,66.6","2,66.6,33.3")
            (dataset/"timestamps.csv").write_text(rows,encoding="utf-8")
            with redirect_stdout(io.StringIO()):r=analyze_dataset(dataset,root/"analysis")
            self.assertEqual(r["matched_frame_count"],2)
            self.assertFalse(r["frames"][1]["input_quality_passed"])
            (dataset/"color/000002.png").unlink()
            with redirect_stdout(io.StringIO()):r=analyze_dataset(dataset,root/"analysis2")
            self.assertEqual(r["matched_frame_count"],1)
            self.assertEqual(r["depth_only"],[2])

    def test_mesh_quality_uses_actual_faces_color_and_meter_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"cube.ply"
            mesh=o3d.geometry.TriangleMesh.create_box(.2,.2,.2);mesh.paint_uniform_color([.2,.3,.4])
            o3d.io.write_triangle_mesh(str(path),mesh)
            stats=runner.quality_stats(path)
            self.assertEqual(stats["triangle_count"],12)
            self.assertEqual(stats["boundary_edge_count"],0)
            self.assertTrue(stats["has_vertex_colors"])
            np.testing.assert_allclose(stats["extent_m"],[.2,.2,.2])

    def test_no_overwrite_and_invalid_cli_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset,_,_,_=inputs(root);out=root/"out";out.mkdir()
            with self.assertRaises(FileExistsError):h.track(dataset,[1,2],1,out,h.settings(dataset))
            for args in (["--step","0"],["--pointcloud-voxel","0"],["--max-loop-candidates","-1"]):
                with self.assertRaises(ValueError):runner.parse_args(["--output",str(root/"new")]+args)


if __name__=="__main__":
    unittest.main()
