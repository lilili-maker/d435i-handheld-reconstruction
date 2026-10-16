"""阶段8：严格验证的长序列轨迹、Pose Graph 和现有家具重建 Pipeline。"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

# OpenMP 必须在加载 Open3D 之前设定；避免默认 32 线程在小图像上反而变慢。
if __name__ == "__main__":
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--cpu-threads", type=int, default=4)
    bootstrap_args, _ = bootstrap.parse_known_args()
    if bootstrap_args.cpu_threads <= 0:
        raise SystemExit("cpu-threads 必须为正整数")
    os.environ["OMP_NUM_THREADS"] = str(bootstrap_args.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(bootstrap_args.cpu_threads)

import cv2
import numpy as np
import open3d as o3d

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from reconstruction import handheld_odometry as odom
from reconstruction import handheld_pose_graph as pg
from reconstruction import clean_mesh as cm
from reconstruction.analyze_handheld_dataset import write_json
from reconstruction.compute_device import select_device
import run_pipeline as pipeline


def quality_stats(path):
    mesh=cm.read_mesh(path)
    vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    edges=np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    _,component_counts,_=mesh.cluster_connected_triangles()
    return {"path":str(Path(path).resolve()),"vertex_count":len(vertices),"triangle_count":len(faces),
            "has_vertex_colors":mesh.has_vertex_colors(),"bounds_m":[vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()],
            "extent_m":np.ptp(vertices,axis=0).tolist(),"surface_area_m2":float(mesh.get_surface_area()),
            "boundary_edge_count":int((counts==1).sum()),"nonmanifold_edge_count":int((counts>2).sum()),
            "component_count":len(component_counts),"largest_component_triangles":max(component_counts,default=0),
            "note":"Boundary/area/components describe observed geometry, not semantic completeness or calibrated accuracy"}


def compare_models(old_path,new_path,output):
    old,new=quality_stats(old_path),quality_stats(new_path)
    report={"old":old,"new":new,"triangle_ratio":new["triangle_count"]/old["triangle_count"],
            "surface_area_ratio":new["surface_area_m2"]/max(old["surface_area_m2"],1e-12),
            "extent_ratio":(np.array(new["extent_m"])/np.maximum(old["extent_m"],1e-9)).tolist(),
            "both_have_rgb":old["has_vertex_colors"] and new["has_vertex_colors"],
            "coordinate_frame":"odometry_camera_500 when reference=500; compare only same frame and crop box",
            "semantic_completeness":"Not automatically measured; inspect actual renders in model_compare.png and CloudCompare",
            "no_hole_filling_or_invented_geometry":True}
    write_json(output/report_name("model_quality"),report)
    return report


def report_name(stem):
    return stem+"_report.json"


def fusion_frames(valid_ids, interval):
    if type(interval) is not int or interval <= 0:
        raise ValueError("fusion-step 必须为正整数")
    # 跟踪仍逐帧；这里只减少用于融合的已验帧，不插值或补齐任何位姿。
    return valid_ids[::interval]


def render_comparison(old_path,new_path,output,filename="model_compare.png"):
    """隐藏窗口渲染真实网格；仅作人工观察，不改变几何。失败也记录，不能假装已看过。"""
    import cv2
    meshes=[cm.read_mesh(old_path),cm.read_mesh(new_path)]
    center=(meshes[0].get_min_bound()+meshes[0].get_max_bound())/2
    target=output/filename
    if target.exists():
        raise FileExistsError("禁止覆盖已有对比图片")
    views=[]
    # Windows 上反复销毁/重建隐藏 GL 窗口会返回空帧；整个对比只使用一个窗口。
    vis=o3d.visualization.Visualizer()
    if not vis.create_window(width=750,height=650,visible=False):
        raise RuntimeError("隐藏 Open3D 渲染窗口创建失败；请在 CloudCompare 人工检查")
    try:
        for front in ([0,0,-1],[-.7,-.1,-.7],[.7,-.1,-.7]):
            images=[]
            for mesh in meshes:
                mesh.compute_vertex_normals()
                vis.clear_geometries()
                vis.add_geometry(mesh)
                opts=vis.get_render_option();opts.mesh_show_back_face=True;opts.background_color=np.array([.95,.95,.95])
                control=vis.get_view_control();control.set_lookat(center);control.set_front(front);control.set_up([0,-1,0]);control.set_zoom(.7)
                for _ in range(3):
                    vis.poll_events();vis.update_renderer()
                image=(np.asarray(vis.capture_screen_float_buffer(do_render=True))*255).astype(np.uint8)
                if np.std(image)<1:
                    raise RuntimeError("渲染返回空白帧，不能据此声称模型质量已验证")
                images.append(image)
            views.append(np.hstack(images))
    finally:
        vis.destroy_window()
    image=np.vstack(views)
    cv2.putText(image,"OLD 500-550 / NEW (same world and crop)",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),2)
    if not cv2.imwrite(str(target),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):
        raise OSError("网格对比预览保存失败")


def run(args):
    start_time=time.perf_counter();output=args.output.resolve();dataset=args.dataset.resolve()
    if output.exists():
        raise FileExistsError(f"禁止覆盖已有手持重建结果：{output}")
    if output==dataset or dataset in output.parents or ROOT/"data" in output.parents:
        raise ValueError("输出不能位于输入 RGB-D 数据目录")
    ids=pipeline.merge.selected_frames(args.start,args.end,args.step)
    if args.reference not in ids:
        raise ValueError("reference 必须属于采样帧列表")
    if args.crop_config and args.crop_reference!=args.reference:
        raise ValueError("裁剪框参考系与轨迹参考帧不同；请转换裁剪框并明确声明 --crop-reference，或使用 --no-crop")
    if args.crop_config and args.crop_config.resolve()==ROOT/"output/mesh/chair_crop_config.json" and dataset!=ROOT/"data/extracted_scan_02":
        raise ValueError("已有裁剪框仅适用于 extracted_scan_02 的 frame500；新数据集必须提供适用的框或 --no-crop")
    output.mkdir(parents=True,exist_ok=False)
    final={"status":"failed","dataset":str(dataset),"requested_frame_ids":ids,"reference_frame":args.reference,
           "pose_source":"rgbd_odometry","error":None,"quality_warning":"Offline partial-scan furniture; no manufacturing precision or complete unobserved geometry claimed"}
    log=logging.FileHandler(output/"handheld.log",mode="x",encoding="utf-8");logging.getLogger().addHandler(log)
    try:
        cv2.setNumThreads(1)
        device, device_info = select_device(args.device)
        final["compute_device"] = device_info
        final["odometry_backend"] = args.odometry_backend
        final["icp_refinement"] = args.icp_refinement
        final["odometry_device"] = device if args.odometry_backend == "tensor" else "CPU:0"
        logging.info("Odometry=%s，设备=%s，深度支持门槛=%.0f%%；ICP=%s，CPU线程=%d",args.odometry_backend,final["odometry_device"],100*args.min_consistency,args.icp_refinement,args.cpu_threads)
        settings=odom.settings(dataset,max_translation=args.max_translation,max_rotation=args.max_rotation,
                               min_consistency=args.min_consistency,odometry_backend=args.odometry_backend,device=device,
                               icp_refinement=args.icp_refinement,
                               max_skip_frames=args.max_skip_frames,max_recovery_gap_ms=args.max_recovery_gap_ms)
        tracking_started = time.perf_counter()
        poses,tracking,metadata,timestamps,intrinsic=odom.track(dataset,ids,args.reference,output/"tracking",settings)
        final["tracking_seconds"] = time.perf_counter() - tracking_started
        final["tracking"]={k:tracking[k] for k in ("valid_frame_count","attempted_pairs","accepted_pairs","attempted_pair_success_ratio","failures","not_estimated_frames","failed_frame_ids","skipped_frame_ids","recovery_bridges")}
        valid=tracking["valid_frame_ids"]
        final["tracking_complete"] = len(valid)==len(ids)
        if not final["tracking_complete"]:
            logging.warning("跟踪完成%d/%d帧；跳过/失跟踪帧不融合。后续complete仅表示通过检查的连通轨迹完成融合。",len(valid),len(ids))
        if len(valid)<3:
            raise ValueError("验证后的相连轨迹不足三帧，停止建模")
        graph_started = time.perf_counter()
        graph=pg.build_and_optimize(poses,tracking,dataset,metadata,timestamps,intrinsic,settings,output/"pose_graph",
                                    min_loop_frame_gap=args.min_loop_gap,max_candidates=args.max_loop_candidates,reference=args.reference)
        final["pose_graph_seconds"] = time.perf_counter() - graph_started
        final["pose_graph"]={k:graph[k] for k in ("nodes","loop_candidates_tested","loop_constraints_accepted","loop_edges_retained","optimization_accepted","max_translation_change_m","max_rotation_change_deg","limitation")}
        # Pose 文件位于 pose_graph，模型输出放在 models，遵守原 Pipeline 对位姿父目录的保护。
        fused_ids = fusion_frames(valid,args.fusion_step)
        if len(fused_ids) < 3:
            raise ValueError("fusion-step 采样后不足三帧，请使用更小的融合间隔")
        final.update(odometry_frame_count=len(valid),fusion_frame_ids=fused_ids,fusion_step=args.fusion_step,
                     standalone_pointcloud_merge_enabled=not args.skip_pointcloud)
        config={"dataset_path":str(dataset),"camera_poses_path":str(output/"pose_graph/poses_for_reconstruction.json"),
                "start_frame":fused_ids[0],"end_frame":fused_ids[-1],"frame_step":args.step*args.fusion_step,
                "frame_ids":fused_ids,"min_depth":.2,"max_depth":3.,
                "pointcloud_voxel_size":args.pointcloud_voxel,"tsdf_voxel_size":.01,"sdf_trunc":.04,
                "enable_pointcloud_merge":not args.skip_pointcloud,"enable_tsdf":True,"enable_mesh_crop":bool(args.crop_config),
                "enable_mesh_clean":bool(args.crop_config),"mesh_crop_config":str(args.crop_config.resolve()) if args.crop_config else None,
                "enable_slicing":False,"output_directory":str(output/"models"),"max_translation":args.max_translation,
                "max_rotation":args.max_rotation,"max_pointcloud_voxels":4_000_000,"compute_device":device}
        write_json(output/"pipeline_config.json",config)
        pipe=pipeline.run_pipeline(output/"pipeline_config.json")
        if pipe["status"]!="complete":
            raise ValueError(f"现有 Pipeline 失败：{pipe['error']}")
        final.update(actual_reconstruction_frame_count=len(fused_ids),pipeline_report=str(Path(pipe["run_directory"])/"pipeline_report.json"),
                     scene_mesh=pipe["summary"]["tsdf"],furniture_mesh=pipe["final_mesh"],pointcloud=pipe["summary"]["pointcloud_merge"])
        if args.crop_config:
            quality=compare_models(args.baseline.resolve(),Path(pipe["final_mesh"]["path"]),output)
            final["model_comparison"]={"surface_area_ratio":quality["surface_area_ratio"],"extent_ratio":quality["extent_ratio"],
                                      "old_boundary_edges":quality["old"]["boundary_edge_count"],"new_boundary_edges":quality["new"]["boundary_edge_count"]}
            try:
                render_comparison(args.baseline.resolve(),Path(pipe["final_mesh"]["path"]),output)
                final["rendered_comparison"]=str(output/"model_compare.png")
            except Exception as exc:
                final["render_warning"]=str(exc)
        else:
            write_json(output/"model_quality_report.json",{"scene":quality_stats(pipe["final_mesh"]["path"]),"warning":"No furniture crop supplied; scene is not segmented furniture"})
        final["status"]="complete"
    except (Exception,KeyboardInterrupt) as exc:
        final["error"]=f"{type(exc).__name__}: {str(exc) or '用户中断'}"
        raise
    finally:
        final["runtime_seconds"]=time.perf_counter()-start_time
        write_json(output/"handheld_report.json",final)
        logging.getLogger().removeHandler(log);log.close()
    print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)
    return final


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description="手持 RGB-D 长序列验证、Pose Graph 和原重建 Pipeline；输出禁止覆盖")
    parser.add_argument("--dataset",type=Path,default=ROOT/"data/extracted_scan_02")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--start",type=int,default=500);parser.add_argument("--end",type=int,default=550)
    parser.add_argument("--step",type=int,default=5);parser.add_argument("--reference",type=int,default=500)
    parser.add_argument("--crop-config",type=Path,default=ROOT/"output/mesh/chair_crop_config.json")
    parser.add_argument("--crop-reference",type=int,default=500,help="所提供裁剪框的参考帧，必须与 --reference 相同")
    parser.add_argument("--no-crop",action="store_true",help="没有适用的裁剪框时只输出场景，不推测家具位置")
    parser.add_argument("--baseline",type=Path,default=ROOT/"output/mesh/chair_cropped.ply")
    parser.add_argument("--pointcloud-voxel",type=float,default=.01)
    parser.add_argument("--max-translation",type=float,default=.15);parser.add_argument("--max-rotation",type=float,default=15.)
    parser.add_argument("--min-consistency",type=float,default=.4,help="双向深度支持门槛，0～1；默认40%%")
    parser.add_argument("--odometry-backend",choices=("tensor","legacy"),default="tensor")
    parser.add_argument("--icp-refinement",choices=("off","geometry"),default="off",help="默认不以纯几何ICP覆盖Hybrid位姿；geometry仅供显式诊断")
    parser.add_argument("--device",choices=("auto","CPU:0","CUDA:0"),default="auto")
    parser.add_argument("--cpu-threads",type=int,default=4)
    parser.add_argument("--fusion-step",type=int,default=1,help="每N个已验跟踪帧融合一次；不改变Odometry采样间隔")
    parser.add_argument("--max-skip-frames",type=int,default=0,help="连续失败时最多跳过的采样帧数；重配准成功才恢复，默认0保持失败即停")
    parser.add_argument("--max-recovery-gap-ms",type=float,default=200.,help="跳帧重配准时距最后可靠帧的两路时间差上限（毫秒）")
    parser.add_argument("--skip-pointcloud",action="store_true",help="只生成Mesh，跳过独立点云融合以节省时间")
    parser.add_argument("--min-loop-gap",type=int,default=100);parser.add_argument("--max-loop-candidates",type=int,default=40)
    args=parser.parse_args(argv)
    if args.no_crop:
        args.crop_config=None
    elif not args.crop_config.is_file():
        raise ValueError("裁剪框不存在：请明确提供 --crop-config，或使用 --no-crop")
    pipeline.merge.selected_frames(args.start,args.end,args.step)
    if args.max_skip_frames < 0 or not np.isfinite(args.max_recovery_gap_ms) or args.max_recovery_gap_ms <= 0:
        raise ValueError("max-skip-frames 须非负，max-recovery-gap-ms 须为正有限数值")
    if not np.isfinite(args.min_consistency) or not 0 < args.min_consistency <= 1 or args.fusion_step <= 0 or args.cpu_threads <= 0:
        raise ValueError("深度支持比例须在(0,1]，fusion-step/cpu-threads必须为正整数")
    if not np.isfinite([args.pointcloud_voxel,args.max_translation,args.max_rotation]).all() or min(args.pointcloud_voxel,args.max_translation,args.max_rotation)<=0 or args.max_rotation>180 or args.min_loop_gap<=args.step or args.max_loop_candidates<0:
        raise ValueError("体素/运动/回环阈值非法")
    return args


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    try:
        run(parse_args())
    except (Exception,KeyboardInterrupt) as exc:
        logging.error("手持流程停止：%s",str(exc) or "用户中断");raise SystemExit(1)
