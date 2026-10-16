"""Open3D PoseGraph：相邻可靠边 + 独立检查的不确定非相邻约束。"""

from functools import lru_cache
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .pose_io import CameraPose, save_poses
from .pose_utils import invert_transform, validate_transform
from . import handheld_odometry as odom


def relative(source_c2w, target_c2w):
    return validate_transform(invert_transform(target_c2w) @ source_c2w)


def delta_metrics(a, b):
    delta = invert_transform(a) @ b
    return {"translation_m": float(np.linalg.norm(delta[:3,3])),
            "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))))}


def orb_depth_matches(source, target, metadata):
    orb = cv2.ORB_create(nfeatures=2000)
    masks = [(frame["depth"] > 0).astype(np.uint8)*255 for frame in (source,target)]
    kp_a, da = orb.detectAndCompute(cv2.cvtColor(source["rgb"],cv2.COLOR_RGB2GRAY), masks[0])
    kp_b, db = orb.detectAndCompute(cv2.cvtColor(target["rgb"],cv2.COLOR_RGB2GRAY), masks[1])
    if da is None or db is None or min(len(da),len(db)) < 30:
        raise ValueError("ORB 有效纹理特征不足")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    def ratio_matches(a,b):
        return {pair[0].queryIdx:pair[0].trainIdx for pair in matcher.knnMatch(a,b,k=2) if len(pair)==2 and pair[0].distance < .7*pair[1].distance}
    forward, backward = ratio_matches(da,db), ratio_matches(db,da)
    xyz_a,xyz_b,uv_a,uv_b = [],[],[],[]
    def lift(keypoint, frame):
        u,v=keypoint.pt
        x,y=int(round(u)),int(round(v))
        if not 0 <= x < metadata["width"] or not 0 <= y < metadata["height"]:
            return None
        z=frame["depth"][y,x]*metadata["depth_scale"]
        if z <= 0:
            return None
        return [(u-metadata["cx"])*z/metadata["fx"],(v-metadata["cy"])*z/metadata["fy"],z]
    for a,b in forward.items():
        if backward.get(b) != a:
            continue
        p,q=lift(kp_a[a],source),lift(kp_b[b],target)
        if p is not None and q is not None:
            xyz_a.append(p); xyz_b.append(q); uv_a.append(kp_a[a].pt); uv_b.append(kp_b[b].pt)
    if len(xyz_a)<30:
        raise ValueError(f"双向唯一 ORB+Depth 对应不足：{len(xyz_a)}")
    return np.asarray(xyz_a),np.asarray(xyz_b),np.asarray(uv_a),np.asarray(uv_b)


def validate_loop(source, target, source_cloud, target_cloud, predicted, metadata, args):
    """不用链式位姿作 RANSAC 初值；独立求变换后检查空间分布、双向几何和循环。"""
    report={"source":source["id"],"target":target["id"],"accepted":False,"uncertain":True,"reason":None}
    try:
        p,q,ua,ub=orb_depth_matches(source,target,metadata)
        clouds=[]
        for points in (p,q):
            cloud=o3d.geometry.PointCloud(); cloud.points=o3d.utility.Vector3dVector(points); clouds.append(cloud)
        reg=o3d.pipelines.registration
        o3d.utility.random.seed(42)
        ransac=reg.registration_ransac_based_on_correspondence(clouds[0],clouds[1],o3d.utility.Vector2iVector(np.column_stack((np.arange(len(p)),np.arange(len(p))))),
                    .025,reg.TransformationEstimationPointToPoint(False),3,
                    [reg.CorrespondenceCheckerBasedOnEdgeLength(.9),reg.CorrespondenceCheckerBasedOnDistance(.025)],reg.RANSACConvergenceCriteria(5000,.999))
        transform=validate_transform(ransac.transformation)
        errors=np.linalg.norm(p@transform[:3,:3].T+transform[:3,3]-q,axis=1)
        inliers=errors<=.025
        coverage=[float(np.prod(np.ptp(uv[inliers],axis=0)/(metadata["width"],metadata["height"]))) if inliers.any() else 0. for uv in (ua,ub)]
        report.update(feature_match_count=len(p),feature_inlier_count=int(inliers.sum()),feature_inlier_ratio=float(inliers.mean()),
                      feature_rmse_m=float(np.sqrt(np.mean(errors[inliers]**2))) if inliers.any() else None,feature_image_bbox_coverage=coverage)
        if inliers.sum()<25 or inliers.mean()<.5 or min(coverage)<.15:
            raise ValueError("3D 特征内点数量/比例或空间覆盖不足，拒绝局部重复纹理回环")
        disagreement=delta_metrics(predicted,transform); report["chain_disagreement"]=disagreement
        if disagreement["translation_m"]>.12 or disagreement["rotation_deg"]>8:
            raise ValueError("独立回环与已验相邻链差异过大；不强制拉回轨迹")
        source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.09,max_nn=30))
        target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.09,max_nn=30))
        forward=reg.registration_icp(source_cloud,target_cloud,.03,transform,reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(max_iteration=40))
        reverse=reg.registration_icp(target_cloud,source_cloud,.03,invert_transform(transform),reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(max_iteration=40))
        transform=validate_transform(forward.transformation)
        refined_errors=np.linalg.norm(p@transform[:3,:3].T+transform[:3,3]-q,axis=1)
        refined_inliers=refined_errors<=.025
        report["refined_feature_inlier_count"]=int(refined_inliers.sum())
        report["refined_feature_inlier_ratio"]=float(refined_inliers.mean())
        if refined_inliers.sum()<25 or refined_inliers.mean()<.5:
            raise ValueError("ICP 精化后不再支持独立的 3D 特征对应，拒绝约束")
        cycle=odom.cycle_check(transform,reverse.transformation,.015,2.)
        geom=odom.geometry_check(source_cloud,target_cloud,transform)
        report.update(cycle=cycle,geometry=geom,photometric=odom.photometric_check(source,target,transform,metadata))
        after_delta=delta_metrics(predicted,transform)
        if not cycle["passed"] or min(geom["source_to_target_fitness"],geom["target_to_source_fitness"])<.7 or max(geom["source_to_target_rmse_m"],geom["target_to_source_rmse_m"])>.015:
            raise ValueError("回环双向 ICP 循环/重叠/距离检查失败")
        if after_delta["translation_m"]>.12 or after_delta["rotation_deg"]>8 or report["photometric"]["mae"] is None or report["photometric"]["mae"]>.10:
            raise ValueError("回环精化后偏离链式位姿或灰度验证失败")
        report.update(accepted=True,transformation=transform.tolist(),information_matrix=odom.information(source_cloud,target_cloud,transform).tolist())
    except (ValueError,RuntimeError) as exc:
        report["reason"]=str(exc)
    return report


def edge_consistency(graph):
    rows=[]
    for edge in graph.edges:
        predicted=relative(graph.nodes[edge.source_node_id].pose,graph.nodes[edge.target_node_id].pose)
        rows.append({"source_node":edge.source_node_id,"target_node":edge.target_node_id,"uncertain":edge.uncertain,
                     **delta_metrics(edge.transformation,predicted)})
    return rows


def optimize_graph(graph, reference_node):
    before=[validate_transform(node.pose) for node in graph.nodes]
    before_consistency=edge_consistency(graph)
    o3d.pipelines.registration.global_optimization(graph,o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=.03,edge_prune_threshold=.25,
                                                            preference_loop_closure=1.,reference_node=reference_node))
    after=[validate_transform(node.pose) for node in graph.nodes]
    # 固定参考应保持原坐标；仅修正求解器的全局规范自由度，不能把参考框悄悄移动。
    gauge=before[reference_node]@invert_transform(after[reference_node])
    after=[validate_transform(gauge@matrix) for matrix in after]
    for node,matrix in zip(graph.nodes,after):
        node.pose=matrix
    if not np.allclose(after[reference_node],before[reference_node],atol=1e-8,rtol=0):
        raise ValueError("优化后的参考坐标发生变化")
    changes=[delta_metrics(a,b) for a,b in zip(before,after)]
    quality={"pose_changes":changes,"edge_consistency_before":before_consistency,"edge_consistency_after":edge_consistency(graph),
             "max_translation_change_m":max(r["translation_m"] for r in changes),
             "max_rotation_change_deg":max(r["rotation_deg"] for r in changes),
             "loop_edges_retained":sum(e.uncertain for e in graph.edges),
             "note":"Edge SE(3) consistency and graph convergence are not independent true pose accuracy"}
    return after,quality


def loop_candidates(valid, min_loop_frame_gap):
    """利用刚体变换不改变距离与旋转迹，向量化原来的距离/角度筛选。"""
    ids=np.array([p.frame_id for p in valid])
    positions=np.array([p.T_world_camera[:3,3] for p in valid])
    rotations=np.array([p.T_world_camera[:3,:3] for p in valid])
    candidates=[]
    for i in range(len(valid)):
        js=np.flatnonzero(ids-ids[i]>=min_loop_frame_gap)
        distances=np.linalg.norm(positions[js]-positions[i],axis=1)
        traces=np.einsum("nij,ij->n",rotations[js],rotations[i])
        angles=np.degrees(np.arccos(np.clip((traces-1)/2,-1,1)))
        mask=(distances<.4)&(angles<40)
        candidates.extend((float(d),i,int(j)) for d,j in zip(distances[mask],js[mask]))
    return candidates


def build_and_optimize(poses, tracking, dataset, metadata, timestamps, intrinsic, args, output,
                       min_loop_frame_gap=100, max_candidates=40, reference=500):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    valid=sorted([p for p in poses if p.validity],key=lambda p:p.frame_id)
    if len(valid)<2:
        raise ValueError("有效轨迹不足两个节点")
    indices={p.frame_id:i for i,p in enumerate(valid)}
    if reference not in indices:
        raise ValueError("优化缺少有效参考帧")
    reg=o3d.pipelines.registration
    graph=reg.PoseGraph()
    for p in valid:
        graph.nodes.append(reg.PoseGraphNode(p.T_world_camera))
    for pair in tracking["pairs"]:
        if pair["accepted"]:
            graph.edges.append(reg.PoseGraphEdge(indices[pair["source"]],indices[pair["target"]],np.asarray(pair["transformation"]),
                                                np.asarray(pair["graph_information_matrix"]),uncertain=False))
    candidates=loop_candidates(valid,min_loop_frame_gap)
    candidates.sort()
    # 避免所有候选只聚集在同一帧：每节点最多三个候选，总数有界。
    chosen=[]; counts={}
    for _,i,j in candidates:
        if max_candidates==0:
            break
        if counts.get(i,0)>=3 or counts.get(j,0)>=3:
            continue
        chosen.append((i,j)); counts[i]=counts.get(i,0)+1; counts[j]=counts.get(j,0)+1
        if len(chosen)>=max_candidates:
            break
    @lru_cache(maxsize=6)
    def cached(i):
        f=odom.read_frame(dataset,valid[i].frame_id,metadata,timestamps,args)
        return f,odom.camera_cloud(f,intrinsic,metadata,args)
    loops=[]
    for number,(i,j) in enumerate(chosen,1):
        sf,sc=cached(i);tf,tc=cached(j)
        loop=validate_loop(sf,tf,sc,tc,relative(valid[i].T_world_camera,valid[j].T_world_camera),metadata,args)
        # 小时间间隔的重复观测不是绕行回环；较大间隔也仅是候选，不能证明真实闭环。
        loop["constraint_type"]="short_temporal_overlap" if valid[j].frame_id-valid[i].frame_id<50 else "nonadjacent_loop_candidate"
        loops.append(loop)
        if loop["accepted"]:
            graph.edges.append(reg.PoseGraphEdge(i,j,np.asarray(loop["transformation"]),np.asarray(loop["information_matrix"]),uncertain=True))
        print(f"非相邻约束候选 {number}/{len(chosen)} {valid[i].frame_id}<->{valid[j].frame_id}：{'接受' if loop['accepted'] else loop['reason']}",flush=True)
    o3d.io.write_pose_graph(str(output/"pose_graph_before.json"),graph)
    matrices,metrics=optimize_graph(graph,indices[reference])
    after_consistency=metrics["edge_consistency_after"]
    accepted=metrics["max_translation_change_m"]<=.15 and metrics["max_rotation_change_deg"]<=10
    if any(r["translation_m"]>.03 or r["rotation_deg"]>3 for r in after_consistency if not r["uncertain"]):
        accepted=False
    result_map={p.frame_id:m for p,m in zip(valid,matrices)}
    optimized=[CameraPose(**(vars(p)|{"T_world_camera":result_map.get(p.frame_id),
                "status":"pose_graph_optimized" if p.validity else p.status,
                "source_metadata":p.source_metadata|{"pose_graph_proposal":True,"optimization_accepted":accepted}})) for p in poses]
    save_poses(output/"poses_optimized_proposal.json",optimized,metadata={"reference_frame":reference,"optimization_accepted":accepted,"pose_source":"rgbd_odometry"})
    # 明确回退记录：只有已经验过的原轨迹可选用，不能用失败位姿或静默更改状态。
    selected=optimized if accepted else poses
    save_poses(output/"poses_for_reconstruction.json",selected,metadata={"reference_frame":reference,"optimization_accepted":accepted,
               "selection":"optimized" if accepted else "validated_original_graph_rejected","pose_source":"rgbd_odometry"})
    o3d.io.write_pose_graph(str(output/"pose_graph_after.json"),graph)
    report={"nodes":len(valid),"odometry_edges":sum(p["accepted"] for p in tracking["pairs"]),
            "candidate_pair_count":len(candidates),"loop_candidates_tested":len(loops),"loop_constraints_accepted":sum(l["accepted"] for l in loops),
            "accepted_short_temporal_constraints":sum(l["accepted"] and l["constraint_type"]=="short_temporal_overlap" for l in loops),
            "accepted_nonadjacent_loop_candidates":sum(l["accepted"] and l["constraint_type"]=="nonadjacent_loop_candidate" for l in loops),
            "loops":loops,"optimization_accepted":accepted,"reference_frame":reference,**metrics,
            "limitation":"No retained nonadjacent constraint: chain-only optimization cannot remove accumulated drift" if not metrics["loop_edges_retained"] else
                         "Verified nonadjacent overlap constraints are not proof of a true physical loop or calibrated pose accuracy"}
    odom.legacy().write_json(output/"pose_graph_report.json",report)
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(10,6));ax=fig.add_subplot(111,projection="3d")
    for label,plist in (("before",[p.T_world_camera for p in valid]),("proposal_after",matrices)):
        xyz=np.array([m[:3,3] for m in plist]);ax.plot(*xyz.T,label=label)
    ax.set(xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)");ax.legend();fig.tight_layout();fig.savefig(output/"trajectory_comparison.png",dpi=130);plt.close(fig)
    return report
