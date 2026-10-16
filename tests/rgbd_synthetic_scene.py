"""已知静止平面/三面角的真实射线求交渲染，仅用于测试，不替代扫描椅子。"""

from pathlib import Path
import csv
import json

import cv2
import numpy as np
import open3d as o3d

from reconstruction.pose_io import CameraPose, save_poses


METADATA = dict(width=160, height=120, fx=145., fy=145., cx=79.5, cy=59.5,
                depth_scale=.001, intrinsics_unit="pixels", depth_scale_unit="meters_per_depth_unit",
                depth_dtype="uint16", depth_aligned_to="color")


def camera_pose(translation=(0,0,0), angles_deg=(0,0,0)):
    matrix=np.eye(4)
    matrix[:3,:3]=o3d.geometry.get_rotation_matrix_from_xyz(np.radians(angles_deg))
    matrix[:3,3]=translation
    return matrix


def render(camera_to_scene, *, corner=True):
    """像素射线以 z=1 参数化；交点参数就是 camera Z，而不是射线欧氏距离。"""
    m=METADATA
    v,u=np.mgrid[:m["height"],:m["width"]]
    rays=np.stack(((u-m["cx"])/m["fx"],(v-m["cy"])/m["fy"],np.ones_like(u)),axis=-1)
    directions=rays@camera_to_scene[:3,:3].T
    origin=camera_to_scene[:3,3]
    planes=[(2,1.6,[-.45,-.8,.6],[.8,.35,1.6])]
    if corner:
        planes += [(0,-.45,[-.45,-.8,.6],[-.45,.35,1.6]),(1,.35,[-.45,.35,.6],[.8,.35,1.6])]
    else:
        planes=[(2,1.6,[-2,-2,1.6],[2,2,1.6])]
    best=np.full(u.shape,np.inf)
    for axis,position,minimum,maximum in planes:
        denom=directions[...,axis]
        distance=np.divide(position-origin[axis],denom,out=np.full_like(denom,np.inf),where=np.abs(denom)>1e-8)
        with np.errstate(invalid="ignore"):
            points=origin+directions*distance[...,None]
        inside=np.isfinite(distance)&(distance>.2)&np.all(points>=np.asarray(minimum)-1e-8,axis=-1)&np.all(points<=np.asarray(maximum)+1e-8,axis=-1)
        best=np.where(inside&(distance<best),distance,best)
    valid=np.isfinite(best)&(best<=3)
    safe=np.where(valid,best,0)
    points=origin+directions*safe[...,None]
    x,y,z=points.transpose(2,0,1)
    # 颜色是物体世界表面的函数，同一个物理点在不同相机视角下颜色相同。
    rgb=np.stack([128+55*np.sin(37*x+19*y+11*z)+35*np.sin(17*x-43*y+7*z),
                  128+65*np.sin(23*x-31*y+13*z),128+65*np.sin(29*y+41*z+17*x)],axis=-1)
    rgb=np.clip(rgb,0,255).astype(np.uint8);rgb[~valid]=0
    depth=np.rint(safe/m["depth_scale"]).astype(np.uint16)
    return rgb,depth


def write_dataset(root, *, corner=True):
    root=Path(root);dataset=root/"synthetic_dataset"
    (dataset/"color").mkdir(parents=True);(dataset/"depth").mkdir()
    (dataset/"camera_intrinsics.json").write_text(json.dumps(METADATA),encoding="utf-8")
    local=[camera_pose(),camera_pose((.02,-.01,.008),(.8,-1.2,1.5)),
           camera_pose((-.02,.012,-.005),(-.6,1.,-1.))]
    gauge=camera_pose((.3,-.1,.25),(12,-15,27))
    records=[]
    with (dataset/"timestamps.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["frame_id","color_timestamp_ms","depth_timestamp_ms"])
        for i,pose in enumerate(local):
            rgb,depth=render(pose,corner=corner)
            if not cv2.imwrite(str(dataset/"color"/f"{i:06d}.png"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)) or not cv2.imwrite(str(dataset/"depth"/f"{i:06d}.png"),depth):
                raise OSError("Synthetic RGB-D write failed")
            writer.writerow([i,1000+i*33.333,1000+i*33.333])
            records.append(CameraPose(i,1+i/30,gauge@pose,"simulation","synthetic_scene_world","synthetic clock",True))
    poses=root/"poses/camera.json";save_poses(poses,records)
    return dataset,poses,local,gauge


def surface_errors(mesh,gauge,*,corner=True):
    # 与物理渲染的已知平面比较，不与生成结果自身比较。
    local=(np.asarray(mesh.vertices)-gauge[:3,3])@gauge[:3,:3]
    errors=np.abs(local[:,2]-1.6)
    if corner:
        errors=np.minimum(errors,np.minimum(np.abs(local[:,0]+.45),np.abs(local[:,1]-.35)))
    return errors
