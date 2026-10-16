"""SE(3) 纯数学工具。标准相机位姿是 T_world_camera（camera-to-world）。

列向量约定：p_A = T_A_B @ p_B。平移和输入/输出三维点统一使用米。
矩阵本身无法辨别单位或坐标系名称，调用者必须保证组合的中间坐标系相同。
"""

import numpy as np


def _numeric_array(value, label):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{label} 必须包含实数，不能包含字符串、布尔值或复数。")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} 含 NaN / Inf。")
    return array


def validate_rotation_matrix(rotation, *, atol=1e-6):
    """验证 R 属于 SO(3)，返回副本；不投影、不悄悄修复旋转矩阵。"""
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol 必须是有限正数。")
    r = _numeric_array(rotation, "旋转矩阵")
    if r.shape != (3, 3):
        raise ValueError(f"旋转矩阵必须是 3x3，实际 {r.shape}。")
    error = float(np.linalg.norm(r.T @ r - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(r))
    if error > atol or abs(determinant - 1.0) > atol:
        raise ValueError(f"非法旋转：正交误差={error:.9g}，det(R)={determinant:.9g}，容差={atol:g}。")
    return r


def validate_transform(transform, *, atol=1e-6):
    """验证 4x4 齐次刚体变换；平移数值必须已转换成米。"""
    t = _numeric_array(transform, "齐次变换")
    if t.shape != (4, 4):
        raise ValueError(f"齐次变换必须是 4x4，实际 {t.shape}。")
    validate_rotation_matrix(t[:3, :3], atol=atol)
    if not np.allclose(t[3], [0, 0, 0, 1], atol=atol, rtol=0):
        raise ValueError(f"齐次变换最后一行必须为 [0,0,0,1]，实际 {t[3].tolist()}。")
    return t


def invert_transform(T_A_B):
    """返回 T_B_A；不把 camera-to-world 与 world-to-camera 当成同一矩阵。"""
    t = validate_transform(T_A_B)
    inverse = np.eye(4)
    inverse[:3, :3] = t[:3, :3].T
    inverse[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return validate_transform(inverse)


def compose_transforms(*transforms):
    """T_A_B @ T_B_C @ T_C_D → T_A_D，严格按参数从左到右组合。

    调用者须确认每个中间坐标系相同；至少传入一个变换。
    """
    if not transforms:
        raise ValueError("至少提供一个已知变换，不能用空组合补齐未知位姿。")
    result = validate_transform(transforms[0])
    for transform in transforms[1:]:
        result = result @ validate_transform(transform)
    return validate_transform(result)


def transform_points(T_A_B, points_B):
    """把 B 系中的 (3,) 或 (N,3) 米制点转换到 A 系；不修改输入。"""
    t = validate_transform(T_A_B)
    points = _numeric_array(points_B, "三维点")
    if points.shape != (3,) and not (points.ndim == 2 and points.shape[1] == 3):
        raise ValueError(f"三维点必须是 (3,) 或 (N,3)，实际 {points.shape}。")
    result = points @ t[:3, :3].T + t[:3, 3]
    if not np.isfinite(result).all():
        raise ValueError("转换后的三维点溢出或包含非有限值。")
    return result


def compute_base_camera(T_base_end, T_end_camera, *, end_frame, calibration_end_frame):
    """纯数学手眼组合：T_base_camera = T_base_end @ T_end_camera。

    必须明确机器人返回的 end 与标定使用的 end 是同一坐标系。
    若机器人返回 TCP、标定使用法兰，应先依据真实工具定义使用
    T_base_flange = T_base_tcp @ T_tcp_flange，不能把 TCP 直接当成法兰。
    本函数不采集机器人位姿，不进行标定，也不验证标定的物理准确性。
    """
    if not isinstance(end_frame, str) or not end_frame.strip():
        raise ValueError("必须明确机器人位姿对应的末端坐标系名称。")
    if not isinstance(calibration_end_frame, str) or not calibration_end_frame.strip():
        raise ValueError("必须明确手眼变换使用的末端坐标系名称。")
    if end_frame != calibration_end_frame:
        raise ValueError(f"末端坐标系不匹配：机器人={end_frame!r}，标定={calibration_end_frame!r}；"
                         "请依据真实工具定义先转换，禁止等同 TCP 与法兰。")
    return compose_transforms(T_base_end, T_end_camera)
