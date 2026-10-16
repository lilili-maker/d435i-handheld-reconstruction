"""统一位姿 JSON v1：T_world_camera、米、秒、显式时钟参考和有效状态。

validity 表示该条记录可用且矩阵通过数值检查，不代表真实位姿精度已验证。
pose_source 只是来源标签，不赋予 robot / odometry / simulation 任何准确性保证。
"""

from dataclasses import dataclass, field
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .pose_utils import validate_transform


def normalize_frame_id(frame_id):
    """接受非负整数或 PNG 的数字文件名编号，如 '000500'，统一为整数。"""
    if isinstance(frame_id, str) and frame_id.isascii() and frame_id.isdigit():
        frame_id = int(frame_id)
    if isinstance(frame_id, bool) or not isinstance(frame_id, Integral) or frame_id < 0:
        raise ValueError(f"frame_id 必须是非负整数或数字字符串：{frame_id!r}")
    return int(frame_id)


@dataclass
class CameraPose:
    frame_id: int
    timestamp: float
    T_world_camera: np.ndarray | None
    pose_source: str
    coordinate_frame: str
    timestamp_reference: str
    validity: bool
    translation_unit: str = "m"
    timestamp_unit: str = "s"
    status: str = "provided"
    reason: str = ""
    source_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.frame_id = normalize_frame_id(self.frame_id)
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, Real) or not np.isfinite(self.timestamp):
            raise ValueError(f"帧 {self.frame_id} 的 timestamp 必须是有限秒数。")
        self.timestamp = float(self.timestamp)
        if self.translation_unit != "m" or self.timestamp_unit != "s":
            raise ValueError("统一格式只接受 translation_unit='m'、timestamp_unit='s'，请显式转换单位。")
        if self.pose_source not in ("robot", "rgbd_odometry", "simulation", "object_registration"):
            raise ValueError(f"未知 pose_source：{self.pose_source!r}")
        for key in ("coordinate_frame", "timestamp_reference", "status"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"必须明确指定 {key}，不能猜测坐标系或时钟参考。")
        if type(self.validity) is not bool or not isinstance(self.reason, str):
            raise ValueError("validity 必须是布尔值，reason 必须是字符串。")
        if self.validity:
            if self.T_world_camera is None:
                raise ValueError(f"帧 {self.frame_id} 声称有效但缺少 T_world_camera。")
            self.T_world_camera = validate_transform(self.T_world_camera)
        elif self.T_world_camera is not None or not self.reason.strip():
            raise ValueError("无效位姿必须使用 T_world_camera=null 并说明原因，禁止填单位矩阵。")
        if not isinstance(self.source_metadata, dict):
            raise ValueError("source_metadata 必须为对象。")
        # 在写文件之前拒绝非 JSON 数值和 NaN，避免生成不可加载的文件。
        json.dumps(self.source_metadata, allow_nan=False)


def _check_collection(records):
    poses = {}
    for pose in records:
        if not isinstance(pose, CameraPose):
            raise ValueError("每条记录必须是 CameraPose。")
        pose.__post_init__()  # 保存/访问前再次验证，防止矩阵被调用者修改成非法值。
        if pose.frame_id in poses:
            raise ValueError(f"重复位姿 frame_id：{pose.frame_id}")
        poses[pose.frame_id] = pose
    if not poses:
        raise ValueError("位姿集合不能为空。")
    worlds = {pose.coordinate_frame for pose in poses.values()}
    if len(worlds) != 1:
        raise ValueError(f"集合混用了不同 World 坐标系：{sorted(worlds)}；需显式完成坐标转换。")
    return poses


def save_poses(path, poses, *, metadata=None):
    """保存位姿列表或 frame_id→CameraPose 字典；拒绝覆盖已有文件。"""
    if isinstance(poses, dict):
        for key, pose in poses.items():
            if not isinstance(pose, CameraPose) or normalize_frame_id(key) != pose.frame_id:
                raise ValueError(f"位姿字典键 {key!r} 与记录的 frame_id 不一致。")
    records = _check_collection(poses.values() if isinstance(poses, dict) else poses)
    payload = {
        "schema_version": 1, "pose_convention": "T_world_camera",
        "transform_direction": "camera_to_world", "translation_unit": "m", "timestamp_unit": "s",
        "validity_definition": "Usable source status and numerically valid SE(3); not a guarantee of physical accuracy",
        "metadata": {} if metadata is None else metadata,
        "poses": [
            {**vars(pose), "T_world_camera": pose.T_world_camera.tolist() if pose.T_world_camera is not None else None}
            for pose in sorted(records.values(), key=lambda p: p.frame_id)
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("统一位姿文件必须使用 .json 扩展名。")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(text)


def _unique_json_keys(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"JSON 中存在重复字段：{key}")
        data[key] = value
    return data


def load_poses(path):
    """加载成 frame_id→CameraPose；显式拒绝旧方向、错误单位或不明确记录。"""
    with Path(path).open(encoding="utf-8") as f:
        def reject_constant(value):
            raise ValueError(f"JSON 不允许非有限数值：{value}")
        data = json.load(f, object_pairs_hook=_unique_json_keys, parse_constant=reject_constant)
    if not isinstance(data, dict):
        raise ValueError("位姿 JSON 顶层必须为对象。")
    expected = {"schema_version": 1, "pose_convention": "T_world_camera",
                "transform_direction": "camera_to_world", "translation_unit": "m", "timestamp_unit": "s"}
    for key, value in expected.items():
        if data.get(key) != value or (key == "schema_version" and type(data[key]) is not int):
            raise ValueError(f"位姿格式 {key} 必须为 {value!r}，实际 {data.get(key)!r}；不自动转换。")
    if not isinstance(data.get("poses"), list):
        raise ValueError("缺少 poses 列表。")
    required = set(CameraPose.__dataclass_fields__)
    records = []
    for index, record in enumerate(data["poses"]):
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError(f"位姿记录 {index} 字段不明确：应包含 {sorted(required)}。")
        records.append(CameraPose(**record))
    return _check_collection(records)


def require_pose(poses, frame_id, *, coordinate_frame=None):
    """按图像 frame_id 获取有效位姿；缺失/无效/World 不匹配时明确报错。"""
    i = normalize_frame_id(frame_id)
    if i not in poses:
        raise KeyError(f"图像帧 {i:06d} 缺少相机位姿，禁止使用单位矩阵补齐。")
    pose = poses[i]
    pose.__post_init__()
    if pose.frame_id != i:
        raise ValueError(f"位姿字典键 {i} 与记录 frame_id={pose.frame_id} 不一致。")
    if not pose.validity:
        raise ValueError(f"帧 {i:06d} 位姿无效：{pose.reason}")
    if coordinate_frame is not None and pose.coordinate_frame != coordinate_frame:
        raise ValueError(f"帧 {i} World={pose.coordinate_frame!r}，期望 {coordinate_frame!r}。")
    return pose
