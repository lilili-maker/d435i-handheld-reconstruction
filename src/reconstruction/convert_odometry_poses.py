"""转换阶段 4A 的真实 CSV 字段到统一位姿 JSON；不估计新位姿、不修改输入。"""

import argparse
import csv
import hashlib
import logging
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reconstruction.pose_io import CameraPose, normalize_frame_id, save_poses, load_poses
    from reconstruction.pose_utils import validate_transform
else:
    from .pose_io import CameraPose, normalize_frame_id, save_poses, load_poses
    from .pose_utils import validate_transform


ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("convert_odometry_poses")
MATRIX_FIELDS = [f"c2w_{i}{j}" for i in range(4) for j in range(4)]
CSV_FIELDS = ["frame_id", "color_timestamp_ms", "depth_timestamp_ms", "status", "reason",
              "pose_convention", "world_frame", "x_m", "y_m", "z_m"] + MATRIX_FIELDS


def _finite_number(value, field, line):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV 第 {line} 行 {field} 不是有效数值：{value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"CSV 第 {line} 行 {field} 含 NaN / Inf：{value!r}")
    return number


def read_odometry_csv(input_path):
    """只接受已核对的阶段 4A 格式；不根据文件名或 pose_source 猜方向。"""
    with Path(input_path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(fields) != set(CSV_FIELDS):
            missing = sorted(set(CSV_FIELDS) - set(fields))
            unknown = sorted(set(fields) - set(CSV_FIELDS))
            raise ValueError(f"CSV 字段与阶段 4A 格式不符：缺少={missing}，未知={unknown}，"
                             f"重复字段={len(fields) != len(set(fields))}；无法确定约定，不转换。")
        rows = list(reader)
    if not rows:
        raise ValueError("旧轨迹没有数据行。")

    poses, previous_id, world_id, segment_stopped = [], None, None, False
    for line, row in enumerate(rows, start=2):
        if set(row) != set(CSV_FIELDS) or any(value is None for value in row.values()):
            raise ValueError(f"CSV 第 {line} 行字段数量不完整或存在额外值。")
        try:
            i = normalize_frame_id(row["frame_id"])
            row_world = normalize_frame_id(row["world_frame"])
        except ValueError as exc:
            raise ValueError(f"CSV 第 {line} 行编号不明确：{exc}") from exc
        if previous_id is not None and i <= previous_id:
            raise ValueError(f"CSV 第 {line} 行 frame_id={i} 重复或不按顺序递增，上一帧={previous_id}。")
        previous_id = i
        if row["pose_convention"] != "camera_to_world":
            raise ValueError(f"CSV 第 {line} 行 pose_convention={row['pose_convention']!r}；"
                             "必须明确为 camera_to_world，不猜测或自动求逆。")
        if world_id is None:
            world_id = i
        if row_world != world_id:
            raise ValueError(f"CSV 第 {line} 行 world_frame={row_world} 与首帧 {world_id} 不一致。")
        color_ms = _finite_number(row["color_timestamp_ms"], "color_timestamp_ms", line)
        depth_ms = _finite_number(row["depth_timestamp_ms"], "depth_timestamp_ms", line)
        status, reason = row["status"], row["reason"]
        if status not in ("reference", "estimated", "failed", "not_estimated"):
            raise ValueError(f"CSV 第 {line} 行未知 status={status!r}，不推断有效性。")
        valid = status in ("reference", "estimated")
        if segment_stopped and valid:
            raise ValueError(f"CSV 第 {line} 行在轨迹失败/停止后出现有效位姿，与阶段 4A 的停止约定矛盾。")
        matrix = None
        if valid:
            values = [_finite_number(row[key], key, line) for key in MATRIX_FIELDS]
            try:
                matrix = validate_transform(np.array(values).reshape(4, 4))
            except ValueError as exc:
                raise ValueError(f"CSV 第 {line} 行帧 {i} 非法矩阵：{exc}") from exc
            xyz = [_finite_number(row[key], key, line) for key in ("x_m", "y_m", "z_m")]
            if not np.allclose(xyz, matrix[:3, 3], atol=1e-9, rtol=0):
                raise ValueError(f"CSV 第 {line} 行 x_m/y_m/z_m={xyz} 与 c2w 平移 {matrix[:3, 3].tolist()} 不符。")
            if i == world_id:
                if status != "reference" or not np.allclose(matrix, np.eye(4), atol=1e-6, rtol=0):
                    raise ValueError("首帧必须明确为 reference 且矩阵为单位矩阵，否则无法确认临时 World。")
            elif status == "reference":
                raise ValueError(f"CSV 第 {line} 行出现第二个 reference，可能包含多个轨迹片段，不自动拼接。")
        else:
            segment_stopped = True
            if i == world_id:
                raise ValueError("首帧位姿无效，无法定义临时 World 坐标系。")
            if any(row[key] != "" for key in MATRIX_FIELDS + ["x_m", "y_m", "z_m"]):
                raise ValueError(f"CSV 第 {line} 行无效帧仍含矩阵或平移，不能悄悄丢弃这些值。")
            reason = reason or f"旧轨迹明确记录 status={status}，没有可用位姿"
            LOG.warning("保留无效帧 %d：%s；矩阵为 null，不补齐。", i, reason)
        # 已确认 c2w 就是 T_world_camera；不求逆、不重新累计、不重置坐标。
        poses.append(CameraPose(
            frame_id=i, timestamp=color_ms / 1000.0, T_world_camera=matrix,
            pose_source="rgbd_odometry", coordinate_frame=f"odometry_camera_{world_id}",
            timestamp_reference="RealSense color frame timestamp; clock domain unspecified, not asserted as Unix time",
            validity=valid, status=status, reason=reason,
            source_metadata={"color_timestamp_ms": color_ms, "depth_timestamp_ms": depth_ms,
                             "legacy_status": status, "legacy_reason": row["reason"], "csv_line": line},
        ))
    return poses, world_id


def check_dataset(dataset, poses):
    """可选检查同编号两路图片，以及原数据集时间戳；不重采集、不读取机器人。"""
    dataset = Path(dataset)
    timestamps_path = dataset / "timestamps.csv"
    if not timestamps_path.is_file():
        raise FileNotFoundError(f"数据集缺少时间戳文件：{timestamps_path}")
    timestamps = {}
    with timestamps_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame_id", "color_timestamp_ms", "depth_timestamp_ms"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("数据集 timestamps.csv 缺少所需字段。")
        for line, row in enumerate(reader, start=2):
            i = normalize_frame_id(row["frame_id"])
            if i in timestamps:
                raise ValueError(f"数据集 CSV 重复帧编号：{i}")
            timestamps[i] = [_finite_number(row[key], key, line) for key in ("color_timestamp_ms", "depth_timestamp_ms")]
    for pose in poses:
        for kind in ("color", "depth"):
            path = dataset / kind / f"{pose.frame_id:06d}.png"
            if not path.is_file():
                raise FileNotFoundError(f"位姿对应的图像不存在：{path}")
        original = pose.source_metadata
        expected = [original["color_timestamp_ms"], original["depth_timestamp_ms"]]
        if pose.frame_id not in timestamps or timestamps[pose.frame_id] != expected:
            raise ValueError(f"帧 {pose.frame_id} 的轨迹时间戳与数据集不一致，可能使用了错误扫描批次。")


def convert_odometry_poses(input_path, output_path, *, dataset=None):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"统一位姿输出已存在，禁止覆盖：{output_path}")
    original_bytes = input_path.read_bytes()
    poses, world_id = read_odometry_csv(input_path)
    if dataset is not None:
        check_dataset(dataset, poses)
    save_poses(output_path, poses, metadata={
        "source_file": str(input_path), "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "legacy_world_frame": world_id,
        "world_definition": f"Temporary World is the original camera coordinate frame of frame {world_id}",
        "conversion": "c2w_ij copied to T_world_camera; timestamps converted ms to s; no inversion or re-estimation",
        "legacy_translation_unit_evidence": "x_m/y_m/z_m and stage 4A save_trajectory use meters",
        "dataset": str(Path(dataset).resolve()) if dataset is not None else None,
    })
    loaded = load_poses(output_path)
    for pose in poses:
        converted = loaded[pose.frame_id]
        if pose.validity:
            if not np.array_equal(pose.T_world_camera, converted.T_world_camera):
                raise ValueError(f"帧 {pose.frame_id} 转换后的矩阵与旧 CSV 不一致。")
        elif converted.T_world_camera is not None:
            raise ValueError(f"帧 {pose.frame_id} 的无效位姿被错误补齐。")
        if converted.timestamp != pose.timestamp:
            raise ValueError(f"帧 {pose.frame_id} 转换后的时间戳发生变化。")
    if input_path.read_bytes() != original_bytes:
        raise ValueError("转换期间旧 CSV 内容发生变化，无法保证转换一致性。")
    LOG.info("转换完成：%d 帧，有效 %d，无效 %d；临时 World=odometry_camera_%d。",
             len(poses), sum(p.validity for p in poses), sum(not p.validity for p in poses), world_id)
    LOG.info("矩阵逐元素一致，时间戳明确从 ms 转为 s；旧 CSV 内容未变。")
    LOG.info("保存位置：%s；pose_source=rgbd_odometry 不代表真实位姿准确。", output_path)
    return loaded


def main(argv=None):
    parser = argparse.ArgumentParser(description="将已确认 camera-to-world 的阶段 4A CSV 转为统一相机位姿 JSON")
    parser.add_argument("--input", type=Path, default=ROOT / "output/odometry/trajectory_500_550.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output/poses/trajectory_500_550.json")
    parser.add_argument("--dataset", type=Path, help="可选：检查位姿与 RGB-D 文件编号及时间戳对应")
    args = parser.parse_args(argv)
    try:
        convert_odometry_poses(args.input, args.output, dataset=args.dataset)
        return 0
    except Exception as exc:
        LOG.error("转换停止（%s）：%s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
