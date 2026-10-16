"""全序列输入质量检查；连续图片不是可靠相机轨迹的证明。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction.merge_pointclouds import single_frame_module, read_timestamps


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def analyze_dataset(dataset, output, min_depth=.2, max_depth=3., max_skew_ms=50., max_gap_ms=100.):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"分析输出已存在：{output}")
    if output == dataset or dataset in output.parents:
        raise ValueError("分析输出不能写入输入数据集。")
    if not 0 <= min_depth < max_depth or not np.isfinite([min_depth, max_depth, max_skew_ms, max_gap_ms]).all() or min(max_skew_ms, max_gap_ms) <= 0:
        raise ValueError("深度/时间阈值非法。")
    metadata = single_frame_module().load_intrinsics(dataset / "camera_intrinsics.json")
    timestamps = read_timestamps(dataset)
    names = {}
    for kind in ("color", "depth"):
        names[kind] = {}
        for path in (dataset / kind).glob("*.png"):
            if not path.stem.isascii() or not path.stem.isdigit():
                raise ValueError(f"RGB-D PNG 的文件编号不明确：{path}")
            frame_id = int(path.stem)
            if path.name != f"{frame_id:06d}.png" or frame_id in names[kind]:
                raise ValueError(f"RGB-D 文件编号重复或非六位标准格式：{path}")
            names[kind][frame_id] = path
    ids = sorted(set(names["color"]) | set(names["depth"]))
    if not ids:
        raise ValueError("没有 RGB-D PNG。")
    rows, manifest, intervals, active = [], {}, [], []
    previous = None
    for i in ids:
        row = {"frame_id": i, "issues": []}
        if i not in names["color"] or i not in names["depth"] or i not in timestamps:
            row["issues"].append("缺少配对图片或时间戳")
        else:
            color = cv2.imread(str(names["color"][i]), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(names["depth"][i]), cv2.IMREAD_UNCHANGED)
            if color is None or depth is None or depth.dtype != np.uint16 or depth.shape != (metadata["height"], metadata["width"]) or color.shape != (*depth.shape, 3):
                row["issues"].append("图片损坏、深度类型或尺寸错误")
            else:
                valid = (depth > 0) & (depth.astype(float) * metadata["depth_scale"] >= min_depth) & (depth.astype(float) * metadata["depth_scale"] <= max_depth)
                row.update(nonzero_depth_ratio=float((depth > 0).mean()), valid_depth_ratio=float(valid.mean()),
                           depth_dtype=str(depth.dtype), grayscale_std=float(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).std() / 255),
                           laplacian_variance=float(cv2.Laplacian(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
                if valid.sum() < 100 or valid.mean() < .1:
                    row["issues"].append("有效深度不足")
            row.update(color_timestamp_ms=timestamps[i][0], depth_timestamp_ms=timestamps[i][1],
                       color_depth_skew_ms=timestamps[i][0] - timestamps[i][1])
            if abs(row["color_depth_skew_ms"]) > max_skew_ms:
                row["issues"].append("RGB-Depth 时间差过大")
            if previous is not None:
                deltas = np.array(timestamps[i]) - np.array(timestamps[previous])
                row.update(color_delta_ms=float(deltas[0]), depth_delta_ms=float(deltas[1]))
                if i != previous + 1 or np.any(deltas <= 0) or np.any(deltas > max_gap_ms):
                    row["issues"].append("帧编号/时间戳不连续、重复或有大间隔")
            previous = i
        row["input_quality_passed"] = not row["issues"]
        if row["input_quality_passed"]:
            active.append(i)
        elif active:
            intervals.append({"start": active[0], "end": active[-1], "frame_count": len(active)})
            active = []
        rows.append(row)
        for kind in ("color", "depth"):
            if i in names[kind]:
                path = names[kind][i]
                with path.open("rb") as f:
                    manifest[str(path)] = hashlib.file_digest(f, "sha256").hexdigest()
        if (i + 1) % 100 == 0:
            print(f"输入分析：{i + 1}/{len(ids)}", flush=True)
    if active:
        intervals.append({"start": active[0], "end": active[-1], "frame_count": len(active)})
    good = [r for r in rows if "valid_depth_ratio" in r]
    report = {"dataset": str(dataset), "color_count": len(names["color"]), "depth_count": len(names["depth"]),
              "matched_frame_count": len(set(names["color"]) & set(names["depth"])), "timestamp_count": len(timestamps),
              "frame_range": [min(ids), max(ids)], "color_only": sorted(set(names["color"]) - set(names["depth"])),
              "depth_only": sorted(set(names["depth"]) - set(names["color"])),
              "input_quality_passed_count": sum(r["input_quality_passed"] for r in rows),
              "problem_frames": [r for r in rows if r["issues"]], "candidate_continuous_intervals": intervals,
              "valid_depth_ratio_min_median_max": np.percentile([r["valid_depth_ratio"] for r in good], [0, 50, 100]).tolist() if good else None,
              "nonzero_depth_ratio_min_median_max": np.percentile([r["nonzero_depth_ratio"] for r in good], [0, 50, 100]).tolist() if good else None,
              "camera_intrinsics": metadata, "depth_range_m": [min_depth, max_depth], "frames": rows,
              "input_sha256": manifest, "note": "Image/time continuity and valid depth are only input checks; pose tracking and loop consistency require separate validation"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "dataset_analysis.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sample_ids = sorted(set(np.linspace(min(ids), max(ids), 12, dtype=int).tolist() + ([500] if 500 in ids else [])))
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for axis, i in zip(axes.flat, sample_ids):
        image = cv2.imread(str(dataset / "color" / f"{i:06d}.png"))
        if image is not None:
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(f"Frame {i}")
        axis.axis("off")
    for axis in list(axes.flat)[len(sample_ids):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "rgb_overview.png", dpi=100)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("matched_frame_count", "input_quality_passed_count", "valid_depth_ratio_min_median_max", "candidate_continuous_intervals")}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手持全数据集检查，不估计或假定全序列可靠位姿")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[2] / "data/extracted_scan_02")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze_dataset(args.dataset, args.output)
