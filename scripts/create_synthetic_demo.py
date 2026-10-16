"""生成已有测试渲染器的三帧三面角RGB-D；不是实际椅子或真实扫描精度示例。"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))


def create(output):
    from rgbd_synthetic_scene import write_dataset
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('模拟输入目录已存在，请换新目录')
    output.mkdir(parents=True, exist_ok=False)
    dataset, poses, _, _ = write_dataset(output)
    description = {
        'data_source': 'simulation', 'geometry': 'three textured planes; not furniture',
        'frame_ids': [0, 1, 2], 'dataset': str(dataset), 'known_camera_poses': str(poses),
        'warning': 'Synthetic correctness tests do not demonstrate real handheld scan accuracy.',
    }
    (output / 'synthetic_source.json').write_text(json.dumps(description, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(description, ensure_ascii=False, indent=2))
    return description


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        create(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')
