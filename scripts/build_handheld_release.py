"""按白名单整理手持环绕扫描项目；算法源码逐字节复制，不包含历史扫描。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'README.md': 'docs/handheld_release/README.md',
    '.gitignore': '.gitignore',
    '.gitattributes': 'docs/handheld_release/gitattributes',
    '.github/workflows/tests.yml': 'docs/handheld_release/tests.yml',
    'docs/LIMITATIONS.md': 'docs/handheld_release/LIMITATIONS.md',
    'docs/SCANNING_GUIDE.md': 'docs/handheld_release/SCANNING_GUIDE.md',
    'docs/VALIDATION.md': 'docs/handheld_release/VALIDATION.md',
    'CONTRIBUTING.md': 'docs/handheld_release/CONTRIBUTING.md',
    'run_handheld_reconstruction.py': 'run_handheld_reconstruction.py',
    'run_pipeline.py': 'run_pipeline.py',
    'scripts/build_handheld_release.py': 'scripts/build_handheld_release.py',
    'scripts/create_synthetic_demo.py': 'scripts/create_synthetic_demo.py',
    **{f'src/{name}.py': f'src/{name}.py' for name in (
        '01_test_camera', '02_record_data', '03_extract_rgbd',
        '04_single_frame_pointcloud', '05_rgbd_odometry')},
    **{f'src/reconstruction/{name}.py': f'src/reconstruction/{name}.py' for name in (
        '__init__', 'analyze_handheld_dataset', 'compute_device', 'pose_utils', 'pose_io',
        'convert_odometry_poses', 'handheld_odometry', 'handheld_pose_graph',
        'merge_pointclouds', 'tsdf_reconstruction', 'clean_mesh', 'mesh_to_svg')},
    **{f'tests/{name}.py': f'tests/{name}.py' for name in (
        'rgbd_synthetic_scene', 'test_handheld_reconstruction', 'test_handheld_recovery',
        'test_rgbd_direction_and_fusion', 'test_rgbd_odometry', 'test_pose_interface',
        'test_merge_pointclouds', 'test_tsdf_reconstruction', 'test_clean_mesh',
        'test_mesh_to_svg', 'test_pipeline', 'test_handheld_release')},
}


def build(output, make_zip=False):
    output = Path(output).resolve()
    archive = output.with_suffix('.zip')
    if output.exists() or (make_zip and archive.exists()):
        raise FileExistsError('发布目录或ZIP已经存在，请选择新目录')
    sources = {}
    # 独立发布版可以重建自身；原项目仍从专用发布文档读取，不改写3DSVG README。
    packaged = ROOT / 'build_manifest.json'
    standalone = packaged.is_file() and json.loads(packaged.read_text(encoding='utf-8')).get('app_name') == 'D435i Handheld Reconstruction'
    for target, source in FILES.items():
        path = ROOT / (target if standalone else source)
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[target] = path
    output.mkdir(parents=True, exist_ok=False)
    for target, source in sources.items():
        path = output / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
    # 复用实际使用的依赖锁定；不带入GPU环境或机器人SDK。
    requirements = ROOT / 'requirements.txt'
    shutil.copyfile(requirements, output / 'requirements.txt')
    manifest = {
        'app_name': 'D435i Handheld Reconstruction',
        'scope': 'Handheld RGB-D odometry, validated pose graph and TSDF furniture reconstruction',
        'accuracy': 'Experimental; drift, distortion and incomplete surfaces are possible. No measured accuracy bound.',
        'algorithm_copy_policy': 'Python files copied byte-for-byte without algorithm modifications',
        'excluded': ['data', 'output', '.venv', '.venv-gpu', 'robot', 'turntable', '3dsvg_studio', 'historical_models'],
        'files': {},
    }
    for path in sorted(output.rglob('*')):
        if path.is_file():
            manifest['files'][path.relative_to(output).as_posix()] = {
                'size_bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (output / 'build_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if make_zip:
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    z.write(path, 'd435i-handheld-reconstruction/' + path.relative_to(output).as_posix())
        with zipfile.ZipFile(archive) as z:
            if z.testzip():
                raise ValueError('ZIP校验失败')
    print(f'手持扫描发布目录：{output}\n文件数：{len(manifest["files"]) + 1}；未复制历史扫描数据或修改原算法。')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zip', action='store_true')
    args = parser.parse_args()
    try:
        build(args.output, args.zip)
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + '\n')
