"""发布整理只复制白名单，不改写算法或带入历史数据。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('handheld_release_builder', ROOT / 'scripts/build_handheld_release.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class HandheldReleaseTests(unittest.TestCase):
    def test_only_allowlisted_files_and_byte_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'repo'; root.mkdir()
            (root / 'source.py').write_bytes(b'def pose():\r\n    return 1\r\n')
            (root / 'requirements.txt').write_bytes(b'numpy\n')
            (root / 'private.db3').write_bytes(b'not published')
            (root / 'data').mkdir(); (root / 'data/raw.png').write_bytes(b'private')
            out = Path(folder) / 'release'
            with patch.object(builder, 'ROOT', root), patch.object(builder, 'FILES', {'run.py': 'source.py'}):
                result = builder.build(out, True)
                self.assertEqual((out / 'run.py').read_bytes(), (root / 'source.py').read_bytes())
                self.assertFalse((out / 'data').exists()); self.assertFalse((out / 'private.db3').exists())
                self.assertEqual(result['files']['run.py']['sha256'], hashlib.sha256((root / 'source.py').read_bytes()).hexdigest())
                with zipfile.ZipFile(out.with_suffix('.zip')) as z: self.assertIsNone(z.testzip())
                with self.assertRaises(FileExistsError): builder.build(out)

    def test_missing_source_stops_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out = root / 'out'
            with patch.object(builder, 'ROOT', root), patch.object(builder, 'FILES', {'run.py': 'missing.py'}):
                with self.assertRaises(FileNotFoundError): builder.build(out)
            self.assertFalse(out.exists())

    def test_synthetic_demo_is_explicit_and_never_overwrites(self):
        import sys
        sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'tests'))
        from scripts.create_synthetic_demo import create
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'demo'
            result = create(out)
            self.assertEqual(result['data_source'], 'simulation')
            self.assertEqual(result['frame_ids'], [0, 1, 2])
            self.assertTrue((out / 'synthetic_dataset/depth/000002.png').is_file())
            poses = json.loads((out / 'poses/camera.json').read_text(encoding='utf-8'))
            self.assertTrue(all(p['pose_source'] == 'simulation' for p in poses['poses']))
            with self.assertRaises(FileExistsError): create(out)


if __name__ == '__main__': unittest.main()
