"""跳过异常帧只允许经过重配准的连通轨迹；验证非连续帧真正进入原融合算法。"""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from reconstruction import handheld_odometry as h, handheld_pose_graph as pg
from reconstruction.pose_io import load_poses
from reconstruction import merge_pointclouds as merge, tsdf_reconstruction as tsdf
from test_tsdf_reconstruction import inputs
import run_handheld_reconstruction as runner
import run_pipeline as pipeline


def translation(x):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix


def dataset_frames(root, count=6):
    dataset, poses, metadata, _ = inputs(root)
    for i in range(3, count + 1):
        for kind in ('color', 'depth'):
            (dataset / kind / f'{i:06d}.png').write_bytes((dataset / kind / '000001.png').read_bytes())
    (dataset / 'timestamps.csv').write_text(
        'frame_id,color_timestamp_ms,depth_timestamp_ms\n' +
        ''.join(f'{i},{i*33.3},{i*33.3}\n' for i in range(1, count + 1)), encoding='utf-8')
    return dataset, poses, metadata


def accepted_pair(source, target, *unused):
    return {'accepted': True, 'failure_reasons': [],
            'transformation': translation((source['id'] - target['id']) * .01).tolist(),
            'graph_information_matrix': (np.eye(6) * 100).tolist()}


class RecoveryTests(unittest.TestCase):
    def track_mock(self, root, ids, reference, rejected=(), **overrides):
        dataset, _, _ = dataset_frames(root)
        calls = []

        def pair(s, t, *args):
            calls.append((s['id'], t['id']))
            if t['id'] in rejected:
                return {'accepted': False, 'failure_reasons': ['synthetic rejection']}
            return accepted_pair(s, t)

        args = h.settings(dataset, **overrides)
        with patch.object(h, 'read_frame', side_effect=lambda dataset, i, *a: {'id': i}), \
                patch.object(h, 'checked_pair', side_effect=pair):
            result = h.track(dataset, ids, reference, root / 'track', args)
        return result, calls, args

    def test_recovery_uses_last_valid_anchor_and_correct_direction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (poses, report, *_), calls, _ = self.track_mock(root, [1, 2, 3, 4, 5], 1, [2], max_skip_frames=1)
            self.assertEqual(calls, [(1, 2), (1, 3), (3, 4), (4, 5)])
            self.assertEqual(report['valid_frame_ids'], [1, 3, 4, 5])
            self.assertEqual(report['skipped_frame_ids'], [2])
            saved = load_poses(root / 'track/poses_before.json')
            self.assertIsNone(saved[2].T_world_camera)
            self.assertFalse(saved[2].validity)
            for p in poses:
                if p.validity:
                    np.testing.assert_allclose(p.T_world_camera, translation((p.frame_id - 1) * .01))

    def test_failure_limit_stops_without_inventing_poses(self):
        with tempfile.TemporaryDirectory() as folder:
            (poses, report, *_), calls, _ = self.track_mock(Path(folder), [1, 2, 3, 4, 5], 1, [2, 3, 4], max_skip_frames=2)
            self.assertEqual(calls, [(1, 2), (1, 3), (1, 4)])
            self.assertEqual(report['valid_frame_ids'], [1])
            self.assertEqual(report['not_estimated_frames'], [5])
            self.assertEqual(report['skipped_frame_ids'], [])  # 没有成功桥接，不冒称恢复。
            self.assertTrue(all(p.T_world_camera is None for p in poses[1:]))

    def test_budget_resets_only_after_success(self):
        with tempfile.TemporaryDirectory() as folder:
            (_, report, *_), calls, _ = self.track_mock(Path(folder), [1, 2, 3, 4, 5], 1, [2, 4], max_skip_frames=1)
            self.assertEqual(calls, [(1, 2), (1, 3), (3, 4), (3, 5)])
            self.assertEqual(report['valid_frame_ids'], [1, 3, 5])
            self.assertEqual(report['skipped_frame_ids'], [2, 4])

    def test_backward_recovery_keeps_common_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            (poses, report, *_), calls, _ = self.track_mock(Path(folder), [1, 2, 3, 4, 5], 5, [4], max_skip_frames=1)
            self.assertEqual(calls, [(5, 4), (5, 3), (3, 2), (2, 1)])
            for p in poses:
                if p.validity:
                    np.testing.assert_allclose(p.T_world_camera, translation((p.frame_id - 5) * .01))
            self.assertEqual(report['recovery_bridges'][0]['direction'], 'backward')

    def test_time_limit_does_not_attempt_distant_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            (_, report, *_), calls, _ = self.track_mock(Path(folder), [1, 2, 3, 4], 1, [2], max_skip_frames=3, max_recovery_gap_ms=60.)
            self.assertEqual(calls, [(1, 2)])
            self.assertIn('60ms', report['failures'][-1]['reason'])
            self.assertEqual(report['not_estimated_frames'], [4])

    def test_bad_input_frame_can_be_skipped_but_not_assigned_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); dataset, _, _ = dataset_frames(root)

            def read(dataset, i, *args):
                if i == 2:
                    raise ValueError('invalid depth')
                return {'id': i}

            with patch.object(h, 'read_frame', side_effect=read), patch.object(h, 'checked_pair', side_effect=accepted_pair):
                _, report, *_ = h.track(dataset, [1, 2, 3], 1, root / 'track', h.settings(dataset, max_skip_frames=1))
            self.assertEqual(report['valid_frame_ids'], [1, 3])
            self.assertEqual(report['skipped_frame_ids'], [2])

    def test_interrupt_and_unexpected_error_are_not_skipped(self):
        for error in (KeyboardInterrupt(), KeyError('programming error')):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); dataset, _, _ = dataset_frames(root)
                with patch.object(h, 'read_frame', side_effect=lambda dataset, i, *a: {'id': i}), \
                        patch.object(h, 'checked_pair', side_effect=error) as check:
                    if isinstance(error, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            h.track(dataset, [1, 2, 3], 1, root / 'track', h.settings(dataset, max_skip_frames=3))
                    else:
                        h.track(dataset, [1, 2, 3], 1, root / 'track', h.settings(dataset, max_skip_frames=3))
                self.assertEqual(check.call_count, 1)
                self.assertIsNone(load_poses(root / 'track/poses_before.json')[2].T_world_camera)

    def test_recovered_edge_is_real_graph_connection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, _, args = self.track_mock(root, [1, 2, 3, 4], 1, [2], max_skip_frames=1)
            poses, tracking, metadata, timestamps, intrinsic = result
            graph = pg.build_and_optimize(poses, tracking, args.dataset, metadata, timestamps, intrinsic,
                                          args, root / 'graph', max_candidates=0, reference=1)
            self.assertEqual(graph['nodes'], 3)
            self.assertEqual(graph['odometry_edges'], 2)
            saved = load_poses(root / 'graph/poses_for_reconstruction.json')
            self.assertFalse(saved[2].validity)
            np.testing.assert_allclose(saved[4].T_world_camera, translation(.03), atol=1e-8)

    def test_cli_limits_and_default_unchanged(self):
        self.assertEqual(runner.parse_args(['--output', 'unused', '--no-crop']).max_skip_frames, 0)
        for option, value in [('--max-skip-frames', '-1'), ('--max-recovery-gap-ms', '0'), ('--max-recovery-gap-ms', 'nan')]:
            with self.assertRaises(ValueError):
                runner.parse_args(['--output', 'unused', '--no-crop', option, value])

    def test_explicit_frame_selection_validation(self):
        self.assertEqual(merge.selected_frames(1, 5, 3, [1, 3, 5]), [1, 3, 5])
        for bad in ([], [1, 1], [3, 1], [0, 3], [1, 6], [True, 3], [1., 3], '1,3'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                merge.selected_frames(1, 5, 1, bad)
        for module in (merge, tsdf):
            args = module.parse_args(['--start', '1', '--end', '5', '--step', '3', '--frames', '1', '3', '5'])
            self.assertEqual(args.frames, [1, 3, 5])

    def test_pipeline_frame_list_affects_resume_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); dataset, poses, _ = dataset_frames(root)
            cfg_path = root / 'cfg.json'
            cfg_path.write_text(json.dumps(dict(dataset_path=str(dataset), camera_poses_path=str(poses),
                start_frame=1, end_frame=3, frame_ids=[1, 3], output_directory=str(root / 'runs'))), encoding='utf-8')
            cfg = pipeline.load_config(cfg_path)
            for stage in (2, 3):
                before, _ = pipeline.fingerprint(stage, cfg, {}, {}, None)
                after, _ = pipeline.fingerprint(stage, cfg | {'frame_ids': [1, 2, 3]}, {}, {}, None)
                self.assertNotEqual(before, after)

    def test_real_tensor_recovery_and_original_tsdf_pipeline(self):
        # 真正运行 Tensor Hybrid 和原 TSDF：2帧深度全零，1→3为已知0.04m运动。
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); dataset, _, metadata = dataset_frames(root, count=3)
            metadata.update(width=128, height=128, fx=150., fy=150., cx=63.5, cy=63.5)
            (dataset / 'camera_intrinsics.json').write_text(json.dumps(metadata), encoding='utf-8')
            rng = np.random.default_rng(3)
            rgb = cv2.GaussianBlur(rng.integers(20, 235, (128, 128, 3), dtype=np.uint8), (5, 5), 0)
            for i in range(1, 4):
                shift = (i - 1) * 3
                color = np.zeros_like(rgb); depth = np.full((128, 128), 1000, np.uint16)
                if shift:
                    color[:, :-shift] = rgb[:, shift:]; depth[:, -shift:] = 0
                else:
                    color[:] = rgb
                if i == 2:
                    depth[:] = 0
                cv2.imwrite(str(dataset / 'color' / f'{i:06d}.png'), color)
                cv2.imwrite(str(dataset / 'depth' / f'{i:06d}.png'), depth)
            _, report, *_ = h.track(dataset, [1, 2, 3], 1, root / 'track', h.settings(dataset, max_skip_frames=1, min_consistency=.6))
            self.assertEqual(report['valid_frame_ids'], [1, 3])
            saved = load_poses(root / 'track/poses_before.json')
            np.testing.assert_allclose(saved[3].T_world_camera, translation(.04), atol=.005)
            self.assertIsNone(saved[2].T_world_camera)
            cfg = dict(dataset_path=str(dataset), camera_poses_path=str(root / 'track/poses_before.json'),
                       start_frame=1, end_frame=3, frame_ids=[1, 3], enable_pointcloud_merge=True,
                       enable_tsdf=True, enable_slicing=False, output_directory=str(root / 'runs'))
            cfg_path = root / 'pipeline.json';cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = pipeline.run_pipeline(cfg_path)
            self.assertEqual(result['status'], 'complete', result.get('error'))
            self.assertEqual(result['selected_frames'], [1, 3])
            self.assertTrue(result['final_mesh']['has_vertex_colors'])
            self.assertGreater(result['final_mesh']['triangle_count'], 0)
            run = Path(result['run_directory'])
            fusion = json.loads((run / 'mesh/tsdf_report_scene_mesh.json').read_text(encoding='utf-8'))
            self.assertEqual(fusion['integrated_frame_count'], 2)
            self.assertEqual([row['frame_id'] for row in fusion['frames']], [1, 3])
            merged = json.loads((run / 'pointcloud/merge_merged_report.json').read_text(encoding='utf-8'))
            self.assertEqual(merged['requested_frame_ids'], [1, 3])


if __name__ == '__main__':
    unittest.main()
