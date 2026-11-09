import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from photo_poses import align_depth, bundle_intrinsics, estimate, match_features, register_views, relative_pose
from photo_sim import simulate
from recon_metrics import camera_errors
from render_views import look_at
from tests.test_photo_sim import _cube_on_floor


def _ring(count=6, radius=3.0):
    return [look_at([radius * np.cos(a), radius * np.sin(a), 1.2], [0, 0, 0.4])
            for a in np.linspace(0, 2 * np.pi, count, endpoint=False)]


def _true_edge(pose_a, pose_b, inliers=100, scale_a=1.0, scale_b=1.0):
    """Relative pose in relative_pose's convention: X_b = R @ X_a + t."""
    rotation = pose_b[:3, :3].T @ pose_a[:3, :3]
    translation = pose_b[:3, :3].T @ (pose_a[:3, 3] - pose_b[:3, 3])
    length = np.linalg.norm(translation)
    return {"rotation": rotation, "translation": translation, "direction": translation / length,
            "inliers": inliers, "scale": length, "scale_a": length * scale_a, "scale_b": length * scale_b}


def _project(points, pose, intrinsics):
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    pixels = camera[:, :2] / camera[:, 2:] * [intrinsics[0, 0], intrinsics[1, 1]] + intrinsics[:2, 2]
    return pixels, camera[:, 2]


class IntrinsicsTests(unittest.TestCase):
    def test_known_fov_beats_focal_hint_and_assumption_is_last(self):
        image = {"width": 640, "height": 480, "focal_length_px": 400.0}
        self.assertEqual(bundle_intrinsics(image)[0][0, 0], 400.0)
        known, source = bundle_intrinsics(image, fov_x=90)
        self.assertAlmostEqual(known[0, 0], 320.0)
        self.assertIn("known", source)
        assumed, source = bundle_intrinsics({"width": 640, "height": 480})
        self.assertIn("assumed", source)
        self.assertAlmostEqual(assumed[0, 2], 319.5)


class PairGeometryTests(unittest.TestCase):
    def test_relative_pose_recovers_rotation_and_depth_scaled_translation(self):
        rng = np.random.default_rng(4)
        points = rng.uniform([-1, -1, 0], [1, 1, 1.2], size=(200, 3))
        pose_a, pose_b = look_at([3, 0, 1.2], [0, 0, 0.4]), look_at([2.4, 1.8, 1.4], [0, 0, 0.4])
        k_a = np.array([[500., 0, 319.5], [0, 500, 239.5], [0, 0, 1]])
        k_b = np.array([[420., 0, 319.5], [0, 420, 239.5], [0, 0, 1]])
        pixels_a, depth_a = _project(points, pose_a, k_a)
        pixels_b, depth_b = _project(points, pose_b, k_b)
        truth = _true_edge(pose_a, pose_b)
        result = relative_pose(pixels_a, pixels_b, k_a, k_b, depth_a, depth_b)
        np.testing.assert_allclose(result["rotation"], truth["rotation"], atol=1e-6)
        np.testing.assert_allclose(result["translation"], truth["translation"], atol=1e-5)
        self.assertGreater(result["parallax_degrees"], 5)
        doubled = relative_pose(pixels_a, pixels_b, k_a, k_b, 2 * depth_a, 2 * depth_b)
        np.testing.assert_allclose(doubled["translation"], 2 * truth["translation"], atol=1e-5)
        self.assertIsNone(relative_pose(pixels_a[:10], pixels_b[:10], k_a, k_b, depth_a[:10], depth_b[:10]))

    def test_mutual_matching_keeps_only_consistent_pairs(self):
        descriptors = np.random.default_rng(5).random((40, 128)).astype(np.float32)
        pairs = match_features(descriptors, descriptors[::-1].copy())
        np.testing.assert_array_equal(pairs[:, 1], 39 - pairs[:, 0])
        self.assertEqual(len(pairs), 40)
        self.assertEqual(len(match_features(None, descriptors)), 0)


K = np.array([[500., 0, 319.5], [0, 500, 239.5], [0, 0, 1]])


def _orbit_capture(count=6, seed=0, depth_factors=None):
    """Ring of cameras seeing shared random points; neighbours are matched like a real orbit."""
    truth = _ring(count)
    world = np.random.default_rng(seed).uniform([-0.8, -0.8, 0], [0.8, 0.8, 0.8], size=(300, 3))
    keypoints = [_project(world, pose, K)[0] for pose in truth]
    factors = np.ones(count) if depth_factors is None else np.asarray(depth_factors)
    matches, edges = {}, {}
    for a in range(count - 1):
        matches[(a, a + 1)] = np.column_stack((np.arange(300), np.arange(300)))
        # Each pair's depth-model baseline is off by its photos' depth errors.
        edges[(a, a + 1)] = _true_edge(truth[a], truth[a + 1], inliers=300 - a,
                                       scale_a=factors[a], scale_b=factors[a + 1])
        edges[(a, a + 1)]["scale"] *= np.sqrt(factors[a] * factors[a + 1])
        edges[(a, a + 1)]["translation"] = edges[(a, a + 1)]["direction"] * edges[(a, a + 1)]["scale"]
    return truth, world, keypoints, matches, edges


class RegistrationTests(unittest.TestCase):
    def test_shared_points_keep_one_scale_despite_per_pair_depth_errors(self):
        truth, _, keypoints, matches, edges = _orbit_capture(depth_factors=[1.0, 1.3, 0.8, 1.2, 0.85, 1.1])
        poses, root, points, tracks, placed_by = register_views(6, edges, matches, keypoints, [K] * 6)
        self.assertEqual(root, 0)
        np.testing.assert_allclose(poses[root], np.eye(4))
        errors = camera_errors(poses, truth)
        self.assertLess(errors["max_center_error"], 1e-6)
        self.assertLess(errors["max_relative_rotation_degrees"], 1e-4)
        self.assertTrue(all("shared points" in p for p in placed_by[2:]))
        self.assertGreater(len(points), 250)
        self.assertTrue(all((track >= 0).sum() > 250 for track in tracks))

    def test_depth_baseline_is_the_fallback_without_shared_points(self):
        truth, _, keypoints, _, edges = _orbit_capture(count=4)
        empty = {key: np.empty((0, 2), dtype=int) for key in edges}
        poses, _, points, _, placed_by = register_views(4, edges, empty, keypoints, [K] * 4)
        self.assertEqual(len(points), 0)
        self.assertTrue(all("depth baseline" in p for p in placed_by[1:]))
        self.assertLess(camera_errors(poses, truth)["max_center_error"], 1e-9)
        self.assertEqual(register_views(3, {}, {}, keypoints[:3], [K] * 3)[0], [None] * 3)

    def test_depth_alignment_sets_one_scale_and_per_photo_multipliers(self):
        truth, world, keypoints, matches, edges = _orbit_capture()
        poses, _, points, tracks, _ = register_views(6, edges, matches, keypoints, [K] * 6)
        factors = np.array([1.3, 0.8, 1.0, 1.1, 0.9, 1.2])  # each photo's depth is this many times too large
        depths = []
        for pose, pixels, factor in zip(truth, keypoints, factors):
            depth = np.full((480, 640), np.nan, dtype=np.float32)
            rounded = np.rint(pixels).astype(int)
            depth[rounded[:, 1], rounded[:, 0]] = _project(world, pose, K)[1] * factor
            depths.append(depth)
        half = [None if pose is None else pose.copy() for pose in poses]
        for pose in half:
            pose[:3, 3] *= 0.5  # reconstruction at an arbitrary size
        aligned, scaled_points, corrections = align_depth(half, points * 0.5, tracks, keypoints, depths)
        # The overall size follows the median depth factor, which lies between 1.0 and 1.1 here.
        size = 1 / camera_errors(aligned, truth)["scale"]
        self.assertTrue(1.0 - 1e-6 <= size <= 1.1 + 1e-6)
        # Every corrected photo depth now matches the reconstruction's size.
        np.testing.assert_allclose(corrections * factors, size, rtol=0.01)
        self.assertEqual(scaled_points.shape, points.shape)


class EstimateTests(unittest.TestCase):
    def test_textureless_capture_registers_nothing_but_still_writes_results(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.glb"
            _cube_on_floor(source)
            simulate(source, Path(directory) / "capture", [0, 0, 0], [1, 1, 1], views=3, size=(64, 48))
            result = estimate(Path(directory) / "capture", Path(directory) / "poses", truth_depth=True)
            self.assertEqual(result["registered_views"], 0)
            self.assertIsNone(result["frame"])
            written = json.loads((Path(directory) / "poses" / "poses.json").read_text())
            self.assertEqual(len(written["views"]), 3)
            self.assertTrue((Path(directory) / "poses" / "depth" / "view_00.npy").is_file())


if __name__ == "__main__":
    unittest.main()
