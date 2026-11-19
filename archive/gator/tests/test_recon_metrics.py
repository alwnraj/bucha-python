from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parents[1]))  # repo root: shared renderer, collision, depth model

from collision import raycast_scene, write_mesh_collisions
from recon_metrics import (camera_errors, chamfer_f1, fit_similarity, fit_translation_scale, mask_iou,
                           points_in_voxels, sample_surface)
from render_views import look_at


def _rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = np.radians(degrees)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


class FittingTests(unittest.TestCase):
    def test_similarity_fit_recovers_scale_rotation_translation(self):
        rng = np.random.default_rng(1)
        source = rng.normal(size=(30, 3))
        rotation, translation = _rotation([1, 2, 3], 40), np.array([0.5, -2, 3])
        target = 2.5 * source @ rotation.T + translation
        scale, fitted_rotation, fitted_translation = fit_similarity(source, target)
        self.assertAlmostEqual(scale, 2.5)
        np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-10)
        np.testing.assert_allclose(fitted_translation, translation, atol=1e-10)

    def test_similarity_fit_rejects_collinear_points(self):
        line = np.outer(np.arange(5.), [1, 0, 0])
        with self.assertRaises(ValueError):
            fit_similarity(line, line)

    def test_translation_scale_fit_keeps_rotation_fixed(self):
        source = np.random.default_rng(2).normal(size=(20, 3))
        scale, translation = fit_translation_scale(source, 0.4 * source + [1, 2, 3])
        self.assertAlmostEqual(scale, 0.4)
        np.testing.assert_allclose(translation, [1, 2, 3], atol=1e-12)

    def test_camera_errors_ignore_frame_and_scale_but_catch_a_bad_camera(self):
        truth = [look_at([3 * np.cos(a), 3 * np.sin(a), 1.5], [0, 0, 0.5]) for a in np.linspace(0, 5, 6)]
        rotation, translation = _rotation([0, 1, 1], 70), np.array([4, 5, 6])
        estimated = []
        for pose in truth:
            moved = np.eye(4)
            moved[:3, :3] = rotation @ pose[:3, :3]
            moved[:3, 3] = 0.25 * rotation @ pose[:3, 3] + translation
            estimated.append(moved)
        exact = camera_errors(estimated, truth)
        self.assertAlmostEqual(exact["scale"], 4.0)
        self.assertLess(exact["max_rotation_degrees"], 1e-5)
        self.assertLess(exact["max_center_error"], 1e-9)
        estimated[2] = estimated[2].copy()
        estimated[2][:3, :3] = _rotation([0, 0, 1], 10) @ estimated[2][:3, :3]
        self.assertAlmostEqual(camera_errors(estimated, truth)["rotation_degrees"][2], 10, places=6)


class ScoringTests(unittest.TestCase):
    def test_surface_samples_stay_on_triangles(self):
        triangle = np.array([[[0, 0, 0], [2, 0, 0], [0, 1, 0]]], dtype=float)
        points = sample_surface(triangle, 500)
        self.assertTrue(np.allclose(points[:, 2], 0))
        self.assertTrue((points[:, 0] / 2 + points[:, 1] <= 1 + 1e-12).all())

    def test_chamfer_and_f1(self):
        points = np.random.default_rng(3).random((400, 3))
        same = chamfer_f1(points, points, 0.01)
        self.assertLess(same["chamfer"], 1e-6)  # the fast distance formula leaves rounding noise
        self.assertEqual(same["f1"], 1.0)
        shifted = chamfer_f1(points + [0.5, 0, 0], points, 0.01)
        self.assertLess(shifted["f1"], 0.2)
        self.assertGreater(shifted["chamfer"], 0.05)

    def test_mask_iou_and_voxel_containment(self):
        a = np.zeros((4, 4), bool)
        b = np.zeros((4, 4), bool)
        a[:2], b[1:3] = True, True
        self.assertAlmostEqual(mask_iou(a, b), 1 / 3)
        occupied = np.zeros((2, 2, 2), bool)
        occupied[0, 0, 0] = True
        inside = points_in_voxels([[0.1, 0.1, 0.1], [1.5, 0.1, 0.1], [-1, 0, 0]], np.zeros(3), 1.0, occupied)
        np.testing.assert_array_equal(inside, [True, False, False])


class MeshCollisionTests(unittest.TestCase):
    def test_reconstructed_mesh_works_with_existing_ray_queries(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            scene = write_mesh_collisions(directory, [{"id": 7, "name": "object_007", "vertices": vertices,
                                                      "faces": [[0, 1, 2]], "closed_solid": False}],
                                          "simulation frame")
            self.assertEqual(scene["objects"][0]["semantic_class"], "unknown")
            hit = raycast_scene(Path(directory) / "collision-scene.json", [0.2, 0.2, 1], [0, 0, -1])
            self.assertEqual(hit["object_id"], 7)
            self.assertEqual(hit["source_face"], -1)
            self.assertAlmostEqual(hit["distance_source_units"], 1.0)
            with self.assertRaises(ValueError):
                write_mesh_collisions(directory, [{"id": 1, "name": "bad", "vertices": vertices,
                                                  "faces": [[0, 1, 3]]}], "frame")


if __name__ == "__main__":
    unittest.main()
