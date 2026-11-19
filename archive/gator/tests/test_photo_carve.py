import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parents[1]))  # repo root: shared renderer, collision, depth model

from photo_carve import carve, estimate, load_views, score, voxel_surface
from photo_sim import simulate
from test_photo_sim import _cube_on_floor


def _signed_volume(vertices, faces):
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def _edge_uses(faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return np.unique(edges, axis=0, return_counts=True)[1]


class SurfaceTests(unittest.TestCase):
    def test_single_voxel_is_a_closed_outward_cube(self):
        vertices, faces = voxel_surface(np.ones((1, 1, 1), bool), np.array([1., 2, 3]), 0.5)
        self.assertEqual((len(vertices), len(faces)), (8, 12))
        self.assertAlmostEqual(_signed_volume(vertices, faces), 0.125)
        np.testing.assert_allclose(vertices.min(axis=0), [1, 2, 3])

    def test_shape_with_a_notch_is_watertight_with_the_right_volume(self):
        occupied = np.ones((3, 2, 2), bool)
        occupied[1, 1, 1] = False
        vertices, faces = voxel_surface(occupied, np.zeros(3), 1.0)
        self.assertAlmostEqual(_signed_volume(vertices, faces), occupied.sum())
        self.assertTrue((_edge_uses(faces) == 2).all())


class CarveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        _cube_on_floor(cls.folder / "scene.glb")
        simulate(cls.folder / "scene.glb", cls.folder / "capture", [-0.1, -0.1, 0.01], [1.1, 1.1, 1.1],
                 views=8, size=(160, 120), elevation=35)
        truth = json.loads((cls.folder / "capture" / "truth" / "truth.json").read_text())
        # Perfect step 2 and step 3 results built from the truth.
        views = [{"image": v["image"], "intrinsics": truth["intrinsics"], "cam_to_world": v["cam_to_world"],
                  "depth": None} for v in truth["views"]]
        (cls.folder / "poses").mkdir()
        (cls.folder / "poses" / "poses.json").write_text(json.dumps(
            {"views": views, "frame": "test world", "units": "test units"}))
        (cls.folder / "masks" / "masks").mkdir(parents=True)
        for index in range(len(views)):
            shutil.copy(cls.folder / "capture" / "truth" / f"view_{index:02d}_mask.png",
                        cls.folder / "masks" / "masks" / f"view_{index:02d}.png")
        (cls.folder / "masks" / "masks.json").write_text(json.dumps(
            {"support_plane": {"normal": [0, 0, 1], "offset": 0.0}}))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder)

    def test_exact_masks_carve_a_hull_that_contains_the_cube(self):
        summary = estimate(self.folder / "poses", self.folder / "masks", self.folder / "carved", resolution=48,
                           dilate=1)
        self.assertTrue(summary["support_plane_used"])
        self.assertEqual(len(summary["views"]), 8)
        # The unit cube has volume 1; its hull from 8 raised views keeps uncarvable corners (1.5 measured).
        self.assertGreater(summary["volume"], 0.95)
        self.assertLess(summary["volume"], 1.65)
        report = score(self.folder / "carved", self.folder / "poses", self.folder / "capture")
        self.assertAlmostEqual(report["alignment_scale_truth_per_estimate"], 1.0, places=6)
        self.assertGreater(report["truth_surface_contained"], 0.97)
        self.assertTrue((np.array(report["box_size_carved_aligned"]) < 1.6).all())
        self.assertGreater(report["surface"]["0.1"]["f1"], 0.5)  # 0.60 measured; the hull is wider than the cube
        scene = json.loads((self.folder / "carved" / "collision-scene.json").read_text())
        self.assertTrue(scene["objects"][0]["closed_solid"])
        with self.assertRaises(ValueError):
            estimate(self.folder / "poses", self.folder / "masks", self.folder / "carved")

    def test_voting_survives_one_empty_mask(self):
        views, _ = load_views(self.folder / "poses", self.folder / "masks")
        support = (np.array([0, 0, 1.]), 0.0)
        good = carve(views, resolution=32, support=support)
        index, intrinsics, pose, mask = views[0]
        views[0] = (index, intrinsics, pose, np.zeros_like(mask))
        voted = carve(views, resolution=32, support=support)
        volume = lambda r: r["occupied"].sum() * r["voxel_size"] ** 3
        self.assertGreater(volume(voted), 0.9 * volume(good))
        with self.assertRaises(ValueError):  # without voting the one bad mask erases everything
            carve(views, agree=1.0, resolution=32, support=support)


if __name__ == "__main__":
    unittest.main()
