import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parents[1]))  # repo root: shared renderer, collision, depth model

from photo_masks import (attention_point, back_project, camera_up, estimate_masks, horizontal_planes,
                         object_cells, score, support_plane)
from photo_sim import orbit_cameras, simulate
from test_photo_sim import _cube_on_floor


def _grid_points(x, y, z):
    xs, ys = np.meshgrid(np.arange(*x), np.arange(*y))
    return np.column_stack((xs.ravel(), ys.ravel(), np.full(xs.size, z)))


class GeometryTests(unittest.TestCase):
    def test_camera_up_and_attention_point_for_an_orbit(self):
        center = np.array([1., -2., 0.5])
        _, poses, _ = orbit_cameras(center, 0.4, 8, (64, 48), elevation=30)
        np.testing.assert_allclose(camera_up(poses), [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(attention_point(poses), center, atol=1e-9)

    def test_back_project_returns_world_points_and_nan_for_missing_depth(self):
        intrinsics = np.array([[100., 0, 1.5], [0, 100, 1.5], [0, 0, 1]])
        pose = np.eye(4)
        pose[:3, 3] = [5, 0, 0]
        depth = np.full((4, 4), 2.0)
        depth[0, 0] = np.nan
        points = back_project(depth, intrinsics, pose)
        self.assertTrue(np.isnan(points[0, 0]).all())
        np.testing.assert_allclose(points[3, 3], [5 + 1.5 / 100 * 2, 1.5 / 100 * 2, 2])

    def test_support_is_the_lowest_large_plane_not_the_tabletop(self):
        floor = _grid_points((-2, 2, 0.05), (-2, 2, 0.05), 0.0)
        table = _grid_points((-0.5, 0.5, 0.05), (-0.5, 0.5, 0.05), 0.7)
        planes = horizontal_planes(np.vstack((floor, table)), np.array([0, 0, 1.]))
        normal, offset, _ = support_plane(planes)
        np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-9)
        self.assertAlmostEqual(offset, 0.0, places=9)

    def test_thin_walls_are_removed_from_the_object_grid(self):
        floor = (np.array([0, 0, 1.]), 0.0, 1)
        block = _grid_points((0, 1, 0.02), (0, 1, 0.02), 0.5)
        wall = np.array([[x, 1.12, z] for x in np.arange(-1, 2, 0.02) for z in np.arange(0.1, 1.5, 0.05)])
        cloud = np.vstack((block, wall))
        grid = object_cells([cloud, cloud], floor, np.array([0.5, 0.5, 0.3]))
        a, b = grid["axes"]
        index = lambda p: tuple(np.floor((np.array([p @ a, p @ b]) - grid["origin"]) / grid["cell"]).astype(int))
        self.assertTrue(grid["cells"][index(np.array([0.5, 0.5, 0]))])
        self.assertFalse(grid["cells"][index(np.array([-0.8, 1.12, 0]))])
        self.assertAlmostEqual(grid["cells"].sum() * 0.05 ** 2, 1.0, delta=0.25)


class EndToEndTests(unittest.TestCase):
    def test_exact_depth_and_poses_give_accurate_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _cube_on_floor(folder / "scene.glb")
            simulate(folder / "scene.glb", folder / "capture", [-0.1, -0.1, 0.01], [1.1, 1.1, 1.1],
                     views=6, size=(160, 120), elevation=35)
            truth = json.loads((folder / "capture" / "truth" / "truth.json").read_text())
            # Fake a perfect photo_poses.py result from the truth.
            (folder / "poses" / "depth").mkdir(parents=True)
            views = []
            for index, view in enumerate(truth["views"]):
                with np.load(folder / "capture" / view["arrays"]) as arrays:
                    np.save(folder / "poses" / "depth" / f"view_{index:02d}.npy", arrays["depth"])
                views.append({"image": view["image"], "intrinsics": truth["intrinsics"],
                              "cam_to_world": view["cam_to_world"], "depth": f"depth/view_{index:02d}.npy",
                              "depth_correction": 1.0})
            (folder / "poses" / "poses.json").write_text(json.dumps({"views": views, "units": "test"}))
            result = estimate_masks(folder / "capture", folder / "poses", folder / "masks", cell=0.1, thin=0.2)
            self.assertEqual(len(result["views"]), 6)
            self.assertAlmostEqual(result["object"]["footprint_area"], 1.0, delta=0.3)
            report = score(folder / "masks", folder / "capture")
            self.assertGreater(report["min_iou"], 0.85)
            with self.assertRaises(ValueError):
                estimate_masks(folder / "capture", folder / "poses", folder / "masks")


if __name__ == "__main__":
    unittest.main()
