import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parents[1]))  # repo root: shared renderer, collision, depth model

from photo_sim import orbit_cameras, simulate
from render_views import intrinsics_from_fov, load_scene, look_at, render_perspective
from segment_objects import write_glb


def _cube_on_floor(path, floor_half=20.0):
    """Untextured unit cube at x, y, z in [0, 1] standing on a large floor at z=0."""
    f = floor_half
    floor = [[-f, -f, 0], [f, -f, 0], [f, f, 0], [-f, f, 0]]
    cube = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    positions = np.asarray(floor + cube, dtype="<f4")
    floor_faces = [0, 1, 2, 0, 2, 3]
    cube_faces = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                  1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    indices = np.asarray(floor_faces + [i + 4 for i in cube_faces], dtype="<u2")
    blob = bytearray(positions.tobytes())
    index_offset = len(blob)
    blob.extend(indices.tobytes())
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
                        {"buffer": 0, "byteOffset": index_offset, "byteLength": indices.nbytes}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
                       "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
                      {"bufferView": 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"}],
    }
    write_glb(path, doc, blob)


def _back_project(intrinsics, pose, x, y, depth):
    ray = np.array([(x - intrinsics[0, 2]) / intrinsics[0, 0], (y - intrinsics[1, 2]) / intrinsics[1, 1], 1.0])
    return pose[:3, :3] @ (ray * depth) + pose[:3, 3]


class PerspectiveRenderTests(unittest.TestCase):
    def test_look_at_builds_right_handed_opencv_axes(self):
        pose = look_at([3, -2, 1], [0, 0, 0.5])
        rotation = pose[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
        forward = np.array([0, 0, 0.5]) - np.array([3, -2, 1])
        np.testing.assert_allclose(rotation[:, 2], forward / np.linalg.norm(forward))
        self.assertLess(rotation[2, 1], 0)  # image +Y (down) points against world up

    def test_depth_back_projects_onto_rendered_faces(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.glb"
            _cube_on_floor(source)
            scene = load_scene(source)
            size = (96, 72)
            intrinsics = intrinsics_from_fov(*size, 60)
            pose = look_at([3, -2.5, 2], [0.5, 0.5, 0.5])
            rgb, depth, face = render_perspective(scene, intrinsics, pose, size)
            self.assertEqual(rgb.shape, (72, 96, 3))
            cube_pixels = face >= 2
            self.assertGreater(cube_pixels.sum(), 200)
            triangles = np.concatenate([chunk["tri"] for chunk in scene["chunks"]])
            for y, x in np.argwhere(face >= 0)[::37]:
                point = _back_project(intrinsics, pose, x, y, depth[y, x])
                tri = triangles[face[y, x]]
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                normal /= np.linalg.norm(normal)
                self.assertLess(abs((point - tri[0]) @ normal), 1e-6)
            self.assertTrue(np.isnan(depth[face < 0]).all())

    def test_floor_crossing_the_camera_plane_is_clipped_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.glb"
            _cube_on_floor(source)
            scene = load_scene(source)
            size = (64, 48)
            intrinsics = intrinsics_from_fov(*size, 60)
            pose = look_at([0.5, -4, 1], [0.5, 0.5, 0.2])
            _, depth, face = render_perspective(scene, intrinsics, pose, size)
            # The floor triangles extend behind the camera; the bottom row must still see floor.
            self.assertTrue((face[-1] >= 0).all() and (face[-1] < 2).all())
            point = _back_project(intrinsics, pose, 32, 47, depth[47, 32])
            self.assertAlmostEqual(point[2], 0.0, places=5)


class PhotoSimulationTests(unittest.TestCase):
    def test_orbit_cameras_all_look_at_center_from_fixed_distance(self):
        center = np.array([1., 2., 3.])
        _, poses, distance = orbit_cameras(center, 0.5, 6, (64, 48), elevation=30)
        for pose in poses:
            self.assertAlmostEqual(np.linalg.norm(pose[:3, 3] - center), distance)
            towards = (center - pose[:3, 3]) / distance
            np.testing.assert_allclose(pose[:3, 2], towards, atol=1e-12)
            self.assertGreater(pose[2, 3], center[2])

    def test_simulation_separates_input_from_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.glb"
            output = Path(directory) / "capture"
            _cube_on_floor(source)
            summary = simulate(source, output, [-0.1, -0.1, -0.1], [1.1, 1.1, 1.1], views=4, size=(64, 48))
            self.assertEqual(summary["target_faces"], 12)
            self.assertEqual(summary["warnings"], [])
            bundle = json.loads((output / "bundle.json").read_text())
            self.assertEqual(bundle["format"], "photo-bundle-v1")
            self.assertEqual(len(bundle["images"]), 4)
            self.assertNotIn("cam_to_world", json.dumps(bundle))
            for image in bundle["images"]:
                self.assertTrue((output / image["file"]).is_file())
            truth = json.loads((output / "truth" / "truth.json").read_text())
            target = np.load(output / "truth" / "target.npz")
            np.testing.assert_array_equal(target["faces"], np.arange(2, 14))
            for view in truth["views"]:
                arrays = np.load(output / view["arrays"])
                masked_faces = set(np.unique(arrays["face"][arrays["mask"]]))
                self.assertTrue(masked_faces and masked_faces <= set(range(2, 14)))
                self.assertEqual(int(arrays["mask"].sum()), view["target_pixels"])
            with self.assertRaises(ValueError):
                simulate(source, output, [0, 0, 0], [1, 1, 1], views=2, size=(32, 24))

    def test_focal_hint_can_be_withheld(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.glb"
            _cube_on_floor(source)
            simulate(source, Path(directory) / "capture", [0, 0, 0], [1, 1, 1], views=3,
                     size=(32, 24), focal_hint=False)
            bundle = json.loads((Path(directory) / "capture" / "bundle.json").read_text())
            self.assertTrue(all("focal_length_px" not in image for image in bundle["images"]))

    def test_empty_target_box_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.glb"
            _cube_on_floor(source)
            with self.assertRaises(ValueError):
                simulate(source, Path(directory) / "capture", [5, 5, 5], [6, 6, 6], views=2, size=(32, 24))


if __name__ == "__main__":
    unittest.main()
