import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segment_objects import write_glb
from split_scene import split_scene


def _grid(origin, u, v, nu, nv):
    """Triangles of a quad split into nu x nv cells; winding gives the normal u x v."""
    origin, u, v = (np.asarray(x, float) for x in (origin, u, v))
    tris = []
    for i in range(nu):
        for j in range(nv):
            a = origin + u * i / nu + v * j / nv
            b, c, d = a + u / nu, a + u / nu + v / nv, a + v / nv
            tris += [[a, b, c], [a, c, d]]
    return np.array(tris)


def _box(x, y, size, height, step=.5):
    """Open-bottomed box whose bottom edges match the floor grid, so it is welded to the floor."""
    n = round(size / step)
    return np.concatenate([
        _grid([x, y, 0], [size, 0, 0], [0, 0, height], n, 1),
        _grid([x + size, y, 0], [0, size, 0], [0, 0, height], n, 1),
        _grid([x + size, y + size, 0], [-size, 0, 0], [0, 0, height], n, 1),
        _grid([x, y + size, 0], [0, -size, 0], [0, 0, height], n, 1),
        _grid([x, y, height], [size, 0, 0], [0, size, 0], n, n)])


def _room(path):
    """4 x 4 room, 2.5 high, with a table-sized and a stool-sized box on the floor."""
    parts = [
        _grid([0, 0, 0], [4, 0, 0], [0, 4, 0], 8, 8),            # floor, facing up
        _grid([0, 0, 2.5], [0, 4, 0], [4, 0, 0], 8, 8),          # ceiling, facing down
        _grid([0, 0, 0], [4, 0, 0], [0, 0, 2.5], 8, 5),
        _grid([4, 0, 0], [0, 4, 0], [0, 0, 2.5], 8, 5),
        _grid([4, 4, 0], [-4, 0, 0], [0, 0, 2.5], 8, 5),
        _grid([0, 4, 0], [0, -4, 0], [0, 0, 2.5], 8, 5),
        _box(1, 1, 1.0, .75),
        _box(2.5, 2.5, .5, .45),
    ]
    points = np.ascontiguousarray(np.concatenate(parts).reshape(-1, 3), dtype="<f4")
    blob = bytearray(points.tobytes())
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
           "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [.7, .6, .5, 1]}}],
           "buffers": [{"byteLength": len(blob)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(blob)}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3",
                          "min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}]}
    write_glb(path, doc, blob)


class SplitSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        _room(cls.folder / "room.glb")
        cls.summary = split_scene(cls.folder / "room.glb", cls.folder / "out")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder)

    def test_room_shell_is_structure_and_each_box_is_one_object(self):
        objects = [o for o in self.summary["objects"] if o["kind"] == "object"]
        self.assertEqual(len(objects), 2)
        heights = sorted(round(o["size_ground_frame"][2], 2) for o in objects)
        self.assertEqual(heights, [.45, .75])
        footprints = sorted(round(max(o["size_ground_frame"][:2]), 2) for o in objects)
        self.assertEqual(footprints, [.5, 1.0])
        faces = self.summary["structure_faces"]
        self.assertEqual(faces, {"floor": 128, "walls": 320, "ceiling": 128})

    def test_outputs_are_written(self):
        out = self.folder / "out"
        for name in ("object_001.glb", "object_002.glb", "structure_floor.glb", "structure_walls.glb",
                     "structure_ceiling.glb", "scene-split.glb", "objects.json", "objects-sheet.png"):
            self.assertTrue((out / name).is_file(), name)
        closed = json.loads((out / "closed" / "collision-scene.json").read_text())
        self.assertEqual(sorted(o["name"] for o in closed["objects"]), ["object_001", "object_002"])
        kinds = {o["name"]: o["kind"] for o in self.summary["objects"]}
        self.assertEqual(kinds["structure_walls"], "walls")


class FakeModel:
    """Stand-in for SegFormer: every pixel is 'table'."""
    labels = {0: "wall", 1: "table"}
    model_sha256 = "fake"

    def __call__(self, rgb):
        probabilities = np.zeros((2, 40, 40), dtype=np.float32)
        probabilities[1] = 1
        return probabilities


class LabelTests(unittest.TestCase):
    def test_labels_are_voted_per_object(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _room(folder / "room.glb")
            summary = split_scene(folder / "room.glb", folder / "out", labels=True, predict=FakeModel(),
                                  positions=4, sheet=False, crop_textures=False)
            objects = [o for o in summary["objects"] if o["kind"] == "object"]
            self.assertEqual([o["label"] for o in objects], ["table", "table"])
            self.assertTrue(all(o["label_share"] == 1.0 for o in objects))
            self.assertTrue((folder / "out" / "work" / "object-votes.npz").is_file())


if __name__ == "__main__":
    unittest.main()
