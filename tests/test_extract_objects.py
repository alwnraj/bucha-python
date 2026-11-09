import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collision import raycast_scene
from extract_objects import (CLASSES, closed_collider, extract, face_adjacency, merge_fragments, smooth_labels,
                             split_objects)
from prepare_glb import decompress
from render_views import intrinsics_from_fov, load_scene, look_at, render_perspective
from segment_objects import estimate_ground_plane, load_geometry, write_glb


GRAY, RED, GREEN = [.5, .5, .5], [.9, .1, .1], [.1, .8, .1]


def _box(low, high):
    (x0, y0, z0), (x1, y1, z1) = low, high
    corners = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
               [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    faces = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
             1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    return np.array(corners, dtype="<f4"), np.array(faces, dtype="<u4")


def _grid_floor(half=12.0, cells=12):
    ticks = np.linspace(-half, half, cells + 1)
    xs, ys = np.meshgrid(ticks, ticks, indexing="ij")
    points = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size))).astype("<f4")
    faces = []
    for i in range(cells):
        for j in range(cells):
            a, b, c, d = i * (cells + 1) + j, (i + 1) * (cells + 1) + j, (i + 1) * (cells + 1) + j + 1, \
                i * (cells + 1) + j + 1
            faces += [a, b, c, a, c, d]
    return points, np.array(faces, dtype="<u4")


def _street(path):
    """Gray floor, two separate red boxes (cars) and one green box (tree)."""
    parts = [(_grid_floor(), GRAY), (_box([-6, -2, 0], [-3, 0, 1.5]), RED),
             (_box([3, -2, 0], [6, 0, 1.5]), RED), (_box([-1, 4, 0], [1, 6, 4]), GREEN)]
    blob, views, accessors, primitives, materials = bytearray(), [], [], [], []

    def add(values, component, kind, target):
        blob.extend(b"\0" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": values.nbytes, "target": target})
        blob.extend(values.tobytes())
        accessors.append({"bufferView": len(views) - 1, "componentType": component,
                          "count": len(values), "type": kind})
        if kind == "VEC3":
            accessors[-1] |= {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}
        return len(accessors) - 1

    for (points, faces), color in parts:
        materials.append({"pbrMetallicRoughness": {"baseColorFactor": color + [1]}})
        primitives.append({"attributes": {"POSITION": add(points, 5126, "VEC3", 34962)},
                           "indices": add(faces, 5125, "SCALAR", 34963), "material": len(materials) - 1})
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": primitives}], "materials": materials,
           "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors}
    write_glb(path, doc, blob)


def _color_labeler(rgb):
    """Stand-in for SegFormer: red is car, green is tree, gray is road, background is other."""
    probabilities = np.full((len(CLASSES), *rgb.shape[:2]), 1e-3, dtype=np.float32)
    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 80)
    green = (rgb[..., 1] > 150) & (rgb[..., 0] < 80)
    gray = (np.abs(rgb.astype(int) - 128).max(axis=2) < 20)
    for name, mask in (("car", red), ("tree", green), ("road", gray)):
        probabilities[CLASSES.index(name)][mask] = 1
    probabilities[CLASSES.index("other")][~(red | green | gray)] = 1
    return probabilities / probabilities.sum(axis=0, keepdims=True)


class ExtractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        _street(cls.folder / "street.glb")
        cls.summary = extract(cls.folder / "street.glb", cls.folder / "out", predict=_color_labeler,
                              positions=4, headings=4, size=(160, 120), min_faces=4)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.folder)

    def test_each_box_becomes_its_own_labeled_object(self):
        by_name = {obj["name"]: obj for obj in self.summary["objects"]}
        self.assertEqual(set(by_name), {"car_01", "car_02", "tree_01", "road"})
        self.assertEqual(self.summary["object_count"], 4)
        self.assertEqual(by_name["road"]["triangles"], 288)
        for name in ("car_01", "car_02", "tree_01"):
            self.assertEqual(by_name[name]["triangles"], 12)
            self.assertTrue((self.folder / "out" / by_name[name]["glb"]).is_file())
        centers = sorted(by_name[name]["position_xyz"][0] for name in ("car_01", "car_02"))
        np.testing.assert_allclose(centers, [-4.5, 4.5], atol=1e-5)
        np.testing.assert_allclose(by_name["tree_01"]["size_ground_frame"][2], 4, atol=.05)  # up is estimated

    def test_textured_and_collision_outputs_are_written(self):
        out = self.folder / "out"
        self.assertTrue((out / "objects-sheet.png").is_file())
        hit = raycast_scene(out / "collision-scene.json", [-4.5, -5, .75], [0, 1, 0])
        self.assertEqual(hit["object_name"], self._car_at(-4.5))
        closed = json.loads((out / "closed" / "collision-scene.json").read_text())
        self.assertEqual({obj["name"] for obj in closed["objects"]}, {"car_01", "car_02", "tree_01"})
        self.assertTrue(all(obj["closed_solid"] for obj in closed["objects"]))

    def _car_at(self, x):
        return next(obj["name"] for obj in self.summary["objects"]
                    if obj["name"].startswith("car") and abs(obj["position_xyz"][0] - x) < .1)

    def test_rerun_reuses_cached_votes(self):
        def fail(_):
            raise AssertionError("cached votes should skip prediction")
        again = extract(self.folder / "street.glb", self.folder / "out", predict=fail, positions=4, headings=4,
                        size=(160, 120), min_faces=4, sheet=False)
        self.assertEqual(again["object_count"], 4)
        closed = list((self.folder / "out" / "closed" / "colliders").iterdir())
        self.assertEqual(len(closed), 6)  # 3 objects x (OBJ + NPZ), nothing left from earlier runs


class StepTests(unittest.TestCase):
    def test_unseen_faces_borrow_neighbour_labels_and_small_islands_merge(self):
        # A strip of 10 faces: 0-5 voted car, 9 voted tree, 6-8 unseen.
        points = np.array([[i, 0, 0] for i in range(7)] + [[i, 1, 0] for i in range(7)], float)
        tri = []
        for i in range(5):
            tri += [[points[i], points[i + 1], points[i + 7]], [points[i + 1], points[i + 8], points[i + 7]]]
        triangles = np.array(tri)
        votes = np.zeros((10, len(CLASSES)))
        votes[:6, CLASSES.index("car")] = 1
        votes[9, CLASSES.index("tree")] = 1
        seen = (votes.sum(axis=1) > 0).astype(float) * 3
        adjacency = face_adjacency(triangles)
        labels, confidence = smooth_labels(votes, seen, adjacency, smooth=0)
        self.assertTrue((labels[:6] == CLASSES.index("car")).all())
        self.assertNotEqual(labels[7], CLASSES.index("other"))
        object_id, named = split_objects(labels, adjacency, triangles, min_faces=3)
        self.assertEqual([name for _, name, _ in named], ["car_01"])
        self.assertTrue((object_id == 1).all())

    def test_closed_collider_is_watertight_and_fills_to_the_bottom(self):
        # An open-bottomed box (no floor faces): the solid must still be closed.
        corners, faces = _box([0, 0, 0], [2, 1, 1])
        triangles = corners[faces.reshape(-1, 3)].astype(float)
        ground = {"right": np.array([1., 0, 0]), "forward": np.array([0, 1., 0]), "up": np.array([0, 0, 1.]),
                  "edge_scale": .01}
        vertices, solid = closed_collider(triangles[2:], ground, resolution=20)
        a, b, c = vertices[solid[:, 0]], vertices[solid[:, 1]], vertices[solid[:, 2]]
        volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
        self.assertAlmostEqual(volume, 2.0, delta=.25)
        edges = np.sort(np.concatenate([solid[:, [0, 1]], solid[:, [1, 2]], solid[:, [2, 0]]]), axis=1)
        self.assertTrue((np.unique(edges, axis=0, return_counts=True)[1] == 2).all())


def _graph(blocks, links):
    """Faces in labeled blocks; ``links`` are (block a, block b, shared edges)."""
    object_id, start, first = [], 0, []
    for block, size in enumerate(blocks, start=1):
        first.append(start)
        object_id += [block] * size
        start += size
    source, target = [], []
    for start, size in zip(first, blocks):  # chain inside each block
        source += list(range(start, start + size - 1))
        target += list(range(start + 1, start + size))
    for a, b, edges in links:
        for k in range(edges):
            source.append(first[a - 1] + k % blocks[a - 1])
            target.append(first[b - 1] + k % blocks[b - 1])
    source, target = np.array(source), np.array(target)
    return np.array(object_id), (np.concatenate([source, target]), np.concatenate([target, source]))


class MergeTests(unittest.TestCase):
    def test_fragments_join_their_main_neighbour_but_shrubs_and_posts_stay(self):
        # 1 building, 2 edge fragment, 3 grass, 4 flat unsure patch, 5 shrub, 6 tall unsure post
        blocks = [1000, 30, 1000, 40, 40, 40]
        object_id, adjacency = _graph(blocks, [(2, 1, 9), (2, 3, 1), (4, 3, 10), (5, 3, 10), (6, 3, 10),
                                               (1, 3, 50)])
        named = [(1, "building_01", "building"), (2, "wall_01", "wall"), (3, "grass", "grass"),
                 (4, "building_02", "building"), (5, "plant_01", "plant"), (6, "pole_01", "pole")]
        confidence = np.repeat([.9, .6, .8, .3, .8, .3], blocks).astype(float)
        heights = np.zeros((len(object_id), 3))
        heights[object_id == 6, 2] = 2.0  # the post stands 2 units tall
        merged, names = merge_fragments(object_id, named, confidence, adjacency, heights)
        by_old = {old: names[merged[object_id == old][0] - 1][1] for old in range(1, 7)}
        self.assertEqual(by_old[2], by_old[1])          # fragment joined the building
        self.assertEqual(by_old[4], "grass")             # unsure flat patch joined the lawn
        self.assertEqual(by_old[5], "plant_01")          # confident shrub stays
        self.assertEqual(by_old[6], "pole_01")           # tall post stays
        self.assertEqual(len(names), 4)

    def test_touching_post_pieces_become_one_pole(self):
        blocks = [1000, 300, 100, 50]  # grass, tree-labeled piece, pole piece, column piece
        object_id, adjacency = _graph(blocks, [(2, 3, 5), (3, 4, 5), (2, 1, 5)])
        named = [(1, "grass", "grass"), (2, "tree_01", "tree"), (3, "pole_01", "pole"), (4, "column_01", "column")]
        heights = np.zeros((len(object_id), 3))
        heights[object_id > 1, 2] = 3.0
        merged, names = merge_fragments(object_id, named, np.full(len(object_id), .8), adjacency, heights)
        self.assertEqual([name for _, name, _ in names], ["grass", "pole_01"])
        self.assertEqual(len(set(merged[object_id > 1])), 1)


def _textured_strip(path):
    """Two separate quads using opposite corners of a 128x128 lossless texture."""
    from io import BytesIO
    from PIL import Image

    texture = np.random.default_rng(1).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    stream = BytesIO()
    Image.fromarray(texture).save(stream, "PNG")
    png = stream.getvalue()
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0]], dtype="<f4")
    uv = np.array([[.05, .05], [.2, .05], [.2, .2], [.05, .2],
                   [.75, .7], [.95, .7], [.95, .9], [.75, .9]], dtype="<f4")
    faces = np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], dtype="<u4")
    blob = bytearray()
    views = []
    for data in (points.tobytes(), uv.tobytes(), faces.tobytes(), png):
        blob.extend(b"\0" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2,
                                       "material": 0}]}],
           "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
           "textures": [{"source": 0}], "images": [{"bufferView": 3, "mimeType": "image/png"}],
           "buffers": [{"byteLength": len(blob)}], "bufferViews": views,
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
                          "min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
                         {"bufferView": 1, "componentType": 5126, "count": 8, "type": "VEC2"},
                         {"bufferView": 2, "componentType": 5125, "count": 12, "type": "SCALAR"}]}
    write_glb(path, doc, blob)


class TextureCropTests(unittest.TestCase):
    def test_packed_texture_is_smaller_and_renders_the_same(self):
        from texture_crop import repack_textures
        from segment_objects import read_glb

        with tempfile.TemporaryDirectory() as directory:
            source, packed = Path(directory) / "strip.glb", Path(directory) / "packed.glb"
            _textured_strip(source)
            before, after = repack_textures(source, packed)
            self.assertLess(after, before / 3)
            doc, _ = read_glb(packed)
            self.assertLess(doc["images"][0]["bufferView"], len(doc["bufferViews"]))
            renders = []
            for path in (source, packed):
                scene = load_scene(path)
                pose = look_at([2, -3, 4], [2, .5, 0])
                renders.append(render_perspective(scene, intrinsics_from_fov(160, 120, 60), pose, (160, 120)))
            np.testing.assert_array_equal(renders[0][2], renders[1][2])
            covered = renders[0][2] >= 0
            self.assertGreater(covered.sum(), 500)
            # Lossless PNG and whole-pixel block moves: colours match except where a
            # sample lands exactly on a texel edge and float32 UV rounding tips it over.
            self.assertGreater((renders[0][0] == renders[1][0]).all(axis=2)[covered].mean(), .99)


class LargeScanTests(unittest.TestCase):
    def test_batched_sub_pixel_triangles_match_the_exact_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fine.glb"
            points, faces = _grid_floor(half=6, cells=160)  # 51,200 tiny triangles
            blob = bytearray(points.tobytes()) + faces.tobytes()
            doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
                   "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
                   "buffers": [{"byteLength": len(blob)}],
                   "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": points.nbytes},
                                   {"buffer": 0, "byteOffset": points.nbytes, "byteLength": faces.nbytes}],
                   "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3",
                                  "min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
                                 {"bufferView": 1, "componentType": 5125, "count": len(faces), "type": "SCALAR"}]}
            write_glb(source, doc, blob)
            scene = load_scene(source)
            size = (120, 90)
            pose = look_at([0, -9, 4], [0, 0, 0])
            exact = render_perspective(scene, intrinsics_from_fov(*size, 60), pose, size, batch=False)
            batched = render_perspective(scene, intrinsics_from_fov(*size, 60), pose, size)
            np.testing.assert_array_equal(batched[2], exact[2])
            np.testing.assert_array_equal(batched[0], exact[0])
            np.testing.assert_allclose(batched[1], exact[1], rtol=1e-6)
            self.assertGreater((exact[2] >= 0).mean(), .3)
            # One sliver whose only candidate pixel center lies outside it draws nothing.
            sliver = {"chunks": [{"tri": np.array([[[.6, .6, 1], [1.3, .6, 1], [.6, 1.3, 1]]]),
                                  "primitive": {"attributes": {}}}],
                      "materials": [], "shading": [((np.ones(3), None, 0), None)], "face_offsets": [0]}
            empty = render_perspective(sliver, np.diag([1., 1, 1]), np.eye(4), (4, 4))
            self.assertTrue((empty[2] == -1).all())

    def test_draco_glb_decodes_to_the_same_triangles(self):
        try:
            import DracoPy
        except ModuleNotFoundError:
            self.skipTest("DracoPy is optional")
        corners, faces = _box([0, 0, 0], [2, 1, 1])
        encoded = DracoPy.encode(corners, faces.reshape(-1, 3), quantization_bits=16)
        blob = bytearray(encoded)
        doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
               "extensionsUsed": ["KHR_draco_mesh_compression"],
               "extensionsRequired": ["KHR_draco_mesh_compression"],
               "meshes": [{"primitives": [{"attributes": {"POSITION": 1}, "indices": 0, "extensions": {
                   "KHR_draco_mesh_compression": {"bufferView": 0, "attributes": {"POSITION": 0}}}}]}],
               "buffers": [{"byteLength": len(blob)}],
               "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(blob)}],
               "accessors": [{"componentType": 5125, "count": 36, "type": "SCALAR"},
                             {"componentType": 5126, "count": 8, "type": "VEC3",
                              "min": [0, 0, 0], "max": [2, 1, 1]}]}
        with tempfile.TemporaryDirectory() as directory:
            write_glb(Path(directory) / "draco.glb", doc, blob)
            self.assertEqual(decompress(Path(directory) / "draco.glb", Path(directory) / "plain.glb"), 12)
            _, _, chunks = load_geometry(Path(directory) / "plain.glb")
            decoded = np.sort(chunks[0]["tri"].reshape(-1, 3), axis=0)
            np.testing.assert_allclose(decoded, np.sort(corners[faces].astype(float), axis=0), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
