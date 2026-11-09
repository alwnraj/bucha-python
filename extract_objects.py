"""Extract labeled objects from an outdoor GLB scan, offline on CPU.

1. Render street-level and aerial views from inside the GLB, with the source
   face behind every pixel.
2. Label each view with a pinned ADE20K SegFormer, grouped into a fixed list of
   common outdoor classes, and vote the labels onto source faces.
3. Smooth the votes over mesh neighbours (unseen faces borrow neighbour labels).
4. Split each class into separate objects; ground-type classes stay one object.
5. Export one textured GLB, an open collision surface and a closed voxel
   collider per object, plus ``objects.json`` and a review contact sheet.

Labels are model proposals, not validated identities. Textured objects keep
scan holes and unscanned backs; closed colliders fill them conservatively.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from render_views import intrinsics_from_fov, load_scene, look_at, render_perspective
from segment_objects import _basis_from_up, estimate_ground_plane


# Fixed outdoor classes, each built from ADE20K class names. Anything else
# (indoor classes, sky, misfires) counts as "other" and stays unknown.
OUTDOOR_CLASSES = {
    "road": ["road", "runway", "dirt track"],
    "sidewalk": ["sidewalk", "path"],
    "ground": ["earth", "land", "field", "floor", "hill", "mountain", "rug"],
    "grass": ["grass"],
    "sand": ["sand"],
    "water": ["water", "sea", "river", "lake", "swimming pool", "waterfall"],
    "building": ["building", "house", "skyscraper", "hovel", "tower", "windowpane", "door",
                 "screen door", "ceiling", "blind", "curtain"],
    "wall": ["wall"],
    "fence": ["fence", "railing", "bannister"],
    "stairs": ["stairs", "stairway", "step"],
    "bridge": ["bridge", "pier"],
    "column": ["column"],
    "awning": ["awning", "canopy", "tent", "booth"],
    "pole": ["pole"],
    "streetlight": ["streetlight"],
    "traffic_light": ["traffic light"],
    "sign": ["signboard", "trade name", "poster", "bulletin board"],
    "bench": ["bench", "seat"],
    "trash_can": ["ashcan"],
    "box_or_barrel": ["box", "barrel", "basket", "bag", "tank"],
    "flag": ["flag"],
    "fountain_or_statue": ["fountain", "sculpture"],
    "car": ["car"],
    "truck": ["truck"],
    "bus": ["bus"],
    "van": ["van"],
    "bicycle": ["bicycle"],
    "motorbike": ["minibike"],
    "boat": ["boat", "ship"],
    "person": ["person"],
    "animal": ["animal"],
    "tree": ["tree", "palm"],
    "plant": ["plant", "flower"],
    "rock": ["rock"],
}
CLASSES = list(OUTDOOR_CLASSES) + ["other"]
OTHER = len(CLASSES) - 1
# Surfaces kept as one object per class instead of split into instances.
STUFF = {"road", "sidewalk", "ground", "grass", "sand", "water"}
LIMITATIONS = [
    "Labels are SegFormer ADE20K proposals voted onto faces; semantic accuracy is unvalidated.",
    "SegFormer labels classes, not instances: touching objects of one class merge into one object.",
    "Mesh splits follow label boundaries, so object edges are only as clean as the votes.",
    "Textured objects keep scan holes and unscanned backs; nothing is invented.",
    "Closed colliders extrude each voxel column down to the object's lowest point and fill overhangs.",
    "Coordinates stay in source GLB units and axes; physical scale and robot registration are unknown.",
    "Rubble and debris have no model class and usually end up as ground, building or unknown.",
]


def ground_frame(chunks):
    """Ground plane whose "up" puts the larger horizontal area at the bottom.

    The shared estimator gives the vertical axis, but its sign and height can
    land on a flat roof above sloped ground (or below a flat floor). Outdoors,
    ground usually has more horizontal area than roofs, so the sign is chosen
    from that and the plane sits at the low end of the horizontal surfaces.
    """
    ground = estimate_ground_plane(chunks)
    triangles = np.concatenate([chunk["tri"] for chunk in chunks])
    up = ground["up"]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(normals, axis=1) / 2
    horizontal = np.abs(normals @ up) >= np.cos(np.radians(35)) * np.maximum(2 * area, 1e-30)
    heights = triangles.mean(axis=1) @ up
    low, high = np.percentile(heights, 2), np.percentile(heights, 98)
    quarter = (high - low) / 4
    if area[horizontal & (heights >= high - quarter)].sum() > area[horizontal & (heights <= low + quarter)].sum():
        up, heights = -up, -heights
    flat = horizontal if horizontal.any() else np.ones(len(heights), bool)
    offset = -float(np.percentile(heights[flat], 5))
    ground["right"], ground["forward"], ground["up"] = _basis_from_up(up)
    ground["normal"] = ground["up"]
    ground["offset"] = offset
    ground["residual_centers"] = heights + offset
    ground["residual_vertices"] = triangles @ ground["up"] + offset
    ground["ground_mask"] = horizontal & (np.abs(ground["residual_centers"]) <= ground["ground_band"])
    ground["scene_height"] = max(float(np.percentile(ground["residual_centers"], 98)), ground["ground_band"])
    return ground


class SegFormer:
    """Pinned ONNX SegFormer returning probabilities per outdoor class."""

    def __init__(self, model_dir):
        from infer_labels import _load_model

        self.session, self.processor, labels, self.entry = _load_model(model_dir)
        lookup = {ade: CLASSES.index(name) for name, ades in OUTDOOR_CLASSES.items() for ade in ades}
        self.group = np.zeros((len(CLASSES), len(labels)), dtype=np.float32)
        for class_id, name in labels.items():
            self.group[lookup.get(name.strip(), OTHER), class_id] = 1
        self.model_sha256 = self.entry["files"]["model.onnx"][1]

    def __call__(self, rgb):
        size = self.processor["size"]
        resized = np.asarray(Image.fromarray(rgb).resize((size["width"], size["height"]),
                                                         Image.Resampling.BILINEAR), dtype=np.float32)
        image = resized * float(self.processor["rescale_factor"])
        image = (image - np.asarray(self.processor["image_mean"], dtype=np.float32)) / np.asarray(
            self.processor["image_std"], dtype=np.float32)
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        logits = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0][0]
        logits -= logits.max(axis=0, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        classes, height, width = probabilities.shape
        return (self.group @ probabilities.reshape(classes, -1)).reshape(len(CLASSES), height, width)


def plan_cameras(triangles, ground, eye_height=1.6, positions=40, headings=6, aerial=12, fov=75.0):
    """Street-level cameras above open ground plus two aerial rings.

    Returns a list of 4x4 camera-to-world poses (OpenCV axes).
    """
    up, right, forward = ground["up"], ground["right"], ground["forward"]
    centers = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    heights = ground["residual_centers"]
    flat = np.column_stack((centers @ right, centers @ forward))
    low, high = np.percentile(flat, 1, axis=0), np.percentile(flat, 99, axis=0)
    floor = (np.abs(normals @ up) >= np.cos(np.radians(35))) & (heights < max(eye_height, ground["scene_height"] / 4))
    poses = []
    if positions > 0 and floor.any():
        spacing = np.sqrt(np.prod(np.maximum(high - low, 1e-9)) / positions)
        cells = np.floor((flat[floor] - low) / spacing).astype(np.int64)
        _, cell_index = np.unique(cells, axis=0, return_inverse=True)
        floor_points = centers[floor]
        eyes = []
        for cell in range(cell_index.max() + 1):
            points = floor_points[cell_index.ravel() == cell]
            middle = points[np.argmin(np.linalg.norm(points - np.median(points, axis=0), axis=1))]
            eye = middle + up * eye_height
            # Skip spots inside walls or rubble: no surface may come this close.
            if np.linalg.norm(centers - eye, axis=1).min() >= .5 * eye_height:
                eyes.append(eye)
        if len(eyes) > positions:
            eyes = [eyes[i] for i in np.linspace(0, len(eyes) - 1, positions).astype(int)]
        pitch = np.radians(-10)
        for eye in eyes:
            for k in range(headings):
                yaw = 2 * np.pi * k / headings
                direction = np.cos(pitch) * (np.cos(yaw) * right + np.sin(yaw) * forward) + np.sin(pitch) * up
                poses.append(look_at(eye, eye + direction, up))
    middle = np.percentile(centers, 50, axis=0)
    radius = max(np.linalg.norm(high - low) / 2, 1e-6)
    distance = radius / np.tan(np.radians(fov) / 2) * 1.1
    for elevation in (np.radians(30), np.radians(55)):
        for k in range(aerial):
            yaw = 2 * np.pi * (k + .5 * (elevation > .6)) / aerial
            offset = np.cos(elevation) * (np.cos(yaw) * right + np.sin(yaw) * forward) + np.sin(elevation) * up
            poses.append(look_at(middle + distance * offset, middle, up))
    return poses


def vote_view(faces, probabilities, votes, seen, min_pixels=4):
    """Add one view: every visible face gets its mean class distribution.

    Faces seen by fewer than ``min_pixels`` pixels get proportionally less weight.
    """
    visible = faces >= 0
    if not visible.any():
        return
    ys, xs = np.nonzero(visible)
    ids = faces[visible]
    height, width = faces.shape
    _, out_h, out_w = probabilities.shape
    sampled = probabilities[:, ys * out_h // height, xs * out_w // width]
    counts = np.bincount(ids, minlength=len(votes))
    hit = np.flatnonzero(counts)
    weight = np.minimum(1.0, counts[hit] / min_pixels)
    for class_id in range(votes.shape[1]):
        sums = np.bincount(ids, weights=sampled[class_id], minlength=len(votes))
        votes[hit, class_id] += weight * sums[hit] / counts[hit]
    seen[hit] += weight


def face_adjacency(triangles):
    """Pairs of faces sharing a full edge (exact world positions), both directions."""
    _, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    vertices = inverse.reshape(-1, 3)
    edges = np.sort(np.concatenate([vertices[:, [0, 1]], vertices[:, [1, 2]], vertices[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(triangles)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    a, b = owner[:-1][same], owner[1:][same]
    keep = a != b
    return np.concatenate([a[keep], b[keep]]), np.concatenate([b[keep], a[keep]])


def _neighbor_sum(values, source, target, length):
    return np.stack([np.bincount(source, weights=values[target, k], minlength=length)
                     for k in range(values.shape[1])], axis=1).astype(np.float32)


def smooth_labels(votes, seen, adjacency, smooth=2, fill=200):
    """Return ``(label, confidence)`` per face from votes smoothed over neighbours."""
    source, target = adjacency
    count = len(votes)
    degree = np.bincount(source, minlength=count).astype(np.float32)
    totals = votes.sum(axis=1, keepdims=True)
    values = (votes / np.maximum(totals, 1e-12)).astype(np.float32)
    values *= np.minimum(1, np.asarray(seen, dtype=np.float32) / 2)[:, None]
    for _ in range(smooth):
        values += _neighbor_sum(values, source, target, count)
        values /= (1 + degree)[:, None]
    for _ in range(fill):  # unseen faces borrow from labeled neighbours
        empty = values.sum(axis=1) <= 1e-9
        if not empty.any():
            break
        edges = empty[source]  # only edges into still-empty faces
        borrowed = _neighbor_sum(values, source[edges], target[edges], count)[empty]
        borrowed /= np.maximum(degree[empty], 1)[:, None]
        if not (borrowed.sum(axis=1) > 1e-9).any():
            break
        values[empty] = borrowed
    mass = values.sum(axis=1)
    label = np.where(mass > 1e-9, values.argmax(axis=1), OTHER)
    confidence = np.where(mass > 1e-9, values.max(axis=1) / np.maximum(mass, 1e-12), 0)
    return label, confidence


def _components(labels, adjacency):
    """Connected same-label face groups; returns a component id per face."""
    source, target = adjacency
    parent = np.arange(len(labels))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    same = labels[source] == labels[target]
    for a, b in zip(source[same].tolist(), target[same].tolist()):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(len(labels))])
    return np.unique(roots, return_inverse=True)[1]


def split_objects(labels, adjacency, triangles, min_faces=20, gap=0.0):
    """Return an object id per face (0 = unknown) and ``[(id, class name)]``.

    Components smaller than ``min_faces`` take the label of the neighbour they
    share most edges with. Ground-type classes become one object per class;
    other classes become one object per connected component, joining
    components of one class whose bounding boxes lie within ``gap``.
    """
    labels = labels.copy()
    source, target = adjacency
    for _ in range(3):
        component = _components(labels, adjacency)
        sizes = np.bincount(component)
        small = sizes[component] < min_faces
        border = small[source] & (component[source] != component[target])
        if not border.any():
            break
        pairs, counts = np.unique(np.column_stack((component[source][border], target[border])),
                                  axis=0, return_counts=True)
        # Best neighbouring label per small component, by shared edge count.
        votes = {}
        for (comp, face), count in zip(pairs.tolist(), counts.tolist()):
            key = (comp, int(labels[face]))
            votes[key] = votes.get(key, 0) + count
        best = {}
        for (comp, label), count in votes.items():
            if comp not in best or count > best[comp][1]:
                best[comp] = (label, count)
        lookup = np.full(len(sizes), -1)
        for comp, (label, _) in best.items():
            lookup[comp] = label
        change = small & (lookup[component] >= 0)
        if not change.any():
            break
        labels[change] = lookup[component[change]]
    component = _components(labels, adjacency)
    sizes = np.bincount(component)
    order = np.argsort(component, kind="stable")
    starts = np.searchsorted(component[order], np.arange(len(sizes)))
    box_low = np.minimum.reduceat(triangles.min(axis=1)[order], starts) - gap
    box_high = np.maximum.reduceat(triangles.max(axis=1)[order], starts) + gap
    comp_label = labels[order][starts]
    group_of = np.zeros(len(sizes), dtype=np.int64)  # component -> object (0 = none)
    objects = []
    for class_id in np.unique(labels):
        if class_id == OTHER:
            continue
        name = CLASSES[class_id]
        comps = np.flatnonzero(comp_label == class_id)
        if name in STUFF:
            groups = [comps]
        else:
            boxes = [(box_low[comp], box_high[comp]) for comp in comps]
            parent = list(range(len(comps)))

            def find(i):
                while parent[i] != i:
                    i = parent[i]
                return i

            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    if np.all(boxes[i][0] <= boxes[j][1]) and np.all(boxes[j][0] <= boxes[i][1]):
                        parent[find(j)] = find(i)
            roots = np.array([find(i) for i in range(len(comps))])
            groups = [comps[roots == root] for root in np.unique(roots)]
        for group in groups:
            if sizes[group].sum() >= min_faces:
                objects.append((name, group, int(sizes[group].sum())))
    for new_id, (name, group, _) in enumerate(objects, start=1):
        group_of[group] = new_id
    return _name_objects(group_of[component], [None] + [name for name, _, _ in objects])


def _name_objects(object_id, classes):
    """Renumber objects by class then size; ``classes[i]`` is object i's class (index 0 unused)."""
    sizes = np.bincount(object_id, minlength=len(classes))
    alive = [i for i in range(1, len(classes)) if sizes[i]]
    alive.sort(key=lambda i: (classes[i], -sizes[i]))
    remap = np.zeros(len(classes), dtype=np.int64)
    result, counter = [], {}
    for new_id, old in enumerate(alive, start=1):
        remap[old] = new_id
        name = classes[old]
        counter[name] = counter.get(name, 0) + 1
        result.append((new_id, name if name in STUFF else f"{name}_{counter[name]:02d}", name))
    return remap[object_id], result


# Thin vertical structures that a post or pole is often split into.
THIN = {"pole", "column", "streetlight", "sign", "traffic_light", "flag", "tree"}


def merge_fragments(object_id, named, confidence, adjacency, heights=None, share=0.75, ratio=0.1,
                    stuff_confidence=0.5, stuff_height=0.5):
    """Merge split-up pieces into whole objects; returns ``(object_id, named)`` like ``split_objects``.

    * Touching pieces of thin structures (``THIN``) become one object, named
      after the most common non-tree class among them (a post with a
      tree-labeled piece stays a pole).
    * A non-ground object whose boundary is at least ``share`` with one
      neighbour, and which has at most ``ratio`` of that neighbour's faces,
      joins it. Ground-type neighbours only absorb flat pieces (height span
      under ``stuff_height`` along up, from per-face ``heights``) whose mean
      label confidence is below ``stuff_confidence``, so shrubs and posts on a
      lawn stay.
    Repeats until nothing changes.
    """
    object_id = object_id.copy()
    classes = [None] + [label for _, _, label in named]
    source, target = adjacency
    for _ in range(20):
        count = len(classes)
        sizes = np.bincount(object_id, minlength=count)
        mean_conf = np.bincount(object_id, weights=confidence, minlength=count) / np.maximum(sizes, 1)
        span = np.zeros(count)
        if heights is not None:
            top, bottom = np.full(count, -np.inf), np.full(count, np.inf)
            np.maximum.at(top, object_id, heights.max(axis=1))
            np.minimum.at(bottom, object_id, heights.min(axis=1))
            span = np.where(sizes > 0, top - bottom, 0)
        cross = object_id[source] != object_id[target]
        pairs, counts = np.unique(np.column_stack((object_id[source][cross], object_id[target][cross])),
                                  axis=0, return_counts=True)
        boundary = np.bincount(pairs[:, 0], weights=counts, minlength=count)
        into = np.arange(count)
        # Thin structures: union touching pieces.
        parent = list(range(count))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for (a, b) in pairs:
            if a and b and classes[a] in THIN and classes[b] in THIN:
                parent[find(a)] = find(b)
        roots = {}
        for i in range(1, count):
            if sizes[i] and classes[i] in THIN:
                roots.setdefault(find(i), []).append(i)
        for members in roots.values():
            if len(members) < 2:
                continue
            totals = {}
            for i in members:
                totals[classes[i]] = totals.get(classes[i], 0) + sizes[i]
            posts = {name: n for name, n in totals.items() if name != "tree"}
            label = max(posts or totals, key=(posts or totals).get)
            keeper = max(members, key=lambda i: sizes[i])
            classes[keeper] = label
            into[members] = keeper
        # Fragments: join the neighbour that holds most of the boundary.
        if (into == np.arange(count)).all():
            best = {}
            for (a, b), n in zip(pairs, counts):
                if a and b and (a not in best or n > best[a][1]):
                    best[a] = (b, n)
            targets = set()
            for a in sorted(best, key=lambda i: sizes[i]):
                b, n = best[a]
                if classes[a] in STUFF or a in targets or into[b] != b or n < share * boundary[a]:
                    continue
                if sizes[a] > ratio * sizes[b]:
                    continue
                if classes[b] in STUFF and (mean_conf[a] >= stuff_confidence or span[a] >= stuff_height):
                    continue
                into[a] = b
                targets.add(b)
        if (into == np.arange(count)).all():
            break
        object_id = into[object_id]
    return _name_objects(object_id, classes)


def closed_collider(triangles, ground, resolution=48):
    """Watertight voxel solid: each occupied column extruded to the object's bottom."""
    from photo_carve import voxel_surface

    basis = np.stack([ground["right"], ground["forward"], ground["up"]])
    local = triangles @ basis.T
    low, high = local.reshape(-1, 3).min(axis=0), local.reshape(-1, 3).max(axis=0)
    size = max(float((high - low).max()) / resolution, ground["edge_scale"] / 2, 1e-9)
    shape = np.maximum(np.ceil((high - low) / size).astype(int), 1)
    # Sample each triangle densely enough to hit every voxel it crosses.
    edges = np.linalg.norm(local - np.roll(local, 1, axis=1), axis=2).max(axis=1)
    steps = np.clip(np.ceil(edges / (size / 2)).astype(int), 1, 64)
    occupied = np.zeros(shape, dtype=bool)
    for n in np.unique(steps):
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        weights = np.column_stack((i[keep], j[keep], n - i[keep] - j[keep])) / n
        points = np.einsum("pk,tkd->tpd", weights, local[steps == n]).reshape(-1, 3)
        cells = np.clip(((points - low) / size).astype(int), 0, shape - 1)
        occupied[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    top = np.where(occupied.any(axis=2), shape[2] - 1 - np.argmax(occupied[:, :, ::-1], axis=2), -1)
    occupied = np.arange(shape[2])[None, None, :] <= top[:, :, None]
    vertices, faces = voxel_surface(occupied, low, size)
    return vertices @ basis, faces


def _render_object(path, ground, size=(200, 150)):
    scene = load_scene(path)
    points = np.concatenate([chunk["tri"].reshape(-1, 3) for chunk in scene["chunks"]])
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) / 2, 1e-6)
    direction = (-.6 * ground["right"] - .6 * ground["forward"] + .55 * ground["up"])
    direction /= np.linalg.norm(direction)
    eye = center + direction * radius / np.tan(np.radians(25)) * 1.05
    rgb, _, _ = render_perspective(scene, intrinsics_from_fov(*size, 50), look_at(eye, center, ground["up"]), size)
    return rgb


def contact_sheet(output, objects, ground, columns=6, limit=60):
    """One labeled thumbnail per object (largest first) for quick review."""
    shown = sorted(objects, key=lambda obj: -obj["triangles"])[:limit]
    tile_w, tile_h, caption = 200, 150, 30
    rows = max(1, -(-len(shown) // columns))
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + caption)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for k, obj in enumerate(shown):
        x, y = (k % columns) * tile_w, (k // columns) * (tile_h + caption)
        sheet.paste(Image.fromarray(_render_object(output / obj["glb"], ground)), (x, y))
        draw.text((x + 4, y + tile_h + 2), obj["name"], fill=(0, 0, 0))
        draw.text((x + 4, y + tile_h + 15), f"{obj['triangles']:,} faces, conf {obj['confidence_mean']:.2f}",
                  fill=(90, 90, 90))
    sheet.save(output / "objects-sheet.png")


def label_scene(source, work, predict, eye_height=1.6, positions=40, headings=6, size=(640, 480), fov=75.0,
                keep_views=24):
    """Render, predict and vote; cached in ``work/votes.npz`` per source, model and camera settings."""
    scene = load_scene(source)
    triangles = np.concatenate([chunk["tri"] for chunk in scene["chunks"]])
    ground = ground_frame(scene["chunks"])
    settings = {"source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "model_sha256": getattr(predict, "model_sha256", "custom"), "classes": CLASSES,
                "eye_height": eye_height, "positions": positions, "headings": headings,
                "size": list(size), "fov": fov}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    cache = work / "votes.npz"
    if cache.is_file():
        with np.load(cache) as data:
            if str(data["key"]) == key:
                return scene, triangles, ground, data["votes"], data["seen"], int(data["views"]), settings
    poses = plan_cameras(triangles, ground, eye_height, positions, headings, fov=fov)
    intrinsics = intrinsics_from_fov(*size, fov)
    votes = np.zeros((len(triangles), len(CLASSES)), dtype=np.float32)
    seen = np.zeros(len(triangles), dtype=np.float32)
    (work / "views").mkdir(parents=True, exist_ok=True)
    palette = np.random.default_rng(3).integers(40, 255, (len(CLASSES), 3)).astype(np.uint8)
    palette[OTHER] = 0
    saved = set(np.linspace(0, len(poses) - 1, min(keep_views, len(poses))).astype(int).tolist())
    started = perf_counter()
    for index, pose in enumerate(poses):
        rgb, _, faces = render_perspective(scene, intrinsics, pose, size)
        if (faces >= 0).mean() >= .02:  # skip views that show almost nothing
            probabilities = predict(rgb)
            vote_view(faces, probabilities, votes, seen)
            if index in saved:
                classes = probabilities.argmax(axis=0)
                labels = np.asarray(Image.fromarray(palette[classes]).resize(size, Image.Resampling.NEAREST))
                labels = np.where((faces >= 0)[..., None], labels, 0)
                Image.fromarray(np.concatenate([rgb, labels], axis=1)).save(work / "views" / f"view_{index:03d}.png")
        if index % 10 == 9 or index == len(poses) - 1:
            print(f"  view {index + 1}/{len(poses)}  ({perf_counter() - started:.0f}s)", flush=True)
    np.savez_compressed(cache, key=key, votes=votes, seen=seen, views=len(poses))
    return scene, triangles, ground, votes, seen, len(poses), settings


def extract(source, output, model_dir=None, predict=None, eye_height=1.6, positions=40, headings=6,
            size=(640, 480), min_faces=20, sheet=True, crop_textures=True):
    """Run the whole pipeline; returns the ``objects.json`` contents."""
    from collision import export_collisions, write_mesh_collisions
    from segment_objects import segment

    source, output = Path(source), Path(output)
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    predict = predict or SegFormer(model_dir)
    print("[1/4] Rendering views and voting labels onto faces...", flush=True)
    scene, triangles, ground, votes, seen, view_count, settings = label_scene(
        source, work, predict, eye_height, positions, headings, size)
    print("[2/4] Smoothing labels and splitting objects...", flush=True)
    adjacency = face_adjacency(triangles)
    labels, confidence = smooth_labels(votes, seen, adjacency)
    object_id, named = split_objects(labels, adjacency, triangles, min_faces, gap=ground["edge_scale"])
    object_id, named = merge_fragments(object_id, named, confidence, adjacency, ground["residual_vertices"])
    print(f"[3/4] Exporting {len(named)} objects (textured GLB + collision surfaces)...", flush=True)
    manifest = segment(source, output, face_objects={
        "object_index": object_id, "confidence": confidence, "method": "street_and_aerial_view_votes_v1",
        "objects": [{"name": name, "semantic_class": label} for _, name, label in named]})
    export_collisions(source, output)
    if crop_textures:
        from texture_crop import repack_textures

        image_bytes = [repack_textures(output / obj["asset"]) for obj in manifest["objects"]]
        manifest["per_object_textures"] = (
            "Object GLBs keep only the texels they use, packed into new images with shifted UVs "
            "(JPEG re-encoded at quality 92). segmented-scene.glb keeps the exact source textures and UVs.")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  textures: {sum(b for b, _ in image_bytes) / 1e6:.0f} MB -> "
              f"{sum(a for _, a in image_bytes) / 1e6:.0f} MB across object GLBs", flush=True)
    by_name = {name: new_id for new_id, name, _ in named}
    basis = np.stack([ground["right"], ground["forward"], ground["up"]])
    closed, objects = [], []
    for obj in manifest["objects"]:
        members = np.flatnonzero(object_id == by_name.get(obj["name"], 0))
        tri = triangles[members]
        local = tri.reshape(-1, 3) @ basis.T
        entry = {"name": obj["name"], "semantic_class": obj["semantic_class"], "triangles": int(len(members)),
                 "confidence_mean": float(confidence[members].mean()) if len(members) else 0.0,
                 "seen_face_fraction": float((seen[members] > 0).mean()) if len(members) else 0.0,
                 "position_xyz": ((tri.reshape(-1, 3).min(axis=0) + tri.reshape(-1, 3).max(axis=0)) / 2).tolist(),
                 "bounds_3d": {"min": tri.reshape(-1, 3).min(axis=0).tolist(),
                               "max": tri.reshape(-1, 3).max(axis=0).tolist()},
                 "size_ground_frame": (local.max(axis=0) - local.min(axis=0)).tolist(),
                 "base_height_above_ground": float(local[:, 2].min() + ground["offset"]),
                 "glb": obj["asset"], "collider_open": f"colliders/{obj['name']}.obj"}
        if obj["semantic_class"] != "unknown" and obj["semantic_class"] not in STUFF:
            vertices, faces = closed_collider(tri, ground)
            closed.append({"id": obj["id"], "name": obj["name"], "vertices": vertices, "faces": faces,
                           "semantic_class": obj["semantic_class"], "closed_solid": True,
                           "status": "voxel_column_fill_of_labeled_surface"})
            entry["collider_closed"] = f"closed/colliders/{obj['name']}.obj"
        objects.append(entry)
    stale = output / "closed" / "colliders"  # earlier runs may have had other objects
    if stale.is_dir():
        for path in stale.iterdir():
            if path.suffix in (".obj", ".npz"):
                path.unlink()
    write_mesh_collisions(output / "closed", closed, manifest["coordinate_frame"],
                          limitations=[LIMITATIONS[4], LIMITATIONS[5]])
    print("[4/4] Writing objects.json and the review sheet...", flush=True)
    if sheet:
        contact_sheet(output, [o for o in objects if o["semantic_class"] not in STUFF | {"unknown"}], ground)
    summary = {
        "source_file": str(source.resolve()), "source_sha256": settings["source_sha256"],
        "method": "street_and_aerial_view_votes_v1",
        "model": {"repository": getattr(predict, "entry", {}).get("repository", "custom"),
                  "model_sha256": settings["model_sha256"]},
        "classes": CLASSES[:-1], "stuff_classes": sorted(STUFF),
        "views": view_count, "camera_settings": {k: settings[k] for k in ("eye_height", "positions", "headings",
                                                                            "size", "fov")},
        "faces": len(triangles), "faces_seen": int((seen > 0).sum()),
        "faces_labeled": int((object_id > 0).sum()),
        "ground_frame": {"up": ground["up"].tolist(), "right": ground["right"].tolist(),
                         "forward": ground["forward"].tolist(), "offset": ground["offset"]},
        "units": "source GLB units", "object_count": len(objects) - any(o["semantic_class"] == "unknown"
                                                                         for o in objects),
        "objects": objects, "limitations": LIMITATIONS,
    }
    (output / "objects.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Source GLB")
    parser.add_argument("--output", type=Path, required=True, help="Result directory")
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parent / "models" / "segformer-b5-ade20k")
    parser.add_argument("--eye-height", type=float, default=1.6,
                        help="Street camera height in source units (glTF files are nominally meters)")
    parser.add_argument("--positions", type=int, default=40, help="Street-level camera positions")
    parser.add_argument("--headings", type=int, default=6, help="Views per street position")
    parser.add_argument("--min-faces", type=int, default=20, help="Smallest object kept, in faces")
    parser.add_argument("--no-sheet", action="store_true", help="Skip the review contact sheet")
    parser.add_argument("--full-textures", action="store_true",
                        help="Keep the whole source texture in every object GLB")
    args = parser.parse_args()
    started = perf_counter()
    try:
        summary = extract(args.input, args.output, args.model_dir, eye_height=args.eye_height,
                          positions=args.positions, headings=args.headings, min_faces=args.min_faces,
                          sheet=not args.no_sheet, crop_textures=not args.full_textures)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Extraction stopped: {error}\n")
    counts = {}
    for obj in summary["objects"]:
        counts[obj["semantic_class"]] = counts.get(obj["semantic_class"], 0) + 1
    print(f"\n{summary['object_count']} objects from {summary['views']} views in {perf_counter() - started:.0f}s:")
    print("  " + ", ".join(f"{name} {count}" for name, count in sorted(counts.items())))
    print(f"Faces seen {summary['faces_seen']:,} / labeled {summary['faces_labeled']:,} of {summary['faces']:,}. "
          f"Results: {args.output}")


if __name__ == "__main__":
    main()
