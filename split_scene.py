"""Split a GLB scan into separate objects by geometry alone, no labels needed.

1. Find the scan's structure: large flat horizontal regions (floors, ground,
   flat roofs, ceilings) and large tall vertical regions (walls, facades).
2. Cut the structure away; what is left falls apart into connected pieces,
   one per object (a sofa, a table, a car, a bush).
3. Export one textured GLB per object, the structure as separate GLBs, a
   colour-coded scene, colliders, ``objects.json`` and a review sheet.

``--labels`` optionally names each object with the pinned SegFormer ADE20K
model (indoor and outdoor classes). Touching objects stay together, and
objects flush against a wall may be cut away with it. Distances and areas
are in source units (glTF files are nominally meters).
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from extract_objects import face_adjacency, ground_frame
from render_views import load_scene

STRUCTURE = ("structure_floor", "structure_walls", "structure_ceiling")
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "segformer-b5-ade20k"
LIMITATIONS = [
    "Objects are connected pieces left after removing large flat structure; they are not verified physical objects.",
    "Touching objects (a chair under a table, pillows on a bed) stay one object.",
    "Objects flush against a wall or floor may be cut away with the structure or come out thin.",
    "Large flat furniture (a long counter, a big table) can be classed as floor structure.",
    "Textured objects keep scan holes and unscanned backs; closed colliders fill them conservatively.",
    "Coordinates stay in source GLB units and axes; physical scale and robot registration are unknown.",
]


def _union_find(count, a, b):
    """Connected components from edge lists; returns a component id per item."""
    parent = np.arange(count)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for x, y in zip(a.tolist(), b.tolist()):
        x, y = find(x), find(y)
        if x != y:
            parent[max(x, y)] = min(x, y)
    roots = np.array([find(i) for i in range(count)])
    return np.unique(roots, return_inverse=True)[1]


def face_normals(triangles):
    """Unit normals and areas."""
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normals, axis=1)
    return normals / np.maximum(length, 1e-30)[:, None], length / 2


def smooth_normals(normals, area, adjacency, iterations=3, corner=30.0):
    """Edge-preserving neighbour average: each face only mixes with neighbours within ``corner`` degrees.

    Removes scan noise so flat regions hold together, without blending floors
    into walls across a corner, whatever the triangle size.
    """
    source, target = adjacency
    smoothed = normals.copy()
    for _ in range(iterations):
        alike = (smoothed[source] * smoothed[target]).sum(axis=1) >= np.cos(np.radians(corner))
        weight = area[target] * alike
        mixed = smoothed * area[:, None] + np.stack(
            [np.bincount(source, weights=smoothed[target, k] * weight, minlength=len(normals)) for k in range(3)],
            axis=1)
        smoothed = mixed / np.maximum(np.linalg.norm(mixed, axis=1), 1e-30)[:, None]
    return smoothed


def _levels(heights, area, component, candidates, big_area, tolerance):
    """Faces of ``candidates`` components at the height of a big component.

    Big components (at least ``big_area``) define levels by their area-weighted
    median height; any candidate component whose median height lies within
    ``tolerance`` of a level joins it (a small bathroom floor, a rug).
    """
    count = component.max() + 1
    members = np.flatnonzero(candidates)
    order = members[np.lexsort((heights[members], component[members]))]
    comps, starts = np.unique(component[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    median = np.full(count, np.nan)
    comp_area = np.bincount(component[members], weights=area[members], minlength=count)
    for comp, a, b in zip(comps, starts, ends):
        cumulative = np.cumsum(area[order[a:b]])
        median[comp] = heights[order[a:b]][np.searchsorted(cumulative, cumulative[-1] / 2)]
    levels = median[comp_area >= big_area]
    if not len(levels):
        return np.zeros(len(heights), bool)
    near = np.zeros(count, bool)
    valid = ~np.isnan(median)
    near[valid] = np.abs(median[valid][:, None] - levels[None, :]).min(axis=1) <= tolerance
    return candidates & near[component]


def find_structure(triangles, ground, adjacency, plane_area=4.0, level_tolerance=0.1, wall_area=2.5,
                   wall_height=1.8, tilt=25.0):
    """Return a structure code per face: 0 none, 1 floor/ground, 2 wall, 3 ceiling.

    Faces are sorted by their smoothed normal into up-facing, down-facing and
    vertical (within ``tilt`` degrees); only faces in the same group connect,
    so floors cannot chain into walls around a corner. Up-facing pieces of at
    least ``plane_area`` set floor levels, and any up-facing piece at a level
    (within ``level_tolerance``) is floor; ceilings likewise. A vertical piece
    of at least ``wall_area`` spanning ``wall_height`` is wall.
    """
    normals, area = face_normals(triangles)
    smoothed = smooth_normals(normals, area, adjacency)
    facing = smoothed @ ground["up"]
    group = np.zeros(len(triangles), dtype=np.int8)
    group[facing >= np.cos(np.radians(tilt))] = 1
    group[np.abs(facing) <= np.sin(np.radians(tilt))] = 2
    group[facing <= -np.cos(np.radians(tilt))] = 3
    source, target = adjacency
    same = (group[source] == group[target]) & (group[source] > 0)
    component = _union_find(len(triangles), source[same], target[same])
    heights = triangles.mean(axis=1) @ ground["up"]
    code = np.zeros(len(triangles), dtype=np.int8)
    code[_levels(heights, area, component, group == 1, plane_area, level_tolerance)] = 1
    code[_levels(heights, area, component, group == 3, plane_area, level_tolerance)] = 3
    count = component.max() + 1
    vertical = group == 2
    wall_piece_area = np.bincount(component[vertical], weights=area[vertical], minlength=count)
    vertex_heights = triangles @ ground["up"]
    top, bottom = np.full(count, -np.inf), np.full(count, np.inf)
    np.maximum.at(top, component[vertical], vertex_heights[vertical].max(axis=1))
    np.minimum.at(bottom, component[vertical], vertex_heights[vertical].min(axis=1))
    wall = (wall_piece_area >= wall_area) & (top - bottom >= wall_height)
    code[vertical & wall[component]] = 2
    return code


def _thickness(points_by_piece, piece, count):
    """Smallest spread of each piece's vertices (4 standard deviations along its thinnest axis)."""
    weights = np.bincount(piece, minlength=count).astype(float)
    mean = np.stack([np.bincount(piece, weights=points_by_piece[:, k], minlength=count) for k in range(3)], 1)
    mean /= np.maximum(weights, 1)[:, None]
    centered = points_by_piece - mean[piece]
    covariance = np.zeros((count, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            value = np.bincount(piece, weights=centered[:, i] * centered[:, j], minlength=count)
            covariance[:, i, j] = covariance[:, j, i] = value / np.maximum(weights, 1)
    return 4 * np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[:, 0], 0))


def split_objects(triangles, structure, adjacency, min_size=0.2, min_area=0.05, thin=0.05):
    """Connected non-structure pieces; returns ``(object id per face, updated structure codes)``.

    Pieces thinner than ``thin`` (baseboards, door frames, wall-edge slivers)
    join the structure they touch most. Pieces whose bounding-box diagonal is
    under ``min_size`` or whose surface is under ``min_area`` are leftovers
    (-1). Object ids run from 1, largest first by face count; 0 is structure.
    """
    structure = structure.copy()
    source, target = adjacency
    free = structure == 0
    keep = free[source] & free[target]
    piece = _union_find(len(triangles), source[keep], target[keep])
    count = piece.max() + 1
    vertex_piece = np.repeat(piece, 3)
    thickness = _thickness(triangles.reshape(-1, 3), vertex_piece, count)
    # Thin pieces become the structure type they share most edges with.
    touch = free[source] & (structure[target] > 0)
    votes = np.zeros((count, 4))
    np.add.at(votes, (piece[source[touch]], structure[target[touch]]), 1)
    sliver = (thickness < thin) & (votes[:, 1:].sum(axis=1) > 0)
    absorbed = free & sliver[piece]
    structure[absorbed] = votes[piece[absorbed], 1:].argmax(axis=1) + 1
    free = structure == 0
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    for k in range(3):
        np.minimum.at(low[:, k], piece, triangles[:, :, k].min(axis=1))
        np.maximum.at(high[:, k], piece, triangles[:, :, k].max(axis=1))
    _, area = face_normals(triangles)
    piece_area = np.bincount(piece[free], weights=area[free], minlength=count)
    sizes = np.bincount(piece[free], minlength=count)
    big = (np.linalg.norm(high - low, axis=1) >= min_size) & (piece_area >= min_area) & (sizes > 0)
    order = [c for c in np.argsort(-sizes) if big[c]]
    remap = np.full(count, -1)
    remap[order] = np.arange(1, len(order) + 1)
    return np.where(free, remap[piece], 0), structure


def label_objects(scene, triangles, ground, object_id, structure, model_dir, work, positions=40, headings=6,
                  size=(640, 480), fov=75.0, predict=None):
    """Name each object with SegFormer ADE20K (150 indoor and outdoor classes).

    Cameras stand on every detected floor (the most open spot per area), plus
    aerial rings when the scan has no substantial ceilings; each view's class
    probabilities are summed per object. Returns ``(labels, shares)`` indexed by
    object id (``None`` for objects no view saw). Votes are cached in ``work``.
    """
    import hashlib
    from extract_objects import SegFormer, plan_cameras
    from render_views import intrinsics_from_fov, render_perspective

    if predict is None:
        predict = SegFormer(model_dir)
        predict.group = np.eye(len(predict.labels), dtype=np.float32)  # keep all 150 classes
    names = [predict.labels[i].strip() for i in range(len(predict.labels))]
    count = int(object_id.max(initial=0))
    floor = structure == 1
    _, area = face_normals(triangles)
    # Indoors (real ceilings), aerial views only look into a roofless dollhouse.
    indoor = area[structure == 3].sum() >= .25 * area[floor].sum()
    poses = plan_cameras(triangles, ground, positions=positions, headings=headings, fov=fov,
                         aerial=0 if indoor else 12, floor=floor if floor.any() else None)
    key = hashlib.sha256(object_id.astype("<i4").tobytes() + np.asarray(poses).tobytes() + json.dumps(
        [predict.model_sha256, list(size), fov]).encode()).hexdigest()
    cache = Path(work) / "object-votes.npz"
    votes = None
    if cache.is_file():
        with np.load(cache) as data:
            if str(data["key"]) == key:
                votes = data["votes"]
    if votes is None:
        intrinsics = intrinsics_from_fov(*size, fov)
        votes = np.zeros((count + 1, len(names)))
        started = perf_counter()
        for index, pose in enumerate(poses):
            rgb, _, faces = render_perspective(scene, intrinsics, pose, size)
            visible = faces >= 0
            if visible.mean() >= .02:
                owners = np.where(visible, object_id[np.maximum(faces, 0)], 0)
                ys, xs = np.nonzero(owners > 0)
                if len(ys):
                    probabilities = predict(rgb)
                    _, out_h, out_w = probabilities.shape
                    sampled = probabilities[:, ys * out_h // size[1], xs * out_w // size[0]]
                    for k in range(len(names)):
                        votes[:, k] += np.bincount(owners[ys, xs], weights=sampled[k], minlength=count + 1)
            if index % 20 == 19 or index == len(poses) - 1:
                print(f"  view {index + 1}/{len(poses)}  ({perf_counter() - started:.0f}s)", flush=True)
        Path(work).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, key=key, votes=votes)
    totals = votes.sum(axis=1)
    labels = [names[k] if total > 0 else None for k, total in zip(votes.argmax(axis=1), totals)]
    shares = np.where(totals > 0, votes.max(axis=1) / np.maximum(totals, 1e-12), 0)
    return labels, shares


def write_colored_scene(path, triangles, groups, colors):
    """GLB with one flat-coloured primitive per group; ``groups`` is a group index per face (-1 = skip)."""
    from segment_objects import write_glb

    blob, views, accessors, primitives, materials = bytearray(), [], [], [], []

    def add(values, component, kind, target):
        blob.extend(b"\0" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": values.nbytes, "target": target})
        blob.extend(values.tobytes())
        accessors.append({"bufferView": len(views) - 1, "componentType": component, "count": len(values),
                          "type": kind})
        return len(accessors) - 1

    for group, color in enumerate(colors):
        points = np.ascontiguousarray(triangles[groups == group].reshape(-1, 3), dtype="<f4")
        if not len(points):
            continue
        position = add(points, 5126, "VEC3", 34962)
        accessors[position] |= {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}
        materials.append({"pbrMetallicRoughness": {"baseColorFactor": list(color) + [1.0], "metallicFactor": 0.0},
                          "doubleSided": True})
        primitives.append({"attributes": {"POSITION": position}, "material": len(materials) - 1})
    doc = {"asset": {"version": "2.0", "generator": "split_scene.py"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"mesh": 0}], "meshes": [{"primitives": primitives}], "materials": materials,
           "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors}
    write_glb(path, doc, blob)


def split_scene(source, output, min_size=0.2, min_area=0.05, thin=0.05, crop_textures=True, sheet=True,
                structure_options=None, labels=False, model_dir=None, positions=80, predict=None):
    """Run the whole split; returns the ``objects.json`` contents."""
    from extract_objects import export_objects

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    print("[1/4] Loading the scan and finding floors, walls and ceilings...", flush=True)
    scene = load_scene(source)
    triangles = np.concatenate([chunk["tri"] for chunk in scene["chunks"]])
    ground = ground_frame(scene["chunks"])
    adjacency = face_adjacency(triangles)
    structure = find_structure(triangles, ground, adjacency, **(structure_options or {}))
    print("[2/4] Splitting what is left into objects...", flush=True)
    pieces, structure = split_objects(triangles, structure, adjacency, min_size, min_area, thin)
    count = int(pieces.max(initial=0))
    width = max(3, len(str(count)))
    named = [(i, f"object_{i:0{width}d}", "object") for i in range(1, count + 1)]
    object_id = np.where(pieces > 0, pieces, 0)
    for code, name in enumerate(STRUCTURE, start=1):
        if (structure == code).any():
            named.append((len(named) + 1, name, name))
            object_id[structure == code] = len(named)
    names, shares = [None] * (count + 1), np.zeros(count + 1)
    if labels:
        print(f"  naming {count} objects with SegFormer ADE20K...", flush=True)
        names, shares = label_objects(scene, triangles, ground, np.maximum(pieces, 0), structure,
                                      model_dir or DEFAULT_MODEL, output / "work", positions=positions,
                                      predict=predict)
    print(f"[3/4] Exporting {count} objects and {len(named) - count} structure parts...", flush=True)

    def caption(entry):
        size = "{:.2f} x {:.2f} x {:.2f}".format(*entry["size_ground_frame"])
        number = int(entry["name"].split("_")[1])
        if names[number]:
            return f"{names[number]} ({shares[number]:.2f}), {size}"
        return f"{entry['triangles']:,} faces, {size}"

    basis = np.stack([ground["right"], ground["forward"], ground["up"]])
    manifest, objects = export_objects(
        source, output, triangles, ground, object_id, named, np.ones(len(triangles), dtype=np.float32),
        "geometry_split_v1", crop_textures=crop_textures, sheet=sheet,
        solid=lambda entry: entry["semantic_class"] == "object", collider_notes=LIMITATIONS[4:],
        caption=caption)
    for entry in objects:
        entry.pop("confidence_mean", None)  # geometry only: no label confidence
        entry.pop("semantic_class")
        entry["kind"] = ("object" if entry["name"].startswith("object_") else
                         "leftover" if entry["name"] == "unknown" else entry["name"].removeprefix("structure_"))
        if labels and entry["kind"] == "object":
            number = int(entry["name"].split("_")[1])
            entry["label"] = names[number]
            entry["label_share"] = round(float(shares[number]), 3)
    rng = np.random.default_rng(7)
    colors = [tuple(rng.uniform(.25, 1, 3)) for _ in range(count)] + [(.55, .5, .45), (.8, .8, .8), (.95, .95, .9)]
    groups = np.where(pieces > 0, pieces - 1, -1)
    for code in (1, 2, 3):
        groups[structure == code] = count + code - 1
    write_colored_scene(output / "scene-split.glb", triangles, groups, colors)
    summary = {
        "source_file": str(source.resolve()), "source_sha256": manifest["source_sha256"],
        "method": "geometry_split_v1", "faces": len(triangles),
        "structure_faces": {name.removeprefix("structure_"): int((structure == code).sum())
                            for code, name in enumerate(STRUCTURE, start=1)},
        "object_faces": int((pieces > 0).sum()), "leftover_faces": int((pieces < 0).sum()),
        "object_count": count, "settings": {"min_size": min_size, "min_area": min_area, "thin": thin},
        "labels": "SegFormer-B5 ADE20K vote share per object (unvalidated)" if labels else None,
        "ground_frame": {"up": ground["up"].tolist(), "right": ground["right"].tolist(),
                         "forward": ground["forward"].tolist(), "offset": ground["offset"]},
        "units": "source GLB units", "objects": objects, "limitations": LIMITATIONS,
        "elapsed_seconds": round(perf_counter() - started, 1),
    }
    (output / "objects.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Source GLB")
    parser.add_argument("--output", type=Path, required=True, help="Result directory")
    parser.add_argument("--min-size", type=float, default=0.2,
                        help="Smallest object kept: bounding-box diagonal in source units")
    parser.add_argument("--min-area", type=float, default=0.05,
                        help="Smallest object kept: surface area in source units squared")
    parser.add_argument("--thin", type=float, default=0.05,
                        help="Pieces thinner than this that touch walls or floors join them")
    parser.add_argument("--labels", action="store_true",
                        help="Also name each object with the pinned SegFormer ADE20K model (slower)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--positions", type=int, default=80, help="Camera spots for --labels")
    parser.add_argument("--no-sheet", action="store_true", help="Skip the review contact sheet")
    parser.add_argument("--full-textures", action="store_true",
                        help="Keep the whole source texture in every object GLB")
    args = parser.parse_args()
    try:
        summary = split_scene(args.input, args.output, args.min_size, args.min_area, args.thin,
                              crop_textures=not args.full_textures, labels=args.labels,
                              model_dir=args.model_dir, positions=args.positions,
                              sheet=not args.no_sheet)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Split stopped: {error}\n")
    print(f"\n{summary['object_count']} objects in {summary['elapsed_seconds']:.0f}s. "
          f"Structure faces: {summary['structure_faces']}. Results: {args.output}")


if __name__ == "__main__":
    main()
