"""Find the photographed object in each photo from depth and camera poses (CPU only).

Step 3 of the CPU photos-to-object pipeline. Reads a photo bundle and the
``photo_poses.py`` output, never ``truth/`` (except with ``--score``):

1. Back-project every registered photo's corrected depth into the shared frame.
2. Estimate "up" as the mean image-up direction of the cameras (casual photos
   are held roughly upright).
3. Find near-horizontal planes by RANSAC and take the lowest large one as the
   support surface (usually the floor).
4. Rasterize points above the support onto a floor grid. A cell is kept when at
   least two photos see something there and almost nothing in it is taller
   than ``--max-height``. A grid opening then removes structures thinner than
   ``--thin`` (partly seen walls). The object is the connected group of kept
   cells nearest the point all cameras aim at.
5. A pixel is in a photo's mask when its 3D point lies above the support inside
   the object's cells.

Masks are binary PNGs at ``masks/view_XX.png`` (XX = bundle image index); a
folder of hand-drawn or GPU-made masks in that layout is a drop-in substitute.

This needs consistent multi-view depth. On the bed capture the masks reach a
median IoU of 0.77 with true depth and 0.69 with the depth model after its
per-photo correction (worst photo 0.36; see README).

Heights are in the poses' estimated meters. Limits: the lowest large plane is
chosen, so an object on a table is merged with the table when the floor is
visible; furniture touching the object is merged with it; objects taller than
``--max-height`` are cut away, and parts thinner than ``--thin`` (legs, poles)
are dropped.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from recon_metrics import mask_iou

MASKS_FORMAT = "photo-masks-v1"


def back_project(depth, intrinsics, cam_to_world, stride=1):
    """World points ``(h, w, 3)`` for every ``stride``-th pixel; NaN where depth is missing."""
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[ys, xs].astype(float)
    z[~(z > 0)] = np.nan
    camera = np.stack(((xs - intrinsics[0, 2]) / intrinsics[0, 0] * z,
                       (ys - intrinsics[1, 2]) / intrinsics[1, 1] * z, z), axis=-1)
    return camera @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def camera_up(poses):
    """Mean image-up direction (camera -Y) of all poses, normalized."""
    up = -np.mean([pose[:3, 1] for pose in poses], axis=0)
    return up / np.linalg.norm(up)


def attention_point(poses):
    """Least-squares point closest to every camera's optical axis."""
    system, target = np.zeros((3, 3)), np.zeros(3)
    for pose in poses:
        direction = pose[:3, 2]
        projector = np.eye(3) - np.outer(direction, direction)
        system += projector
        target += projector @ pose[:3, 3]
    return np.linalg.lstsq(system, target, rcond=None)[0]


def horizontal_planes(points, up, threshold=0.05, max_tilt=25.0, iterations=400, count=3, seed=0):
    """Up to ``count`` near-horizontal planes found by repeated RANSAC.

    Each plane is ``(normal, offset, inliers)`` with ``normal`` oriented along
    ``up`` and height ``normal @ p - offset``. Inliers are removed between searches.
    """
    rng = np.random.default_rng(seed)
    remaining = points[np.isfinite(points).all(axis=1)]
    planes = []
    min_cos = np.cos(np.radians(max_tilt))
    for _ in range(count):
        if len(remaining) < 3:
            break
        best = None
        for _ in range(iterations):
            sample = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(normal)
            if length < 1e-12:
                continue
            normal /= length
            if normal @ up < 0:
                normal = -normal
            if normal @ up < min_cos:
                continue
            inliers = np.abs(remaining @ normal - normal @ sample[0]) < threshold
            if best is None or inliers.sum() > best[1].sum():
                best = (normal, inliers)
        if best is None:
            break
        support = remaining[best[1]]
        # Least-squares refit of the inliers, keeping the normal oriented upward.
        center = support.mean(axis=0)
        normal = np.linalg.svd(support - center, full_matrices=False)[2][2]
        normal = normal if normal @ up > 0 else -normal
        planes.append((normal, float(normal @ center), int(best[1].sum())))
        remaining = remaining[~best[1]]
    return planes


def support_plane(planes, min_fraction=0.1):
    """The lowest plane holding at least ``min_fraction`` of the largest plane's inliers."""
    if not planes:
        return None
    largest = max(plane[2] for plane in planes)
    candidates = [plane for plane in planes if plane[2] >= min_fraction * largest]
    reference = candidates[0][0]
    return min(candidates, key=lambda plane: plane[1] / (plane[0] @ reference))


def _plane_basis(normal):
    reference = np.array([1., 0, 0]) if abs(normal[0]) < .9 else np.array([0, 1., 0])
    a = np.cross(normal, reference)
    a /= np.linalg.norm(a)
    return a, np.cross(normal, a)


def object_cells(clouds, plane, center, cell=0.05, min_height=0.03, max_height=2.0, max_tall_fraction=0.05,
                 min_views=2, thin=0.2):
    """Grid cells (on the support plane) occupied by the object nearest ``center``.

    ``clouds`` is a list of ``(n, 3)`` world points, one per photo. Returns a
    dict with the grid geometry and a boolean ``cells`` array, or None.
    """
    normal, offset, _ = plane
    a, b = _plane_basis(normal)
    stacked = np.concatenate(clouds)
    heights = stacked @ normal - offset
    above = np.isfinite(heights) & (heights > min_height)
    if not above.any():
        return None
    coords = np.column_stack((stacked @ a, stacked @ b))
    low = coords[above].min(axis=0) - cell
    shape = tuple(np.ceil((coords[above].max(axis=0) - low) / cell).astype(int) + 2)
    counts, tall = np.zeros(shape, int), np.zeros(shape, int)
    views = np.zeros(shape, int)
    start = 0
    for cloud in clouds:
        end = start + len(cloud)
        part = above[start:end]
        index = np.floor((coords[start:end][part] - low) / cell).astype(int)
        np.add.at(counts, tuple(index.T), 1)
        np.add.at(tall, tuple(index.T), heights[start:end][part] > max_height)
        seen = np.zeros(shape, bool)
        seen[tuple(index.T)] = True
        views += seen
        start = end
    kept = (views >= min_views) & (tall <= max_tall_fraction * np.maximum(counts, 1))
    # Walls seen only partway up form thin lines on the grid; an opening removes
    # structures thinner than ``thin`` and cuts thin bridges between objects.
    size = int(round(thin / cell)) | 1  # odd, so the opening does not shift the grid

    kept = cv2.morphologyEx(kept.astype(np.uint8), cv2.MORPH_OPEN, np.ones((size, size), np.uint8)) > 0
    count, labels = cv2.connectedComponents(kept.astype(np.uint8), connectivity=8)
    if count < 2:
        return None
    target = np.floor((np.array([center @ a, center @ b]) - low) / cell).astype(int)
    occupied = np.argwhere(labels > 0)
    nearest = occupied[np.argmin(np.linalg.norm(occupied - target, axis=1))]
    return {"origin": low, "cell": cell, "axes": (a, b), "cells": labels == labels[tuple(nearest)],
            "min_height": min_height, "max_height": max_height}


def mask_from_points(points, plane, grid):
    """Pixels whose 3D point is above the support inside the object's cells."""
    normal, offset, _ = plane
    heights = points @ normal - offset
    a, b = grid["axes"]
    index = np.floor((np.stack((points @ a, points @ b), axis=-1) - grid["origin"]) / grid["cell"])
    valid = np.isfinite(heights) & np.isfinite(index).all(axis=-1)
    valid &= (heights > grid["min_height"]) & (heights <= grid["max_height"])
    index = np.where(valid[..., None], index, 0).astype(int)
    shape = np.array(grid["cells"].shape)
    valid &= ((index >= 0) & (index < shape)).all(axis=-1)
    mask = np.zeros(valid.shape, bool)
    mask[valid] = grid["cells"][index[valid][:, 0], index[valid][:, 1]]
    # Remove speckle and fill pinholes left by depth noise.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)


def estimate_masks(bundle_dir, poses_dir, output, cell=0.05, min_height=0.03, max_height=2.0, thin=0.2):
    """Write ``masks/view_XX.png`` for registered photos and ``masks.json``."""
    bundle_dir, poses_dir, output = Path(bundle_dir), Path(poses_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; choose a new output folder.")
    poses = json.loads((poses_dir / "poses.json").read_text())
    views = [(i, view) for i, view in enumerate(poses["views"]) if view["cam_to_world"] is not None]
    if len(views) < 2:
        raise ValueError("At least two registered photos are needed.")
    matrices = {i: (np.array(view["intrinsics"]), np.array(view["cam_to_world"])) for i, view in views}
    depths = {i: np.load(poses_dir / view["depth"]) * view.get("depth_correction", 1.0) for i, view in views}
    clouds = [back_project(depths[i], *matrices[i], stride=4).reshape(-1, 3) for i, _ in views]
    up = camera_up([pose for _, pose in matrices.values()])
    center = attention_point([pose for _, pose in matrices.values()])
    sample = np.concatenate([cloud[::4] for cloud in clouds])
    planes = horizontal_planes(sample, up)
    plane = support_plane(planes)
    if plane is None:
        raise ValueError("No near-horizontal support surface was found.")
    grid = object_cells(clouds, plane, center, cell, min_height, max_height, thin=thin)
    if grid is None:
        raise ValueError("Nothing was found standing on the support surface.")
    (output / "masks").mkdir(parents=True)
    records = []
    for index, view in views:
        points = back_project(depths[index], *matrices[index])
        mask = mask_from_points(points, plane, grid)
        name = f"masks/view_{index:02d}.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(output / name)
        records.append({"view": index, "image": view["image"], "mask": name, "pixels": int(mask.sum())})
    normal, offset, inliers = plane
    footprint = grid["cells"].sum() * cell ** 2
    result = {"format": MASKS_FORMAT, "bundle": str(bundle_dir), "poses": str(poses_dir),
              "up": up.tolist(), "attention_point": center.tolist(),
              "support_plane": {"normal": normal.tolist(), "offset": offset, "inliers_sampled": inliers,
                                "tilt_from_up_degrees": float(np.degrees(np.arccos(np.clip(normal @ up, -1, 1))))},
              "planes_found": [{"height_along_up": float(o / (n @ up)), "inliers_sampled": c} for n, o, c in planes],
              "object": {"footprint_area": float(footprint), "cell": cell, "min_height": min_height,
                         "max_height": max_height, "thin": thin},
              "views": records, "units": poses["units"]}
    (output / "masks.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def score(output, bundle_dir):
    """Per-photo IoU, precision and recall against truth masks; writes ``score.json``."""
    output, bundle_dir = Path(output), Path(bundle_dir)
    result = json.loads((output / "masks.json").read_text())
    truth = json.loads((bundle_dir / "truth" / "truth.json").read_text())
    rows = []
    for record in result["views"]:
        predicted = np.asarray(Image.open(output / record["mask"])) > 0
        with np.load(bundle_dir / truth["views"][record["view"]]["arrays"]) as arrays:
            true_mask = arrays["mask"]
        overlap = (predicted & true_mask).sum()
        rows.append({"view": record["view"], "iou": mask_iou(predicted, true_mask),
                     "precision": float(overlap / max(predicted.sum(), 1)),
                     "recall": float(overlap / max(true_mask.sum(), 1))})
    report = {"views": rows, "median_iou": float(np.median([row["iou"] for row in rows])),
              "min_iou": float(min(row["iou"] for row in rows))}
    (output / "score.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True, help="Folder containing bundle.json")
    parser.add_argument("--poses", type=Path, required=True, help="photo_poses.py output folder")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output folder")
    parser.add_argument("--cell", type=float, default=0.05, help="Floor grid cell size (estimated meters)")
    parser.add_argument("--min-height", type=float, default=0.03, help="Ignore points this close to the support")
    parser.add_argument("--max-height", type=float, default=2.0, help="Cells with taller content count as walls")
    parser.add_argument("--thin", type=float, default=0.2, help="Drop grid structures thinner than this")
    parser.add_argument("--score", action="store_true", help="After estimating, compare with truth/")
    args = parser.parse_args()
    result = estimate_masks(args.bundle, args.poses, args.output, args.cell, args.min_height, args.max_height,
                            args.thin)
    print(f"Masked {len(result['views'])} photos; object footprint "
          f"{result['object']['footprint_area']:.2f} square units into {args.output}.")
    if args.score:
        report = score(args.output, args.bundle)
        print(f"Mask IoU median {report['median_iou']:.3f}, worst {report['min_iou']:.3f}")
        for row in report["views"]:
            print(f"  view {row['view']:02d}: IoU {row['iou']:.3f}  precision {row['precision']:.3f}  "
                  f"recall {row['recall']:.3f}")


if __name__ == "__main__":
    main()
