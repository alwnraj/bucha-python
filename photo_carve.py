"""Carve a voxel model of the object from per-photo masks and camera poses (CPU only).

Step 4 of the CPU photos-to-object pipeline. Reads ``poses.json`` from
``photo_poses.py`` and a mask folder in the ``masks/view_XX.png`` layout (from
``photo_masks.py``, drawn by hand, or made by the later GPU version):

1. A coarse cube around the point every camera aims at is carved, then the
   grid is shrunk to what survived and carved again at full resolution.
2. Each voxel center is projected into every photo that has a mask. A photo
   "sees" the voxel when it lands inside the image in front of the camera.
   The voxel stays when at least ``--agree`` of the photos that see it put it
   inside the mask (voting, so one bad mask cannot erase the object) and at
   least ``--min-views`` photos see it (default: three quarters of them). A
   photo whose frame misses a voxel casts no vote, so without that rule
   slivers behind the object, seen inside the mask by only a few photos, survive.
3. When ``masks.json`` gives a support plane, voxels below it are removed;
   cameras looking down from one ring cannot carve under the object otherwise.
4. The surface between kept and empty voxels becomes a closed triangle mesh,
   written in the ``collision-scene.json`` format (``collision.py`` can ray-cast it).

Carving gives the visual hull: the object plus any concave part no photo sees
past (for example the space under a bed or inside a bowl). ``--score``
aligns the estimated cameras to ``truth/`` and compares the mesh with the
hidden target surface.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from collision import write_mesh_collisions
from photo_masks import attention_point
from recon_metrics import chamfer_f1, fit_similarity, points_in_voxels, sample_surface

CARVE_FORMAT = "photo-carve-v1"


def load_views(poses_dir, masks_dir, dilate=2):
    """Registered photos that have a mask: ``[(index, intrinsics, cam_to_world, mask)]``."""
    import cv2

    poses = json.loads((Path(poses_dir) / "poses.json").read_text())
    views = []
    for index, view in enumerate(poses["views"]):
        path = Path(masks_dir) / "masks" / f"view_{index:02d}.png"
        if view["cam_to_world"] is None or not path.is_file():
            continue
        mask = np.asarray(Image.open(path).convert("L")) > 127
        if dilate > 0:
            # Small growth so slightly misplaced cameras do not carve away object edges.
            kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        views.append((index, np.array(view["intrinsics"]), np.array(view["cam_to_world"]), mask))
    return views, poses


def vote(centers, views):
    """Per voxel: how many photos see it and how many of those put it inside the mask."""
    seen = np.zeros(len(centers), dtype=np.int32)
    inside = np.zeros(len(centers), dtype=np.int32)
    for _, intrinsics, pose, mask in views:
        camera = (centers - pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        front = z > 1e-6
        safe = np.where(front, z, 1.0)
        x = np.rint(camera[:, 0] / safe * intrinsics[0, 0] + intrinsics[0, 2]).astype(np.int64)
        y = np.rint(camera[:, 1] / safe * intrinsics[1, 1] + intrinsics[1, 2]).astype(np.int64)
        visible = front & (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
        seen += visible
        inside[visible] += mask[y[visible], x[visible]]
    return seen, inside


def carve_grid(origin, voxel_size, shape, views, agree, min_views, support=None):
    """Boolean occupancy for a regular grid; ``support`` is ``(normal, offset)`` with height ``normal @ p - offset``."""
    axes = [origin[k] + (np.arange(shape[k]) + 0.5) * voxel_size for k in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    seen, inside = vote(centers, views)
    keep = (seen >= min_views) & (inside >= agree * np.maximum(seen, 1))
    if support is not None:
        keep &= centers @ support[0] - support[1] > 0
    return keep.reshape(shape), seen.reshape(shape)


def carve(views, agree=0.85, min_views=None, resolution=128, coarse=48, support=None, center=None, half=None):
    """Two-pass carving. Returns a dict with ``occupied``, ``origin``, ``voxel_size`` and ``seen``."""
    if len(views) < 2:
        raise ValueError("At least two registered photos with masks are needed.")
    if min_views is None:
        min_views = max(2, int(np.ceil(0.75 * len(views))))
    poses = [pose for _, _, pose, _ in views]
    center = attention_point(poses) if center is None else np.asarray(center, float)
    if half is None:
        half = float(np.median([np.linalg.norm(pose[:3, 3] - center) for pose in poses]))
    size = 2 * half / coarse
    occupied, _ = carve_grid(center - half, size, (coarse,) * 3, views, agree, min_views, support)
    if not occupied.any():
        raise ValueError("Every voxel was carved away; check masks and poses.")
    found = np.argwhere(occupied)
    # Shrink to the surviving block plus one coarse voxel, then carve finely.
    low = center - half + (found.min(axis=0) - 1) * size
    high = center - half + (found.max(axis=0) + 2) * size
    voxel_size = float((high - low).max() / resolution)
    shape = tuple(int(n) for n in np.ceil((high - low) / voxel_size))
    occupied, seen = carve_grid(low, voxel_size, shape, views, agree, min_views, support)
    return {"occupied": occupied, "origin": low, "voxel_size": voxel_size, "seen": seen}


def voxel_surface(occupied, origin, voxel_size):
    """Closed triangle mesh of the faces between occupied and empty voxels (outward winding)."""
    padded = np.pad(occupied, 1)
    corners = {}
    vertices, faces = [], []

    def vertex(key):
        if key not in corners:
            corners[key] = len(vertices)
            vertices.append(key)
        return corners[key]

    for axis in range(3):
        u, v = [k for k in range(3) if k != axis]
        for step in (1, -1):
            neighbour = np.roll(padded, -step, axis=axis)
            exposed = np.argwhere(padded & ~neighbour) - 1
            for cell in exposed:
                base = cell.copy()
                if step == 1:
                    base[axis] += 1
                quad = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    corner = base.copy()
                    corner[u] += du
                    corner[v] += dv
                    quad.append(vertex(tuple(corner)))
                # The quad's normal is +axis when (u, v, axis) is a cyclic order of (0, 1, 2).
                if (step == 1) == ((u, v) in ((1, 2), (2, 0), (0, 1))):
                    faces += [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
                else:
                    faces += [[quad[0], quad[2], quad[1]], [quad[0], quad[3], quad[2]]]
    vertices = np.asarray(origin, float) + np.array(vertices, dtype=float).reshape(-1, 3) * voxel_size
    return vertices, np.array(faces, dtype=np.int64).reshape(-1, 3)


def estimate(poses_dir, masks_dir, output, agree=0.85, min_views=None, resolution=128, dilate=2):
    """Carve, then write ``voxels.npz``, ``carve.json`` and the collision mesh."""
    poses_dir, masks_dir, output = Path(poses_dir), Path(masks_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; choose a new output folder.")
    views, poses = load_views(poses_dir, masks_dir, dilate)
    support, center = None, None
    info_path = masks_dir / "masks.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        if "support_plane" in info:
            plane = info["support_plane"]
            support = (np.array(plane["normal"]), float(plane["offset"]))
        center = info.get("attention_point")
    result = carve(views, agree, min_views, resolution, support=support, center=center)
    vertices, faces = voxel_surface(result["occupied"], result["origin"], result["voxel_size"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "voxels.npz", occupied=result["occupied"], origin=result["origin"],
                        voxel_size=result["voxel_size"], seen=result["seen"].astype(np.int16))
    write_mesh_collisions(output, [{"id": 1, "name": "carved_object", "vertices": vertices, "faces": faces,
                                    "closed_solid": True}],
                          f"{poses['frame']}, {poses['units']}",
                          limitations=["Visual hull: unseen concave space is filled.",
                                       "Scale comes from monocular metric depth and is not measured."])
    summary = {"format": CARVE_FORMAT, "poses": str(poses_dir), "masks": str(masks_dir),
               "views": [index for index, *_ in views], "agree": agree, "min_views": min_views,
               "dilate_pixels": dilate, "support_plane_used": support is not None,
               "grid_shape": list(result["occupied"].shape), "voxel_size": result["voxel_size"],
               "origin": result["origin"].tolist(), "occupied_voxels": int(result["occupied"].sum()),
               "volume": float(result["occupied"].sum() * result["voxel_size"] ** 3),
               "mesh": {"vertices": len(vertices), "triangles": len(faces), "file": "colliders/carved_object.obj"},
               "units": poses["units"]}
    (output / "carve.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def score(output, poses_dir, bundle_dir, thresholds=(0.05, 0.1), samples=20000):
    """Align the estimated cameras to the truth, then compare mesh and target surface; writes ``score.json``."""
    output, poses_dir, bundle_dir = Path(output), Path(poses_dir), Path(bundle_dir)
    poses = json.loads((poses_dir / "poses.json").read_text())
    truth = json.loads((bundle_dir / "truth" / "truth.json").read_text())
    registered = [i for i, view in enumerate(poses["views"]) if view["cam_to_world"] is not None]
    scale, rotation, translation = fit_similarity([np.array(poses["views"][i]["cam_to_world"])[:3, 3] for i in registered],
                                                  [np.array(truth["views"][i]["cam_to_world"])[:3, 3] for i in registered])
    with np.load(output / "colliders" / "carved_object.npz") as mesh:
        vertices = scale * mesh["vertices"] @ rotation.T + translation
        faces = mesh["faces"]
    with np.load(bundle_dir / truth["target"]["file"]) as target:
        triangles = target["triangles"]
    truth_points = sample_surface(triangles, samples, seed=1)
    carved_points = sample_surface(vertices[faces], samples, seed=2)
    # Containment: share of the true surface inside the carved voxels (a visual hull should contain it).
    with np.load(output / "voxels.npz") as voxels:
        back = ((truth_points - translation) @ rotation) / scale
        contained = points_in_voxels(back, voxels["origin"], float(voxels["voxel_size"]), voxels["occupied"])
    true_box = triangles.reshape(-1, 3)
    report = {"alignment_scale_truth_per_estimate": scale,
              "surface": {str(t): chamfer_f1(carved_points, truth_points, t) for t in thresholds},
              "truth_surface_contained": float(contained.mean()),
              "box_size_truth": (true_box.max(axis=0) - true_box.min(axis=0)).tolist(),
              "box_size_carved_aligned": (vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
              "notes": ["The carved mesh is moved into the truth frame by the camera-center similarity fit.",
                        "The truth target is every source face inside the capture box (may include nearby parts).",
                        "Distances are in truth (source GLB) units."]}
    (output / "score.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses", type=Path, required=True, help="photo_poses.py output folder")
    parser.add_argument("--masks", type=Path, required=True, help="Folder containing masks/view_XX.png")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output folder")
    parser.add_argument("--agree", type=float, default=0.85, help="Share of seeing photos that must agree")
    parser.add_argument("--min-views", type=int, help="Photos that must see a kept voxel (default: 3/4 of them)")
    parser.add_argument("--resolution", type=int, default=128, help="Voxels along the longest side")
    parser.add_argument("--dilate", type=int, default=2, help="Grow masks by this many pixels")
    parser.add_argument("--bundle", type=Path, help="With --score: the bundle folder holding truth/")
    parser.add_argument("--score", action="store_true", help="After carving, compare with truth/")
    args = parser.parse_args()
    if args.score and args.bundle is None:
        parser.error("--score needs --bundle")
    summary = estimate(args.poses, args.masks, args.output, args.agree, args.min_views, args.resolution, args.dilate)
    print(f"Carved {summary['occupied_voxels']} voxels ({summary['volume']:.3f} cubic units) from "
          f"{len(summary['views'])} photos into {args.output}.")
    if args.score:
        report = score(args.output, args.poses, args.bundle)
        for threshold, row in report["surface"].items():
            print(f"Within {threshold}: precision {row['precision']:.3f}, recall {row['recall']:.3f}, F1 {row['f1']:.3f}")
        print(f"Chamfer {report['surface']['0.05']['chamfer']:.3f}; true surface inside carving "
              f"{report['truth_surface_contained']:.3f}; scale truth/estimate {report['alignment_scale_truth_per_estimate']:.3f}")


if __name__ == "__main__":
    main()
