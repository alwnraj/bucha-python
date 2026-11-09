"""Simulate a casual RGB photo capture of one GLB object, with hidden ground truth.

The output folder has two parts:

* ``bundle.json`` + ``images/``: the only input a photos-only pipeline may read.
  ``focal_length_px`` stands in for EXIF focal length and can be withheld.
* ``truth/``: cameras, depth, target masks and target faces, for scoring only.

The target is every source face whose centroid lies inside an axis-aligned box.
Cameras orbit the target center and look at it. Distances and depths are in
source GLB units; the source scale is not verified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from render_views import intrinsics_from_fov, load_scene, look_at, render_perspective

BUNDLE_FORMAT = "photo-bundle-v1"
AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def select_target(scene, box_min, box_max):
    """Boolean mask over global faces whose centroid lies inside the box."""
    centers = np.concatenate([chunk["tri"].mean(axis=1) for chunk in scene["chunks"]])
    box_min, box_max = np.asarray(box_min, dtype=float), np.asarray(box_max, dtype=float)
    if (box_min >= box_max).any():
        raise ValueError("Every box minimum must be smaller than its maximum.")
    return ((centers >= box_min) & (centers <= box_max)).all(axis=1)


def orbit_cameras(center, radius, count, size, fov_x=60.0, elevation=25.0, up=(0, 0, 1), distance=None):
    """Intrinsics and camera-to-world poses on a ring around ``center``.

    Without ``distance``, cameras stand far enough for a sphere of ``radius``
    to fit the narrower field of view with a 10% margin.
    """
    if count < 1:
        raise ValueError("At least one view is required.")
    if not -89 <= elevation <= 89:
        raise ValueError("Elevation must stay between -89 and 89 degrees.")
    width, height = size
    intrinsics = intrinsics_from_fov(width, height, fov_x)
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)
    if distance is None:
        half_fov = min(np.arctan((width / 2) / intrinsics[0, 0]), np.arctan((height / 2) / intrinsics[1, 1]))
        distance = 1.1 * radius / np.sin(half_fov)
    reference = np.array([1., 0, 0]) if abs(up[0]) < .9 else np.array([0, 1., 0])
    a = np.cross(up, reference)
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    tilt = np.radians(elevation)
    poses = []
    for k in range(count):
        azimuth = 2 * np.pi * k / count
        direction = np.cos(tilt) * (np.cos(azimuth) * a + np.sin(azimuth) * b) + np.sin(tilt) * up
        poses.append(look_at(center + distance * direction, center, up))
    return intrinsics, poses, float(distance)


def simulate(input_glb, output, box_min, box_max, views=8, size=(640, 480), fov_x=60.0,
             elevation=25.0, up="z", distance=None, focal_hint=True, min_mask_pixels=200):
    """Render the capture and return a short summary dict."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; choose a new output folder.")
    scene = load_scene(input_glb)
    target = select_target(scene, box_min, box_max)
    if not target.any():
        raise ValueError("No source faces have a centroid inside the target box.")
    all_triangles = np.concatenate([chunk["tri"] for chunk in scene["chunks"]])
    target_triangles = all_triangles[target]
    points = target_triangles.reshape(-1, 3)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = float(np.linalg.norm(points - center, axis=1).max())
    intrinsics, poses, distance = orbit_cameras(center, radius, views, size, fov_x, elevation, AXES[up], distance)

    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "truth").mkdir()
    images, truth_views, warnings = [], [], []
    for index, pose in enumerate(poses):
        rgb, depth, face = render_perspective(scene, intrinsics, pose, size)
        mask = face >= 0
        mask[mask] = target[face[mask]]
        name = f"view_{index:02d}"
        Image.fromarray(rgb).save(output / "images" / f"{name}.png")
        np.savez_compressed(output / "truth" / f"{name}.npz", depth=depth, mask=mask, face=face)
        Image.fromarray((mask * 255).astype(np.uint8)).save(output / "truth" / f"{name}_mask.png")
        image = {"file": f"images/{name}.png", "width": size[0], "height": size[1]}
        if focal_hint:
            image["focal_length_px"] = float(intrinsics[0, 0])
        images.append(image)
        pixels = int(mask.sum())
        if pixels < min_mask_pixels:
            warnings.append(f"{name}: only {pixels} target pixels visible; the camera may be blocked.")
        truth_views.append({"image": image["file"], "cam_to_world": pose.tolist(),
                            "arrays": f"truth/{name}.npz", "mask_png": f"truth/{name}_mask.png",
                            "target_pixels": pixels})
    np.savez_compressed(output / "truth" / "target.npz", faces=np.flatnonzero(target).astype(np.int64),
                        triangles=target_triangles)

    bundle = {"format": BUNDLE_FORMAT, "images": images,
              "notes": ["Only images and optional focal lengths are pipeline input.",
                        "focal_length_px stands in for EXIF data and may be absent."]}
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    truth = {
        "source": Path(input_glb).name,
        "source_sha256": hashlib.sha256(Path(input_glb).read_bytes()).hexdigest(),
        "units": "source GLB units (scale unverified)", "up": AXES[up],
        "camera_convention": "OpenCV pinhole; +X right, +Y down, +Z forward; pixel centers at integer coordinates",
        "intrinsics": intrinsics.tolist(), "views": truth_views,
        "target": {"box_min": list(map(float, box_min)), "box_max": list(map(float, box_max)),
                   "faces": int(target.sum()), "center": center.tolist(), "radius": radius,
                   "file": "truth/target.npz"},
        "capture": {"views": views, "fov_x_degrees": fov_x, "elevation_degrees": elevation,
                    "distance": distance, "focal_hint": focal_hint},
        "depth_meaning": "camera-axis Z in source units; NaN where nothing was rendered",
        "warnings": warnings,
    }
    (output / "truth" / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return {"output": str(output), "target_faces": int(target.sum()), "distance": distance,
            "target_pixels": [view["target_pixels"] for view in truth_views], "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Source GLB")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output folder")
    parser.add_argument("--box", type=float, nargs=6, required=True,
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                        help="Target box in source GLB world coordinates")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=float, default=60.0, help="Horizontal field of view in degrees")
    parser.add_argument("--elevation", type=float, default=25.0, help="Camera height angle in degrees")
    parser.add_argument("--distance", type=float, help="Camera distance; default fits the target")
    parser.add_argument("--up", choices=tuple(AXES), default="z")
    parser.add_argument("--no-focal-hint", action="store_true", help="Withhold focal length, like stripped EXIF")
    args = parser.parse_args()
    box = args.box
    summary = simulate(args.input, args.output, box[:3], box[3:], args.views, (args.width, args.height),
                       args.fov, args.elevation, args.up, args.distance, not args.no_focal_hint)
    print(f"Rendered {args.views} views of {summary['target_faces']} target faces at distance "
          f"{summary['distance']:.3f} into {summary['output']}.")
    print("Target pixels per view:", summary["target_pixels"])
    for warning in summary["warnings"]:
        print("Warning:", warning)


if __name__ == "__main__":
    main()
