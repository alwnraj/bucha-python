"""Estimate per-photo metric depth and camera poses from an RGB photo bundle (CPU only).

Step 2 of the CPU photos-to-object pipeline. Reads only ``bundle.json`` and
its images (see ``photo_sim.py``):

1. Intrinsics come from a known field of view (``--fov``), ``focal_length_px``
   (an EXIF stand-in) or, as a poor last resort, an assumed 60 degrees. The
   principal point is the image center. A wrong focal length badly bends the
   pose geometry, and it could not be recovered from the photos (see README).
2. Each photo gets depth from the pinned indoor/outdoor metric ONNX model.
3. For every photo pair, SIFT matches give an essential matrix, so rotation and
   travel direction come from image geometry alone.
4. Photos are placed one at a time. Each new photo's travel distance comes
   from 3D points it shares with already placed photos, so all cameras share
   one scale. Per-pair depth guesses (10-15% off per photo) are only a
   fallback. No bundle adjustment is done yet.
5. The depth model sets the one overall scale, and each photo's depth gets a
   multiplier (``depth_correction``) fitted to the triangulated points.

Results are in the root photo's camera frame; units are the depth model's
estimated meters. ``--score`` compares against ``truth/`` after estimation.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from download_metric_models import MODELS, ROOT as METRIC_ROOT
from image_depth import MEAN, STD, sha256
from recon_metrics import camera_errors

POSES_FORMAT = "photo-poses-v1"
ASSUMED_FOV_X = 60.0


def bundle_intrinsics(image, fov_x=None):
    """3x3 intrinsics for one bundle image entry and where the focal length came from.

    Priority: a known camera field of view (``fov_x``, from camera specs), then
    the bundle's ``focal_length_px`` (EXIF stand-in), then an assumed 60 degrees.
    """
    width, height = image["width"], image["height"]
    if fov_x is not None:
        focal, source = (width / 2) / np.tan(np.radians(fov_x) / 2), f"known {fov_x:g} degree FOV"
    elif "focal_length_px" in image:
        focal, source = float(image["focal_length_px"]), "focal_length_px"
    else:
        focal, source = (width / 2) / np.tan(np.radians(ASSUMED_FOV_X) / 2), f"assumed {ASSUMED_FOV_X:g} degree FOV"
    return np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1.]]), source


def load_depth_model(scene="indoor"):
    import onnxruntime as ort

    model = MODELS[scene]
    directory = METRIC_ROOT / scene
    graph = next(name for name in model["files"] if name.endswith(".onnx"))
    for name, expected in model["files"].items():
        if not (directory / name).is_file() or sha256(directory / name) != expected:
            raise FileNotFoundError(f"Missing or changed {scene} metric model; run download_metric_models.py.")
    session = ort.InferenceSession(str(directory / graph), providers=["CPUExecutionProvider"])
    return session, {"scene": scene, "repo": model["repo"], "revision": model["revision"], "file": graph}


def estimate_depth(session, rgb):
    """Metric depth (model meters) resized to the photo's own size."""
    height, width = rgb.shape[:2]
    pixels = np.asarray(Image.fromarray(rgb).resize((518, 518), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    tensor = ((pixels - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    raw = np.squeeze(session.run(None, {session.get_inputs()[0].name: tensor})[0])
    if raw.ndim != 2 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("The metric model returned an invalid depth map.")
    return np.asarray(Image.fromarray(raw.astype(np.float32)).resize((width, height), Image.Resampling.BILINEAR))


def detect_features(rgb, count=4000):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=count).detectAndCompute(gray, None)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2)
    return points, descriptors


def match_features(desc_a, desc_b, ratio=0.75):
    """Index pairs passing Lowe's ratio test in both directions."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty((0, 2), dtype=int)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def one_way(a, b):
        found = {}
        for pair in matcher.knnMatch(a, b, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                found[pair[0].queryIdx] = pair[0].trainIdx
        return found

    forward, backward = one_way(desc_a, desc_b), one_way(desc_b, desc_a)
    pairs = [(a, b) for a, b in forward.items() if backward.get(b) == a]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _sample(depth, points):
    height, width = depth.shape
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    return depth[y, x]


def relative_pose(points_a, points_b, intrinsics_a, intrinsics_b, depth_a, depth_b, min_inliers=30):
    """Pose of camera b relative to a, with ``X_b = R @ X_a + t`` in depth units.

    ``depth_a``/``depth_b`` are depth values at each matched point. Returns
    None when the geometry is too weak.
    """
    if len(points_a) < max(min_inliers, 8):
        return None
    # Normalize to a shared unit focal length so different intrinsics are allowed.
    norm_a = cv2.undistortPoints(points_a.reshape(-1, 1, 2), intrinsics_a, None).reshape(-1, 2)
    norm_b = cv2.undistortPoints(points_b.reshape(-1, 1, 2), intrinsics_b, None).reshape(-1, 2)
    pixel_threshold = 1.0 / np.sqrt(intrinsics_a[0, 0] * intrinsics_b[0, 0])
    essential, mask = cv2.findEssentialMat(norm_a, norm_b, np.eye(3), cv2.RANSAC, 0.999, pixel_threshold)
    if essential is None or essential.shape[0] < 3:
        return None
    count, rotation, direction, mask = cv2.recoverPose(essential[:3], norm_a, norm_b, np.eye(3), mask=mask)
    inliers = mask.ravel() > 0
    if count < min_inliers:
        return None
    direction = direction.ravel()
    projection_a = np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_b = np.hstack((rotation, direction[:, None]))
    homogeneous = cv2.triangulatePoints(projection_a, projection_b, norm_a[inliers].T, norm_b[inliers].T)
    in_a = (homogeneous[:3] / homogeneous[3]).T
    in_b = in_a @ rotation.T + direction
    ratios = np.concatenate((depth_a[inliers] / in_a[:, 2], depth_b[inliers] / in_b[:, 2]))
    usable = np.concatenate((in_a[:, 2] > 0, in_b[:, 2] > 0)) & np.isfinite(ratios) & (ratios > 0)
    if usable.sum() < min_inliers:
        return None
    half = len(in_a)
    scale_a = float(np.median(ratios[:half][usable[:half]])) if usable[:half].any() else np.nan
    scale_b = float(np.median(ratios[half:][usable[half:]])) if usable[half:].any() else np.nan
    ratios = ratios[usable]
    scale = float(np.median(ratios))
    # Parallax: median angle between the two viewing rays of each inlier, in degrees.
    ray_a = in_a / np.linalg.norm(in_a, axis=1, keepdims=True)
    ray_b = (in_a + rotation.T @ direction) / np.linalg.norm(in_a + rotation.T @ direction, axis=1, keepdims=True)
    parallax = float(np.degrees(np.median(np.arccos(np.clip((ray_a * ray_b).sum(axis=1), -1, 1)))))
    return {"rotation": rotation, "translation": direction * scale, "direction": direction,
            "inliers": int(count), "scale": scale, "scale_a": scale_a, "scale_b": scale_b,
            "scale_spread": float(np.median(np.abs(ratios / scale - 1))), "parallax_degrees": parallax}


def _invert(transform):
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _pair_edges(matches, keypoints, intrinsics, depths=None, min_inliers=30):
    """Run :func:`relative_pose` on every matched pair; depth ``None`` uses unit depth."""
    edges = {}
    for (a, b), pairs in matches.items():
        points_a, points_b = keypoints[a][pairs[:, 0]], keypoints[b][pairs[:, 1]]
        depth_a = np.ones(len(pairs)) if depths is None else _sample(depths[a], points_a)
        depth_b = np.ones(len(pairs)) if depths is None else _sample(depths[b], points_b)
        result = relative_pose(points_a, points_b, intrinsics[a], intrinsics[b], depth_a, depth_b, min_inliers)
        if result is not None:
            edges[(a, b)] = result
    return edges


def _matches_between(matches, a, b):
    """Keypoint index pairs ``(index in a, index in b)``."""
    if (a, b) in matches:
        return matches[(a, b)]
    if (b, a) in matches:
        return matches[(b, a)][:, ::-1]
    return np.empty((0, 2), dtype=int)


def _camera_points(pose, world):
    return (world - pose[:3, 3]) @ pose[:3, :3]


def _triangulate(a, b, poses, matches, keypoints, intrinsics, tracks, points, max_error=2.0, min_angle=2.0):
    """Add 3D points for matches between two placed photos that have no point yet."""
    pairs = _matches_between(matches, a, b)
    pairs = pairs[(tracks[a][pairs[:, 0]] < 0) & (tracks[b][pairs[:, 1]] < 0)]
    if not len(pairs):
        return 0
    pixels = [keypoints[a][pairs[:, 0]], keypoints[b][pairs[:, 1]]]
    projections = [intrinsics[v] @ _invert(poses[v])[:3] for v in (a, b)]
    homogeneous = cv2.triangulatePoints(projections[0], projections[1], pixels[0].T, pixels[1].T)
    with np.errstate(divide="ignore", invalid="ignore"):
        world = (homogeneous[:3] / homogeneous[3]).T
    keep = np.isfinite(world).all(axis=1)
    world[~keep] = 0
    rays = []
    for view, pixel in zip((a, b), pixels):
        camera = _camera_points(poses[view], world)
        keep &= camera[:, 2] > 1e-9
        z = np.where(keep, camera[:, 2], 1.0)[:, None]
        projected = camera[:, :2] / z * [intrinsics[view][0, 0], intrinsics[view][1, 1]] + intrinsics[view][:2, 2]
        keep &= np.linalg.norm(projected - pixel, axis=1) < max_error
        rays.append(camera @ poses[view][:3, :3].T)
    lengths = np.linalg.norm(rays[0], axis=1) * np.linalg.norm(rays[1], axis=1)
    keep &= (rays[0] * rays[1]).sum(axis=1) < np.cos(np.radians(min_angle)) * np.maximum(lengths, 1e-300)
    ids = np.arange(len(points), len(points) + keep.sum())
    points.extend(world[keep])
    tracks[a][pairs[keep, 0]] = ids
    tracks[b][pairs[keep, 1]] = ids
    return int(keep.sum())


def register_views(count, edges, matches, keypoints, intrinsics, min_shared=5):
    """Place photos one at a time with camera spacing carried by shared 3D points.

    The strongest pair starts the reconstruction (its baseline comes from the
    depth model). Each further photo joins through its strongest pair with a
    placed photo: rotation and travel direction come from that pair's essential
    matrix, and only the travel distance is solved, from 3D points the new
    photo shares with the placed one. This keeps one consistent scale across
    all photos instead of a separate depth-model guess per pair. When too few
    points are shared, the pair's depth baseline is used instead.

    ``edges`` maps ``(a, b)``, ``a < b``, to :func:`relative_pose` results and
    ``matches`` maps the same keys to keypoint index pairs. Returns
    ``(poses, root, points, tracks, placed_by)``; ``tracks[v][k]`` is the point
    index of keypoint ``k`` in photo ``v`` or -1.
    """
    poses, placed_by = [None] * count, [None] * count
    tracks = [np.full(len(points), -1) for points in keypoints]
    points = []
    if not edges:
        return poses, None, np.empty((0, 3)), tracks, placed_by
    (root, other), edge = max(edges.items(), key=lambda item: item[1]["inliers"])
    root_to_other = np.eye(4)
    root_to_other[:3, :3], root_to_other[:3, 3] = edge["rotation"], edge["translation"]
    poses[root], poses[other] = np.eye(4), _invert(root_to_other)
    placed_by[root], placed_by[other] = "start", f"depth baseline from view {root}"
    _triangulate(root, other, poses, matches, keypoints, intrinsics, tracks, points)
    while True:
        candidates = []
        for (i, j), edge in edges.items():
            if (poses[i] is None) != (poses[j] is None):
                placed, new = (i, j) if poses[i] is not None else (j, i)
                pairs = _matches_between(matches, new, placed)
                shared = pairs[tracks[placed][pairs[:, 1]] >= 0]
                candidates.append((len(shared) >= min_shared, edge["inliers"], placed, new, edge, shared))
        if not candidates:
            break
        _, _, placed, new, edge, shared = max(candidates, key=lambda c: c[:2])
        relative = np.eye(4)  # X_new = R @ X_placed + t with a unit-length t
        relative[:3, :3], relative[:3, 3] = edge["rotation"], edge["direction"]
        if placed > new:
            relative = _invert(relative)
        rotation = poses[placed][:3, :3] @ relative[:3, :3].T
        direction = -rotation @ relative[:3, 3]  # world direction from the placed to the new camera
        inverse = np.linalg.inv(intrinsics[new])
        distances = []
        for k_new, k_placed in shared:
            ray = rotation @ inverse @ np.array([*keypoints[new][k_new], 1.0])
            ray /= np.linalg.norm(ray)
            if np.linalg.norm(np.cross(direction, ray)) < np.sin(np.radians(1)):
                continue  # ray nearly parallel to the travel direction: distance undetermined
            # placed center + distance * direction + along * ray = shared 3D point
            target = points[tracks[placed][k_placed]] - poses[placed][:3, 3]
            (distance, along), *_ = np.linalg.lstsq(np.column_stack((direction, ray)), target, rcond=None)
            if along > 0:
                distances.append(distance)
        if len(distances) >= min_shared and np.median(distances) > 0:
            distance = float(np.median(distances))
            placed_by[new] = f"{len(distances)} shared points with view {placed}"
        else:
            distance = edge["scale"]
            placed_by[new] = f"depth baseline from view {placed}"
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, poses[placed][:3, 3] + distance * direction
        poses[new] = pose
        for view in range(count):
            if poses[view] is None or view == new:
                continue
            pairs = _matches_between(matches, new, view)
            known = pairs[(tracks[view][pairs[:, 1]] >= 0) & (tracks[new][pairs[:, 0]] < 0)]
            tracks[new][known[:, 0]] = tracks[view][known[:, 1]]
            _triangulate(new, view, poses, matches, keypoints, intrinsics, tracks, points)
    return poses, root, np.array(points, dtype=float).reshape(-1, 3), tracks, placed_by


def align_depth(poses, points, tracks, keypoints, depths, min_points=10):
    """Put the reconstruction in depth-model meters and fit each photo's depth to the 3D points.

    One overall scale makes the median model depth match the triangulated
    depth across all photos. Each photo then gets a multiplier that makes its
    own depth agree with the points it sees (photos seeing fewer than
    ``min_points`` keep 1.0). Returns ``(poses, points, corrections)``.
    """
    ratios = {}
    for view, pose in enumerate(poses):
        if pose is None:
            continue
        found = np.flatnonzero(tracks[view] >= 0)
        z = _camera_points(pose, points[tracks[view][found]])[:, 2]
        model = _sample(depths[view], keypoints[view][found])
        usable = (z > 0) & np.isfinite(model) & (model > 0)
        ratios[view] = model[usable] / z[usable]
    combined = np.concatenate(list(ratios.values())) if ratios else np.empty(0)
    scale = float(np.median(combined)) if len(combined) else 1.0
    scaled = [None if pose is None else pose.copy() for pose in poses]
    for pose in scaled:
        if pose is not None:
            pose[:3, 3] *= scale
    corrections = np.ones(len(poses))
    for view, ratio in ratios.items():
        if len(ratio) >= min_points:
            corrections[view] = float(np.median(scale / ratio))
    return scaled, points * scale, corrections


def depth_errors(estimated, truth, mask=None):
    """Median relative error and median scale ratio where truth depth exists."""
    valid = np.isfinite(truth) & (truth > 0) & np.isfinite(estimated)
    if mask is not None:
        valid &= mask
    if not valid.any():
        return None
    ratio = estimated[valid] / truth[valid]
    return {"pixels": int(valid.sum()), "median_ratio": float(np.median(ratio)),
            "median_abs_rel": float(np.median(np.abs(ratio - 1)))}


def estimate(bundle_dir, output, scene="indoor", truth_depth=False, min_inliers=30, fov_x=None):
    """Run steps 1-4 and write ``poses.json`` plus per-photo depth arrays."""
    bundle_dir, output = Path(bundle_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; choose a new output folder.")
    bundle = json.loads((bundle_dir / "bundle.json").read_text())
    if bundle.get("format") != "photo-bundle-v1":
        raise ValueError("Expected a photo-bundle-v1 bundle.json.")
    images = bundle["images"]
    if truth_depth:
        truth = json.loads((bundle_dir / "truth" / "truth.json").read_text())
        model_info = {"diagnostic": "truth depth from truth/ (not a photos-only result)"}
    else:
        session, model_info = load_depth_model(scene)
    (output / "depth").mkdir(parents=True)
    intrinsics, sources, depths, features, view_records = [], [], [], [], []
    for index, image in enumerate(images):
        with Image.open(bundle_dir / image["file"]) as source:
            rgb = np.asarray(source.convert("RGB"))
        if rgb.shape[:2] != (image["height"], image["width"]):
            raise ValueError(f"{image['file']} size differs from bundle.json.")
        matrix, source = bundle_intrinsics(image, fov_x)
        if truth_depth:
            with np.load(bundle_dir / truth["views"][index]["arrays"]) as arrays:
                depth = arrays["depth"].astype(np.float32)
        else:
            depth = estimate_depth(session, rgb)
        depth_file = f"depth/view_{index:02d}.npy"
        np.save(output / depth_file, depth, allow_pickle=False)
        intrinsics.append(matrix)
        sources.append(source)
        depths.append(depth)
        features.append(detect_features(rgb))
        view_records.append({"image": image["file"], "intrinsics": matrix.tolist(), "intrinsics_source": source,
                             "depth": depth_file, "features": len(features[-1][0])})

    matches, match_counts = {}, {}
    for a in range(len(images)):
        for b in range(a + 1, len(images)):
            pairs = match_features(features[a][1], features[b][1])
            match_counts[(a, b)] = len(pairs)
            if len(pairs):
                matches[(a, b)] = pairs
    keypoints = [points for points, _ in features]
    edges = _pair_edges(matches, keypoints, intrinsics, depths, min_inliers)
    edge_records = []
    for (a, b), count in match_counts.items():
        record = {"views": [a, b], "matches": count, "registered": (a, b) in edges}
        if (a, b) in edges:
            result = edges[(a, b)]
            record.update(inliers=result["inliers"], scale=result["scale"],
                          scale_spread=result["scale_spread"], parallax_degrees=result["parallax_degrees"])
        edge_records.append(record)
    poses, root, points, tracks, placed_by = register_views(len(images), edges, matches, keypoints, intrinsics)
    poses, points, corrections = align_depth(poses, points, tracks, keypoints, depths)
    np.save(output / "points.npy", points.astype(np.float32), allow_pickle=False)
    for view, (record, pose, correction) in enumerate(zip(view_records, poses, corrections)):
        record["cam_to_world"] = None if pose is None else pose.tolist()
        record["depth_correction"] = float(correction)
        record["placed_by"] = placed_by[view]
        record["tracked_points"] = int((tracks[view] >= 0).sum())
    result = {"format": POSES_FORMAT, "bundle": str(bundle_dir), "depth_model": model_info,
              "frame": None if root is None else f"camera of view {root} (OpenCV axes)",
              "units": "estimated meters from the depth model", "root_view": root,
              "registered_views": sum(pose is not None for pose in poses), "views": view_records,
              "points": "points.npy", "point_count": len(points), "pairs": edge_records,
              "method": ["Photos placed one at a time: rotation and travel direction from each pair's essential matrix, "
                         "travel distance from 3D points shared with already placed photos.",
                         "One overall scale from the depth model; each photo's depth_correction fits its depth "
                         "to the triangulated points (multiply depth by depth_correction)."],
              "limitations": ["No bundle adjustment; small rotation errors accumulate around the orbit.",
                              "Overall scale comes from monocular metric depth and is not measured."]}
    (output / "poses.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def score(output, bundle_dir):
    """Compare an estimate with ``truth/`` and write ``score.json``."""
    output, bundle_dir = Path(output), Path(bundle_dir)
    estimate_result = json.loads((output / "poses.json").read_text())
    truth = json.loads((bundle_dir / "truth" / "truth.json").read_text())
    views = []
    for index, (view, true_view) in enumerate(zip(estimate_result["views"], truth["views"])):
        depth = np.load(output / view["depth"])
        with np.load(bundle_dir / true_view["arrays"]) as arrays:
            corrected = depth * view.get("depth_correction", 1.0)
            views.append({"view": index, "registered": view["cam_to_world"] is not None,
                          "depth_all": depth_errors(depth, arrays["depth"]),
                          "depth_target": depth_errors(depth, arrays["depth"], arrays["mask"]),
                          "corrected_depth_target": depth_errors(corrected, arrays["depth"], arrays["mask"])})
    registered = [i for i, view in enumerate(estimate_result["views"]) if view["cam_to_world"] is not None]
    cameras = None
    if len(registered) >= 3:
        cameras = camera_errors([estimate_result["views"][i]["cam_to_world"] for i in registered],
                                [truth["views"][i]["cam_to_world"] for i in registered])
        cameras["views"] = registered
    true_focal = truth["intrinsics"][0][0]
    report = {"registered_views": registered, "total_views": len(truth["views"]), "cameras": cameras,
              "focal_ratio": [view["intrinsics"][0][0] / true_focal for view in estimate_result["views"]],
              "depth": views,
              "notes": ["cameras.scale is the truth/estimate size ratio after alignment; 1.0 means correct metric scale.",
                        "Center errors are in truth (source GLB) units."]}
    (output / "score.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True, help="Folder containing bundle.json")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output folder")
    parser.add_argument("--scene", choices=tuple(MODELS), default="indoor", help="Metric depth model")
    parser.add_argument("--min-inliers", type=int, default=30)
    parser.add_argument("--fov", type=float, help="Known horizontal field of view in degrees for all photos")
    parser.add_argument("--truth-depth", action="store_true",
                        help="Diagnostic only: use truth/ depth to isolate pose-chaining error")
    parser.add_argument("--score", action="store_true", help="After estimating, compare with truth/")
    args = parser.parse_args()
    result = estimate(args.bundle, args.output, args.scene, args.truth_depth, args.min_inliers, args.fov)
    assumed = [v["image"] for v in result["views"] if v["intrinsics_source"].startswith("assumed")]
    if assumed:
        print(f"WARNING: {len(assumed)} photos have no focal length; an assumed {ASSUMED_FOV_X:g} degree FOV was used. "
              "Poses are unreliable unless that matches the camera; pass --fov from the camera specs.")
    print(f"Registered {result['registered_views']} of {len(result['views'])} photos "
          f"(root view {result['root_view']}) into {args.output}.")
    if args.score:
        report = score(args.output, args.bundle)
        if report["cameras"]:
            cams = report["cameras"]
            print(f"Camera rotation error: max {cams['max_rotation_degrees']:.2f} deg; center error: max "
                  f"{cams['max_center_error']:.3f} units; scale truth/estimate {cams['scale']:.3f}")
        for view in report["depth"]:
            target, fixed = view["depth_target"], view["corrected_depth_target"]
            if target:
                print(f"  view {view['view']:02d}: target depth ratio {target['median_ratio']:.3f} raw, "
                      f"{fixed['median_ratio']:.3f} corrected; registered {view['registered']}")


if __name__ == "__main__":
    main()
