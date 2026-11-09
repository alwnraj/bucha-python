"""Shared fitting and scoring for photo-based object reconstruction.

Both the CPU and GPU pipeline versions are scored with these functions against
``photo_sim.py`` ground truth. Everything is NumPy only. Photo reconstructions
have an arbitrary frame and scale, so camera errors are measured after a
similarity alignment of camera centers.
"""
import numpy as np


def fit_similarity(source, target, with_scale=True):
    """Least-squares ``target ≈ scale * source @ R.T + t`` (Umeyama, 1991).

    Returns ``(scale, R, t)``. Needs at least three non-collinear points.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Expected two matching arrays of at least three 3D points.")
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    s, t = source - mean_s, target - mean_t
    u, singular, vt = np.linalg.svd(t.T @ s / len(source))
    if singular[1] < 1e-12 * max(singular[0], 1e-300):
        raise ValueError("Points are collinear or coincident; the rotation is undetermined.")
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u @ vt)) or 1.0
    rotation = u @ sign @ vt
    variance = (s ** 2).sum() / len(source)
    scale = float(np.trace(np.diag(singular) @ sign) / variance) if with_scale else 1.0
    return scale, rotation, mean_t - scale * rotation @ mean_s


def fit_translation_scale(source, target):
    """Least-squares ``target ≈ scale * source + t`` with rotation held fixed.

    This is GATOR's placement step: orientation and shape stay unchanged.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.shape != target.shape or len(source) < 2:
        raise ValueError("Expected two matching arrays of at least two points.")
    s, t = source - source.mean(axis=0), target - target.mean(axis=0)
    variance = (s ** 2).sum()
    if variance < 1e-300:
        raise ValueError("Source points are coincident; scale is undetermined.")
    scale = float((s * t).sum() / variance)
    return scale, target.mean(axis=0) - scale * source.mean(axis=0)


def _rotation_degrees(rotation):
    cosine = np.clip((np.trace(rotation) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def camera_errors(estimated, truth):
    """Compare camera-to-world poses after aligning estimated centers to the truth.

    Returns the alignment scale and per-view rotation (degrees) and center
    errors, with the center errors in truth units.
    """
    estimated = [np.asarray(pose, dtype=float) for pose in estimated]
    truth = [np.asarray(pose, dtype=float) for pose in truth]
    if len(estimated) != len(truth) or len(truth) < 3:
        raise ValueError("Need matching lists of at least three camera poses.")
    scale, rotation, translation = fit_similarity([p[:3, 3] for p in estimated], [p[:3, 3] for p in truth])
    rotation_errors, center_errors = [], []
    for est, true in zip(estimated, truth):
        aligned_rotation = rotation @ est[:3, :3]
        aligned_center = scale * rotation @ est[:3, 3] + translation
        rotation_errors.append(_rotation_degrees(true[:3, :3].T @ aligned_rotation))
        center_errors.append(float(np.linalg.norm(aligned_center - true[:3, 3])))
    # Frame-independent check: rotation between every pair of cameras.
    relative = [_rotation_degrees((truth[i][:3, :3].T @ truth[j][:3, :3]).T @ (estimated[i][:3, :3].T @ estimated[j][:3, :3]))
                for i in range(len(truth)) for j in range(i + 1, len(truth))]
    return {"scale": scale, "rotation_degrees": rotation_errors, "center_error": center_errors,
            "max_rotation_degrees": max(rotation_errors), "max_center_error": max(center_errors),
            "max_relative_rotation_degrees": max(relative),
            "median_relative_rotation_degrees": float(np.median(relative))}


def sample_surface(triangles, count, seed=0):
    """Area-weighted uniform points on triangles shaped ``(n, 3, 3)``."""
    triangles = np.asarray(triangles, dtype=float)
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if areas.sum() <= 0:
        raise ValueError("Triangles have no area to sample.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(triangles), size=count, p=areas / areas.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    tri = triangles[chosen]
    return tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])


def nearest_distances(points, reference, block=2048):
    """Distance from each point to its nearest reference point (brute force, chunked)."""
    points, reference = np.asarray(points, dtype=float), np.asarray(reference, dtype=float)
    reference_sq = (reference ** 2).sum(axis=1)
    result = np.empty(len(points))
    for start in range(0, len(points), block):
        part = points[start:start + block]
        squared = (part ** 2).sum(axis=1)[:, None] - 2 * part @ reference.T + reference_sq[None, :]
        result[start:start + block] = np.sqrt(np.maximum(squared.min(axis=1), 0))
    return result


def chamfer_f1(predicted, truth, threshold):
    """Symmetric chamfer distance and F-score at ``threshold`` between point sets.

    Chamfer is the mean of the two directional mean distances. Precision is the
    share of predicted points within ``threshold`` of the truth; recall is the
    reverse.
    """
    to_truth = nearest_distances(predicted, truth)
    to_predicted = nearest_distances(truth, predicted)
    precision = float((to_truth <= threshold).mean())
    recall = float((to_predicted <= threshold).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"chamfer": float((to_truth.mean() + to_predicted.mean()) / 2),
            "precision": precision, "recall": recall, "f1": f1, "threshold": float(threshold)}


def mask_iou(predicted, truth):
    predicted, truth = np.asarray(predicted, dtype=bool), np.asarray(truth, dtype=bool)
    union = (predicted | truth).sum()
    return 1.0 if union == 0 else float((predicted & truth).sum() / union)


def points_in_voxels(points, origin, voxel_size, occupied):
    """True where a point falls in an occupied voxel; points outside the grid are False."""
    index = np.floor((np.asarray(points, dtype=float) - origin) / voxel_size).astype(np.int64)
    inside = ((index >= 0) & (index < occupied.shape)).all(axis=1)
    result = np.zeros(len(index), dtype=bool)
    result[inside] = occupied[tuple(index[inside].T)]
    return result
