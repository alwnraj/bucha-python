"""Render textured GLB views with source face IDs.

Six fixed orthographic views feed image segmentation models; the pinhole
renderer feeds the simulated photo capture in ``photo_sim.py``. This small CPU
rasterizer is not a physically based renderer and only handles opaque triangle
primitives loaded by :func:`segment_objects.load_geometry`.
"""
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from segment_objects import accessor, load_geometry


VIEWS = {
    "+X": ([1, 0, 0], [0, 0, 1]), "-X": ([-1, 0, 0], [0, 0, 1]),
    "+Y": ([0, 1, 0], [0, 0, 1]), "-Y": ([0, -1, 0], [0, 0, 1]),
    "+Z": ([0, 0, 1], [0, 1, 0]), "-Z": ([0, 0, -1], [0, 1, 0]),
}


def _embedded_image(doc, blob, image_index):
    image = doc["images"][image_index]
    if "bufferView" not in image:
        raise ValueError("Only embedded bufferView images are supported.")
    view = doc["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    end = start + view["byteLength"]
    with Image.open(BytesIO(blob[start:end])) as decoded:
        return np.asarray(decoded.convert("RGB"))


def _materials(doc, blob):
    cache = {}
    result = []
    for material in doc.get("materials", []):
        if material.get("alphaMode", "OPAQUE") != "OPAQUE":
            raise ValueError("Semantic view rendering does not support transparent glTF materials.")
        pbr = material.get("pbrMetallicRoughness", {})
        factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=float)
        info = pbr.get("baseColorTexture")
        texture = None
        texcoord = 0
        if info is not None:
            if "KHR_texture_transform" in info.get("extensions", {}):
                raise ValueError("Semantic view rendering does not support KHR_texture_transform.")
            texcoord = info.get("texCoord", 0)
            texture_index = info["index"]
            image_index = doc["textures"][texture_index]["source"]
            if image_index not in cache:
                cache[image_index] = _embedded_image(doc, blob, image_index)
            texture = cache[image_index]
        result.append((factor, texture, texcoord))
    return result


def _chunk_material(doc, blob, chunk, materials):
    """Return ``(material, per-face UVs or None)`` for one loaded chunk."""
    prim = chunk["primitive"]
    attrs = prim.get("attributes", {})
    index = prim.get("material")
    material = materials[index] if index is not None and index < len(materials) else (np.ones(3), None, 0)
    if material[1] is None:
        return material, None
    attr_name = f"TEXCOORD_{material[2]}"
    if attr_name not in attrs:
        raise ValueError(f"Textured primitive is missing {attr_name}.")
    return material, accessor(doc, blob, attrs[attr_name], decode_normalized=True)[chunk["faces"]]


def _shade(material, uv, shape):
    """Base color (texture times factor) as uint8 for UVs shaped ``shape + (2,)``."""
    color = np.broadcast_to(material[0], (*shape, 3)).copy()
    if uv is not None:
        texture = material[1]
        # glTF UV (0, 0) is the top-left image corner, matching decoded row 0.
        # Coordinates wrap (glTF's default REPEAT sampler).
        tx = np.floor(uv[..., 0] % 1.0 * texture.shape[1]).astype(int) % texture.shape[1]
        ty = np.floor(uv[..., 1] % 1.0 * texture.shape[0]).astype(int) % texture.shape[0]
        color *= texture[ty, tx] / 255.
    return np.clip(np.rint(color * 255), 0, 255).astype(np.uint8)


def render_view(path, direction, resolution=512, background=(20, 22, 26), exclude_names=None):
    """Return ``(rgb, source_instance, source_face)`` for one orthographic view.

    ``direction`` is one of ``+X``, ``-X``, ``+Y``, ``-Y``, ``+Z`` or ``-Z``.
    Pixel ID arrays are int32 and use -1 for background. Resolution may be an
    integer or ``(width, height)``. Geometry is fitted to the image with a 3%
    border and all faces are treated as double-sided.
    """
    if direction not in VIEWS:
        raise ValueError(f"Unknown view {direction!r}; expected one of {tuple(VIEWS)}.")
    if isinstance(resolution, int):
        width = height = resolution
    else:
        width, height = map(int, resolution)
    if width < 1 or height < 1:
        raise ValueError("Resolution dimensions must be positive.")

    doc, blob, chunks = load_geometry(path)
    excluded = {str(name).casefold() for name in (exclude_names or [])}
    if excluded:
        chunks = [chunk for chunk in chunks if not any(
            term in " ".join((doc["nodes"][chunk["source_node"]].get("name", ""),
                               doc["meshes"][chunk["source_mesh"]].get("name", ""))).casefold()
            for term in excluded)]
    if not chunks:
        raise ValueError("No geometry remains after applying exclude_names.")
    view_dir, up = (np.asarray(v, dtype=float) for v in VIEWS[direction])
    right = np.cross(view_dir, up)
    all_points = np.concatenate([chunk["tri"].reshape(-1, 3) for chunk in chunks])
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    projected = np.column_stack(((all_points - center) @ right, (all_points - center) @ up))
    span = np.maximum(projected.max(axis=0) - projected.min(axis=0), 1e-8)
    scale = min((width - 1) * .94 / span[0], (height - 1) * .94 / span[1])
    screen_all = np.column_stack((width / 2 + projected[:, 0] * scale,
                                  height / 2 - projected[:, 1] * scale))

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = background
    source_instance = np.full((height, width), -1, dtype=np.int32)
    source_face = np.full((height, width), -1, dtype=np.int32)
    depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    materials = _materials(doc, blob)

    point_offset = 0
    for chunk in chunks:
        tri = chunk["tri"]
        count = len(tri)
        screen = screen_all[point_offset:point_offset + count * 3].reshape(count, 3, 2)
        point_offset += count * 3
        depths = tri @ view_dir
        material, uv = _chunk_material(doc, blob, chunk, materials)

        for face_index, vertices in enumerate(screen):
            min_x = max(0, int(np.floor(vertices[:, 0].min())))
            max_x = min(width - 1, int(np.ceil(vertices[:, 0].max())))
            min_y = max(0, int(np.floor(vertices[:, 1].min())))
            max_y = min(height - 1, int(np.ceil(vertices[:, 1].max())))
            if min_x > max_x or min_y > max_y:
                continue
            ax, ay = vertices[0]
            bx, by = vertices[1]
            cx, cy = vertices[2]
            denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denom) < 1e-12:
                continue
            ys, xs = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
            px, py = xs + .5, ys + .5
            w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
            face_depth = w0 * depths[face_index, 0] + w1 * depths[face_index, 1] + w2 * depths[face_index, 2]
            visible = inside & (face_depth >= depth_buffer[min_y:max_y + 1, min_x:max_x + 1])
            if not visible.any():
                continue
            target_depth = depth_buffer[min_y:max_y + 1, min_x:max_x + 1]
            target_depth[visible] = face_depth[visible]
            target_inst = source_instance[min_y:max_y + 1, min_x:max_x + 1]
            target_face = source_face[min_y:max_y + 1, min_x:max_x + 1]
            target_inst[visible] = chunk["source_instance"]
            target_face[visible] = face_index

            interpolated = None
            if uv is not None:
                face_uv = uv[face_index]
                interpolated = (w0[..., None] * face_uv[0] + w1[..., None] * face_uv[1]
                                + w2[..., None] * face_uv[2])
            color = _shade(material, interpolated, visible.shape)
            rgb_region = rgb[min_y:max_y + 1, min_x:max_x + 1]
            rgb_region[visible] = color[visible]
    return rgb, source_instance, source_face


def load_scene(path):
    """Load geometry and materials once for repeated pinhole renders.

    Global face indices follow the concatenated chunk order of
    :func:`segment_objects.load_geometry`, the same order used by
    ``face-labels.npz`` and collision exports.
    """
    doc, blob, chunks = load_geometry(path)
    counts = [len(chunk["tri"]) for chunk in chunks]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    materials = _materials(doc, blob)
    return {"doc": doc, "blob": blob, "chunks": chunks, "materials": materials,
            "shading": [_chunk_material(doc, blob, chunk, materials) for chunk in chunks],
            "face_offsets": offsets[:-1], "face_count": int(offsets[-1])}


def intrinsics_from_fov(width, height, fov_x_degrees=60.0):
    """Pinhole matrix with square pixels and the principal point at the image center."""
    if not 0 < fov_x_degrees < 180:
        raise ValueError("Horizontal field of view must be between 0 and 180 degrees.")
    focal = (width / 2) / np.tan(np.radians(fov_x_degrees) / 2)
    return np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1.]])


def look_at(eye, target, up=(0, 0, 1)):
    """Camera-to-world 4x4 in OpenCV axes: +X right, +Y down, +Z toward ``target``."""
    eye, target, up = (np.asarray(value, dtype=float) for value in (eye, target, up))
    forward = target - eye
    if np.linalg.norm(forward) < 1e-12:
        raise ValueError("Camera eye and target must differ.")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("Viewing direction must not be parallel to the up vector.")
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, np.cross(forward, right), forward, eye
    return pose


def _clip_near(z, near):
    """Split a triangle at camera ``z >= near``.

    Returns sub-triangles as 3x3 rows of barycentric weights over the original
    vertices, so positions and UVs can be interpolated after clipping.
    """
    corners = np.eye(3)
    if (z >= near).all():
        return [corners]
    polygon = []
    for i in range(3):
        j = (i + 1) % 3
        if z[i] >= near:
            polygon.append(corners[i])
        if (z[i] >= near) != (z[j] >= near):
            t = (near - z[i]) / (z[j] - z[i])
            polygon.append(corners[i] + t * (corners[j] - corners[i]))
    return [np.array([polygon[0], polygon[k], polygon[k + 1]]) for k in range(1, len(polygon) - 1)]


def _splat_single_pixel(faces, camera, sx, sy, px, py, uv, material, offset,
                        rgb, depth_buffer, face_ids, width):
    """Draw triangles whose bounding box holds exactly one pixel center, all at once.

    Same coverage test, perspective-correct depth and texture lookup as the
    per-triangle loop; the nearest triangle wins each pixel.
    """
    (ax, bx, cx), (ay, by, cy) = sx[faces].T, sy[faces].T
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    with np.errstate(divide="ignore", invalid="ignore"):
        l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
        l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    l2 = 1 - l0 - l1
    inside = (np.abs(denom) >= 1e-12) & (l0 >= -1e-8) & (l1 >= -1e-8) & (l2 >= -1e-8)
    if not inside.any():
        return
    faces, px, py, l0, l1, l2 = faces[inside], px[inside], py[inside], l0[inside], l1[inside], l2[inside]
    z = camera[faces, :, 2]
    inverse = l0 / z[:, 0] + l1 / z[:, 1] + l2 / z[:, 2]
    depth = 1 / inverse
    pixel = py * width + px
    order = np.lexsort((depth, pixel))
    first = order[np.r_[True, pixel[order][1:] != pixel[order][:-1]]]
    first = first[depth[first] < depth_buffer.ravel()[pixel[first]]]
    if not len(first):
        return
    depth_buffer.ravel()[pixel[first]] = depth[first]
    face_ids.ravel()[pixel[first]] = offset + faces[first]
    pixel_uv = None
    if uv is not None:
        weights = np.column_stack((l0[first] / z[first, 0], l1[first] / z[first, 1], l2[first] / z[first, 2]))
        pixel_uv = np.einsum("nk,nkd->nd", weights / inverse[first, None], uv[faces[first]])
    rgb.reshape(-1, 3)[pixel[first]] = _shade(material, pixel_uv, (len(first),))


def render_perspective(scene, intrinsics, cam_to_world, size, background=(20, 22, 26), near=1e-3, batch=True):
    """Render one pinhole view of a scene from :func:`load_scene`.

    ``size`` is ``(width, height)``. Pixel column x, row y is centered at image
    coordinate ``(x, y)`` (OpenCV convention). Returns ``(rgb, depth, face)``:
    depth is camera-axis Z in source units (NaN where empty) and face is the
    global source-face index (-1 where empty). Faces are double-sided, and
    triangles crossing the near plane are clipped rather than dropped.
    With ``batch`` (default), triangles that can cover at most one pixel are
    drawn together in numpy; large scans are mostly such triangles.
    """
    width, height = map(int, size)
    if width < 1 or height < 1 or near <= 0:
        raise ValueError("Image size and near distance must be positive.")
    intrinsics = np.asarray(intrinsics, dtype=float)
    pose = np.asarray(cam_to_world, dtype=float)
    if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("Expected a 3x3 intrinsic matrix and a 4x4 camera-to-world pose.")
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    rotation, eye = pose[:3, :3], pose[:3, 3]

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = background
    depth_buffer = np.full((height, width), np.inf)
    face_ids = np.full((height, width), -1, dtype=np.int32)

    for chunk_index, (chunk, offset) in enumerate(zip(scene["chunks"], scene["face_offsets"])):
        camera = (chunk["tri"] - eye) @ rotation
        z = camera[..., 2]
        keep = (z >= near).any(axis=1)
        front = (z >= near).all(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = fx * camera[..., 0] / z + cx
            sy = fy * camera[..., 1] / z + cy
        outside = front & ((sx.max(axis=1) < -.5) | (sx.min(axis=1) > width - .5)
                           | (sy.max(axis=1) < -.5) | (sy.min(axis=1) > height - .5))
        keep &= ~outside
        if not keep.any():
            continue
        material, uv = scene["shading"][chunk_index] if "shading" in scene else _chunk_material(
            scene["doc"], scene["blob"], chunk, scene["materials"])
        if batch:
            with np.errstate(invalid="ignore"):
                x0, x1 = np.ceil(sx.min(axis=1)), np.floor(sx.max(axis=1))
                y0, y1 = np.ceil(sy.min(axis=1)), np.floor(sy.max(axis=1))
                tiny = keep & front
                no_pixel = tiny & ((x0 > x1) | (y0 > y1))  # covers no pixel center
                single = tiny & (x0 == x1) & (y0 == y1) & (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
            chosen = np.flatnonzero(single)
            if len(chosen):
                _splat_single_pixel(chosen, camera, sx, sy, x0[chosen].astype(int), y0[chosen].astype(int),
                                    uv, material, offset, rgb, depth_buffer, face_ids, width)
            keep &= ~(single | no_pixel)
        for face_index in np.flatnonzero(keep):
            for weights in _clip_near(z[face_index], near):
                points = weights @ camera[face_index]
                zz = points[:, 2]
                screen = np.column_stack((fx * points[:, 0] / zz + cx, fy * points[:, 1] / zz + cy))
                min_x = max(0, int(np.ceil(screen[:, 0].min())))
                max_x = min(width - 1, int(np.floor(screen[:, 0].max())))
                min_y = max(0, int(np.ceil(screen[:, 1].min())))
                max_y = min(height - 1, int(np.floor(screen[:, 1].max())))
                if min_x > max_x or min_y > max_y:
                    continue
                (ax, ay), (bx, by), (cx_, cy_) = screen
                denom = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
                if abs(denom) < 1e-12:
                    continue
                ys, xs = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
                l0 = ((by - cy_) * (xs - cx_) + (cx_ - bx) * (ys - cy_)) / denom
                l1 = ((cy_ - ay) * (xs - cx_) + (ax - cx_) * (ys - cy_)) / denom
                l2 = 1 - l0 - l1
                inside = (l0 >= -1e-8) & (l1 >= -1e-8) & (l2 >= -1e-8)
                inverse = l0 / zz[0] + l1 / zz[1] + l2 / zz[2]
                with np.errstate(divide="ignore"):
                    pixel_depth = 1 / inverse
                region = depth_buffer[min_y:max_y + 1, min_x:max_x + 1]
                visible = inside & (pixel_depth < region)
                if not visible.any():
                    continue
                region[visible] = pixel_depth[visible]
                face_ids[min_y:max_y + 1, min_x:max_x + 1][visible] = offset + face_index
                pixel_uv = None
                if uv is not None:
                    # Perspective-correct weights over the sub-triangle, mapped
                    # back to barycentric weights over the original face.
                    sub = np.stack((l0[visible] / zz[0], l1[visible] / zz[1], l2[visible] / zz[2]), axis=-1)
                    sub /= inverse[visible][:, None]
                    pixel_uv = (sub @ weights) @ uv[face_index]
                rgb[min_y:max_y + 1, min_x:max_x + 1][visible] = _shade(material, pixel_uv, (int(visible.sum()),))
    depth = np.where(np.isfinite(depth_buffer), depth_buffer, np.nan).astype(np.float32)
    return rgb, depth, face_ids


def render_views(path, output_dir, resolution=512, exclude_names=None, directions=None):
    """Save RGB PNGs and paired ID arrays for selected canonical views."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directions = tuple(VIEWS if directions is None else directions)
    if not directions or any(direction not in VIEWS for direction in directions):
        raise ValueError(f"Directions must be chosen from {tuple(VIEWS)}.")
    for direction in directions:
        rgb, instances, faces = render_view(path, direction, resolution, exclude_names=exclude_names)
        stem = direction.replace("+", "p").replace("-", "m").lower()
        Image.fromarray(rgb).save(output_dir / f"view_{stem}.png")
        np.savez_compressed(output_dir / f"view_{stem}.npz",
                            source_instance=instances, source_face=faces)
    metadata = {
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "views": {
            direction: {"rgb": f"view_{direction.replace('+', 'p').replace('-', 'm').lower()}.png",
                        "ids": f"view_{direction.replace('+', 'p').replace('-', 'm').lower()}.npz"}
            for direction in directions
        },
    }
    (output_dir / "views.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--direction", action="append", choices=tuple(VIEWS),
                        help="Repeat to render a subset; default is all six")
    parser.add_argument("--exclude-name", action="append", default=[],
                        help="Exclude source node or mesh names containing this text")
    args = parser.parse_args()
    render_views(args.input, args.output, args.resolution, args.exclude_name, args.direction)
    print(f"Rendered {len(args.direction or VIEWS)} views in {args.output}.")


if __name__ == "__main__":
    main()
