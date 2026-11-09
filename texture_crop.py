"""Shrink a GLB's textures to the texels its triangles actually use.

Scanned objects cut from a large scene keep UVs that point all over the scene's
texture atlas, so cropping to a bounding box saves little. Instead the used
texture is found in small tiles (with a pixel margin for filtering), each
connected block of tiles is copied into a new, tightly packed image, and the
UVs are shifted to match. Geometry, indices, materials and every other
attribute are unchanged. Images that share UVs through one material (for
example a base color and a specular map) get the same layout.

Images no triangle uses become 1x1 placeholders. JPEG textures are re-encoded
(quality 92); PNG stays lossless. Groups with
wrapped UVs (outside 0..1), texture transforms, mismatched image sizes or no
real saving are left as they were.
"""
from io import BytesIO
import math

import numpy as np
from PIL import Image

from segment_objects import accessor, read_glb, write_glb


def _texture_refs(value):
    """Yield ``(texture index, texCoord, info)`` for every textureInfo in a material."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("Texture") and isinstance(item, dict) and "index" in item:
                yield item["index"], item.get("texCoord", 0), item
            else:
                yield from _texture_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texture_refs(item)


def _components(mask):
    """4-connected components of a boolean grid; returns a label grid (-1 outside)."""
    labels = np.full(mask.shape, -1, dtype=np.int64)
    count = 0
    for start in zip(*np.nonzero(mask)):
        if labels[start] >= 0:
            continue
        labels[start] = count
        stack = [start]
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and labels[ny, nx] < 0:
                    labels[ny, nx] = count
                    stack.append((ny, nx))
        count += 1
    return labels, count


def _shelf_pack(sizes):
    """Place ``(height, width)`` rectangles on shelves; returns positions and atlas size."""
    total = sum(h * w for h, w in sizes)
    width = max(max(w for _, w in sizes), math.ceil(math.sqrt(total * 1.15)))
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0])
    positions = [None] * len(sizes)
    x = y = shelf = 0
    for i in order:
        h, w = sizes[i]
        if x + w > width:
            x, y, shelf = 0, y + shelf, 0
        positions[i] = (y, x)
        x += w
        shelf = max(shelf, h)
    return positions, (y + shelf, width)


def _layout(uv_sets, size, tile, pad):
    """Tile clusters for all faces in a group; returns per-set vertex shifts and atlas size in pixels.

    ``uv_sets`` is a list of ``(uv, faces)``; ``size`` is the reference image ``(width, height)``.
    Returns ``None`` when packing would not save space.
    """
    width, height = size
    grid = (math.ceil(height / tile), math.ceil(width / tile))
    marks = np.zeros((grid[0] + 1, grid[1] + 1), dtype=np.int64)
    spans = []
    for uv, faces in uv_sets:
        pixels = uv[faces] * (width, height)  # (faces, 3, 2), x then y
        low = np.floor((pixels.min(axis=1) - pad) / tile).astype(np.int64)
        high = np.floor((pixels.max(axis=1) + pad) / tile).astype(np.int64)
        low = np.clip(low, 0, (grid[1] - 1, grid[0] - 1))
        high = np.clip(high, 0, (grid[1] - 1, grid[0] - 1))
        spans.append(low)
        # Mark each face's tile rectangle with a 2D difference array.
        np.add.at(marks, (low[:, 1], low[:, 0]), 1)
        np.add.at(marks, (low[:, 1], high[:, 0] + 1), -1)
        np.add.at(marks, (high[:, 1] + 1, low[:, 0]), -1)
        np.add.at(marks, (high[:, 1] + 1, high[:, 0] + 1), 1)
    used = marks.cumsum(axis=0).cumsum(axis=1)[:grid[0], :grid[1]] > 0
    labels, count = _components(used)
    if not count:
        return None
    boxes = []
    for cluster in range(count):
        ys, xs = np.nonzero(labels == cluster)
        boxes.append((ys.min(), xs.min(), ys.max() + 1, xs.max() + 1))
    sizes = [(y1 - y0, x1 - x0) for y0, x0, y1, x1 in boxes]
    positions, (rows, columns) = _shelf_pack(sizes)
    if rows * columns >= .7 * grid[0] * grid[1]:
        return None
    shifts = []
    for (uv, faces), low in zip(uv_sets, spans):
        face_cluster = labels[low[:, 1], low[:, 0]]
        vertex_cluster = np.zeros(len(uv), dtype=np.int64)
        vertex_cluster[faces.ravel()] = np.repeat(face_cluster, 3)
        offset = np.array([(positions[c][1] - boxes[c][1], positions[c][0] - boxes[c][0]) for c in range(count)])
        shifts.append(offset[vertex_cluster] * tile)  # pixels, x then y
    blocks = [((y0 * tile, x0 * tile, y1 * tile, x1 * tile), (py * tile, px * tile))
              for (y0, x0, y1, x1), (py, px) in zip(boxes, positions)]
    return shifts, (columns * tile, rows * tile), blocks


def _decode(doc, blob, image_index):
    image = doc["images"][image_index]
    view = doc["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return Image.open(BytesIO(blob[start:start + view["byteLength"]]))


def repack_textures(path, output=None, tile=16, pad=2, quality=92):
    """Rewrite ``path`` (or write ``output``) with packed textures; returns image bytes before and after."""
    doc, blob = read_glb(path)
    output = output or path
    uses = []  # (accessor, face indices, image)
    blocked = set()
    for mesh in doc.get("meshes", []):
        for primitive in mesh["primitives"]:
            if primitive.get("material") is None:
                continue
            faces = (accessor(doc, blob, primitive["indices"]).reshape(-1, 3) if "indices" in primitive
                     else None)
            for texture, texcoord, info in _texture_refs(doc["materials"][primitive["material"]]):
                image = doc["textures"][texture].get("source")
                attribute = primitive["attributes"].get(f"TEXCOORD_{texcoord}")
                if image is None or attribute is None or "bufferView" not in doc["images"][image]:
                    continue
                if faces is None:
                    faces = np.arange(doc["accessors"][attribute]["count"]).reshape(-1, 3)
                uses.append((attribute, faces, image))
                if "extensions" in info:  # e.g. KHR_texture_transform
                    blocked.add(image)
    # Group images that share UV accessors.
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            key = parent[key]
        return key

    for attribute, _, image in uses:
        parent[find(("a", attribute))] = find(("i", image))
    groups = {}
    for attribute, faces, image in uses:
        group = groups.setdefault(find(("i", image)), {"images": set(), "uvs": {}})
        group["images"].add(image)
        group["uvs"].setdefault(attribute, []).append(faces)
    new_uvs, new_images = {}, {}
    embedded = [i for i, image in enumerate(doc.get("images", [])) if "bufferView" in image]
    before = sum(doc["bufferViews"][doc["images"][i]["bufferView"]]["byteLength"] for i in embedded)
    # Images no triangle here samples (a cut-out object keeps every scene image) become
    # 1x1 placeholders, so texture and material indices stay valid.
    used = {image for _, _, image in uses}
    for i in embedded:
        if i not in used:
            stream = BytesIO()
            Image.new("RGB", (1, 1)).save(stream, "PNG")
            new_images[i] = stream.getvalue()
            doc["images"][i]["mimeType"] = "image/png"
    for group in groups.values():
        images = sorted(group["images"])
        if blocked & group["images"]:
            continue
        decoded = {i: _decode(doc, blob, i) for i in images}
        reference = max(images, key=lambda i: decoded[i].width * decoded[i].height)
        size = decoded[reference].size
        if any(decoded[i].size != size for i in images):
            continue  # keep it simple: only identical image sizes share a layout
        attributes = sorted(group["uvs"])
        uv_sets = [(accessor(doc, blob, a, decode_normalized=True).astype(np.float64),
                    np.concatenate(group["uvs"][a])) for a in attributes]
        if any(uv.size and (uv.min() < -1e-6 or uv.max() > 1 + 1e-6) for uv, _ in uv_sets):
            continue  # wrapped UVs cannot be cut into blocks
        packed = _layout(uv_sets, size, tile, pad)
        if packed is None:
            continue
        shifts, atlas, blocks = packed
        for a, (uv, _), shift in zip(attributes, uv_sets, shifts):
            new_uvs[a] = ((uv * size + shift) / atlas).astype("<f4")
        for i in images:
            mode = "RGBA" if "A" in decoded[i].getbands() else "RGB"
            source = decoded[i].convert(mode)
            target = Image.new(mode, atlas)
            for (y0, x0, y1, x1), (py, px) in blocks:
                target.paste(source.crop((x0, y0, x1, y1)), (px, py))
            stream = BytesIO()
            if doc["images"][i].get("mimeType") == "image/jpeg" and mode == "RGB":
                target.save(stream, "JPEG", quality=quality)
            else:
                target.save(stream, "PNG")
            new_images[i] = stream.getvalue()
    if not new_images:
        return before, before
    after = sum(len(data) for data in new_images.values()) + sum(
        doc["bufferViews"][doc["images"][i]["bufferView"]]["byteLength"] for i in set(embedded) - set(new_images))
    _rewrite(doc, blob, new_uvs, new_images, output)
    return before, after


def _rewrite(doc, blob, new_uvs, new_images, output):
    """Write the GLB with replaced UV accessors and images, dropping unreferenced buffer views."""
    keep = {item["bufferView"] for index, item in enumerate(doc.get("accessors", []))
            if index not in new_uvs and "bufferView" in item}
    keep |= {item["bufferView"] for index, item in enumerate(doc.get("images", []))
             if index not in new_images and "bufferView" in item}
    out, views, mapping = bytearray(), [], {}

    def append(data, extra=None):
        out.extend(b"\0" * (-len(out) % 4))
        views.append({"buffer": 0, "byteOffset": len(out), "byteLength": len(data)} | (extra or {}))
        out.extend(data)
        return len(views) - 1

    for index in sorted(keep):
        view = doc["bufferViews"][index]
        start = view.get("byteOffset", 0)
        mapping[index] = append(blob[start:start + view["byteLength"]],
                                {k: view[k] for k in ("byteStride", "target") if k in view})
    for index, item in enumerate(doc.get("accessors", [])):
        if index in new_uvs:
            item.update({"bufferView": append(new_uvs[index].tobytes(), {"target": 34962}), "byteOffset": 0,
                         "componentType": 5126})
            item.pop("normalized", None)
            item["min"], item["max"] = new_uvs[index].min(axis=0).tolist(), new_uvs[index].max(axis=0).tolist()
        elif "bufferView" in item:
            item["bufferView"] = mapping[item["bufferView"]]
    for index, item in enumerate(doc.get("images", [])):
        if index in new_images:
            item["bufferView"] = append(new_images[index])
        elif "bufferView" in item:
            item["bufferView"] = mapping[item["bufferView"]]
    doc["bufferViews"] = views
    write_glb(output, doc, out)
