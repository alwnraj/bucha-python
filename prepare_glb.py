"""Convert a Draco-compressed GLB into a plain GLB the loader can read.

Decodes every ``KHR_draco_mesh_compression`` primitive with the pinned DracoPy
package (``requirements-vision.txt``) and writes ordinary accessors. Images,
materials, nodes and transforms are copied unchanged.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from segment_objects import write_glb

DRACO = "KHR_draco_mesh_compression"


def _read_raw_glb(path):
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} is not a binary glTF file.")
    length = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20:20 + length])
    offset = 20 + length
    blob = b""
    if offset < len(data):
        blob_length = struct.unpack("<I", data[offset:offset + 4])[0]
        blob = data[offset + 8:offset + 8 + blob_length]
    return doc, blob


def decompress(source, output):
    """Write ``output`` with Draco primitives decoded; returns the decoded triangle count."""
    try:
        import DracoPy
    except ModuleNotFoundError as error:
        raise ValueError("Install the pinned DracoPy package from requirements-vision.txt.") from error
    doc, blob = _read_raw_glb(source)
    if DRACO not in doc.get("extensionsUsed", []):
        raise ValueError(f"{source} has no Draco-compressed primitives.")
    draco_views = {prim["extensions"][DRACO]["bufferView"] for mesh in doc["meshes"]
                   for prim in mesh["primitives"] if DRACO in prim.get("extensions", {})}
    new_blob, new_views, view_map = bytearray(), [], {}

    def append(data, target=None):
        new_blob.extend(b"\0" * (-len(new_blob) % 4))
        view = {"buffer": 0, "byteOffset": len(new_blob), "byteLength": len(data)}
        if target:
            view["target"] = target
        new_blob.extend(data)
        new_views.append(view)
        return len(new_views) - 1

    for index, view in enumerate(doc.get("bufferViews", [])):
        if index in draco_views:
            continue
        start = view.get("byteOffset", 0)
        copied = append(blob[start:start + view["byteLength"]])
        new_views[copied].update({k: view[k] for k in ("byteStride", "target") if k in view})
        view_map[index] = copied
    for accessor in doc.get("accessors", []):
        if "bufferView" in accessor and accessor["bufferView"] in view_map:
            accessor["bufferView"] = view_map[accessor["bufferView"]]
    for image in doc.get("images", []):
        if "bufferView" in image:
            image["bufferView"] = view_map[image["bufferView"]]

    names = {"POSITION": "points", "NORMAL": "normals", "TEXCOORD_0": "tex_coord", "COLOR_0": "colors"}
    triangles = 0
    for mesh in doc["meshes"]:
        for prim in mesh["primitives"]:
            extension = prim.get("extensions", {}).pop(DRACO, None)
            if extension is None:
                continue
            if not prim["extensions"]:
                del prim["extensions"]
            view = doc["bufferViews"][extension["bufferView"]]
            start = view.get("byteOffset", 0)
            decoded = DracoPy.decode(bytes(blob[start:start + view["byteLength"]]))
            faces = np.asarray(decoded.faces, dtype="<u4")
            triangles += len(faces)
            index_accessor = doc["accessors"][prim["indices"]]
            index_accessor.update({"bufferView": append(faces.tobytes(), 34963), "byteOffset": 0,
                                   "componentType": 5125, "count": int(faces.size)})
            for name, accessor_index in prim["attributes"].items():
                if name not in names or getattr(decoded, names[name]) is None:
                    raise ValueError(f"Cannot decode Draco attribute {name}.")
                values = np.ascontiguousarray(getattr(decoded, names[name]), dtype="<f4")
                accessor = doc["accessors"][accessor_index]
                accessor.pop("normalized", None)
                accessor.update({"bufferView": append(values.tobytes(), 34962), "byteOffset": 0,
                                 "componentType": 5126, "count": int(len(values))})
                if name == "POSITION":
                    accessor.update({"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()})
    for key in ("extensionsUsed", "extensionsRequired"):
        if key in doc:
            doc[key] = [name for name in doc[key] if name != DRACO]
            if not doc[key]:
                del doc[key]
    doc["bufferViews"] = new_views
    doc["buffers"] = [{"byteLength": len(new_blob)}]
    write_glb(output, doc, new_blob)
    return triangles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Decoded {decompress(args.input, args.output):,} triangles into {args.output}")


if __name__ == "__main__":
    main()
