#!/usr/bin/env python3
"""Visually guided instance/semantic face partition for the supplied Bucha scan.

Requires NumPy. Does not run a trained segmentation model. Edit annotations.json
to revise the provisional masks. Source geometry and UVs are preserved exactly.
"""
import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path
import numpy as np


def read_glb(path):
    raw = Path(path).read_bytes()
    assert struct.unpack_from('<III', raw) == (0x46546c67, 2, len(raw))
    size, kind = struct.unpack_from('<II', raw, 12)
    assert kind == 0x4e4f534a
    doc = json.loads(raw[20:20 + size])
    length, kind = struct.unpack_from('<II', raw, 20 + size)
    assert kind == 0x004e4942
    return doc, raw[28 + size:28 + size + length]


def accessor(doc, blob, index):
    a = doc['accessors'][index]
    v = doc['bufferViews'][a['bufferView']]
    assert not a.get('sparse')
    dtype = np.dtype({5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}[a['componentType']])
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
    return np.ndarray((a['count'], width), dtype=dtype, buffer=blob,
                      offset=v.get('byteOffset', 0) + a.get('byteOffset', 0),
                      strides=(v.get('byteStride', width * dtype.itemsize), dtype.itemsize)).copy()


def write_glb(path, doc, blob):
    blob = bytes(blob) + b'\0' * (-len(blob) % 4)
    doc['buffers'][0]['byteLength'] = len(blob)
    payload = json.dumps(doc, separators=(',', ':')).encode()
    payload += b' ' * (-len(payload) % 4)
    Path(path).write_bytes(struct.pack('<III', 0x46546c67, 2, 28 + len(payload) + len(blob))
                          + struct.pack('<II', len(payload), 0x4e4f534a) + payload
                          + struct.pack('<II', len(blob), 0x004e4942) + blob)


def polygon_mask_and_distance(points, polygon):
    """Even-odd containment plus nearest boundary distance, in original world XY."""
    vertices = np.asarray(polygon, dtype=float)
    vertices[:, 0] = vertices[:, 0] / 40 - 5
    vertices[:, 1] = 27 - vertices[:, 1] / 40
    inside = np.zeros(len(points), dtype=bool)
    distance = np.full(len(points), np.inf)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = b - a
        if abs(edge[1]) > 1e-12:
            crossing = ((a[1] > points[:, 1]) != (b[1] > points[:, 1]))
            inside ^= crossing & (points[:, 0] < a[0] + edge[0] * (points[:, 1] - a[1]) / edge[1])
        t = np.clip(((points - a) @ edge) / max(edge @ edge, 1e-12), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(points - a - t[:, None] * edge, axis=1))
    return inside, distance


def load_geometry(path):
    doc, blob = read_glb(path)
    assert doc['nodes'][0]['children'] == [1] and doc['nodes'][1]['children'] == [2, 3, 4, 5]
    transform = np.array(doc['nodes'][0]['matrix']).reshape(4, 4, order='F')
    chunks = []
    for mi, mesh in enumerate(doc['meshes']):
        assert len(mesh['primitives']) == 1 and doc['nodes'][mi + 2]['mesh'] == mi
        p = mesh['primitives'][0]
        vertices = accessor(doc, blob, p['attributes']['POSITION'])
        world = vertices @ transform[:3, :3].T + transform[:3, 3]
        faces = accessor(doc, blob, p['indices']).reshape(-1, 3)
        tri = world[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(cross, axis=1) / 2
        chunks.append(dict(mesh=mi, primitive=p, faces=faces, tri=tri, centers=tri.mean(1),
                           area=area, normals=cross / np.maximum(area[:, None] * 2, 1e-12)))
    return doc, blob, chunks


def segment(input_glb, output_directory, annotations):
    source = Path(input_glb)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != '488e2890ab5fa2b464d7b794b37e1b663d88ce7edd2d478135850108d3d7ae93':
        raise ValueError('These annotations target the supplied Bucha scan only. Use its original GLB.')
    cfg = json.loads(Path(annotations).read_text())
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    doc, blob, chunks = load_geometry(source)
    plane_n = np.array(cfg['plane']['normal'])
    plane_d = cfg['plane']['offset']
    centers = np.concatenate([c['centers'] for c in chunks])
    tri = np.concatenate([c['tri'] for c in chunks])
    normals = np.concatenate([c['normals'] for c in chunks])
    area = np.concatenate([c['area'] for c in chunks])
    residual = tri @ plane_n + plane_d
    height = residual.mean(1) / plane_n[2]
    labels = np.full(len(centers), 6, dtype=np.uint8)
    review = np.zeros(len(centers), dtype=bool)
    regions = {}
    for obj in cfg['objects']:
        mask = np.zeros(len(centers), dtype=bool)
        distance = np.full(len(centers), np.inf)
        for polygon in obj['polygons_pixels']:
            inside, dist = polygon_mask_and_distance(centers[:, :2], polygon)
            mask |= inside
            distance = np.minimum(distance, dist)
        regions[obj['id']] = mask
        review |= distance < cfg['boundary_review_distance']
    # Low debris is selected by inspected texture regions, overriding flat-ground heuristics.
    rubble = regions[5] & (height > -.15) & (height < cfg['rubble_max_height'])
    labels[rubble] = 5
    ground = ((np.abs(residual.mean(1)) <= cfg['ground_centroid_distance'])
              & (np.max(np.abs(residual), axis=1) <= cfg['ground_vertex_distance'])
              & (np.abs(normals @ plane_n) >= np.cos(np.deg2rad(cfg['ground_max_normal_angle_degrees'])))
              & (area > 1e-12))
    labels[ground & ~regions[5]] = 4
    building_surface = ((height > cfg['building_height_minimum'])
                        | ((height > cfg['building_low_face_minimum'])
                           & (np.abs(normals[:, 2]) < cfg['building_low_face_max_abs_normal_z'])))
    for object_id in [3, 2, 1]:
        labels[regions[object_id] & building_surface] = object_id
    review |= np.abs(height - cfg['building_height_minimum']) < .08
    cursor = 0
    for c in chunks:
        count = len(c['faces'])
        c['labels'] = labels[cursor:cursor + count]
        cursor += count

    def export(filename, objects):
        d = copy.deepcopy(doc)
        binary = bytearray()
        d['bufferViews'], d['accessors'], d['meshes'] = [], [], []
        for image in d['images']:
            old = doc['bufferViews'][image['bufferView']]
            data = blob[old.get('byteOffset', 0):old.get('byteOffset', 0) + old['byteLength']]
            binary.extend(b'\0' * (-len(binary) % 4))
            image['bufferView'] = len(d['bufferViews'])
            d['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(data)})
            binary.extend(data)

        def add(array, original=None, target=34962):
            binary.extend(b'\0' * (-len(binary) % 4))
            vi = len(d['bufferViews'])
            data = array.tobytes()
            d['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(data), 'target': target})
            binary.extend(data)
            a = {'bufferView': vi, 'componentType': original['componentType'] if original else 5125,
                 'count': len(array), 'type': original['type'] if original else 'SCALAR'}
            if original and original.get('normalized'):
                a['normalized'] = True
            if original and original['type'] == 'VEC3':
                a.update(min=array.min(0).astype(float).tolist(), max=array.max(0).astype(float).tolist())
            index = len(d['accessors'])
            d['accessors'].append(a)
            return index

        d['nodes'] = copy.deepcopy(doc['nodes'][:2])
        d['nodes'][1]['children'] = []
        checks = []
        for obj in objects:
            parent_index = len(d['nodes'])
            parent = {'name': obj['name'], 'children': [], 'extras': {
                'objectId': obj['id'], 'semanticClass': obj['semantic_class'], 'labelStatus': obj['status']}}
            d['nodes'].append(parent)
            d['nodes'][1]['children'].append(parent_index)
            for c in chunks:
                chosen = np.flatnonzero(c['labels'] == obj['id'])
                if not len(chosen):
                    continue
                used, inverse = np.unique(c['faces'][chosen], return_inverse=True)
                p = copy.deepcopy(c['primitive'])
                p['attributes'] = {}
                for name, ai in c['primitive']['attributes'].items():
                    p['attributes'][name] = add(accessor(doc, blob, ai)[used], doc['accessors'][ai])
                p['indices'] = add(inverse.astype('<u4').reshape(-1, 1), target=34963)
                mi = len(d['meshes'])
                d['meshes'].append({'name': obj['name'] + '_chunk_' + str(c['mesh']), 'primitives': [p]})
                parent['children'].append(len(d['nodes']))
                d['nodes'].append({'name': obj['name'] + '_chunk_' + str(c['mesh']), 'mesh': mi})
                checks.append((c, used, inverse))
        d['asset']['generator'] = 'Bucha visually guided object partition; provisional labels'
        d.setdefault('extras', {})['semanticSegmentation'] = {
            'method': cfg['method'], 'scale': 'unverified source units', 'labelValidation': 'provisional'}
        write_glb(out / filename, d, binary)
        check, raw = read_glb(out / filename)
        assert check['materials'] == doc['materials']
        assert check['nodes'][0]['matrix'] == doc['nodes'][0]['matrix']
        for mesh, (c, used, inverse) in zip(check['meshes'], checks):
            p = mesh['primitives'][0]
            assert np.array_equal(accessor(check, raw, p['indices']).ravel(), inverse.ravel())
            for name, ai in c['primitive']['attributes'].items():
                assert np.array_equal(accessor(check, raw, p['attributes'][name]), accessor(doc, blob, ai)[used])
            assert p['material'] == c['primitive']['material']
        for a, b in zip(doc['images'], check['images']):
            va, vb = doc['bufferViews'][a['bufferView']], check['bufferViews'][b['bufferView']]
            assert blob[va.get('byteOffset', 0):va.get('byteOffset', 0) + va['byteLength']] == raw[vb.get('byteOffset', 0):vb.get('byteOffset', 0) + vb['byteLength']]

    export('segmented-scene.glb', cfg['objects'])
    entries = []
    for obj in cfg['objects']:
        export(obj['name'] + '.glb', [obj])
        mask = labels == obj['id']
        selected = tri[mask].reshape(-1, 3)
        entries.append({k: obj[k] for k in ['id', 'name', 'semantic_class', 'status', 'color']} | {
            'asset': obj['name'] + '.glb', 'triangles': int(mask.sum()),
            'surface_area_source_units_squared': float(area[mask].sum()),
            'bounds_world': [selected.min(0).tolist(), selected.max(0).tolist()] if len(selected) else None,
            'review_faces': int((mask & review).sum())})
    assert len(labels) == 183051 and sum(e['triangles'] for e in entries) == 183051
    np.savez_compressed(out / 'face-labels.npz', object_id=labels,
                        source_mesh=np.concatenate([np.full(len(c['faces']), c['mesh'], dtype=np.uint8) for c in chunks]),
                        source_face=np.concatenate([np.arange(len(c['faces']), dtype=np.uint32) for c in chunks]),
                        boundary_review=review.astype(np.uint8))
    manifest = {'source_sha256': source_hash, 'method': cfg['method'], 'validated_semantic_accuracy': False,
                'coordinate_frame': 'Original GLB world coordinates, +Z up; common transforms preserved',
                'meters_per_source_unit': None, 'objects': entries,
                'validation': {'all_source_faces_partitioned_once': True, 'original_vertex_attributes_exact': True,
                               'original_materials_and_embedded_images_exact': True, 'source_root_transform_exact': True},
                'limitations': ['Masks guided by inspection of this scan; no trained model run',
                                'Rubble regions can include ground between debris fragments',
                                'Perimeter facades are incomplete and their physical instance membership is unverified',
                                'No Unity physics or Magic Leap 2/robot integration test has run']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_glb')
    parser.add_argument('output_directory')
    parser.add_argument('--annotations', default=str(Path(__file__).with_name('annotations.json')))
    args = parser.parse_args()
    result = segment(args.input_glb, args.output_directory, args.annotations)
    for obj in result['objects']:
        print(f"{obj['name']}: {obj['triangles']:,} triangles")
