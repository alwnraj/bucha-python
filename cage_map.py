"""Procedural maps for the mocap cage, written as JSON for the Unity loader.

A map is the cage floor, a start zone, a goal zone and objects (an asset name plus a 4x4 pose).
A *ruleset* (JSON in rulesets/) says what kind of map to make: the layout, which object types,
how many, the spacing and the robot size. The same ruleset and seed always give the same map,
and every map is checked to have a path for a ground robot from start to goal.

Frame of every map ("cage" frame): metres, Y up, origin on the floor at the cage's back-left
corner, X along the 22 ft side, Z along the 30 ft side, Unity's left-handed axes. Matrices are
flat row-major lists of 16 numbers and map an asset's own coordinates (base at Y=0) into this frame.

Make maps:      python cage_map.py generate --ruleset rulesets/aisles.json --seeds 1-20 --out-dir DIR
Make a variant: python cage_map.py variant --map DIR/aisles-1.json --remove obj-03 --out DIR/aisles-1-v1.json
"""
import argparse
import copy
import fnmatch
import json
import math
import random
import struct
from collections import deque
from pathlib import Path

import numpy as np

CAGE_X, CAGE_Z = 6.71, 9.14  # 22 x 30 ft
KIT = Path(__file__).parent / 'data' / 'kenney-city-kit-industrial' / 'Models' / 'GLB format'
FRAME = ('metres; Y up; origin on the floor at the cage back-left corner; X along the 22 ft side, '
         'Z along the 30 ft side; Unity left-handed axes; matrix = 16 numbers row-major, asset base at Y=0')
CELL = 0.05  # path-check grid (m)
DEFAULTS = {
    'layout': {'type': 'scatter'},
    'objects': {'include': ['*'], 'count': [8, 8]},
    'spacing': {'gap_m': 0.6, 'border_m': 0.5},
    'robot': {'radius_m': 0.25},
    'zones': {'size_m': 0.8, 'min_separation_m': 5.0},
    'attempts': 200,
}


# ---------------------------------------------------------------- assets

def _node_matrix(node):
    m = np.eye(4)
    if 'matrix' in node:
        return np.array(node['matrix'], float).reshape(4, 4).T
    if 'rotation' in node:
        x, y, z, w = node['rotation']
        m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, :3] = m[:3, :3] * np.array(node.get('scale', [1, 1, 1]))
    m[:3, 3] = node.get('translation', [0, 0, 0])
    return m


def glb_bounds(path):
    """Axis-aligned bounds (min, max) of a GLB with node transforms applied, from accessor min/max."""
    data = Path(path).read_bytes()
    doc = json.loads(data[20:20 + struct.unpack('<I', data[12:16])[0]])
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)

    def visit(index, parent):
        nonlocal lo, hi
        node = doc['nodes'][index]
        world = parent @ _node_matrix(node)
        for prim in doc['meshes'][node['mesh']]['primitives'] if 'mesh' in node else []:
            acc = doc['accessors'][prim['attributes']['POSITION']]
            corners = np.array([[x, y, z, 1] for x in (acc['min'][0], acc['max'][0])
                                for y in (acc['min'][1], acc['max'][1]) for z in (acc['min'][2], acc['max'][2])])
            pts = (corners @ world.T)[:, :3]
            lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
        for child in node.get('children', []):
            visit(child, world)

    for root in doc['scenes'][doc.get('scene', 0)]['nodes']:
        visit(root, np.eye(4))
    return lo, hi


def catalog(kit=KIT):
    """{asset name: (min, max)} in Unity's axes for every GLB in the kit.

    glTF is right-handed; Unity imports the kit (FBX or glTF) with X mirrored, which matters for
    models whose bounds are off-centre. Checked in Unity batch mode against renderer bounds."""
    out = {}
    for p in sorted(Path(kit).glob('*.glb')):
        lo, hi = glb_bounds(p)
        out[p.stem] = (np.array([-hi[0], lo[1], lo[2]]), np.array([-lo[0], hi[1], hi[2]]))
    return out


def pose(x, z, yaw_deg, scale=1.0):
    """Flat row-major 4x4 (16 numbers): scale, yaw about +Y (Unity's convention), then move."""
    a = math.radians(yaw_deg)
    c, s = round(math.cos(a)), round(math.sin(a))  # yaws are multiples of 90
    return [c * scale, 0, s * scale, x, 0, scale, 0, 0, -s * scale, 0, c * scale, z, 0, 0, 0, 1]


def footprint(bounds, matrix):
    """(xmin, zmin, xmax, zmax) of the asset's bounds placed by `matrix`."""
    lo, hi = bounds
    corners = np.array([[x, 0, z, 1] for x in (lo[0], hi[0]) for z in (lo[2], hi[2])])
    pts = corners @ np.array(matrix, float).reshape(4, 4).T
    return pts[:, 0].min(), pts[:, 2].min(), pts[:, 0].max(), pts[:, 2].max()


def place_at(bounds, cx, cz, yaw):
    """Matrix that puts the asset's footprint centre at (cx, cz) with the given yaw."""
    x0, z0, x1, z1 = footprint(bounds, pose(0, 0, yaw))
    return pose(cx - (x0 + x1) / 2, cz - (z0 + z1) / 2, yaw)


# ---------------------------------------------------------------- rules

def load_ruleset(path_or_dict):
    rules = json.loads(Path(path_or_dict).read_text()) if not isinstance(path_or_dict, dict) else path_or_dict
    merged = copy.deepcopy(DEFAULTS)
    for key, value in rules.items():
        merged[key] = {**merged[key], **value} if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    merged.setdefault('name', Path(path_or_dict).stem if not isinstance(path_or_dict, dict) else 'custom')
    return merged


def _overlaps(a, b, gap):
    gap -= 1e-3  # footprints are stored rounded to 0.1 mm
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def _zone_rect(zone):
    (x, z), h = zone['center'], zone['size'] / 2
    return (x - h, z - h, x + h, z + h)


def _pick_zones(rng, rules, start_z=None, goal_z=None):
    """Random start and goal squares inside the cage; optional (zmin, zmax) range for each centre."""
    size, border = rules['zones']['size_m'], rules['spacing']['border_m']
    lo = max(border, 0.1) + size / 2
    start_z, goal_z = start_z or (lo, CAGE_Z - lo), goal_z or (lo, CAGE_Z - lo)
    for _ in range(1000):
        s = (rng.uniform(lo, CAGE_X - lo), rng.uniform(*start_z))
        g = (rng.uniform(lo, CAGE_X - lo), rng.uniform(*goal_z))
        if math.dist(s, g) >= rules['zones']['min_separation_m']:
            r = lambda p: [round(p[0], 3), round(p[1], 3)]
            return {'center': r(s), 'size': size}, {'center': r(g), 'size': size}
    raise RuntimeError('cannot fit start and goal zones; lower zones.min_separation_m')


def _candidates(rng, rules, names, cat):
    """Yield (asset, matrix) proposals for the layout; `place` decides which are kept."""
    layout = rules['layout']
    kind = layout['type']
    border = rules['spacing']['border_m']
    if kind == 'scatter':
        while True:
            name = rng.choice(names)
            yield name, pose(rng.uniform(border, CAGE_X - border), rng.uniform(border, CAGE_Z - border),
                             rng.choice((0, 90, 180, 270)))
    elif kind == 'rows':
        # Rows of objects along Z separated by aisles; each slot may be left empty.
        aisle, fill = layout.get('aisle_m', 1.2), layout.get('fill', 0.8)
        depth = layout.get('row_depth_m', 0.9)
        x = border + depth / 2 + rng.uniform(0, 0.3)
        while x + depth / 2 <= CAGE_X - border:
            z = border
            while z < CAGE_Z - border:
                name = rng.choice(names)
                yaw = _long_side_along(cat[name], 'z', rng)
                x0, z0, x1, z1 = footprint(cat[name], pose(0, 0, yaw))
                length = z1 - z0
                if rng.random() < fill:
                    yield name, place_at(cat[name], x, z + length / 2, yaw)
                z += length + rules['spacing']['gap_m']
            x += depth + aisle
    elif kind == 'baffles':
        # Walls across the cage (along X), each with one opening at a random place.
        joint = layout.get('joint_m', 0.05)
        for zc, gap_at in _baffle_walls(rng, rules):
            x, segment_end = border, gap_at - layout.get('opening_m', 1.2) / 2
            while True:
                room = segment_end - x
                fitting = [n for n in names if _width_along(cat[n], 'x') <= room + 1e-9]
                if not fitting:
                    if segment_end >= CAGE_X - border:
                        break
                    x, segment_end = gap_at + layout.get('opening_m', 1.2) / 2, CAGE_X - border
                    continue
                name = rng.choice(fitting)
                yaw = _long_side_along(cat[name], 'x', rng)
                width = _width_along(cat[name], 'x')
                yield name, place_at(cat[name], x + width / 2, zc, yaw)
                x += width + joint
    else:
        raise ValueError(f'unknown layout type {kind!r}; use scatter, rows or baffles')


def _baffle_walls(rng, rules):
    """(z, opening centre x) of each wall; drawn once per attempt so zones can avoid them."""
    layout, border = rules['layout'], rules['spacing']['border_m']
    count, opening = layout.get('walls', 3), layout.get('opening_m', 1.2)
    if '_walls' not in layout:
        zs = np.linspace(0, CAGE_Z, count + 2)[1:-1]
        layout['_walls'] = [(float(z + rng.uniform(-0.3, 0.3)),
                             rng.uniform(border + opening / 2, CAGE_X - border - opening / 2)) for z in zs]
    return layout['_walls']


def _width_along(bounds, axis):
    """Length of the asset's longer footprint side (it is turned to lie along `axis`)."""
    x0, z0, x1, z1 = footprint(bounds, pose(0, 0, 0))
    return max(x1 - x0, z1 - z0)


def _long_side_along(bounds, axis, rng):
    x0, z0, x1, z1 = footprint(bounds, pose(0, 0, 0))
    along_x = (x1 - x0) >= (z1 - z0)
    yaws = (0, 180) if along_x == (axis == 'x') else (90, 270)
    return rng.choice(yaws)


def path_exists(objects, start, goal, radius, cat=None):
    """Is there a route for a disc of `radius` from the start centre to the goal centre?"""
    nx, nz = int(math.ceil(CAGE_X / CELL)), int(math.ceil(CAGE_Z / CELL))
    xs, zs = (np.arange(nx) + 0.5) * CELL, (np.arange(nz) + 0.5) * CELL
    free = np.ones((nx, nz), bool)
    free[(xs < radius) | (xs > CAGE_X - radius), :] = False
    free[:, (zs < radius) | (zs > CAGE_Z - radius)] = False
    for o in objects:
        x0, z0, x1, z1 = o['footprint']
        bx = (xs > x0 - radius) & (xs < x1 + radius)
        bz = (zs > z0 - radius) & (zs < z1 + radius)
        free[np.ix_(bx, bz)] = False
    cell = lambda p: (min(int(p[0] / CELL), nx - 1), min(int(p[1] / CELL), nz - 1))
    s, g = cell(start['center']), cell(goal['center'])
    if not free[s] or not free[g]:
        return False
    seen = np.zeros_like(free)
    seen[s] = True
    queue = deque([s])
    while queue:
        i, j = queue.popleft()
        if (i, j) == g:
            return True
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < nx and 0 <= b < nz and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                queue.append((a, b))
    return False


def _object(i, name, matrix, cat):
    lo, hi = cat[name]
    return {'id': f'obj-{i:02d}', 'asset': name, 'scale': 1.0, 'matrix': [round(float(v), 6) for v in matrix],
            'size': [round(float(v), 4) for v in hi - lo],
            'footprint': [round(float(v), 4) for v in footprint(cat[name], matrix)]}


def _fits(fp, placed, zones, rules):
    border, gap = rules['spacing']['border_m'], rules['spacing']['gap_m']
    clear = max(gap, 2 * rules['robot']['radius_m'])
    if fp[0] < border - 1e-9 or fp[1] < border - 1e-9 or fp[2] > CAGE_X - border + 1e-9 or fp[3] > CAGE_Z - border + 1e-9:
        return False
    if any(_overlaps(fp, _zone_rect(z), clear) for z in zones):
        return False
    return not any(_overlaps(fp, o['footprint'], gap) for o in placed)


def generate(rules, seed, kit=KIT):
    """One map from a ruleset (dict or path) and a seed. Retries until start-to-goal is passable."""
    rules = load_ruleset(rules)
    rng = random.Random(seed)
    cat = catalog(kit)
    names = sorted(n for n in cat if any(fnmatch.fnmatch(n, p) for p in rules['objects']['include']))
    if not names:
        raise ValueError(f'no kit model matches {rules["objects"]["include"]}')
    lo_count, hi_count = rules['objects']['count']
    for attempt in range(rules['attempts']):
        rules['layout'].pop('_walls', None)
        if rules['layout']['type'] == 'baffles':
            walls = _baffle_walls(rng, rules)
            room = rules['zones']['size_m'] / 2 + 2 * rules['robot']['radius_m'] + 0.45
            ends = [(0.5, walls[0][0] - room), (walls[-1][0] + room, CAGE_Z - 0.5)]
            if rng.random() < 0.5:
                ends.reverse()
            start, goal = _pick_zones(rng, rules, *ends)
        else:
            start, goal = _pick_zones(rng, rules)
        want = rng.randint(lo_count, hi_count) if rules['layout']['type'] == 'scatter' else None
        placed, proposals = [], 0
        for name, m in _candidates(rng, rules, names, cat):
            proposals += 1
            if want is not None and (len(placed) == want or proposals > 2000):
                break
            fp = footprint(cat[name], m)
            if _fits(fp, placed, (start, goal), rules):
                placed.append(_object(len(placed), name, m, cat))
        if want is not None and len(placed) < want:
            continue
        if placed and path_exists(placed, start, goal, rules['robot']['radius_m']):
            rules['layout'].pop('_walls', None)
            return {'version': 2, 'frame': FRAME, 'seed': seed, 'ruleset': rules, 'attempt': attempt,
                    'floor': {'size_x': CAGE_X, 'size_z': CAGE_Z}, 'start': start, 'goal': goal,
                    'path_check': {'robot_radius_m': rules['robot']['radius_m'], 'passable': True},
                    'asset_source': 'Kenney City Kit (Industrial) 2.0, CC0', 'objects': placed}
    raise RuntimeError(f'no passable map in {rules["attempts"]} attempts; loosen the ruleset')


def variant(base, remove=(), move=None, swap=None, kit=KIT):
    """A copy of `base` with objects removed, moved ({id: (x, z)} footprint centre) or swapped
    ({id: asset}); everything else identical. Raises if the result overlaps or blocks the path."""
    cat = catalog(kit)
    m = copy.deepcopy(base)
    rules = load_ruleset(m['ruleset'])
    ids = {o['id'] for o in m['objects']}
    for i in list(remove) + list(move or {}) + list(swap or {}):
        if i not in ids:
            raise KeyError(f'no object {i} in map')
    kept = []
    for o in m['objects']:
        if o['id'] in remove:
            continue
        if o['id'] in (move or {}) or o['id'] in (swap or {}):
            asset = (swap or {}).get(o['id'], o['asset'])
            fp = o['footprint']
            cx, cz = (move or {}).get(o['id'], ((fp[0] + fp[2]) / 2, (fp[1] + fp[3]) / 2))
            yaw = round(math.degrees(math.atan2(o['matrix'][2], o['matrix'][0]))) % 360
            o = {**_object(0, asset, place_at(cat[asset], cx, cz, yaw), cat), 'id': o['id']}
            others = [p for p in m['objects'] if p['id'] != o['id'] and p['id'] not in remove]
            if not _fits(o['footprint'], others, (m['start'], m['goal']), rules):
                raise ValueError(f'{o["id"]} would overlap another object, a zone or the border')
        kept.append(o)
    if not path_exists(kept, m['start'], m['goal'], rules['robot']['radius_m']):
        raise ValueError('variant blocks the path from start to goal')
    m['objects'] = kept
    m['variant_of'] = {'seed': base['seed'], 'ruleset': base['ruleset']['name'],
                       'removed': list(remove), 'moved': {k: list(v) for k, v in (move or {}).items()},
                       'swapped': dict(swap or {})}
    return m


def preview(m, path, k=60):
    """Top-down PNG: objects grey, start green, goal red, the cage's back edge (Z = 0) at the bottom."""
    from PIL import Image, ImageDraw
    w, h = int(CAGE_X * k) + 20, int(CAGE_Z * k) + 40
    im = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(im)
    at = lambda x, z: (10 + x * k, h - 10 - z * k)
    d.rectangle((*at(0, CAGE_Z), *at(CAGE_X, 0)), outline='black', width=2)
    for o in m['objects']:
        x0, z0, x1, z1 = o['footprint']
        d.rectangle((*at(x0, z1), *at(x1, z0)), fill=(150, 150, 170), outline='black')
        d.text(at(x0 + 0.03, z1 - 0.03), o['id'][4:], fill='black')
    for name, colour in (('start', (0, 160, 0)), ('goal', (220, 0, 0))):
        (x, z), r = m[name]['center'], m[name]['size'] / 2
        d.rectangle((*at(x - r, z + r), *at(x + r, z - r)), outline=colour, width=4)
        d.text(at(x - r + 0.05, z + r - 0.05), name, fill=colour)
    d.text((10, 6), f"{m['ruleset']['name']} seed {m['seed']}", fill='black')
    im.save(path)


# ---------------------------------------------------------------- command line

def _seeds(text):
    out = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def _write(m, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=1) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate', help='make one map per seed from a ruleset')
    g.add_argument('--ruleset', required=True)
    g.add_argument('--seeds', default='1', help='e.g. 1-20 or 1,5,9')
    g.add_argument('--out-dir', required=True)
    g.add_argument('--png', action='store_true', help='also save a top-down picture of each map')
    v = sub.add_parser('variant', help='copy a map with objects removed, moved or swapped')
    v.add_argument('--map', required=True)
    v.add_argument('--remove', nargs='*', default=[], metavar='ID')
    v.add_argument('--move', nargs=3, action='append', default=[], metavar=('ID', 'X', 'Z'))
    v.add_argument('--swap', nargs=2, action='append', default=[], metavar=('ID', 'ASSET'))
    v.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.cmd == 'generate':
        rules = load_ruleset(args.ruleset)
        for seed in _seeds(args.seeds):
            m = generate(rules, seed)
            out = Path(args.out_dir) / f'{rules["name"]}-{seed}.json'
            _write(m, out)
            if args.png:
                preview(m, out.with_suffix('.png'))
            print(f'{out}: {len(m["objects"])} objects, start {m["start"]["center"]}, goal {m["goal"]["center"]}')
    else:
        base = json.loads(Path(args.map).read_text())
        m = variant(base, args.remove, {i: (float(x), float(z)) for i, x, z in args.move}, dict(args.swap))
        _write(m, args.out)
        print(f'{args.out}: {len(m["objects"])} objects ({m["variant_of"]})')


if __name__ == '__main__':
    main()
