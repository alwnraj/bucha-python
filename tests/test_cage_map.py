import fnmatch
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import cage_map

ROOT = Path(cage_map.__file__).parent
RULESETS = sorted((ROOT / 'rulesets').glob('*.json'))


@unittest.skipUnless(cage_map.KIT.is_dir(), 'Kenney kit not downloaded')
class CageMapTest(unittest.TestCase):
    def check_map(self, m):
        cat = cage_map.catalog()
        rules = m['ruleset']
        border, gap = rules['spacing']['border_m'], rules['spacing']['gap_m']
        boxes = []
        for o in m['objects']:
            self.assertTrue(any(fnmatch.fnmatch(o['asset'], p) for p in rules['objects']['include']))
            x0, z0, x1, z1 = cage_map.footprint(cat[o['asset']], o['matrix'])
            np.testing.assert_allclose([x0, z0, x1, z1], o['footprint'], atol=1e-3)
            self.assertTrue(border - 1e-6 <= x0 and x1 <= cage_map.CAGE_X - border + 1e-6, o['id'])
            self.assertTrue(border - 1e-6 <= z0 and z1 <= cage_map.CAGE_Z - border + 1e-6, o['id'])
            base = (np.array(o['matrix']).reshape(4, 4) @ [0, cat[o['asset']][0][1], 0, 1])[1]
            self.assertAlmostEqual(base, 0, places=3)
            for zone in (m['start'], m['goal']):
                self.assertFalse(cage_map._overlaps((x0, z0, x1, z1), cage_map._zone_rect(zone), 0), o['id'])
            for b in boxes:
                self.assertFalse(cage_map._overlaps((x0, z0, x1, z1), b, gap), o['id'])
            boxes.append((x0, z0, x1, z1))
        self.assertTrue(cage_map.path_exists(m['objects'], m['start'], m['goal'], rules['robot']['radius_m']))

    def test_every_ruleset_makes_valid_maps(self):
        self.assertEqual({p.stem for p in RULESETS}, {'scatter', 'aisles', 'baffles'})
        for path in RULESETS:
            for seed in range(4):
                with self.subTest(ruleset=path.stem, seed=seed):
                    m = cage_map.generate(path, seed)
                    self.assertGreater(len(m['objects']), 3)
                    self.assertNotIn('_walls', m['ruleset']['layout'])
                    self.check_map(m)

    def test_same_seed_same_map(self):
        for path in RULESETS:
            self.assertEqual(cage_map.generate(path, 3), cage_map.generate(path, 3))
            self.assertNotEqual(cage_map.generate(path, 3)['objects'], cage_map.generate(path, 4)['objects'])

    def test_baffles_force_the_robot_through_openings(self):
        m = cage_map.generate(ROOT / 'rulesets' / 'baffles.json', 2)
        zs = sorted({round((o['footprint'][1] + o['footprint'][3]) / 2, 1) for o in m['objects']})
        lo, hi = sorted((m['start']['center'][1], m['goal']['center'][1]))
        self.assertTrue(lo < zs[0] and zs[-1] < hi)  # start and goal lie beyond the walls
        # Closing every opening with a wide object blocks the path.
        blocked = m['objects'] + [{'footprint': [0, z - 0.2, cage_map.CAGE_X, z + 0.2]} for z in zs]
        self.assertFalse(cage_map.path_exists(blocked, m['start'], m['goal'], 0.25))

    def test_path_check(self):
        start, goal = {'center': [1, 1], 'size': 0.5}, {'center': [1, 8], 'size': 0.5}
        wall = {'footprint': [0, 4, cage_map.CAGE_X, 4.5]}
        self.assertTrue(cage_map.path_exists([], start, goal, 0.25))
        self.assertFalse(cage_map.path_exists([wall], start, goal, 0.25))
        gap = [{'footprint': [0, 4, 3, 4.5]}, {'footprint': [3.6, 4, cage_map.CAGE_X, 4.5]}]
        self.assertTrue(cage_map.path_exists(gap, start, goal, 0.25))   # 0.6 m opening, 0.5 m robot
        self.assertFalse(cage_map.path_exists(gap, start, goal, 0.35))  # 0.7 m robot does not fit

    def test_variant_remove_move_swap(self):
        base = cage_map.generate(ROOT / 'rulesets' / 'scatter.json', 1)
        first, second = base['objects'][0]['id'], base['objects'][1]['id']
        v = cage_map.variant(base, remove=[first])
        self.assertEqual(len(v['objects']), len(base['objects']) - 1)
        self.assertEqual(v['objects'], base['objects'][1:])
        self.assertEqual(v['variant_of']['removed'], [first])
        v = cage_map.variant(base, swap={second: 'chimney-basic'})
        self.assertEqual(next(o for o in v['objects'] if o['id'] == second)['asset'], 'chimney-basic')
        self.check_map(v)
        with self.assertRaises(ValueError):  # onto the start zone
            cage_map.variant(base, move={second: tuple(base['start']['center'])})
        with self.assertRaises(KeyError):
            cage_map.variant(base, remove=['obj-99'])

    def test_command_line_writes_maps_and_pictures(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, 'cage_map.py', 'generate', '--ruleset', 'rulesets/aisles.json',
                            '--seeds', '1-2', '--out-dir', tmp, '--png'], check=True, capture_output=True,
                           cwd=Path(cage_map.__file__).parent)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ['aisles-1.json', 'aisles-1.png', 'aisles-2.json', 'aisles-2.png'])
            self.check_map(json.loads((Path(tmp) / 'aisles-1.json').read_text()))

    def test_all_assets_have_base_at_zero(self):
        for name, (lo, hi) in cage_map.catalog().items():
            self.assertAlmostEqual(lo[1], 0, places=3, msg=name)

    def test_pose_yaw_matches_unity(self):
        # Unity: Quaternion.Euler(0, 90, 0) maps +X to -Z and +Z to +X.
        m = np.array(cage_map.pose(1, 2, 90)).reshape(4, 4)
        np.testing.assert_allclose(m @ [1, 0, 0, 1], [1, 0, 1, 1])
        np.testing.assert_allclose(m @ [0, 0, 1, 1], [2, 0, 2, 1])


if __name__ == '__main__':
    unittest.main()
