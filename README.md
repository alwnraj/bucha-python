# 3D object identification and mocap cage maps

New agent session: read **`AGENTS.md` → `agent.md` → `handoff.md`** (latest section first). Local Git repository, no remote.

## Procedural cage maps in Unity (current work, branch `unity-maps`)

A **map** is the mocap cage floor (22 × 30 ft = 6.71 × 9.14 m), a green **start** square, a red **goal** square and virtual objects, each a Kenney *City Kit (Industrial)* model with a 4×4 pose. Python writes maps from **rulesets**; Unity builds them with colliders; tracked bodies from Motive move in them live and every touch with an object is logged.

**Map frame** (written into every map file): metres, Y up, origin on the floor at the cage's back-left corner, X along the 22 ft side, Z along the 30 ft side, Unity's left-handed axes. `matrix` is 16 numbers, row-major.

1. **Make maps from a ruleset.** A ruleset (`rulesets/*.json`) describes a *kind* of map; each seed gives a different map of that kind, and the same ruleset + seed always gives the same map. `--png` also saves a top-down picture of each map.

   ```sh
   .venv/bin/python cage_map.py generate --ruleset rulesets/baffles.json --seeds 1-20 --out-dir unity/CageMaps/Assets/StreamingAssets/maps --png
   ```

   | Ruleset | Layout |
   |---|---|
   | `scatter.json` | 6–10 random objects anywhere |
   | `aisles.json` | warehouse rows along the long side with aisles between, some slots empty |
   | `baffles.json` | three walls across the cage, each with one 1.2 m opening; start and goal at opposite ends, so the robot must weave |

   Every map is **checked to be passable**: a ground robot (a disc of `robot.radius_m`, default 0.25 m) must be able to get from the start square to the goal square while staying inside the cage. A map that fails is redrawn. Start and goal are random anywhere in the cage (at least `zones.min_separation_m` apart) and kept clear of objects.

   Ruleset fields: `layout.type` (`scatter`, `rows`, `baffles`) and its options, `objects.include` (model-name patterns such as `"shipping-container-*"`; names are the files in `Assets/Resources/Kenney/`), `objects.count` (`[min, max]`, scatter only), `spacing.gap_m` / `spacing.border_m`, `robot.radius_m`, `zones.size_m` / `zones.min_separation_m`. Copy a ruleset file to make a new kind of map.

   **Variants** (same map with one change, for counterfactual runs): remove, move (to a footprint centre X Z) or swap objects by id. The new file records what changed, and is rejected if it overlaps something or blocks the path.

   ```sh
   M=unity/CageMaps/Assets/StreamingAssets/maps
   .venv/bin/python cage_map.py variant --map $M/baffles-1.json --remove obj-03 --out $M/baffles-1-no03.json
   .venv/bin/python cage_map.py variant --map $M/baffles-1.json --swap obj-05 detail-tank --move obj-00 1.0 4.5 --out $M/baffles-1-v2.json
   ```

2. **Open the project in Unity:** Unity Hub → *Projects* → *Add* → *Add project from disk* → choose `unity/CageMaps` (editor 6000.6.9f1). The first open takes a minute while Unity imports.
3. **Build the scene from the map:** menu *Cage → Build scene from map...* → pick the JSON. This writes `Assets/Scenes/Cage.unity` (open it from the *Project* window if it isn't open) and prints a check to the *Console* window. Each object is static with convex mesh colliders, so a body inside one also counts as touching.
4. **Press Play** (▶ at the top). The *Mocap (VRPN)* object connects read-only to Motive at `192.168.1.42:3883`. The white box *Tracked modalai* follows that rigid body; without mocap data, drive it with the arrow keys and PageUp/PageDown. Touches print `CONTACT start ...` / `CONTACT end ...` in the Console, and the current state shows in the top-left of the Game view. Add more tracked bodies by duplicating *Tracked modalai* and changing *Body Name*.

**Motive → cage calibration** (measured 2026-11-18, about ±0.15 m): Motive +Z points to the cage front, Motive −X to the right side, and *Motive Origin In Cage* is (3.01, 0, 4.74). The cage frame's origin is the back-left corner, so the left side is inferred from the 22 ft width. If Motive's ground plane is reset, re-measure by resting a tracked body at edge middles and update those fields on *Tracked modalai*. Motive only streams bodies its cameras currently see.

**Checks without opening Unity** (batch mode; the second needs the scene from the first):

```sh
U=/Applications/Unity/Hub/Editor/6000.6.9f1/Unity.app/Contents/MacOS/Unity
$U -batchmode -nographics -projectPath unity/CageMaps -executeMethod CageBuild.BuildAndCheck -map scatter-1.json -logFile -
$U -batchmode -nographics -projectPath unity/CageMaps -executeMethod CageBuild.PlayCheck -logFile -
```

The first checks every object is found by a collision query, sits on the floor and matches its map footprint within 1 cm, and that the start and goal squares are shown without colliders; the second enters Play mode, drives the tracked box into an object and out again, and expects both contact events. `tests/test_cage_map.py` checks the generator: every ruleset over several seeds (inside the cage, no overlaps, zones clear, path exists), repeatability, baffle openings, the path check itself and variants.

**Assets:** Kenney City Kit (Industrial) 2.0, CC0, zip SHA-256 `5b381164e5760f3830a2dbee43b972deee38b2a695d091b56e238ab2910c96d2`. FBX copies are in `unity/CageMaps/Assets/Resources/Kenney/`; the generator reads sizes from the GLB copies in `data/kenney-city-kit-industrial/` (not committed; re-download from kenney.nl). The Unity project uses only packages built into the editor (`Packages/manifest.json`).

Read live poses from Motive without Unity (read-only, no packages): `python3 vrpn_peek.py modalai 192.168.1.42 5`.

## Setup

Python 3.12. In VS Code, select the `.venv` interpreter.

```sh
python3.12 setup_env.py                                                # NumPy 2.3.5, Pillow 11.3.0
.venv/bin/python -m pip install --no-deps -r requirements-vision.txt   # ONNX Runtime 1.23.2, DracoPy 2.2.0
.venv/bin/python -m pip install --no-deps -r requirements-live.txt     # OpenCV 4.11 (camera, archive)
.venv/bin/python download_vision_model.py --model b5                   # SegFormer-B5, 340 MB
```

`models/`, `results*/` and the scans in `data/` (except the Bucha asset) are not committed.

## GLB scan tools (paused)

Input is any triangle `.glb`. Convert Draco-compressed files first with `prepare_glb.py --input in.glb --output plain.glb`.

| Tool | What it does |
|---|---|
| `split_scene.py --input X.glb --output DIR` | **No labels.** Cuts away floors, walls and ceilings; each remaining connected piece becomes `object_NNN.glb`. `--labels` names objects with SegFormer (experimental, weak indoors). |
| `extract_objects.py --input X.glb --output DIR` | **Outdoor, labeled.** Renders views inside the scan, votes SegFormer labels onto faces, splits and merges objects (`building_01.glb`, `tree_02.glb` …). |
| `run.py --inventory --input X.glb --output DIR` | Connected-surface candidates only, all `unknown`. |
| `run.py` | Bucha regression with hand annotations (`config.json`, `annotations.json`); `--labels`, `--corrections` and `--review` take hash-guarded JSON. |
| `collision.py --scene DIR/collision-scene.json --origin X Y Z --direction DX DY DZ` | Ray query against exported colliders. |

Outputs per run: one textured GLB per object (only the texels it uses), `objects.json` (position, bounds, size), `colliders/` (open surfaces), `closed/` (watertight voxel solids), `objects-sheet.png` thumbnails; `split_scene.py` adds `structure_*.glb` and a colour-coded `scene-split.glb`.

Results (eyeballed, not validated against hand labels):

- **HM3D house, `split_scene`:** 327 objects in 36 s. Furniture separates; blinds, stair parts and door frames also appear as objects.
- **Flat-roof drone scan (5.5M faces), `split_scene`:** 519 objects in 3.5 min. HVAC units, shrubs and posts are clean; roof-edge trims are noise.
- **Flat-roof, `extract_objects`:** 85 labeled objects in 31 min.
- **Bucha, `extract_objects`:** 17 objects; buildings and ground are fine, rubble is mislabeled.

Limits: touching objects stay one object; scans keep holes and unscanned backs; coordinates are in source units with no robot registration; ray queries are not physics or navigation.

## Single-photo depth tools

- **`image_depth.py photo.jpg --output DIR`:** relative depth from one photo (Depth Anything V2 Small; run `download_depth_model.py` once first).
- **`depth_demo_server.py`:** local web demo at http://127.0.0.1:8765, with optional metric depth from `download_metric_models.py`.
- **`live_depth.py`:** webcam benchmark.

Depth values are model estimates, not measurements.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests                            # 46 tests
PYTHONPATH=. .venv/bin/python -m unittest discover -s archive/gator/tests  # 31 archived
```

Tests are structural (geometry, files, pipelines on synthetic scenes), not accuracy evaluations.

## Archived: GATOR-style photos-to-object pipeline

Paused and moved to [`archive/gator/`](archive/gator/README.md), with its own README and tests.

## Assets and licences

- **Bucha scan:** "Destroyed Apartments, Bucha Ukraine War 2022" by LibanCiel ([Sketchfab](https://sketchfab.com/3d-models/destroyed-apartments-bucha-ukraine-war-2022-f9dbc156ba95471c840c2d3e52880629)), Standard licence; see `source-license.json`.
- **Flat-roof scan:** [Matt1Up drone-building-scans](https://huggingface.co/datasets/Matt1up/drone-building-scans), CC BY 4.0, attribution required.
- **HM3D homes:** Matterport's terms apply. Not committed.
- **SegFormer (ADE20K):** NVIDIA licence, noncommercial research or evaluation only.
- **Depth Anything V2 Small:** Apache-2.0. Metric-model provenance is pinned in `download_metric_models.py`.
