# Handoff

Prepared 2026-09-22. Read `AGENTS.md`, `agent.md`, then this file.

## Purpose

Continue a Python-first workflow for splitting the supplied Bucha photogrammetry scan into separate objects, preserving their textures and relative coordinates, and giving each object queryable collision geometry. Everything in the current workflow must run from Python scripts in VS Code.

The user wants a new agent session with this repository as its workspace. No further implementation change was specified after requesting this handoff; start by explaining the current state briefly and asking what they want to improve next.

## Current state

The source project, input scan, annotations, VS Code configuration and pinned requirements are self-contained. `run.py` regenerates the full output folder. It uses NumPy and Pillow; no browser, Node.js, GPU model, Unity project or ML2 connection is needed.

This is a **visually guided, scan-specific baseline**, not automatic semantic inference on arbitrary files. Editable polygons and geometric thresholds in `annotations.json` assign faces to six groups:

| ID | Object | Faces |
| --- | --- | ---: |
| 1 | building_01_central | 78,967 |
| 2 | building_02_south_east | 21,408 |
| 3 | building_03_north | 21,280 |
| 4 | ground_roads | 18,976 |
| 5 | rubble | 37,877 |
| 6 | uncertain_other | 4,543 |

All 183,051 triangles are partitioned once. The exports preserve original positions, normals, UVs, materials, image bytes and the root transform. Individual GLBs retain a common coordinate frame.

Collision exports are one static triangle surface per group, in NPZ and OBJ formats. `collision.py` returns the nearest ray hit with object ID, class, position and source-face reference. It does not implement robot dynamics, clearance, path planning or sensor transport.

The user explicitly abandoned the website prototype. Earlier slides, browser experiments and Unity C# helpers remain outside this repository in the original workspace; they are not dependencies and should not be revived without a request.

## Important files

| File | Read it for |
| --- | --- |
| `README.md`, `config.json`, `.vscode/` | Setup, commands, paths and F5 configurations |
| `run.py`, `segment_objects.py`, `annotations.json` | Pipeline orchestration, face partition/export, editable scan-specific masks |
| `collision.py`, `render_previews.py` | Collision surfaces/rays and Python PNG previews |
| `tests/test_workflow.py`, `requirements.txt`, `setup_env.py` | Meaningful regression checks and reproducible environment |
| `data/*.glb`, `source-license.json` | User-supplied source asset and provenance evidence |

Generated results are under `results/`, which is ignored by Git. Inspect `results/previews/scene-labeled.png`, `objects-preview.png`, `annotation-map.png`, `manifest.json` and `run-report.json` after a run. A fresh checkout has no results or virtual environment.

## Run and verification

Use Python 3.11–3.13; tested with 3.12.14. On this machine a local `.venv` is already available. For a fresh checkout:

```sh
python3.12 setup_env.py
# Select .venv in VS Code, or activate it in the terminal.
python run.py
python -m unittest discover -s tests -v
python collision.py --origin 10 0 20 --direction 0 0 -1
```

The full run completed in approximately 5.7 seconds on the current Mac. Seven tests passed with NumPy 2.3.5 and Pillow 11.3.0. Checks cover the source-face partition, exact collision-file geometry, nearest-hit identity, non-unit directions, both triangle sides, misses, limits and invalid rays.

The example ray above hits rubble (ID 5). A ray from `7 8 20` in direction `0 0 -1` hits the central building (ID 1). Running from a different working directory with an explicit output folder also passed.

The F5 launch configurations were prepared; the scripts were verified from the terminal, not through an automated VS Code UI test. Previews were visually inspected. PNG rendering uses average-depth face ordering and per-face texture sampling, so it is approximate; GLB texture mapping is preserved exactly.

Default count tests intentionally assume the supplied annotations. If masks change, review the new result and update expected counts deliberately, not just to silence a failure.

## Known limits and decisions

- Physical scale is unknown. Buildings protrude toward source world **+Z** after the original GLB root transform. Never call source units meters without calibration.
- The central building is a clearer instance than the incomplete perimeter facades. South/east facade membership as one physical building is provisional.
- Rubble masks describe debris-covered regions and can include ground or vegetation between fragments. Ground labels do not establish traversability. No ground-truth semantic accuracy has been measured.
- The scan has holes and incomplete backs. Collision meshes retain these, are not watertight solids, and are not simplified.
- ML2 means **Magic Leap 2**. The user has no Unity project and asked for the easiest Python workflow for now. Future engine components, coordinate registration, robot interface and ML2 integration are unimplemented.

The segmentation script intentionally checks the supplied GLB's SHA-256:
`488e2890ab5fa2b464d7b794b37e1b663d88ce7edd2d478135850108d3d7ae93`.
Removing this guard alone would apply the wrong spatial masks to other scans.

## Source and research context

The asset is “Destroyed Apartments, Bucha Ukraine War 2022” by LibanCiel. Public [Sketchfab metadata](https://api.sketchfab.com/v3/models/f9dbc156ba95471c840c2d3e52880629) lists the Standard license; the matching title and 183,051-face count support the match. Full license-page terms and the original acquisition receipt were not retrieved. Keep `source-license.json` and check acquisition terms before public redistribution.

Potential learned approaches were researched, not executed: [SensatUrban](https://github.com/QingyongHu/SensatUrban), [STPLS3D](https://www.stpls3d.com/data), [OpenMask3D](https://openmask3d.github.io/), and [Open3DIS](https://open3dis.github.io/). The first two are datasets with baseline methods, not plug-in segmenters. Any learned approach needs an appropriate checkpoint, preprocessing, face-label projection and evaluation on damaged outdoor scans.

## Next session

1. Read this handoff and `agent.md`; use this repository as the working directory.
2. Confirm the user's next focus. Candidate directions are improving masks, a lightweight Python review tool, or evaluating a learned segmentation baseline. These are suggestions, not approved new implementation scope.
3. Reproduce the baseline if the environment changed or work requires it, then make one small requested improvement while preserving face mapping and coordinate consistency.

No additional skill is needed beyond the default Python coding workflow. Use a debugging skill only if an actual failure needs diagnosis. Follow the user's constraints in `agent.md`; do not spawn subagents.

## Repository handoff

The repository is initialized locally on `main` with a baseline commit. No remote, pull request or hosted repository has been created. Use `git log -1` for the current commit. The source GLB is tracked for reproducibility; `.venv/`, caches and generated results are excluded. The accompanying ZIP is a source snapshot and the Git bundle retains history.
