# Session journal

A dated log of what was done, newest first. One entry per working session: what was asked, what was done, what was found, what is left. `handoff.md` holds the current state and details; this file is the running history.

## 2026-11-19

- **Unity walkthrough.** Labelled screenshots of the Unity window. Covered Scene-view navigation on a trackpad (Map + F to reset the view) and how the floor is solid: a 10 cm box collider; objects are static with colliders. Explained why a hand-placed building went through the floor: Play mode was on, editor moves ignore physics, and the model had no collider.
- **Rulesets for simulation maps.** The maps are for a ground robot. Layout, object types and start/goal should vary between runs.
  - `cage_map.py` now has `generate` and `variant` commands.
  - Three rulesets in `rulesets/`: scatter, aisles and baffles.
  - Random start (green) and goal (red) zones.
  - A path check for a 0.25 m-radius robot (placeholder size).
  - `--png` previews.
  - Unity shows the zones.
  - Fixed: zones knocking holes in baffle walls; wall ends left open.
  - 49 Python tests pass; Unity build and contact checks pass for one map of each kind.
- The user generated 20 baffle maps and tried them in Unity.
- **Repo reorganised by the user:** GATOR photo pipeline moved to `archive/gator/`, and `voxel_surface` copied into `extract_objects.py`. Checked: all tests pass and the command-line tools start.
- **Committed** `8bc89d1` on branch `unity-maps`. The scan GLBs in `data/` are not committed.
- **To do:** remove Unity's AI Assistant packages, which the editor added when the user opened the project. They are committed in `Packages/manifest.json`: `com.unity.ai.assistant` 2.22.0-pre.1 (a pre-release) and `com.unity.ai.inference` 2.6.1. The user chose to remove them later. Unity must be closed when removing them.
- Started this journal.

## 2026-11-18

- **Pivot** from photo-to-object work (GATOR still unreleased) to procedural Unity maps for the OptiTrack mocap cage (22 × 30 ft).
- **Lab facts:**
  - Motive streams VRPN at `192.168.1.42:3883`.
  - Rigid bodies are `modalai` and `starscream`.
  - `vrpn_peek.py` was written as a read-only reader.
- **The user's answers:** collision response is someone else's call; the virtual world must be live; start with the Kenney City Kit (Industrial); Unity licence active and batch mode allowed; same repo, new branch.
- **Downloaded** with approval: Kenney City Kit (Industrial) 2.0 (CC0, 5 MB).
- **Built:**
  - a map generator (scatter)
  - the Unity project `unity/CageMaps`: map loader with colliders, C# VRPN client, tracked-body contact logging, batch build and Play checks
- **Problems found and fixed:**
  - Unity mirrors X on import.
  - Non-convex colliders missed a body inside an object.
  - Unity once fetched 10 default packages, which were then removed.
- **Live mocap:** Unity decoded the same `modalai` pose as Python. Calibration was measured from edge middles: Motive +Z is the cage front, −X the right side, origin at about (3.01, 0, 4.74) in the cage frame, accurate to about ±0.15 m. Drone rotation is not yet checked.

## Before 2026-11-18

See `handoff.md` for the earlier work: GLB object extraction and inventory, the photo-depth tools and demo, the GATOR-style CPU photo pipeline (now in `archive/gator/`), and the label-free scene split. Commits: `c68a49f` (2026-09-22), `0116116` and `a8732be` (2026-11-08).
