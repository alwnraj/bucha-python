# Handoff

## Webcam photo input — 2026-10-29

The local depth website accepts a single webcam photo through browser camera permission and processes it with the same relative-depth, metric-depth and rotatable relief pipeline as an uploaded image. The camera stops on capture, close or page exit. This is a still-photo workflow, not live inference or multi-view reconstruction. The upload option and GLB workflows remain available. See README for usage; no new packages are needed.

Updated 2026-10-15. Read `AGENTS.md`, `agent.md`, then this file.

## Goal

Build a Python-first **offline GLB-to-3D-object workflow** for indoor and outdoor scenes. The desired result is one record per physical object with an ID, a supported semantic label or `unknown`, XYZ position, 3D bounds, source-face references and collision geometry in a shared scan coordinate frame. A 2D map is not a deliverable, and real-time processing is not required now. LiDAR/depth input may be considered later; GLB is the current supported format. Physical scale and scan-to-robot registration are required before coordinates can guide a robot.

## Current implementation

- `segment_objects.py` loads triangle GLBs across roots, nested transforms, meshes and primitives. It retains source instance/face IDs, UVs, textures, local attributes and world-space positions when exporting labeled 3D assets.
- `object_inventory.py`, selected by `run.py --inventory`, creates source-linked 3D surface candidates using exact shared world-space edges across all chunks. Disconnected surfaces within chunks are split; source mesh chunks are not treated as objects. Every semantic label stays `unknown`, and all physical boundaries are unverified. The inventory records stable source/membership hashes, observed bounds centers as XYZ, 3D bounds, source frame/units, explicit uncertainty and inclusive source-face ranges backed by `face-labels.npz`.
- `run.py` is the entry point for the Bucha annotation regression, generic outdoor geometry proposals, or hash-guarded imported source-face labels. It exports GLBs, a manifest, face labels, collision surfaces and a run report. `collision.py` performs static ray queries against the exported surfaces.
- `render_views.py` and `infer_labels.py` form an optional *internal* image-model proposal path. They render textured views with face IDs and vote provisional classes back onto 3D faces. `run.py --labels` imports those proposals while leaving every other face `unknown`. This is not a validated 3D object segmenter.
- `data/TEEsavR23oF.glb` is a furnished indoor living room. Its source mesh occurrences may be reconstruction chunks, not physical objects. The old building/ground/rubble labels are out of domain for it. The previous five-view model run labeled only 883 of 395,018 faces, without human semantic validation; no sofa or chair was established.
- `data/destroyed_apartments_bucha_ukraine_war_2022.glb` and `annotations.json` remain as a scan-specific outdoor regression. The default `python run.py` runs that regression; it is not evidence of general scene understanding.

## Cleanup on 2026-10-15

The user removed the need for a 2D map. `project_map.py`, `project_labels.py`, the map CLI flags and map-only tests were removed. All ignored `results*` directories, including 3.2 GB of living-room smoke outputs, were deleted because they are generated artifacts. Both input GLBs, `.venv/`, `models/`, the internal label-proposal path, 3D exports and collision code were retained. Generated outputs must be rerun when needed. The repo had substantial pre-existing uncommitted changes; do not discard them while continuing work.

## Verification

`.venv/bin/python -m unittest discover -s tests -v` passed **18 tests** after inventory implementation (4.6 seconds). The four new tests cover connectivity across and within chunks, exact versus point/near contacts, degenerate/nonmanifold faces, stable IDs, bounds, face ranges, unchanged source bytes, texture attributes, collision queries and stale-output cleanup. Tests use temporary files. The unchanged default Bucha regression also completed with `.venv/bin/python run.py --no-previews` in 4.2 seconds. No packages were downloaded.

## Run the inventory

```sh
.venv/bin/python run.py --inventory --input data/TEEsavR23oF.glb \
  --output results-living-room-inventory
```

Verified 2026-10-15 in **8.6 seconds**: **395,018 faces, 209 source occurrences, 159 candidates**. The largest candidate contains **381,667 faces (96.62%) across 186 source occurrences**, with observed bounds center `[-4.158688, 4.409025, 2.816410]`. It spans the scene bounds and is unresolved structure, not one physical object. **134 candidates have ten faces or fewer** and may be fragments. There are 48 candidates joining multiple source occurrences, while 67 source occurrences split across candidates. All labels are `unknown`; no sofa, chair or other semantic object has been established, and zero physical objects are verified.

Outputs are retained under ignored `results-living-room-inventory/` (about 63 MB): `inventory.json`, `face-labels.npz`, one combined `segmented-scene.glb`, collision files, manifest and run report. Individual visual GLBs are deliberately not duplicated in this mode; `objectId` extras identify each group in the combined scene. Existing collision query commands work with this folder's `collision-scene.json`. Inventory mode always skips images, does not run the outdoor heuristics, and runs separately from annotations, labels, corrections and review. Reusing an output directory removes stale inventory data along with other generated files.

Every real-scan inventory range, bound and collision triangle was independently checked against the loader's source geometry; all 395,018 source faces occur exactly once. Export checks confirm textures, UV/local attributes and world transforms are preserved, and the derived collision ray returned the expected candidate and source face. This verifies structure, not semantic or physical-instance accuracy.

## Verified limitations and remaining work

Shared-edge connectivity cannot separate touching furniture, floors and walls, and reconstruction bridges may join unrelated surfaces. Holes, disconnected object parts, unequal edge tessellation and tiny transform rounding differences may fragment one object. No tolerance, gap closing, semantic classifier or name-based labeling is used. Zero-area triangles are retained as separate candidates. A candidate is never asserted to be a complete or verified physical object.

XYZ is the observed axis-aligned bounds center in **source GLB world units and axes**, with all nested node transforms applied. It is not a calibrated object origin or robot pose. Meters per source unit, position error and scan-to-robot registration remain unknown. IDs are stable for identical source bytes and source-face membership; they do not identify objects across edited GLBs. Keep the next work focused on reviewed physical boundaries and supported semantics before claiming object recognition. No additional feature is authorized by this handoff alone.

## Constraints

Follow `agent.md`: no subagents, no overengineering, no edits for question-only requests, and pin any downloaded packages to versions at least two weeks old. Preserve source hashes, world-coordinate consistency and original geometry attributes. Do not claim source units are meters or that a collision surface implies traversability. The Bucha source has license evidence in `source-license.json`; verify its acquisition terms before redistributing it publicly. The local repo has no configured remote.

## Single-photo depth prototype — 2026-10-26

`image_depth.py` accepts a local image and runs the pinned ONNX Community Depth Anything V2 Small relative model using the existing ONNX Runtime, NumPy and Pillow. `download_depth_model.py` verifies the pinned revision and SHA-256. See README for the command, model provenance, preprocessing and interpretation. The raw `.npy` contains the model-resolution output, where larger is relatively nearer; the grayscale visualization is separately normalized and resized. The original photo remains read only, output is restricted to a new `results-*` directory, and no GLB face mapping is claimed.

The checkpoint was downloaded and verified; local inference ran on a synthetic solid-color image only. A user photograph has not yet been supplied or processed. The model cannot infer metric scale or unseen surfaces, and its results do not establish object boundaries or reproduce GATOR. No packages were installed. Existing GLB workflows were left unchanged.

Verification on 2026-10-26: all 21 unit tests passed, including three depth preprocessing/output checks. The default Bucha regression completed with `--no-previews` in a separate output directory; the living-room inventory also completed with 159 unknown candidates. Both are structural regressions, not semantic validation. Synthetic inference produced a finite float32 array of shape 294 × 518 for a 640 × 360 input.

On 2026-10-26, the user attached a 1174 × 1462 interior image and local inference succeeded. Outputs are in ignored `results-attached-photo/`: the raw float32 model output is 518 × 420, with a separate original-size grayscale visualization and report. The foreground sofa appears nearer and the back of the room farther in the visualization; no ground-truth depth validation was performed. The attached input was a temporary file, so its original path may not persist.

A later requested experimental single-image relief preview was generated in ignored `results-photo-surface/` from the attached image and relative depth. It includes a vertex-colored PLY, oblique PNG, generation script and notes documenting arbitrary depth and camera assumptions. This is not measured 3D geometry or collision data; occluded surfaces are missing. The script depends on the original temporary screenshot path recorded in the depth report.

## Local depth web demo

The user requested an image-upload demo with depth comparison and a rotatable relief; object labels are deferred. Run `.venv/bin/python depth_demo_server.py` and open `http://127.0.0.1:8765`. The localhost Python server uses the existing verified ONNX checkpoint, processes uploaded images in memory, and serves `depth_demo/`. The comparison offers a downloadable depth PNG. The WebGL relief uses an arbitrary relative-depth mapping and assumed 60° field of view; it is not measured geometry. No new packages, external hosting, or changes to the GLB workflow were introduced.

## Metric depth demo

The local demo supports an indoor/outdoor metric model selection before image upload and click/keyboard sampling of estimated camera-axis depth in meters. Run `.venv/bin/python download_metric_models.py` once and restart the server. Indoor and outdoor ONNX exports are pinned and checksum-verified; the existing relative model still supplies the comparison and 3D relief. See README for source revisions, licenses, preprocessing, output semantics and limitations. Fixed 518 × 518 model inputs require stretching the image; metric output is resampled to 256 × 256 for click lookup. No per-camera calibration, true-scale validation, or range accuracy is established. On the user's outdoor image both metric checkpoints produced finite output, but their numbers differed significantly. The outdoor export lacks a model card; some official model weights were independently matched, but numerical equivalence has not been verified.

## Live camera benchmark

`live_depth.py` is a separate local OpenCV camera preview and timed benchmark, leaving the website intact. It selects the existing indoor/outdoor metric or relative ONNX model, displays camera/depth side by side, and reports end-to-end FPS and inference time after warm-up. Frames are not saved. The optional `requirements-live.txt` pins OpenCV 4.11.0.86, released 2025-01-16; installed with `--no-deps`. 27 tests passed, including live preprocessing and depth polarity. A five-frame repeated still-image check on the indoor model took 0.533 seconds per frame for preprocessing, inference and depth-image preparation, excluding capture and window display. This is not a live benchmark. On 2026-10-26 the camera attempt failed with OpenCV `not authorized to capture video (status 0)` and no frames processed; the user needs to grant macOS camera access before an actual camera benchmark can be reported.

After the user enabled camera access, `live_depth.py --mode indoor --seconds 15` ran successfully on 2026-10-26: 26 camera frames in 15.30 seconds, 1.70 end-to-end FPS, mean inference 520.4 ms (1.92 inference FPS), after two warm-up frames. Camera opening initially returned an empty frame; the script now retries for up to eight seconds during warm-up. Three live-processing unit tests passed after the change. Frames were not saved.

The user requested a simpler 1990s appearance for the local website. The upload, depth comparison, metric estimate and 3D controls remain in `depth_demo/`; the interface now uses a Windows 95-inspired layout. The separate live camera prototype remains available and unchanged. The local endpoint and demo tests passed after the visual update.

The user then requested much less CSS. The website now uses plain browser-style presentation with only the layout rules needed for comparison, the 3D canvas and click selection. Demo tests and JavaScript syntax check pass.

On 2026-10-26 the user asked to keep the website simple and reduce CSS. The page styling and heading/footer markup were simplified without removing upload, relative depth, 3D rotation or metric selection. All 27 tests passed; a synthetic local HTTP request returned relative depth, metric depth and surface data. The page was visually checked in a browser.

## GATOR-inspired photos-to-object plan, step 1 — 2026-11-02

GATOR (NVIDIA, arXiv 2610.11215, Oct 2026) reconstructs complete, scene-placed object meshes from casual photos using estimated cameras/depth (Pi3X), masks (SAM3), a trained TRELLIS.2-based generator and an optional proprietary agent. Its code and weights are not released. With the user, a reduced version was planned in two versions sharing one test harness: **CPU-only** (EXIF/assumed intrinsics, ONNX metric depth, feature-matched pose chaining, support-surface masks, voting space carving) and later an **NVIDIA GPU** version (learned cameras/depth, SAM3 masks, off-the-shelf image-to-3D generator, placement). The only input is RGB photos. The user has no GPU yet. The CPU version is built first; GPU stages must later write the same bundle files.

Step 1 (shared base) is implemented: `render_views.render_perspective` (pinhole, near-plane clipping, depth, global face IDs), `photo_sim.py` (input bundle plus separate `truth/`), `recon_metrics.py` and `collision.write_mesh_collisions`. 42 tests pass (15 new). On the living room, `results-sim-bed/` (ignored, about 10 MB) holds 8 views of a bed (2,688 faces in the box above; pillows fall outside it). Eight 640×480 views render in about 16 seconds. View 0 is blocked by a wall (12,017 target pixels versus 87k–157k elsewhere), a realistic case later steps must tolerate. No packages were installed.

Next, step 2 of the CPU version: per-photo metric depth on the bundle images and feature-matched pose chaining, scored with `recon_metrics.camera_errors` against `truth/`. Not yet authorized beyond what the user approves.

## CPU step 2: depth and camera positions from photos — 2026-11-02

`photo_poses.py` implements step 2 (see README section "Estimate depth and camera positions from photos"). It reads only the bundle, runs the pinned indoor metric ONNX depth model per photo, matches SIFT (OpenCV 4.11, already pinned) across all pairs, takes rotation/direction from the essential matrix and baseline length from depth, chains rotations along a maximum spanning tree, solves per-photo depth-scale corrections from pair agreement, and re-solves centers from all pairs with Huber-style reweighted least squares. `recon_metrics.camera_errors` now also reports frame-independent pairwise rotation errors; aligned per-camera rotation errors are inflated by center errors and should not be read alone. 49 tests pass (7 new). No packages were installed.

Bed results (16 views, 22.5° apart): 13/16 registered (the 3 wall-blocked views are correctly excluded), camera-pair rotation 0.95° median / 2.58° max, worst center error 0.32 source units, scale 11% small (truth÷estimate 0.885). With true depth the worst center error is 0.08, so depth error dominates position error. 8 views at 45° register only 2 photos. Without a known focal length (60° assumed for a 75° camera) rotations fail (28° median); three self-calibration signals were tried and failed, so the CPU version requires a known FOV (`--fov`) or EXIF focal length. Run outputs are in ignored `results-sim-bed*` and `results-poses-bed*` folders (about 140 MB total; safe to delete and regenerate).

Possible next work, not yet authorized: CPU step 3 (masks from the support surface using fused depth and poses) and step 4 (voting space carving). Bundle adjustment could reduce the 0.32 position error but adds complexity; the GPU version's learned cameras/depth remain the planned alternative.

## CPU step 3: object masks from depth and poses — 2026-11-02

`photo_masks.py` implements step 3 (see README section "Find the object in each photo"). It uses only the bundle and step 2 output: camera-up estimate, lowest large horizontal RANSAC plane as support, a floor grid of above-support points (two-photo rule, wall cutoff, odd-sized opening to drop thin walls), and the connected component nearest the cameras' common aim point. 54 tests pass (5 new). No packages were installed.

Bed result: median IoU 0.82 / worst 0.67 with true depth, but 0.46 / 0.10 with the monocular depth model, because per-photo depth errors (10–15%) prevent a consistent floor. Two bugs were found and fixed during the work: an even-sized grid opening shifted the grid and cut one side of objects, and the thin-wall rule was needed because walls are only seen up to about 2 m. A per-photo floor fit was tried and removed (no gain). The mask folder layout `masks/view_XX.png` is the contract for the next step, so hand-drawn or GPU masks can replace these. `results-masks-bed16/comparison.png` (ignored) shows true-depth vs model-depth masks.

Open decision for the user: improve multi-photo depth consistency on CPU (for example aligning each photo's depth to points triangulated from SIFT matches with the known poses), accept hand-drawn masks, or proceed to step 4 (voting space carving) using any mask folder.

## Depth consistency fix — 2026-11-02

The user chose to fix depth consistency. Diagnosis: the main problem was camera spacing, not depth shape. Pair baselines came from the depth model, and since each photo matches only 2–3 neighbours, per-photo depth errors went straight into camera positions; points triangulated with those poses disagreed in scale by 0.86–1.09 between photos. `photo_poses.py` now places photos one at a time (`register_views`): rotation and direction from the pair's essential matrix, travel distance from 3D points shared with placed photos (depth baseline only as fallback), then `align_depth` sets one overall scale from the depth model and a per-photo `depth_correction` from the triangulated points. `chain_poses`, `depth_corrections` and `refine_centers` were removed. Also writes `points.npy`. Tried and dropped: direction-only center solving (undetermined on a chain), full PnP (one camera 104° wrong), smooth per-photo depth fits (extrapolate badly from sparse points).

Results (bed, 16 views): worst camera position error 0.32 → 0.095, median 0.14 → 0.044, scale 0.885 → 0.945, identical with true depth. Masks with the depth model 0.46/0.10 → 0.69/0.36 (median/worst IoU); with true depth 0.82/0.67 → 0.77/0.70 (slightly worse median because of no bundle adjustment). Remaining limit: ~6–9% in-photo depth shape error of the model. Outputs in ignored `results-*-v2` folders; `results-masks-bed16-v2/comparison.png` shows before (top) and after (bottom). 54 tests pass. Nothing committed. Next candidates: step 4 (voting space carving), or bundle adjustment.

## CPU step 4: voxel carving from masks — 2026-11-02

`photo_carve.py` implements step 4 (see README section "Carve the object from masks"): two-pass voting space carving from poses plus any `masks/view_XX.png` folder, support-plane cut from `masks.json` when present, exposed-voxel-face closed mesh written with `collision.write_mesh_collisions`, and `--score` against `truth/target.npz` (camera-similarity alignment, chamfer/F1, containment). Defaults chosen by sweep on the bed: agree 0.85, min views 3/4 of photos (half let slivers behind the object survive), dilate 2 px. 58 tests pass (4 new). No packages installed. Nothing committed.

Bed results (F1 within 0.1 / true surface inside): perfect masks and cameras 0.62 / 0.96 (the visual-hull limit of one raised camera ring), true-depth masks 0.42 / 0.87, photos only 0.47 / 0.40. The photos-only result is limited by masks that miss parts of the bed. `results-carve-bed16/comparison.png` (ignored) projects each carving onto two photos with the true outline in cyan.

Possible next work, not yet authorized: better photos-only masks (the main limit now), GATOR-style placement refinement (translation and scale only), smoothing the voxel mesh, or bundle adjustment. The GPU version would replace masks (SAM3) and the carved shape (image-to-3D generator) while reusing the same files and scoring.

## GLB object extraction — 2026-11-04

The user pivoted from photos to GLB input: decompose an outdoor GLB into labeled, extracted objects, offline (time is not a constraint). `extract_objects.py` implements it (see README "Extract objects from an outdoor GLB"): virtual street-level and aerial views rendered from the GLB, SegFormer-B5 ADE20K labels grouped into a fixed outdoor class list, per-view face votes, neighbour smoothing, per-class splitting at label boundaries, then the existing `segment(..., labels=)` export, open colliders, closed voxel colliders, `objects.json` and a contact sheet.

- Downloaded with the user's approval: `Xenova/segformer-b5-finetuned-ade-640-640` revision `02e76595880a` (2025-07-01), 340 MB, checksum-pinned in `download_vision_model.py --model b5`. No packages installed.
- Fixed a bug in `render_views._shade`: it flipped glTF V and clamped instead of wrapping. glTF UV (0,0) is the top-left texel. All earlier renders of textured GLBs (including the 883-face living-room model run) were wrongly textured. `tests/test_render_views.py` now checks the correct convention and wrapping.
- `infer_labels._load_model` now accepts either pinned model and returns its entry.
- `extract_objects.ground_frame` flips the estimated up when most non-ground geometry sits below the plane (a flat synthetic floor came out upside down). `segment_objects.estimate_ground_plane` itself is unchanged.
- Bucha result (`results-objects-bucha`, ignored): 33 objects in 21 minutes. Buildings and ground separate plausibly; small objects are mostly mislabeled rubble at low confidence, and "water" is likely wrong. Not validated against hand labels. Do not tune thresholds on Bucha alone.
- 63 tests pass (5 new in `tests/test_extract_objects.py`, using a color-based stand-in for the model). Nothing committed.
- Open options: confidence cutoff for small objects, cropping textures per object GLB (each currently embeds all source textures), instance-aware model, a scan with distinct objects (cars, trees, benches) to evaluate on.

## Large-scan support and second test scan — 2026-11-04

- Downloaded with the user's approval: `data/flat-roof.glb` (21.8 MB, CC BY 4.0, Matt1Up drone-building-scans revision `fde325fbd1d3`, SHA-256 `6c712d16…`). It is Draco-compressed with 5.5M triangles. `data/flat-roof-plain.glb` (170 MB) is the decoded copy. Neither is committed.
- Installed `DracoPy==2.2.0` (2026-10-07, Apache-2.0), pinned in `requirements-vision.txt`. `prepare_glb.py` decodes Draco primitives into plain accessors.
- `render_views.render_perspective(batch=True)` draws triangles whose bounding box holds one pixel center in one numpy step; output is pixel-identical to the per-triangle loop in tests and on two Bucha views. `load_scene` caches per-chunk UVs. Flat-roof views render in 1–6 s.
- `segment(..., face_objects=)` takes per-face labels as arrays (no per-face JSON); `extract_objects.py` uses it. Votes are float32, fill only touches empty faces, and splitting uses vectorized component bounds.
- `extract_objects.ground_frame` now picks "up" so the larger horizontal area is at the bottom, and puts the plane at the 5th percentile of horizontal surfaces. The shared estimator put flat-roof's plane on the roof (sloped lot below). Bucha still comes out Z-up.
- Flat-roof result (`results-objects-flatroof`, 2.8 GB, ignored): 204 objects in 31 minutes. Building, lawn, ground, road, sidewalk, shrubs, wall/fence panels, rocks and poles are good. Shade-sail posts split across pole/column/tree, small edge fragments, and three spurious "car" fragments. Not validated.
- 65 tests pass. Nothing committed.
- Next options: crop textures per object (each GLB embeds the whole 8 MB texture), merge or drop small low-confidence fragments, join adjacent pieces of thin vertical structures.

## Fragment merging — 2026-11-05

`extract_objects.merge_fragments` runs after `split_objects`. Touching THIN-class pieces (pole, column, streetlight, sign, traffic_light, flag, tree) merge into one object named after the most common non-tree class. A non-ground object with at least 75% of its boundary on one neighbour and at most 10% of that neighbour's faces joins it. Ground-type neighbours only absorb pieces under 0.5 units tall with mean confidence under 0.5; without the height rule, a whole shade-sail post was absorbed into the lawn. Flat-roof went from 204 to 85 objects (both posts whole, building includes facade walls); Bucha from 33 to 17. Thresholds were checked on flat-roof only and are not validated. Closed colliders from earlier runs are now deleted before writing. 67 tests pass. Nothing committed.

## Per-object texture packing — 2026-11-08

`texture_crop.repack_textures` rewrites each object GLB with only the texture blocks its faces use (16 px tiles, 2 px margin, 4-connected blocks shelf-packed into a new image, UVs shifted per block). Images sharing UV accessors through a material get the same layout; groups with wrapped UVs, texture extensions, mismatched image sizes or under 30% saving are left unchanged. JPEG is re-encoded at quality 92. `extract_objects.py` runs it by default (`--full-textures` skips it) and notes it in `manifest.json` (`per_object_textures`); `segmented-scene.glb` keeps exact source textures, so the manifest's exactness flags describe that file and the per-face exports, not the packed object textures. Object GLBs: Bucha about 300 MB to 35 MB, flat-roof about 700 MB to 170 MB. Five sampled objects render with mean pixel differences of 0.01 to 0.06; the lossless synthetic test matches exactly. 68 tests pass. Nothing committed.

