# 3D object identification — Python in VS Code

For a new agent session, open this repository as the workspace and read **`AGENTS.md` → `agent.md` → `handoff.md`**. The handoff records the tested baseline, limits and next-session context.

The project is a local Git repository on `main`. The accompanying `bucha-python.zip` is a source snapshot; `bucha-python.bundle` includes Git history and can be cloned with `git clone bucha-python.bundle bucha-python`. No remote has been configured.

## Goal and current status

The goal is to process **indoor or outdoor GLB scans offline**, identify physical objects, label them when the evidence is sufficient, and report each object's identity, XYZ position and 3D bounds in a shared coordinate frame for later robot use. Unknown objects remain unknown. A 2D image or top-down map is not required as an output, and real-time processing is not a current requirement. LiDAR point clouds and depth scans are possible future inputs, not supported formats today. Robot use also requires physical scale and registration between the scan and robot frames, which are not implemented.

The offline inventory now creates source-linked **3D surface candidates** with stable IDs, XYZ positions, bounds, uncertainty and collision surfaces. It conservatively separates connected surfaces across mesh chunks. These candidates are not verified physical objects; the inventory keeps every semantic label `unknown`. Semantic recognition and reliable boundaries between touching physical objects remain unresolved.

## Run the offline 3D inventory

Use the existing environment; no additional packages or model downloads are needed:

```sh
.venv/bin/python run.py --inventory --input data/TEEsavR23oF.glb \
  --output results-living-room-inventory
```

This mode skips the configured Bucha annotations and all preview generation. It produces only 3D assets and metadata, with no 2D map or real-time service. Run it separately from `--automatic`, `--annotations`, `--labels`, `--corrections` and `--review`.

| Output | Contents |
| --- | --- |
| `inventory.json` | One candidate record with a stable `id`, `semantic_label`, `position_xyz`, `bounds_3d`, coordinate-frame reference, uncertainty and source-face ranges |
| `face-labels.npz` | Every original `(source_instance, source_node, source_mesh, source_primitive, source_face)` and its numeric `object_id`; shared with collision exports |
| `segmented-scene.glb` | One textured scene containing all candidate groups, identified by numeric `objectId` extras; avoids duplicating textures into hundreds of assets |
| `colliders/*.npz`, `colliders/*.obj`, `collision-scene.json` | Unmodified world-space triangle surfaces per candidate; the existing Python ray-query command still works |
| `manifest.json`, `run-report.json` | Source hash, geometry-preservation checks, run timing and a face-derived collision verification ray |

The grouping rule joins triangles that share a complete edge with **exactly equal world-space endpoints**, even across primitives, source chunks and duplicated UV vertices. Disconnected surfaces inside one chunk become separate candidates. Point contact and proximity alone do not merge groups. Zero-area faces remain separate candidates. No geometry, texture bytes, local attributes or transforms are changed by the adjacency calculation or export.

Each candidate's string ID hashes the source GLB SHA-256 and its sorted source-face membership. It is reproducible for the same source bytes and membership, including a different output directory; it does not track an object across edited GLBs. Numeric `object_id` links the inventory to GLB extras, the face mapping and collision queries. `source_faces` gives node/mesh/primitive/instance references with **inclusive** `[first_face, last_face]` ranges, starting at face zero. The matching full face list can also be read from `face-labels.npz` by selecting that numeric object ID.

**XYZ is the center of the observed axis-aligned 3D bounds**, not a verified object origin, solid center of mass or robot pose. All nested source transforms have been applied, with source axes and units unchanged. Physical meters per source unit, object-position error and scan-to-robot registration are unknown. Bounds cover observed surfaces only. `physical_object_verified` and `semantic_label_verified` are false for every candidate; confidence is not a calibrated probability.

Verified on `TEEsavR23oF.glb` on 2026-10-15: **395,018 faces in 209 source mesh occurrences → 159 candidates**, in **8.6 seconds**, with about **63 MB** of outputs. There are 48 candidates spanning multiple source occurrences, and 67 source occurrences are split across candidates. This demonstrates that the result is not one candidate per mesh chunk.

The largest candidate contains **381,667 faces (96.62%) across 186 source occurrences**, covering the whole scene bounds. It remains unresolved scene structure, not one identified object. Its bounds center is approximately `[-4.158688, 4.409025, 2.816410]` in source XYZ units. Of the other groups, **134 have ten faces or fewer** and may be reconstruction fragments. **No sofa, chair or other semantic object is established; all 159 labels are `unknown`.** Connectivity cannot separate touching furniture from a floor or wall, and holes or differently tessellated seams can split one physical object. Source node names do not count as verified labels. Human boundary and semantic review is still required.

All 18 tests pass. The real-scan check also expanded every inventory source-face range and checked every bound and collision triangle against the original GLB; all 395,018 faces are retained exactly once. Texture/attribute/transform export checks and a candidate collision ray passed. These establish geometry and mapping preservation, not physical-object accuracy.

**Open this folder in VS Code and run `run.py`.** Segmentation, asset export, collision geometry, collision ray queries and preview images all execute in Python. The supplied GLB is included under `data/`.

The default command reproduces the Bucha **scan-specific, visually guided regression**. The pipeline also has an automatic geometry-proposal mode for a new scan, so hand-drawn polygons are optional. That mode estimates an up/ground plane, proposes building-like instances, ground, rubble and unknown faces, and records confidence values. It is not a trained semantic model and has not been validated on a second real scan.

## Run on this computer

1. Open the **bucha-python** folder itself in VS Code using **File → Open Folder**.
2. Choose **Python: Select Interpreter** and select **`.venv/bin/python`**. The environment is already installed on this computer. On Windows its equivalent is `.venv/Scripts/python.exe`.
3. Open `run.py` and use **Run Python File**, or press **F5** with **Bucha: full pipeline** selected. The F5 configurations require VS Code's Python and Python Debugger extensions.

In a new VS Code terminal with that environment selected:

```sh
python run.py
```

For a different GLB with no annotation file, use the conservative inventory:

```sh
python run.py --inventory --input path/to/other-scan.glb --output results-other
```

The full run took about **6 seconds on this machine**. It regenerates the files in `results/`; rerunning replaces generated files with the same names. The source GLB is read only.

## Set up a fresh copy

Use Python **3.11–3.13**, preferably **3.12**, which was tested. Run the setup script once with that Python:

```sh
python3.12 setup_env.py
```

On Windows, use `py -3.12 setup_env.py`. Afterwards select the generated `.venv` interpreter in VS Code. Setup downloads exactly **NumPy 2.3.5** and **Pillow 11.3.0**. Their releases predate this project by more than two weeks; subsequent pipeline runs work offline. No browser, JavaScript, Node.js or Unity is involved.

[NumPy release](https://pypi.org/project/numpy/2.3.5/) · [Pillow release](https://pypi.org/project/pillow/11.3.0/) · [VS Code interpreter settings](https://code.visualstudio.com/docs/python/settings-reference)

## What the command produces

| Output under `results/` | Purpose |
| --- | --- |
| `segmented-scene.glb` and individual `.glb` files | Original textures, local attributes and shared world coordinates; inventory mode uses only the combined scene |
| `face-labels.npz`, `manifest.json`, `annotations-used.json`, `review-used.json`, `corrections-used.json` | Object IDs, source instance/node/mesh/primitive/face references, confidence, review flags, bounds and labeling metadata |
| `colliders/*.npz` and `colliders/*.obj` | One geometry-only collision surface per object, in source world coordinates |
| `collision-scene.json` | Maps each collision surface to its object ID and semantic class |
| `previews/*.png` and `run-report.json` | Label/texture views, an annotation map, separate-object previews and execution/verification results; matching source provenance is copied only when available |

All source triangles are retained exactly once across the visual objects. The Bucha regression contains **183,051 source triangles**. The exporters verify original vertex attributes, UVs, material assignments, embedded images and world-space coordinates. Collision exports merge identical vertex positions but retain exactly the same world-space triangles.

Open **`results/previews/scene-labeled.png`** or **`objects-preview.png`** directly in VS Code to inspect the result. The lightweight PNG renderer uses depth-sorted triangles and one texture sample per face; its texture view is approximate. The GLBs retain the complete original texture mapping.

## Try local indoor/outdoor semantic proposals

The optional path renders the original textured GLB from selected directions, runs a compact ADE20K image segmentation model locally through ONNX Runtime, and votes image labels back onto **visible source faces**. It starts every GLB face as `unknown`; weak image evidence stays unknown. Labels such as `chair` or `wall` are provisional semantic classes. The generated `chair_source_0012`-style names group a class within a source mesh occurrence; they do not establish physical object instances. This path does not use the outdoor building/ground/rubble heuristic.

Install the pinned optional packages after `setup_env.py`, then download the pinned checkpoint. The model and generated results are ignored by Git. The checkpoint's [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) limits use to noncommercial research or evaluation; check it before using the model in another setting.

```sh
.venv/bin/python -m pip install --no-deps -r requirements-vision.txt
.venv/bin/python download_vision_model.py
```

The following example uses the indoor living-room scan. `--exclude-name ceiling` uses its source node names to reveal more of the room; omit that filter for scans without those names. At 256 pixels, five views took about 76 seconds on the M4 Mac. The model pass took about 2 seconds. Increase `--resolution` to 512 if the first view set is too coarse; rendering will take longer.

```sh
.venv/bin/python render_views.py --input data/TEEsavR23oF.glb \
  --output results-my-views --resolution 256 \
  --direction=+Z --direction=+X --direction=-X --direction=+Y --direction=-Y \
  --exclude-name ceiling
.venv/bin/python infer_labels.py --input data/TEEsavR23oF.glb \
  --views results-my-views --model-dir models/segformer-b0-ade20k \
  --output results-my-labels.json
.venv/bin/python run.py --input data/TEEsavR23oF.glb \
  --output results-my-labeled --labels results-my-labels.json --no-previews
```

The last command exports labeled 3D GLBs, collision surfaces, face references and object metadata. The source hash is checked when importing labels.

On this scan, the 256-pixel five-view smoke run labeled **883 of 395,018** faces (before a texture-orientation bug in `render_views.py` was fixed on 2026-11-04; the views were wrongly textured), mostly as wall, box, mirror, windowpane, shelf or cabinet; no sofa or chair was established. This is a working local proposal path, not a validated semantic result. Many triangles are occluded or smaller than a pixel, and the orthographic views differ from the model's training photos. The model confidence and vote threshold are filtering signals, not calibrated correctness probabilities. Review a representative set of source faces before relying on any class or collision identity. The existing `--labels` importer also accepts reviewed/manual source-face JSON without running this model.

For manual imports, use `{"source_sha256":"<GLB SHA-256>","labeled_faces":[{"source_instance":0,"source_face":12,"semantic_class":"chair","object_name":"chair_source_0000","confidence":0.8}]}`. An omitted confidence defaults to 1.0. Every unlisted face stays `unknown` with confidence 0. The renderer currently supports opaque triangle surfaces with embedded base color textures; transparency and advanced glTF material effects are not rendered for inference.

## Extract objects from an outdoor GLB

`extract_objects.py` takes a GLB and writes one textured GLB per detected object. It needs no photos or camera data: it places virtual cameras inside the scan (street level at `--eye-height`, default 1.6 source units, plus two aerial rings), renders each view with the source face behind every pixel, labels the views with the pinned SegFormer-B5 ADE20K model, and votes those labels onto the faces. Labels are smoothed over neighbouring faces (unseen faces borrow neighbour labels). Each class is then split into separate objects at label boundaries; ground-type classes (road, sidewalk, ground, grass, sand, water) stay one object each. Finally, split-up pieces are merged: touching pole/column/streetlight/sign/traffic-light/flag/tree pieces become one object (named after the most common non-tree class), and an object that shares at least 75% of its boundary with one neighbour and has at most 10% of its faces joins that neighbour. Ground-type neighbours only absorb flat (under 0.5 units tall), low-confidence (under 0.5) pieces, so shrubs and posts on a lawn stay separate. The fixed outdoor class list is `OUTDOOR_CLASSES` in the script; anything else stays `unknown`.

```sh
.venv/bin/python -m pip install --no-deps -r requirements-vision.txt   # once
.venv/bin/python download_vision_model.py --model b5                  # once, 340 MB
.venv/bin/python extract_objects.py --input data/destroyed_apartments_bucha_ukraine_war_2022.glb \
  --output results-objects-bucha
```

Outputs in the result folder:

| File | Contents |
|---|---|
| `<name>.glb` | One textured object per file (e.g. `building_01.glb`, `road.glb`), source UVs and textures kept |
| `objects.json` | Per object: class, faces, mean label confidence, seen fraction, XYZ, bounds, size along the ground frame, base height above ground, file paths |
| `objects-sheet.png` | One thumbnail per non-ground object for quick review |
| `colliders/`, `collision-scene.json` | Open collision surface per object (works with `collision.py`) |
| `closed/` | Watertight voxel collider per non-ground object: each column filled down to the object's lowest point |
| `work/` | Cached votes (`votes.npz`, reused when the input, model and camera settings match), label JSON and 24 sample views (render beside predicted labels) |

On the Bucha scan (183,051 faces) the run used 204 views and took 21 minutes on the M4 Mac (about 3 s to render and 3 s to label each view). It found 33 objects: two buildings (the large one is the connected apartment blocks, the second is the red-roofed block), ground, road, sidewalk, water, plus small pieces labeled rock, wall, fence, person, plant and fountain/statue. Building and ground separation looks right from the air. The small pieces are mostly rubble fragments with low confidence (0.2–0.4) and wrong classes; the 12,103-face "water" is very likely rubble or wet ground. Street-level textures in this scan are heavily smeared, which limits what any image model can see. Nothing has been checked against hand labels.

**Draco-compressed or very large GLBs.** Convert Draco files once with `prepare_glb.py` (needs the pinned `DracoPy` from `requirements-vision.txt`). The renderer draws triangles that cover at most one pixel in one numpy step, so multi-million-triangle scans are practical:

```sh
.venv/bin/python prepare_glb.py --input data/flat-roof.glb --output data/flat-roof-plain.glb
.venv/bin/python extract_objects.py --input data/flat-roof-plain.glb --output results-objects-flatroof
```

Second test scan: `flat-roof.glb` from [Matt1Up drone-building-scans](https://huggingface.co/datasets/Matt1up/drone-building-scans) (revision `fde325f`, CC BY 4.0, attribution to Matt1Up required when shared). It is a metric drone scan of a single-storey commercial building with lawn, shrubs, parking lot, sidewalk, fences and a shade sail, 5,500,320 triangles. The run used 264 views and took 31 minutes, peaking at about 5 GB of memory. Before fragment merging it produced 204 objects; with merging, 85 (36 with 1,000+ faces). The building (4.15M faces including its facade walls, confidence 0.85), lawn, ground, road and sidewalk separate well. Individual shrubs (`plant_*`), the corrugated screen walls, fence panels, rocks and both shade-sail posts (`pole_01`, `pole_02`) come out as separate objects. Remaining problems: a tail of small low-confidence pieces (mostly under 500 faces) labeled wall, building, rock or plant, and two "cars" that are 300-face fragments at confidence 0.3; the lot shows no parked cars. Bucha drops from 33 to 17 objects with merging. Rooftop HVAC units stay part of the building. Not checked against hand labels.

Known limits: SegFormer gives classes, not instances, so touching objects of one class merge, and one object can split across classes. Textured objects keep scan holes. Object GLBs keep only the texels they use: `texture_crop.py` finds the used texture in 16-pixel tiles (2-pixel margin), packs those blocks into a new image and shifts the UVs; geometry is untouched and `segmented-scene.glb` keeps the exact source. JPEG textures are re-encoded at quality 92 (renders differ by under 0.1 of 255 on average); PNG stays lossless. Objects that use most of the atlas, such as the flat-roof building, keep the original image. Object GLBs total about 35 MB for Bucha (was about 300 MB) and 170 MB for flat-roof (was about 700 MB; the 4M-face building is 125 MB of geometry). Pass `--full-textures` to skip this. The model is under NVIDIA's noncommercial license.

## Try a collision query in Python

After running the pipeline, inspect `run-report.json` for a ray derived from the generated geometry, or supply your own source-coordinate ray:

```sh
python collision.py --scene results/collision-scene.json \
  --origin X Y Z --direction DX DY DZ
```

You can also call it from another Python script:

```python
from pathlib import Path
from collision import raycast_scene

scene = Path(__file__).resolve().parent / "results" / "collision-scene.json"
hit = raycast_scene(scene, origin=[x, y, z], direction=[dx, dy, dz])
if hit:
    print(hit["object_id"], hit["object_name"], hit["position_world"])
```

This tests which object's triangle surface a ray hits. It is **not** a physics simulation, a robot-clearance test or a navigation planner. The collision surfaces retain scan holes and incomplete backs. A ground label does not establish traversability.

## Change and rerun

Edit `annotations.json` to change the Bucha masks or thresholds. Its `source_sha256` prevents applying those masks to another scan. For an unseen scan, use `--inventory` for conservative 3D surface candidates. The older `--automatic` mode derives outdoor geometry labels from the input. `results/previews/annotation-map.png` is created only for an annotation-guided run.

To evaluate predictions, use a separate review-only JSON so it cannot change the labels being measured. It must include the same `source_sha256` and a `reviewed_faces` list, for example `{"source_instance": 0, "source_face": 12, "expected_instance_id": "building_01_auto", "semantic_class": "building"}`. Run `python run.py --automatic --input path/to/scan.glb --review review.json --output results-review`. Every reviewed face must resolve to exactly one source face; otherwise the evaluation is invalid and its accuracy fields remain null. A valid automatic run sets `validated_semantic_accuracy` only for the supplied review samples, not as evidence of broad generalization.

## Correct labels on an automatic scan

For a scan that needs human fixes, use a hash-guarded correction JSON. Correct a whole source mesh occurrence with `corrected_instances`, or correct individual triangles by the stable `(source_instance, source_face)` pair in `corrected_faces`:

Replace the hash placeholder and source references with values from your reviewed input. The example JSON is illustrative only.

```json
{
  "source_sha256": "replace-with-the-input-file-sha256",
  "corrected_instances": [
    {
      "source_instance": 7,
      "object_name": "building_west",
      "semantic_class": "building"
    }
  ],
  "corrected_faces": [
    {
      "source_instance": 0,
      "source_face": 12,
      "object_name": "building_west",
      "semantic_class": "building"
    }
  ]
}
```

Use actual source references from your review; the numbers above only illustrate the fields. An instance correction assigns every face in that source mesh occurrence. Face corrections apply after instance corrections, so they can make local exceptions. `object_name` can select an existing automatic object or create a new instance. Every corrected face remains tied to its original source-face ID, and exports still use the original textures, transforms and collision geometry. The correction file must match the input GLB hash. Apply it with:

```sh
python run.py --automatic --input path/to/scan.glb \
  --output ./results-my-scan \
  --corrections ./reviews/scan-corrections.json
```

The run copies the applied file to `corrections-used.json` and records the number of corrected faces in the manifest and run report. Keep this labeling run separate from `--review`: review-only evaluation measures unmodified automatic predictions and cannot be combined with corrections.

Do not apply the earlier TEEsavR23oF correction set. The supplied screenshot shows a furnished indoor living room, not an outdoor ruin. The earlier “façade” interpretation was wrong, so those scan-specific building assignments have been withdrawn. Its automatic outdoor geometry proposals are also semantically invalid for this scene and remain unvalidated; use neither result as a successful label set. The generic correction seam remains available for future scans after their scene type and label taxonomy are established.

`config.json` selects the input, annotations, output folder and whether to generate previews. Relative paths resolve from that config file, so the project can be moved without changing user-specific paths. For a quick run without images:

```sh
python run.py --no-previews
```

For a separate result folder:

```sh
python run.py --output results-experiment
```

Use `--scene results-experiment/collision-scene.json` when querying an alternate result folder. The source hash is checked intentionally: applying these hand-guided masks to a different scan would produce misleading labels.

| Script | Responsibility |
| --- | --- |
| `run.py` | Run automatic or annotation-guided labeling, exports, previews and verification |
| `segment_objects.py` | Load general triangle GLBs, classify faces and export textured assets |
| `object_inventory.py` | Group exact shared 3D edges and write source-linked candidate records |
| `collision.py` | Export collision surfaces and query nearest ray hits by object ID |
| `render_previews.py` | Generate all PNG previews in Python |
| `render_views.py`, `infer_labels.py` | Render textured source-face views and make local model proposals |
| `setup_env.py` | Create the local environment and install pinned dependencies |

## Checks and current limits

Run the self-contained tests (68 at the latest count; the paragraph below describes the original 18):

```sh
python -m unittest discover -s tests -v
```

The 18 tests cover the default face partition, collision-file geometry and source-face mapping, ray behavior, a synthetic multi-root/multi-mesh/multi-primitive textured GLB, source-face corrections/imports, texture view IDs and model-vote fusion. Inventory tests also check splitting within chunks, joining across chunks, point contacts, near surfaces, degenerate and nonmanifold faces, ID reproducibility, source guards, bounds, face ranges, preservation and stale-output cleanup. These are structural checks, not a semantic accuracy evaluation. Count-based Bucha regression checks intentionally expect the default annotations; update their expected counts if you change those masks.

Labels remain provisional. Automatic geometry proposals can fragment building instances or confuse low structures, rubble, vegetation and ground. The TEEsavR23oF scan is an indoor living room and falls outside the outdoor labeling scope; it is not a semantic generalization result. No representative reviewed semantic test set is stored in this repository, so automatic class and instance accuracy remain unvalidated. Review-only files report sample accuracy only when every reviewed face resolves to the current source and the automatic prediction path was used. The estimated up direction and real-world scale are recorded per run.

The Python pipeline creates collision **geometry and queryable object identities**. Attaching engine-specific components such as Unity MeshColliders and connecting a Magic Leap 2 feed are later integration steps. An RGB feed alone does not transmit these object identities. No robot or ML2 connection has been implemented or tested.

## Source asset

**Destroyed Apartments, Bucha Ukraine War 2022**, by **LibanCiel**. [Matching Sketchfab model](https://sketchfab.com/3d-models/destroyed-apartments-bucha-ukraine-war-2022-f9dbc156ba95471c840c2d3e52880629). The saved metadata lists the **Standard** license; evidence and limitations are in `source-license.json`. Keep the asset's applicable acquisition terms with any later redistribution.

## Estimate depth from one photograph

This independent prototype uses the existing NumPy, Pillow and ONNX Runtime environment. Download the pinned 99.1 MB checkpoint once; inference then runs locally on CPU:

```sh
.venv/bin/python download_depth_model.py
.venv/bin/python image_depth.py /path/to/photo.jpg --output results-my-photo
```

Choose a new `results-*` directory directly inside the repository; existing depth output files are never overwritten. The input is read only. The command saves `raw-output.npy` (unmodified model output at model resolution), `depth-visualization.png` (resized to the EXIF-oriented photo, white = relatively near and black = relatively far), and `report.json` (input hash, checkpoint, preprocessing, output shape and limitations). For example, load the numerical array with `numpy.load('results-my-photo/raw-output.npy')`.

The pinned [ONNX Community export](https://huggingface.co/onnx-community/depth-anything-v2-small/tree/34f40a4c93991327074907b51792dbc27e7d6f1c) of [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) uses revision `34f40a4c93991327074907b51792dbc27e7d6f1c`, file `onnx/model.onnx`, SHA-256 `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c`, and Apache-2.0 licensing. This is model provenance, separate from Python package requirements. The output is **relative inverse depth / disparity-like**: larger means nearer; values are not meters or calibrated camera-space Z, and their scale is not comparable across photos. The visualization is min-max normalized per image.

The image is EXIF-transposed, converted to RGB, resized with Pillow bicubic interpolation so its long edge is 518 pixels and dimensions are rounded to multiples of 14, then normalized by ImageNet mean and standard deviation. Pillow interpolation may differ from the original OpenCV implementation. A single image cannot establish true scale or reveal hidden surfaces; predictions may be incorrect and do not identify physical objects. Image pixels have no GLB source-face references. This is not a reproduction of the GATOR pipeline.

## Local photo depth demo

Start the local web demo using the existing environment and downloaded depth checkpoint:

```sh
.venv/bin/python depth_demo_server.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), choose a JPG, PNG or WebP image (up to 20 MB and 30 megapixels), and drag the comparison divider or 3D surface. The server binds to localhost; uploaded image bytes are processed in memory. Stop it with Ctrl+C. No new packages are needed. The preview uses WebGL and a heuristic, arbitrary depth scale with an assumed 60° horizontal field of view; it is a single-view relief, not measured geometry or a collision mesh. The original image is not changed. The depth image can be downloaded from the page; the Python CLI remains available for raw output and a detailed report.

To use a webcam, choose the scene, select **Use webcam**, allow browser camera access, and select **Take photo**. One frame is processed by the same local pipeline as an uploaded photo; the camera stops after capture. **Close camera** stops it without taking a photo. Camera access requires localhost or HTTPS and a supported browser. Captures are processed in memory and are not saved by the server.

### Estimated metric depth

Download the two pinned metric ONNX exports once, then start or restart the demo:

```sh
.venv/bin/python download_metric_models.py
.venv/bin/python depth_demo_server.py
```

Choose **Indoor scenes** or **Outdoor scenes** before uploading. Click the photo in the metric panel (or focus it and use arrow keys) to display estimated camera-axis depth in meters. This is a model prediction, not a measured straight-line range; a single RGB image cannot establish true scale. Sky, hidden surfaces, unusual cameras and scenes outside the model's training domain can produce misleading values. Do not use it for navigation or measurement decisions without independent validation. The comparison and 3D relief continue to use the separate relative model. If a metric checkpoint is unavailable, those features still work and the metric panel reports the missing model.

Both metric exports accept fixed 518 × 518 RGB tensors. We resize the oriented photo directly to that square with Pillow bicubic interpolation, divide by 255 and apply ImageNet normalization. The output is a metric depth map, larger meaning farther. It is bilinearly resized to a 256 × 256 grid for click sampling; image coordinates are mapped to that grid. Square resizing changes aspect ratio and can reduce accuracy; no camera calibration or per-image correction is applied. Both models are run locally with ONNX Runtime on CPU.

The small indoor model is the [77ukhtar ONNX export](https://huggingface.co/77ukhtar/depth-anything-v2-metric-onnx), revision `a4259a3c45137b6eb32c84fcd95b86cd54c255b9`, `model.onnx` SHA-256 `badcaa28c923da4b0bfaa370ed709acfa00e9f743d295d5443e2149a383413c9`. Its model card identifies the official Hypersim Small weights and reports a PyTorch export comparison. The outdoor model is the [sollaholla ONNX export](https://huggingface.co/sollaholla/depth-anything-v2-metric-vits-vkitti), revision `f3f42bbfb266b8c7dbd0ef6cf24bfc4b167717f5`; its graph and external-data SHA-256 values are pinned in `download_metric_models.py`. It has no model card: its filename identifies the VKITTI Small model, and a spot check found 45 large byte-identical tensors from the [official outdoor checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf) in its external weights; full numerical equivalence has not been verified. Both original Small checkpoint pages list Apache-2.0. The upstream metric documentation lists a 20 m indoor and 80 m outdoor output range; these are model limits, not accuracy guarantees.

## Live camera depth benchmark

This is a separate Python camera preview; the photo-upload website remains unchanged. Install the optional pinned camera dependency once (released 2025-01-16), using the existing NumPy and ONNX Runtime environment:

```sh
.venv/bin/python -m pip install --no-deps -r requirements-live.txt
.venv/bin/python live_depth.py --mode indoor --seconds 15
```

A window shows camera frames beside a grayscale depth estimate; white means relatively near. The command runs two warm-up frames, then reports actual end-to-end frames per second (capture, preprocessing, inference and display) and model-only inference time. Use `--mode outdoor` for the outdoor metric model, `--mode relative` for the original relative model, `--camera 1` for another camera, or `--seconds 0` to run until you press Q. OpenCV and the OS may require camera permission for the Python process or terminal. It does not save frames or transmit them. Both metric models are fixed to 518 × 518, so rectangular camera frames are stretched and estimated depth is not calibrated or verified. The display normalizes each frame separately and may flicker; it is not suitable for flight or obstacle avoidance.

## Simulate a photo capture (photos-to-object groundwork)

This is step 1 of a GATOR-inspired photos-to-collision-shape pipeline (planned in two versions: CPU-only and, later, NVIDIA GPU). It renders "photos" of one object from a GLB so later steps can be scored against hidden ground truth. No new packages are needed.

```sh
.venv/bin/python photo_sim.py --input data/TEEsavR23oF.glb --output results-sim-bed \
  --box 0.55 6.25 0.06 2.85 8.2 1.4 --views 8 --fov 75 --distance 2.2 --elevation 30
```

The target is every source face whose centroid lies inside `--box` (source GLB world coordinates, Z up by default). Cameras orbit the target center and look at it. The output folder must be new or empty:

- `bundle.json` and `images/` are the **only pipeline input**. `focal_length_px` stands in for EXIF focal length; `--no-focal-hint` withholds it.
- `truth/` holds camera poses and intrinsics, per-view depth (camera-axis Z, NaN where empty), target masks, global face IDs and the target triangles. It is for scoring only.

Cameras use OpenCV axes (+X right, +Y down, +Z forward) with pixel centers at integer coordinates. Distances are in source units; scale is unverified. A view whose camera stands behind a wall prints a warning with its small target-pixel count. The target is "faces in the box", not a reviewed physical-object boundary.

`recon_metrics.py` provides the shared scoring: similarity and translation-plus-scale fits, camera error after frame/scale alignment, area-weighted surface sampling, chamfer distance with F-score, mask IoU and voxel containment. `collision.write_mesh_collisions` writes reconstructed meshes into the existing `collision-scene.json` format, so `collision.py` ray queries work on them. The pinhole renderer is `render_views.render_perspective`; it clips triangles at the near plane rather than dropping them.

## Estimate depth and camera positions from photos (CPU step 2)

`photo_poses.py` reads only a photo bundle (`bundle.json` plus images) and estimates per-photo metric depth and camera poses on the CPU. It uses the already pinned indoor/outdoor metric ONNX model and OpenCV 4.11 (SIFT); no new packages.

```sh
.venv/bin/python photo_sim.py --input data/TEEsavR23oF.glb --output results-sim-bed16 \
  --box 0.55 6.25 0.06 2.85 8.2 1.4 --views 16 --fov 75 --distance 2.2 --elevation 30
.venv/bin/python photo_poses.py --bundle results-sim-bed16 --output results-poses-bed16 --score
```

Method: mutual SIFT matches for every photo pair → essential matrix (rotation and travel direction from geometry alone) → photos placed one at a time through their strongest pair, with each camera's travel distance solved from 3D points it shares with already placed photos (so all cameras share one scale; the per-pair depth baseline is only a fallback) → one overall scale from the depth model → a per-photo `depth_correction` that fits each photo's depth to the triangulated points. Output is `poses.json` (camera-to-world in the root photo's frame, OpenCV axes, estimated meters, plus `depth_correction` and `placed_by` per photo), `points.npy` (the triangulated SIFT points) and `depth/*.npy`. Photos that match nothing stay unregistered. `--score` then compares with `truth/` and writes `score.json`; `--truth-depth` is a diagnostic that swaps in true depth to isolate pose error from depth error.

The focal length must be known: pass `--fov` from the camera specs or provide `focal_length_px` (EXIF). Without either, a 60° field of view is assumed and a warning is printed. On the bed capture, assuming 60° for a 75° camera raised the camera-pair rotation error from 0.95° to 28° median. Recovering the focal length from these photos was tried (triplet rotation consistency, inlier count, depth-based 3D agreement) and none of the signals identified it, so it was not kept.

Results on the 16-photo bed capture (22.5° apart, 2.2 m away; source units appear to be meters but are unverified):

| Run | Registered | Camera-pair rotation error (median / max) | Worst camera position error | Scale (truth ÷ estimate) |
|---|---|---|---|---|
| True depth (diagnostic) | 13 / 16 | 0.95° / 2.58° | 0.095 | 1.00 |
| Depth model, focal known | 13 / 16 | 0.95° / 2.58° | 0.095 | 0.945 |
| Depth model, 60° assumed | 13 / 16 | 27.9° / 71.4° | 1.25 | 0.67 |

The three unregistered photos are the ones blocked by the wall. With 8 photos (45° apart) only 2 photos registered: SIFT found too few matches between neighbours, so captures need roughly 20–25° spacing. Rotations and camera spacing no longer depend on depth, so true depth and model depth give the same camera positions; only the overall scale (5.5% small) comes from the depth model. An earlier version took each pair's baseline from the depth model and was much worse (worst position error 0.32, median 0.14, against 0.095 and 0.044 now): the model's per-photo depth is off by 10–15%, and each photo only matches its 2–3 neighbours, so those errors could not average out. There is no bundle adjustment yet, which is why true depth no longer improves positions.

## Find the object in each photo (CPU step 3)

`photo_masks.py` turns the step 2 depth and poses into one object mask per registered photo, without a segmentation model and without reading `truth/` (except with `--score`):

```sh
.venv/bin/python photo_masks.py --bundle results-sim-bed16 --poses results-poses-bed16 \
  --output results-masks-bed16 --score
```

It back-projects every photo's corrected depth into the shared frame, estimates "up" from how the cameras are held, takes the lowest large near-horizontal RANSAC plane as the support surface, and builds a floor grid of what stands on it. Cells need points from at least two photos and almost nothing taller than `--max-height` (2.0). An odd-sized grid opening removes structures thinner than `--thin` (0.2), which drops partly seen walls. The object is the connected group of cells nearest the point all cameras aim at. Masks are written as `masks/view_XX.png` (XX = bundle image index). A folder of hand-drawn or GPU-made masks in the same layout is a drop-in substitute for the next step.

Results on the 16-photo bed capture (IoU against the hidden truth mask):

| Depth source | Median IoU | Worst IoU |
|---|---|---|
| True depth (diagnostic) | 0.77 | 0.70 |
| Depth model (photos only) | 0.69 | 0.36 |
| Depth model, before shared-point placement | 0.46 | 0.10 |

With true depth the whole bed is captured. The extra area is the pillows, headboard and lamps, which lie outside the truth box, plus some wall. With the depth model, consistent camera positions plus per-photo depth corrections from the triangulated points make the method usable, but the depth model also distorts shape inside each photo (about 6–9% error remains even after a perfect smooth per-photo correction), so the worst photos still miss part of the bed. A per-photo floor fit and smooth (position-dependent) depth corrections were tried and did not help, so they were removed. Limits by design: an object on a visible-floor table is merged with the table; touching furniture merges; thin parts (legs, poles) and objects taller than `--max-height` are cut.

## Carve the object from masks (CPU step 4)

`photo_carve.py` builds a voxel model of the object from the step 2 cameras and any mask folder in the `masks/view_XX.png` layout (step 3 output, hand-drawn, or later GPU masks):

```sh
.venv/bin/python photo_carve.py --poses results-poses-bed16-v2 --masks results-masks-bed16-v2 \
  --output results-carve-bed16 --bundle results-sim-bed16 --score
```

A coarse cube around the point the cameras aim at is carved, shrunk to what survived, and carved again at `--resolution` (128) voxels along the longest side. Each voxel is projected into every masked photo. It stays when at least `--agree` (0.85) of the photos that see it put it inside the mask, and at least `--min-views` photos see it (default three quarters). Voting means one bad mask cannot erase the object; the "seen by most photos" rule removes slivers behind the object that only a few photos see. Masks are grown by `--dilate` (2) pixels to tolerate small camera errors, and voxels under the support plane from `masks.json` are removed when it is present. The surface between kept and empty voxels is written as a closed mesh in the `collision-scene.json` format (`colliders/carved_object.obj`), plus `voxels.npz` and `carve.json`. It takes about 3 seconds for 13 photos at 640×480. `--score` aligns the cameras to `truth/` and compares the mesh with the hidden target surface.

Results on the 16-photo bed capture (13 registered photos; distances in source units):

| Masks and cameras | F1 within 0.05 | F1 within 0.1 | Chamfer | True surface inside carving |
|---|---|---|---|---|
| Perfect masks and cameras (reference) | 0.34 | 0.62 | 0.12 | 0.96 |
| True-depth masks (diagnostic) | 0.25 | 0.42 | 0.19 | 0.87 |
| Photos only | 0.30 | 0.47 | 0.14 | 0.40 |

Even perfect inputs give a modest score: carving from one ring of raised cameras cannot remove space the photos never see past (under the bed, around the base, between pillows), so the result is always fatter than the object. A unit cube carved from 8 such photos comes out about 1.5 times its volume. With true-depth masks the extra pillows, headboard and lamp area in the masks become extra volume. Photos-only carving is smaller than the bed: its masks miss parts of the bed in several photos, and carving cannot add back what a mask left out, so only 40% of the true surface is inside. Its F1 is similar because it is tight where it is right. Stricter agreement (1.0) gives tighter shapes with perfect masks but cuts more of the object with imperfect ones; 0.75 left the hull much too wide (F1 0.40 even with perfect inputs).

