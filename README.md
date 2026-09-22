# Bucha segmentation — Python in VS Code

For a new agent session, open this repository as the workspace and read **`AGENTS.md` → `agent.md` → `handoff.md`**. The handoff records the tested baseline, limits and next-session context.

The project is a local Git repository on `main`. The accompanying `bucha-python.zip` is a source snapshot; `bucha-python.bundle` includes Git history and can be cloned with `git clone bucha-python.bundle bucha-python`. No remote has been configured.

**Open this folder in VS Code and run `run.py`.** Segmentation, asset export, collision geometry, collision ray queries and preview images all execute in Python. The supplied GLB is included under `data/`.

This reproduces the current **scan-specific, visually guided segmentation**. The editable masks in `annotations.json` are inputs; the script does not automatically discover semantic objects in arbitrary new scans. No trained segmentation model is used.

## Run on this computer

1. Open the **bucha-python** folder itself in VS Code using **File → Open Folder**.
2. Choose **Python: Select Interpreter** and select **`.venv/bin/python`**. The environment is already installed on this computer. On Windows its equivalent is `.venv/Scripts/python.exe`.
3. Open `run.py` and use **Run Python File**, or press **F5** with **Bucha: full pipeline** selected. The F5 configurations require VS Code's Python and Python Debugger extensions.

In a new VS Code terminal with that environment selected:

```sh
python run.py
```

The full run took **5.7 seconds on this machine**. It regenerates the files in `results/`; rerunning replaces generated files with the same names. The source GLB is read only.

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
| `segmented-scene.glb` and six individual `.glb` files | Three building groups, ground/roads, rubble and uncertain geometry, with original textures and shared coordinates |
| `face-labels.npz`, `manifest.json`, `annotations-used.json` | Object IDs, original face references, review flags, bounds and the exact annotation settings used |
| `colliders/*.npz` and `colliders/*.obj` | One geometry-only collision surface per object, in source world coordinates |
| `collision-scene.json` | Maps each collision surface to its object ID and semantic class |
| `previews/*.png` and `run-report.json` | Label/texture views, an annotation map, separate-object previews and execution/verification results |

All **183,051 source triangles** are retained exactly once across the visual objects. The exporters verify original vertex attributes, UVs, material assignments and embedded images. Collision exports merge identical vertex positions but retain exactly the same world-space triangles.

Open **`results/previews/scene-labeled.png`** or **`objects-preview.png`** directly in VS Code to inspect the result. The lightweight PNG renderer uses depth-sorted triangles and one texture sample per face; its texture view is approximate. The GLBs retain the complete original texture mapping.

## Try a collision query in Python

After running the pipeline:

```sh
python collision.py --origin 10 0 20 --direction 0 0 -1
```

With the default annotations, this downward ray returns object **5**, `rubble`, including hit position, distance and the original source face. A second example hits the central building:

```sh
python collision.py --origin 7 8 20 --direction 0 0 -1
```

The F5 configuration **Bucha: collision ray example** runs the first query. You can also call it from another Python script:

```python
from pathlib import Path
from collision import raycast_scene

scene = Path(__file__).resolve().parent / "results" / "collision-scene.json"
hit = raycast_scene(scene, origin=[7, 8, 20], direction=[0, 0, -1])
if hit:
    print(hit["object_id"], hit["object_name"], hit["position_world"])
```

This tests which object's triangle surface a ray hits. It is **not** a physics simulation, a robot-clearance test or a navigation planner. The collision surfaces retain scan holes and incomplete backs. A ground label does not establish traversability.

## Change and rerun

Edit `annotations.json` to change masks or height/normal thresholds. Keep the existing six object IDs and names. `results/previews/annotation-map.png` shows the polygon outlines in the file's pixel coordinate system; each pixel maps to world XY as documented in `annotations.json`.

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
| `run.py` | Run the complete pipeline and write its report |
| `segment_objects.py` | Classify original faces and export textured assets |
| `collision.py` | Export collision surfaces and query nearest ray hits by object ID |
| `render_previews.py` | Generate all PNG previews in Python |
| `setup_env.py` | Create the local environment and install pinned dependencies |

## Checks and current limits

After running the default pipeline:

```sh
python -m unittest discover -s tests -v
```

Seven tests passed on Python 3.12.14 with the pinned dependencies. They cover the default face partition, collision-file geometry and source-face mapping, two distinct object hits on the actual scan, nearest-hit ordering, non-unit ray directions, both triangle sides, misses, distance limits and invalid directions. Count-based regression checks intentionally expect the default annotations; update their expected counts if you change those masks.

Labels remain provisional, especially debris mixed with ground and the south/east perimeter facade grouped as one building. Accuracy has not been measured against ground truth. Source +Z is up, and real-world scale is unknown.

The Python pipeline creates collision **geometry and queryable object identities**. Attaching engine-specific components such as Unity MeshColliders and connecting a Magic Leap 2 feed are later integration steps. An RGB feed alone does not transmit these object identities. No robot or ML2 connection has been implemented or tested.

## Source asset

**Destroyed Apartments, Bucha Ukraine War 2022**, by **LibanCiel**. [Matching Sketchfab model](https://sketchfab.com/3d-models/destroyed-apartments-bucha-ukraine-war-2022-f9dbc156ba95471c840c2d3e52880629). The saved metadata lists the **Standard** license; evidence and limitations are in `source-license.json`. Keep the asset's applicable acquisition terms with any later redistribution.
