# Archived: GATOR-style photos-to-object pipeline (CPU)

Paused on 2026-11-18. The GATOR paper (NVIDIA, arXiv 2610.11215) has no released code, so the project moved to procedural Unity maps for the mocap cage. Kept here because it runs and is tested.

**What it does:** rebuild one object from a few photos, entirely on CPU.

| Step | File | Method |
|---|---|---|
| 1 | `photo_sim.py` | Render test "photos" of one object from a GLB, with hidden ground truth |
| 2 | `photo_poses.py` | Metric depth (ONNX) + SIFT matching → camera poses |
| 3 | `photo_masks.py` | Support-plane masks from depth and poses (no segmentation model) |
| 4 | `photo_carve.py` | Voting voxel carving → closed mesh collider |
| — | `recon_metrics.py` | Scoring: camera error, mask IoU, chamfer/F1 |

**Results on a simulated 16-photo capture of a bed** (from the living-room scan):
- **Cameras:** 13/16 placed, with 0.95° median rotation error. The focal length must be known.
- **Masks:** median IoU 0.69 from photos alone.
- **Carved shape:** F1 0.47 at 0.1 units. Always fatter than the object; with only 4 spread-out photos, known camera poses are needed.

It was never tried on real photos. The planned GPU version (learned cameras/depth, SAM 3 masks, an image-to-3D generator) was not built.

**Run** from the repo root; the scripts still use shared modules there (`render_views.py`, `collision.py`, `image_depth.py`, `download_metric_models.py`):

```sh
PYTHONPATH=. .venv/bin/python archive/gator/photo_sim.py --input data/TEEsavR23oF.glb --output results-sim-bed16 \
  --box 0.55 6.25 0.06 2.85 8.2 1.4 --views 16 --fov 75 --distance 2.2 --elevation 30
PYTHONPATH=. .venv/bin/python archive/gator/photo_poses.py --bundle results-sim-bed16 --output results-poses-bed16 --score
PYTHONPATH=. .venv/bin/python archive/gator/photo_masks.py --bundle results-sim-bed16 --poses results-poses-bed16 --output results-masks-bed16 --score
PYTHONPATH=. .venv/bin/python archive/gator/photo_carve.py --poses results-poses-bed16 --masks results-masks-bed16 --output results-carve-bed16 --bundle results-sim-bed16 --score
PYTHONPATH=. .venv/bin/python -m unittest discover -s archive/gator/tests   # 31 tests
```

Step 2 needs the metric depth models (`download_metric_models.py`). Full method notes and result tables are in `handoff.md` (sections dated 2026-11-02) and in git history.
