# SynthTree — paired calm/windy synthetic RGB-D foliage dataset

A fully scripted, reproducible pipeline that renders procedural potted saplings as
**calm and windy sequences with identical camera paths** and exact per-leaf ground
truth: the synthetic twin of a real RealSense-style capture. It is built to benchmark 3D
foliage reconstruction (3DGS / NeRF / RGB-D SLAM) and to split the error into
**occlusion/method**, **sensor** and **wind** components.

Verified on: macOS 27 / Apple M5 / 16 GB, **Blender 5.2.2 LTS** (primary) and **Blender 4.5.14 LTS**,
host Python 3.9 (any 3.9–3.13 works). Ground truth is bit-identical between the two Blender versions.

---

## Design decisions (where this deviates from the original spec, and why)

| Spec asked for | Implemented instead | Why |
|---|---|---|
| Sapling Tree Gen / GN tree | Own NumPy generator (`synthtree/tree.py`) | Needs branch hierarchy, node frames, attachment points and parent ids as first-class data; Sapling exposes none of these and is a separately-installed extension since 4.2; NumPy version is deterministic and unit-testable without Blender. |
| Wind in Geometry Nodes | Closed-form NumPy wind (`synthtree/wind.py`), vertices pushed into Blender each frame | The *same* function moves the rendered vertices and produces the exported per-leaf transforms ⇒ GT is exact by construction (validated to 0.006 px); no GN↔GT formula duplication; immune to GN API changes. |
| Blender motion blur / rolling shutter | Sub-frame rendering (average in linear light) / row-band rendering at per-row times | Native MB/RS only see depsgraph animation, not per-frame vertex writes; ours is exact and works in EEVEE too. |
| Depth/normal/ID from the beauty render | Separate Cycles data pass, 1 spp, 0.01 px box filter, 32-bit EXR AOVs (world position, normal, leaf id, semantic, next-frame position, rest position) | AA would mix fore/background in GT depth & ids; the world-position AOV makes depth exactly z-depth independent of Blender's pass definitions. |
| Blender Vector pass for flow | Flow from a per-vertex `P_next` attribute | Blender's vector pass is wrong for script-driven deformation. |
| Compositor File Output | None — multilayer EXR straight from the render | The compositor API changed completely in Blender 5.0; avoiding it keeps 4.5 and 5.x on one code path. |

Everything else follows the spec. Known limitations are listed at the end.

---

## Repository layout

```
configs/
  master.yaml            master config: sensor, capture, wind levels/model, render, outputs, noise, splits, 10 trees, smoke overrides
  debug_2x2.yaml         start-small plan (2 trees x 2 wind levels x 60 frames), extends master.yaml
  sensor_d435i.yaml      real-sensor intrinsics/depth settings  >>> fill in your calibration <<<
  species.yaml           10 species presets (broad/small/compound/needle x sparse/medium/dense, autumn)
synthtree/               pure-NumPy core (runs in Blender's Python and host Python)
  rng.py                 named deterministic random substreams
  transforms.py          rotations, quaternions, Blender<->OpenCV<->OpenGL conventions, look-at
  intrinsics.py          K <-> Blender camera (lens, shift, pixel aspect), project/backproject
  leaves.py              real-geometry leaf meshes: ovate/elliptic/..., compound pinnate, needles
  meshing.py             branch tubes, pot, soil, ground, room
  tree.py                procedural sapling generator (skeleton, wood mesh, leaves, per-leaf table)
  wind.py                exact procedural wind: hierarchical branch sway + rigid leaf flutter
  camera_path.py         orbit + seeded hand-held jitter + static ring
  config.py              YAML master -> resolved per-sequence JSON, hashing, `extends`
  noise.py               depth/RGB sensor-noise model
  imageio.py             EXR (multi-part aware) / PNG I/O, normal & KITTI-flow encodings (host only)
  pose_formats.py        transforms.json, TUM, npy pose exports
  colour.py              exact sRGB transfer functions
  ply.py                 binary PLY writer/reader with per-face properties
blender/                 run with: blender -b --factory-startup -P <script> -- <args>
  bl_common.py           version shims (4.5/5.x), scene build, materials + AOVs, render settings
  render_sequence.py     renders one sequence: data pass + beauty pass, per-frame GT, resumable
  check_render_setup.py  renders one frame to PNG + EXR to verify colour management
  build_scene.py         saves an inspectable .blend of a sequence at a given time
scripts/                 host Python
  make_tree.py           generate + export one canonical tree (npz, PLYs, leaf table)
  run_batch.py           batch runner: tree -> render -> post -> noise -> validate; resumable; --smoke
  postprocess.py         raw EXR -> dataset products, poses, transforms.json, leaf tracks, metadata, qc
  noise_sim.py           clean -> noisy/{rgb,depth}
  calibrate_noise.py     fit noise profile from real wall/leaf recordings (+ --demo self-test, plots)
  validate_sequence.py   reprojection / z-depth / id / alignment / flow+leaf-GT / noise / pairing checks
  wind_preview.py        side-by-side low-res clip of every wind level for one tree
  benchmark.py           10 frames per engine -> time & disk extrapolation for the full config
  make_overlay.py        visual QC montage of one frame
  check_colour.py        compares Blender's Standard view transform with our sRGB conversion
  package_dataset.py     Kaggle dataset folder + dataset card + licence + splits + size report
  eval/eval_geometry.py  chamfer/F-score, leaf area, leaf-angle distribution, per-leaf centroid error
  eval/eval_images.py    leaf-masked PSNR / SSIM / LPIPS and depth metrics on held-out views
  eval/collect_results.py  per-run JSONs -> results.csv
  eval/decompose_errors.py three-way decomposition + plots (error component vs U per method)
kaggle/
  synthtree_dataset.py   self-contained reader + PyTorch Dataset
  convert.py             -> nerfstudio (splatfacto / depth-nerfacto) or MonoGS (TUM) layout
  SANITY_SPLATFACTO.md   quick T4 sanity run and a list of likely bugs if PSNR is low
dataset_card/            README template + CC BY 4.0 licence text for the Kaggle dataset
docs/PITFALLS.md         known pitfalls and how each is handled/verified
tests/                   pytest (no Blender needed): geometry, wind, conventions, noise, eval
```

---

## Commands, in order

```bash
# 0. setup (once)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export BLENDER=/Applications/Blender.app/Contents/MacOS/Blender        # or pass --blender
.venv/bin/python -m pytest                                             # 33 tests, ~3 s

# 1. fill in your real setup
#    configs/sensor_d435i.yaml : fx fy cx cy, resolution, depth range  (rs-enumerate-devices -c)
#    configs/master.yaml       : capture.duration_s, orbit radius/heights, wind.levels (m/s at canopy)
#    configs/species.yaml      : height_m ranges matching your saplings

# 2. smoke test: 2 trees x calm/high x 6 frames at half resolution, all stages + validation (~20 s;
#    the very first Cycles render compiles Metal kernels for ~2 min, once per Blender version)
.venv/bin/python scripts/run_batch.py --master configs/master.yaml --smoke

# 3. look at the trees and the wind
.venv/bin/python scripts/make_tree.py --master configs/master.yaml --tree 1
.venv/bin/python scripts/wind_preview.py --master configs/master.yaml --tree 1     # data/previews/tree_01/*.mp4
.venv/bin/python scripts/make_overlay.py --seq data/smoke/tree_01/wind_high --frame 2 --out overlay.png
$BLENDER -b --factory-startup -P blender/build_scene.py -- --config data/smoke/tree_01/wind_high/config.json --out tree01.blend --time 1.0

# 4. start small: 2 trees x 2 wind levels x 60 frames at full resolution (~2-4 min)
.venv/bin/python scripts/run_batch.py --master configs/debug_2x2.yaml

# 5. benchmark + disk estimate for the full config
.venv/bin/python scripts/benchmark.py --master configs/master.yaml

# 6. full dataset (resumable: just run it again after a crash / Ctrl-C)
caffeinate -i .venv/bin/python scripts/run_batch.py --master configs/master.yaml --cooldown-s 20

# 7. once you have real recordings: calibrate noise, then re-run only the noise + validate stages
.venv/bin/python scripts/calibrate_noise.py --wall rec/wall --leaves rec/leaves --intrinsics rec/intrinsics.json \
    --out configs/noise_profile_d435i.json --plots calib_plots/
#    set noise.profile: noise_profile_d435i.json in master.yaml (path relative to the repo root or absolute), then
.venv/bin/python scripts/run_batch.py --master configs/master.yaml --stages noise,validate --force

# 8. package for Kaggle and upload
.venv/bin/python scripts/package_dataset.py --master configs/master.yaml --estimate
.venv/bin/python scripts/package_dataset.py --master configs/master.yaml --out data/kaggle/synthtree-v1 --kaggle-id <user>/synthtree-v1
kaggle datasets create -p data/kaggle/synthtree-v1 --dir-mode zip

# 9. on Kaggle: convert + train (see kaggle/SANITY_SPLATFACTO.md), then evaluate and decompose
python kaggle/convert.py nerfstudio --seq /kaggle/input/synthtree-v1/tree_01/wind_calm --variant clean --out /kaggle/working/ns
python scripts/eval/eval_geometry.py --recon splat.ply --tree-dir <dataset>/trees/tree_01 --out results/splatfacto__tree_01__calm__clean__geom.json
python scripts/eval/eval_images.py --seq <dataset>/tree_01/wind_calm --renders renders/ --out results/splatfacto__tree_01__calm__clean__img.json
python scripts/eval/collect_results.py --dir results/ --out results.csv
python scripts/eval/decompose_errors.py --results results.csv --metric chamfer_mm --out decomposition/
```

Other useful flags: `run_batch.py --only tree_01/wind_high`, `--stages render`, `--max-frames 10`,
`--dry-run`, `--force`; `package_dataset.py --variants clean --no-flow --frame-stride 2 --zip`.

---

## What one sequence contains

`data/output/tree_XX/wind_<level>/` — see `dataset_card/README_TEMPLATE.md` for the full table.
Clean RGB / depth (mm PNG + float EXR subset) / camera-space normals / leaf-id / semantic / KITTI
flow; noisy RGB + depth; poses as npy (OpenCV and Blender axes), TUM and Nerfstudio
`transforms.json` (with train/test filenames); `gt/leaf_tracks.npz` (per frame × leaf: 3×4
transform from canonical, centroid, normal, visible pixels); `metadata.json` (config, seeds,
wind, render settings, Blender version + build hash, git commit, timings, conventions),
`qc.json`, `validation.json`. The canonical tree (`tree_XX/canonical/`) is shared by all wind
levels: full / leaves-only (face `leaf_id`) / wood-only PLYs and the per-leaf table.

**Splits**: every 8th frame of every sequence is a held-out view (`splits.json`, also in
transforms.json); trees 8 and 9 are the held-out tree set (`splits.heldout_trees`).

**Three-way decomposition** (scripts/eval/decompose_errors.py):
E_method = E(calm, clean); E_sensor = E(calm, noisy) − E(calm, clean);
E_wind(U) = E(U, noisy) − E(calm, noisy). Also run windy+clean: the script then reports the
interaction term [E(U,noisy) − E(U,clean)] − E_sensor; if it is large the components are not
additive and should be reported as such. Evaluate windy reconstructions against the canonical
(rest) tree — the wind error is precisely the deviation from it.

---

## Wind model and its defaults

Per node of the branch skeleton a bend rotation about the horizontal axis ⊥ wind (static
deflection + oscillation, per-branch frequency 0.6–2 Hz and phase, coherent along the branch) and
a smaller cross-wind sway, accumulated root→tip (forward kinematics), so displacement grows with
height and hierarchy depth. Node compliance ∝ (r_trunk/r)·segment length. Each leaf additionally
rotates rigidly about its petiole attachment (pitch + twist, 2.5–9 Hz, random phase, plus a mean
downwind reconfiguration). Gusts: U_eff(t) = U·(1 + 0.35·n(t)) with smooth seeded noise n.

Amplitude laws: branch ∝ (U/5)² (drag ∝ U², linear-elastic deflection ∝ force);
leaf ∝ (U/5) with tanh saturation (flutter grows ~linearly then saturates). Resulting motion
(10 s, defaults): broad-leaved tree 1 — median leaf displacement 6 / 19 / 48 mm and leaf
rotation 8° / 15° / 26° at 1.5 / 3 / 5 m/s; the stiffer conifer (tree 7) moves about a third as
much. **These are plausible, not calibrated.** To calibrate against your fan recordings,
compare the per-pixel temporal std of real vs synthetic RGB (or optical-flow magnitude) on
leaf pixels at each U and tune `wind.branch.static_deg/osc_deg` and `wind.leaf.pitch_deg/twist_deg`
(`wind_preview.py` makes the side-by-side clip).

---

## Performance (measured on this laptop: M5, 16 GB, Blender 5.2, Metal)

Full resolution 848×480, densest tree (8192 needles), per frame: Cycles data pass 0.26 s,
EEVEE beauty 0.23 s (32 samples) or Cycles beauty 1.25 s (64 samples + OIDN), post+noise
0.25 s. Full config (10 trees × 4 wind levels × 150 frames = 6000 frames): **≈ 1.3 h with
EEVEE, ≈ 3 h with Cycles; ≈ 12 GB** (1.95 MB/frame; noisy RGB is the largest product because
noise does not compress). First launch per Blender version compiles Cycles Metal kernels (~2 min).

Recommendations: EEVEE for the bulk set; a Cycles beauty subset (e.g. two trees) for a
realism check — the GT is identical either way. GPU (Metal) for both. Keep the laptop on
power, run under `caffeinate -i`, use `--cooldown-s 20` against thermal throttling. Crashes
are harmless: frames are written atomically and every stage resumes. Scaling up: more frames
(`frame_stride: 2`), longer orbits, `resolution_scale`, Cycles samples — rerun the benchmark.

---

## Assumptions to verify on your machine / data

- **Sensor values in `configs/sensor_d435i.yaml` are typical D435i colour intrinsics, not yours.**
  Depth is assumed aligned to colour (rs2::align), like the synthetic depth.
- Blender path defaults to `/Applications/Blender.app/...` (macOS); set `BLENDER` elsewhere.
  Tested on 4.5 LTS and 5.2 LTS; 4.2 should work (same API points as 4.5) but is untested.
- Noise defaults are D435-like guesses until `calibrate_noise.py` runs on your recordings.
- Wind amplitudes are uncalibrated (see above). Gusts are spatially uniform over the tree.
- `nerfstudio==1.1.5` flags in `kaggle/SANITY_SPLATFACTO.md` and the expected PSNR there were
  not run here (no CUDA GPU on this machine) — treat them as a starting point.
- HDRI backgrounds need a file you supply (`render.background.hdri_path`, e.g. Poly Haven CC0).

## Known limitations

- Leaves can interpenetrate each other (no collision handling) — rare, visible only up close.
- Compound leaves move rigidly as one unit (leaflets do not flutter independently).
- Branch bending is piecewise-rigid per skeleton segment (rings follow node frames); no gravity sag.
- Motion blur / rolling shutter affect RGB only; GT is at frame-centre time (global-shutter depth).
- Rendered pixels are deterministic per machine, not across GPUs; geometry, poses and GT are bit-identical everywhere.
