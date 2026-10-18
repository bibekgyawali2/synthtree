# Real-data capture protocol

Everything here is recorded with `scripts/capture/rs_capture.py`, which saves colour + depth
**aligned to colour** (16-bit PNG, millimetres) in the folder layout the pipeline reads. Use the
**same resolution, fps and visual preset for every recording** (defaults come from
`configs/sensor_d435i.yaml`; preset `high_accuracy`).

Setup (once):
```bash
# Linux/Windows: .venv/bin/pip install pyrealsense2
# macOS arm64:   .venv/bin/pip install pyrealsense2-macosx   (run the commands below with sudo if the camera is not found)
brew install librealsense && rs-enumerate-devices -c          # optional: confirms the camera and firmware
```

## Step 1 — intrinsics (2 min)

```bash
.venv/bin/python scripts/capture/rs_capture.py intrinsics --write-config
```
Writes `rec/intrinsics.json` and updates fx/fy/cx/cy, resolution, fps and `depth.effective_scale`
in `configs/sensor_d435i.yaml`. Then edit `configs/master.yaml` with your real capture geometry:
`capture.duration_s`, `capture.orbit.radius_m`, `capture.orbit.heights_m`, and `wind.levels`
(anemometer reading **at the canopy**, not at the fan). Re-render afterwards (~1.5 h).

## Step 2 — noise calibration (≈ 1 h)

**Wall** — flat, matte, *textured* wall (poster or printed noise pattern; stereo needs texture), room lit
normally, camera on a tripod facing the wall squarely, measure the distance with a tape:
```bash
for d in 0.4 0.8 1.2 1.8 2.5 3.0; do
  read -p "place camera at $d m and press enter"
  .venv/bin/python scripts/capture/rs_capture.py wall --distance $d --frames 30
done
```
**Leaves** — a real sapling with **≥ 1 m of free space behind it** (then a wall), tripod, calm air:
```bash
.venv/bin/python scripts/capture/rs_capture.py leaves --frames 30
```
Then:
```bash
.venv/bin/python scripts/calibrate_noise.py --wall rec/wall --leaves rec/leaves --intrinsics rec/intrinsics.json \
    --out configs/noise_profile_d435i.json --plots data/calib_noise/
```
Check `data/calib_noise/*.png` (real vs simulated histograms), set `noise.profile:
configs/noise_profile_d435i.json` in `configs/master.yaml`, and re-run only the noise stage:
`run_batch.py --master configs/master.yaml --stages noise,validate --force`.

## Step 3 — wind calibration (≈ 15 min)

Camera on a tripod at roughly your orbit radius, whole sapling in view, fan at each level, wait ~10 s
for the motion to settle before recording:
```bash
for lvl in calm low medium high; do
  read -p "set fan to $lvl and press enter"
  .venv/bin/python scripts/capture/rs_capture.py clip --name fan_$lvl --seconds 10
done
.venv/bin/python scripts/calibrate_wind.py --clips rec/clips --master configs/master.yaml --tree <closest synthetic tree id>
```
Copy `suggested_master_overrides` from `data/calib_wind/wind_calibration.json` into `master.yaml`
(`wind.branch` / `wind.leaf`) and re-render. Pick `--tree` with the same leaf type/size as the filmed sapling.

## Step 4 — the real dataset (per sapling × wind level)

Mark the orbit circle on the floor (radius = `capture.orbit.radius_m`) and the camera height on a
monopod; walk one loop in `capture.duration_s` seconds at constant speed (a metronome helps), camera
always pointed at the tree centre. Same start position and direction for every wind level.
```bash
.venv/bin/python scripts/capture/rs_capture.py orbit --tree 0 --wind calm --seconds 20
.venv/bin/python scripts/capture/rs_capture.py orbit --tree 0 --wind high --seconds 20
```
Real poses are not known — estimate them per sequence with COLMAP or an RGB-D SLAM on the **calm**
sequence (the windy one shares the path only approximately; hand-held orbits are never identical).
Record for each sapling: species, height, leaf type, a photo, and the canopy wind speed per level.

Tips: fixed indoor lighting (no sunlight flicker), disable auto white balance drift by letting the
camera warm up (the script discards 30 frames), keep the emitter on, avoid reflective pots.
