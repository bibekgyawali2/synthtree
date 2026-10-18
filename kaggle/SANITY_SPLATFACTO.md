# Sanity training check: splatfacto on one calm + clean sequence (Kaggle T4)

Goal: prove the data + poses + intrinsics are consistent before running the benchmark.
A calm, clean, noise-free synthetic sequence with exact poses is the easiest possible case:
splatfacto should reach **PSNR > 30 dB on the held-out views within ~7k steps** (typically
32–38 dB full-image; leaf-masked PSNR a few dB lower because leaves are the hard part).

## Notebook cells (GPU T4, Internet on)

```bash
# 1) install (nerfstudio pulls gsplat; the first splatfacto run JIT-compiles CUDA kernels, ~5-10 min on T4)
pip install -q nerfstudio==1.1.5
# 2) the loader / converter from this repo (add the repo as a Kaggle dataset, or copy the kaggle/ folder)
cp -r /kaggle/input/synthtree-code/kaggle /kaggle/working/st && cd /kaggle/working/st
python convert.py nerfstudio --seq /kaggle/input/synthtree-v1/tree_01/wind_calm --variant clean \
    --out /kaggle/working/ns_t01_calm --init-ply clean
# 3) train (metric world kept: no re-orientation / re-centering / auto-scaling)
ns-train splatfacto --data /kaggle/working/ns_t01_calm --output-dir /kaggle/working/runs \
    --max-num-iterations 7000 --viewer.quit-on-train-completion True \
    --pipeline.model.background-color white \
    nerfstudio-data --orientation-method none --center-method none --auto-scale-poses False --eval-mode filename
# 4) evaluate on the held-out views + export Gaussians for the geometry metrics
ns-eval --load-config /kaggle/working/runs/ns_t01_calm/splatfacto/*/config.yml --output-path /kaggle/working/eval.json \
    --render-output-path /kaggle/working/renders
ns-export gaussian-splat --load-config /kaggle/working/runs/ns_t01_calm/splatfacto/*/config.yml --output-dir /kaggle/working/export
```

Then on any machine with this repo:

```bash
python scripts/eval/eval_geometry.py --recon export/splat.ply --tree-dir <dataset>/trees/tree_01 --out geom.json
python scripts/eval/eval_images.py --seq <dataset>/tree_01/wind_calm --renders renders_dir --out img.json
```

Notes: T4 has 15 GB VRAM; at 848x480 with ~130 train views splatfacto uses ~4-6 GB. `ns-eval`
render file names differ between nerfstudio versions; rename them to `NNNNN.png` (the frame index
from the test filenames) before `eval_images.py`. If `--background-color white` makes the sky
pixels harder, use `random` (default) instead.

## If PSNR is low (< 25 dB) — likely bugs, in order

1. **Pose convention**: transforms.json must be used as OpenGL c2w (it is). If you converted to
   OpenCV somewhere, the cameras look backwards → PSNR ~10–12 dB, Gaussians scattered.
2. **Auto-orientation / scaling**: forgetting `--orientation-method none --center-method none
   --auto-scale-poses False` does not hurt PSNR but breaks the geometry metrics (scene is
   normalised); pass `dataparser_transforms.json` to `eval_geometry.py --dataparser-transforms`.
3. **Intrinsics / resolution mismatch**: smoke data is rendered at half resolution; transforms.json
   of that sequence carries the matching fx/cx — don't mix files from different renders.
4. **Wrong variant**: `transforms_noisy.json` points at noisy images (JPEG + noise) → lower PSNR is
   expected (~28–32 dB).
5. **Windy sequence by mistake**: the scene is not static → blurred leaves, PSNR on leaves drops a lot.
6. **Depth units** (depth-nerfacto only): PNGs are millimetres; nerfstudio's default
   `depth_unit_scale_factor = 1e-3` matches. Do not use TUM's 5000.
7. **Images with alpha / colour space**: RGB PNGs are 8-bit sRGB without alpha; no extra gamma.
8. **Too few init points**: with `--init-ply none` splatfacto starts from random points; fine but slower.
