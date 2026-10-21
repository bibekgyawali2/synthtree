# Deformable-3DGS re-run (for a co-author's Kaggle account)

Runs the 16 deformable 3D Gaussian Splatting runs of the SynthTree benchmark (~3.5 GPU-hours on 2x T4).

1. kaggle.com -> Create -> **New Notebook**
2. Right panel: **Add Input -> Datasets -> search `bibekgyawali2/synthtree-v1`** (shared with you) and add it.
   Session options: **Accelerator = GPU T4 x2**, **Internet = On**.
3. **File -> Import Notebook** is not needed: paste the whole content of `synthtree_dyn_rerun.py`
   (from https://github.com/bibekgyawali2/synthtree/tree/main/kaggle/kernels/dyn_collab) into one cell.
4. **Save Version -> Save & Run All (Commit)** and close the tab; it runs in the background (~4 h).
5. When it shows *Complete*: open the version -> **Output** tab -> download `results/` (or the whole output as zip)
   and send it to Bibek. Only the small `*__geom.json` files are needed.
