# Software versions of the Kaggle training runs

Read from the Kaggle kernel logs (`data/kaggle_out/*/*.log`, not tracked by git).
Hardware for every kernel: Kaggle `NvidiaTeslaT4` machine (2x T4 16 GB, one run per GPU).

## Section 6 (motion-model study, sauravkatwal17 kernels) - confirmed from logs

Kernels: `synthtree-paper2-stage1`, `synthtree-stage1-conditions`, `synthtree-stage1-gtmotion`,
`synthtree-stage1-multiview`, `synthtree-stage1-seeds`.

| Package | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| gsplat | 1.5.3 |
| NumPy | 2.1.3 |

`synthtree-stage1-seeds` also ran a version check with gsplat 1.4.0 on the same PyTorch
(`results/paper2_stage1_seeds/gsplat140`).

## Benchmark and baseline study (bibekgyawali / bibekgyawali2 kernels) - PENDING

Logs not yet downloaded; versions below are unconfirmed.

| Kernel | Models | Expected (unconfirmed) |
|---|---|---|
| `synthtree-bench` | splatfacto, depth-nerfacto | Nerfstudio 1.1.5 (unpinned `pip install nerfstudio`, runs dated 2026-10-20), gsplat 1.4.0 (pinned by Nerfstudio 1.1.5) |
| `synthtree-dyn` | static and deformable 3DGS | gsplat 1.4.0 (paper, Section 6) |
| `synthtree-mitigation` | trimmed-loss 3DGS | gsplat 1.4.0 |
| `synthtree-models` | check whether used in the paper | - |

Still needed from each log: Python, PyTorch/CUDA, gsplat, Nerfstudio (bench only).
