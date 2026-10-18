# {{TITLE}}

Synthetic twin of a real potted-sapling RGB-D capture. Every tree is rendered **calm and
at several wind speeds with the identical camera path**, together with exact per-leaf
ground truth, so 3D foliage reconstruction error can be split into
**occlusion/method error, sensor error and wind error**.

- {{N_TREES}} procedural trees (broad, small, compound and needle leaves; sparse to dense canopies), real leaf geometry (no alpha cards)
- {{N_SEQ}} sequences, {{FRAMES}} frames each ({{FPS}} fps timeline, every {{STRIDE}}th frame rendered), {{SIZE_GB}} GB
- Wind levels: {{WIND_LEVELS}}
- Camera: {{SENSOR}} intrinsics, {{W}}x{{H}}, fx={{FX}}, fy={{FY}}, cx={{CX}}, cy={{CY}} (OpenCV pixel-centre convention, no distortion)
- Each sequence comes CLEAN and NOISY (simulated depth/RGB sensor noise; noise draws are identical across the wind levels of a tree)
- Rendered with Blender {{BLENDER}}; generator commit `{{GIT}}`; built {{DATE}}; per-sequence zips: {{ZIPPED}}

## Trees

| id | preset | description | split |
|----|--------|-------------|-------|
{{TREE_TABLE}}

## Layout

```
trees/tree_XX/canonical/           rest-pose ground truth, shared by all wind levels of a tree
    canonical_tree.npz             all generator arrays (nodes, branches, meshes, leaf table)
    tree_full.ply                  wood + leaves + pot; face props: semantic, leaf_id, branch_id
    leaves.ply / wood.ply          leaves only (face leaf_id) / wood only (face branch_id)
    leaf_table.csv / .npz          leaf_id, attachment point, centroid, normal, area, parent branch
tree_XX/wind_<level>/
    clean/rgb/NNNNN.png            8-bit sRGB
    clean/depth/NNNNN.png          16-bit z-depth in millimetres, 0 = no geometry
    clean/depth_exr/NNNNN.exr      float32 z-depth in metres (channel "Z", subset of frames)
    clean/normal/NNNNN.png         16-bit camera-space normals (OpenCV axes, facing camera): n = (v-1)/65534*2-1, 0 = invalid
    clean/leaf_id/NNNNN.png        16-bit leaf id, 0 = not a leaf
    clean/semantic/NNNNN.png       0 background, 1 leaf, 2 wood, 3 pot/soil, 4 ground, 5 room
    clean/flow/NNNNN.png           forward optical flow to the next frame, KITTI encoding (R,G = flow*64+32768, B = valid)
    noisy/rgb, noisy/depth         sensor-noise versions (noisy/noise_meta.json has the profile and seed)
    poses/c2w_opencv.npy           (N,4,4) camera-to-world, OpenCV camera axes; also per-frame .npy
    poses/c2w_blender.npy          (N,4,4) camera-to-world, Blender/OpenGL camera axes
    poses/groundtruth_tum.txt      TUM format "t tx ty tz qx qy qz qw" (OpenCV axes)
    transforms.json                Nerfstudio (clean); transforms_noisy.json (noisy); train/val/test filenames included
    gt/leaf_tracks.npz             per frame x leaf: transform_3x4 [R|t] from canonical, centroid, normal, visible_px
    intrinsics.json, splits.json, metadata.json, qc.json, validation.json
splits.json                        view split rule + train / held-out trees
```

## Conventions

- World: right-handed, +Z up, metres; the pot stands on z = 0 at the origin.
- Poses are camera-to-world. `transforms.json` uses OpenGL camera axes (x right, y up, looking -z);
  `c2w_opencv` and TUM use OpenCV axes (x right, y down, looking +z). `c2w_opencv = c2w_gl @ diag(1,-1,-1,1)`.
- Depth is z-depth along the optical axis (not distance along the ray). Pixels without geometry are 0.
- Leaf motion is rigid: `x_t = R x_canonical + t` with `[R|t] = transform_3x4[frame, leaf_id-1]`. Calm sequences have identity transforms exactly.
- Nerfstudio: train with `--orientation-method none --center-method none --auto-scale-poses False` so outputs stay in metric world coordinates comparable with the GT meshes.

## Validation

Every sequence was checked automatically (see `validation.json`): world positions re-project onto their pixel to < 0.01 px;
cross-frame depth re-projection on static geometry < 0.05 px; back-projected ground lies on z = 0 also in image corners
(z-depth, not ray distance); leaf ids match the semantic mask; per-leaf transforms predict the rendered optical flow; RGB
edges align with id edges; calm/windy sequences share bit-identical poses.

## Licence

CC BY 4.0 (see LICENSE). Please cite the accompanying work if you use the dataset.
