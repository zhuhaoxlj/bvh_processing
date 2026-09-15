# Vendored Motion Diffusion Model

This directory contains the inference-only subset used by BVH merging. It was
adapted from `https://github.com/zhuhaoxlj/motion-diffusion-model` at commit
`fa715d7` and remains covered by the included MIT license.

The service-specific changes namespace imports under `bvh_processing`, remove
training and visualization code, skip unused SMPL mesh initialization for the
HumanML3D feature path, and expose an in-process merge API.

To update the vendor, copy the corresponding inference files from a pinned
upstream commit, rewrite imports under `bvh_processing.vendor.mdm`, reapply the
SMPL removal and explicit resource-path patches, then run the focused MDM tests,
the full service suite, and one real CUDA merge before changing this commit ID.
