# Vendored Motion Diffusion Model

This directory contains the inference-only subset used by BVH merging. It was
adapted from the local `motion-diffusion-model` repository at commit `fa715d7`
and remains covered by the included MIT license.

The service-specific changes namespace imports under `bvh_processing`, remove
training and visualization code, skip unused SMPL mesh initialization for the
HumanML3D feature path, and expose an in-process merge API.
