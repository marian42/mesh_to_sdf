# Calculate signed distance fields for arbitrary meshes

This project calculates SDFs for meshes.
It works for **non-watertight** meshes (meshes with holes), **self-intersecting** meshes, meshes with **non-manifold geometry** and meshes with **inconsistently oriented faces**.

The general pipeline for calculating SDF in this project is as follows:

1. Create virtual laser scans of the shape from multiple angles
2. Use the inverse MVP matrix and depth buffer of each scan to calculate a world-space surface point cloud
3. Determine the value of the SDF for each query point by finding the closest surface point using a kd-tree
4. Determine the sign of the SDF using either the normal of the closest surface point or by checking it against the depth buffers of the scans