# Voxel Renderer Optimization Plan

This document captures the current performance options for `lptd.demos.vox`, in priority order.

## 1) Chunk Shell Cache (Implemented)

Build and cache only surface/shell voxel points per chunk (not all non-air voxels).

- Idea: keep voxels where at least one neighbor is air/out-of-bounds.
- Rebuild cache only when a chunk is modified.
- Why: biggest reduction in per-frame candidates and projection work.

Expected impact: high.

## 2) Chunk-Local Sprite Instance Cache (Implemented)

Reduce per-frame Python object churn for billboard entities.

- Cache reusable sprite-instance inputs per chunk where possible.
- Keep dynamic values (depth/projection) updated per frame, but avoid repeated allocations/work for static metadata.

Expected impact: medium.

## 3) GPU Batching Upgrade (Implemented: Client-Array Batching)

Replace `glBegin/glEnd` style drawing with vertex arrays/VBOs.

- Batch block splats and sprite quads in larger buffers.
- Reduce CPU->driver call overhead.

Expected impact: high (especially at larger view ranges), but bigger refactor.

## 4) More Aggressive Culling (Implemented)

Tighten chunk and point rejection before draw-list append.

- Stronger chunk-level rejection.
- Optional stricter per-point screen bounds for far points.

Expected impact: low-to-medium.

## 5) Distance-Based LOD Sampling (Implemented)

Draw fewer points in far chunks.

- Keep full density nearby.
- Skip pattern/sample decimation farther away.

Expected impact: medium, configurable quality/perf tradeoff.

## Recommended Sequence

1. Implement chunk shell cache first.
2. Then profile and decide between sprite-instance caching vs VBO migration.
3. Add distance-based LOD only if needed after the above.
