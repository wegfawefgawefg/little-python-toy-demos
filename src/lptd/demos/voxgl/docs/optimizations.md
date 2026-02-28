# Voxel Renderer Optimization Plan

This document captures performance options for `lptd.demos.voxgl`.

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

## Next Optimization Backlog

All items above are now implemented in `voxgl`. These are the next candidates, ordered by expected payoff.

## 6) GPU Vertex Shader Culling + Compaction

Move more rejection logic to GPU so CPU sends chunk payloads with minimal per-block work.

- Submit packed block buffers per visible chunk.
- Do frustum/depth reject in vertex shader and optionally write compacted visible lists.
- Keep CPU focused on chunk visibility and streaming.

Expected impact: high.
Complexity: high.

## 7) Persistent GPU Chunk Buffers

Stop rebuilding/uploading large block arrays every frame.

- Keep one VBO/SSBO per chunk (or pooled arena).
- Upload only when a chunk is generated/edited.
- Draw with offsets into persistent buffers each frame.

Expected impact: high.
Complexity: high.

## 8) Draw-Indirect / Multi-Draw Batching

Reduce driver overhead from many draw calls.

- Build command buffers for chunk draws.
- Use `glMultiDraw*`/indirect where available.

Expected impact: medium-high.
Complexity: medium-high.

## 9) Frustum-Driven Streaming Queue

Generate chunks by visibility priority instead of radius-only.

- Queue chunk jobs by projected screen size and front-facing score.
- Fill nearest/most-visible first to reduce pop and wasted generation.

Expected impact: medium.
Complexity: medium.

## 10) Chunk Occupancy Hierarchy

Skip empty/near-empty chunk work earlier.

- Maintain per-chunk occupancy bits and optional 4x4x4 mini-brick masks.
- Early-out chunks that cannot contribute visible points.

Expected impact: medium.
Complexity: medium.

## 11) Better LOD Strategy (Coverage-Preserving)

Replace hash-drop sampling with a screen-coverage target.

- Compute target points-per-pixel or per-solid-angle.
- Use blue-noise or stratified patterns to reduce visible fracture.
- Keep close chunks full-res, blend transitions across LOD bands.

Expected impact: medium (quality + perf stability).
Complexity: medium.

## 12) Sprite Path Optimization

Further reduce billboard overhead.

- Instance all billboard sprites in one pass by type.
- GPU-side billboard expansion from center + size.
- Optional sprite atlas packing to reduce binds/state churn.

Expected impact: low-medium.
Complexity: medium.

## 13) Worldgen/IO Parallelism

Hide generation and meshing cost behind background workers.

- Generate chunks in worker threads/processes.
- Main thread only performs finalized uploads/cache insertion.

Expected impact: medium (frame-time stability).
Complexity: medium-high.

## 14) Optional Cython/Native Hot Paths

If Python-side prep is still the bottleneck.

- Move hottest loops (packing, culling prep, queueing) to Cython/Rust/C++ extension.

Expected impact: medium.
Complexity: high.

## Recommended Next Sequence

1. Persistent GPU chunk buffers.
2. Draw-indirect / multi-draw batching.
3. Better LOD strategy (coverage-preserving).
4. Frustum-driven streaming queue.
