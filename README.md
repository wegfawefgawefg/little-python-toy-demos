# little-python-toy-demos

Tiny toy demos meant to be run as scripts via `uv`.

## Setup

```bash
uv sync
```

## Run

Snake (manual):

```bash
uv run python demos/snake_manual.py
```

Snake (auto/BFS, very fast):

```bash
uv run python demos/snake_auto.py
```

Voxel-ish world (software render):

```bash
uv run python demos/vox_soft.py
```

Voxel-ish world (software render, v3):

```bash
uv run python demos/vox_soft_v3.py
```

Voxel-ish world (OpenGL):

```bash
uv run python demos/vox_gl.py
```
