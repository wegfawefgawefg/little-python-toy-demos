# little-python-toy-demos

Tiny toy demos meant to be run as scripts via `uv`.

## Setup

```bash
uv sync
```

## Run

Run directly as a module:

```bash
uv run python -m lptd.demos.snake_manual
```

Voxel-ish world (hardware path, active development):

```bash
uv run python -m lptd.demos.voxgl
uv run python -m lptd.demos.voxgl --render-scale 2
uv run python -m lptd.demos.voxgl --render-scale 1
```

Voxel-ish world (legacy/soft-hybrid path):

```bash
uv run python -m lptd.demos.vox
```

Snake (manual):

```bash
uv run python -m lptd.demos.snake_manual
```

Snake (auto/BFS, very fast):

```bash
uv run python -m lptd.demos.snake_auto
```
