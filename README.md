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

Voxel-ish world (canonical, supports both renderers):

```bash
uv run python -m lptd.demos.vox
uv run python -m lptd.demos.vox --renderer gl
uv run python -m lptd.demos.vox --renderer gl --render-scale 2
uv run python -m lptd.demos.vox --render-scale 1
```

Snake (manual):

```bash
uv run python -m lptd.demos.snake_manual
```

Snake (auto/BFS, very fast):

```bash
uv run python -m lptd.demos.snake_auto
```
