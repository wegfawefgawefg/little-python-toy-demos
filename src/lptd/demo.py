from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").is_file() and (p / "demos").is_dir():
            return p
    return None


def _list_demos(demos_dir: Path) -> list[str]:
    names: list[str] = []
    for p in demos_dir.glob("*.py"):
        if p.name.startswith("_"):
            continue
        names.append(p.stem)
    return sorted(names)


def _resolve_demo_path(repo_root: Path, demo: str) -> Path:
    # Allow passing a path (absolute or relative) for convenience.
    if os.sep in demo or (os.altsep and os.altsep in demo) or demo.endswith(".py"):
        p = Path(demo)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"demo script not found: {p}")
        return p

    p = repo_root / "demos" / f"{demo}.py"
    if not p.is_file():
        raise FileNotFoundError(f"unknown demo: {demo}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demo",
        description="Run a demo from ./demos (installed as a uv console script).",
    )
    parser.add_argument("--list", action="store_true", help="List available demos and exit.")
    parser.add_argument("demo", nargs="?", help="Demo name (e.g. snake_manual) or path to a .py file.")
    parser.add_argument(
        "demo_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the demo. Use `--` before the first forwarded arg.",
    )

    ns = parser.parse_args(argv)

    repo_root = _find_repo_root(Path.cwd())
    if repo_root is None:
        print("error: could not find repo root (expected pyproject.toml + demos/ in current dir or parents)", file=sys.stderr)
        return 2

    demos_dir = repo_root / "demos"
    available = _list_demos(demos_dir)

    if ns.list or ns.demo is None:
        for name in available:
            print(name)
        return 0 if ns.list else 2

    try:
        script_path = _resolve_demo_path(repo_root, ns.demo)
    except FileNotFoundError as e:
        msg = str(e)
        if msg.startswith("unknown demo:"):
            print(f"error: {msg.split(':', 1)[1].strip()}", file=sys.stderr)
            if available:
                print("\navailable demos:", file=sys.stderr)
                for name in available:
                    print(f"  {name}", file=sys.stderr)
        else:
            print(f"error: {msg}", file=sys.stderr)
        return 2

    demo_args = list(ns.demo_args)
    if demo_args and demo_args[0] == "--":
        demo_args = demo_args[1:]

    # Use a subprocess so the demo behaves like `python demos/foo.py`
    # (pygame/OpenGL scripts typically assume __main__ semantics).
    cmd = [sys.executable, str(script_path), *demo_args]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

