"""Utility script for updating the STCK installation.

The update command tries to keep the project current with a single shell
invocation. When run inside a Git checkout it performs a ``git pull``;
otherwise it falls back to ``pip install --upgrade stck``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the STCK installation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--git", action="store_true", help="Force using git pull to update")
    group.add_argument(
        "--pip",
        action="store_true",
        help="Force using pip to install the latest published package",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )
    return parser.parse_args(argv)


def is_git_checkout(path: Path) -> bool:
    return (path / ".git").exists()


def _run_command(command: Sequence[str], *, cwd: Optional[Path] = None, dry_run: bool = False) -> int:
    if dry_run:
        print("Dry run:", " ".join(command))
        return 0
    try:
        subprocess.run(command, check=True, cwd=cwd)
    except FileNotFoundError as exc:
        print(f"Required executable not found: {command[0]}")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(command)}")
        return exc.returncode or 1
    return 0


def update_with_git(*, dry_run: bool = False) -> int:
    print("Updating STCK via git pull...")
    return _run_command(["git", "pull", "--ff-only"], cwd=PROJECT_ROOT, dry_run=dry_run)


def update_with_pip(*, dry_run: bool = False) -> int:
    print("Updating STCK via pip...")
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "stck"]
    return _run_command(command, dry_run=dry_run)


def choose_strategy(force_git: bool, force_pip: bool) -> str:
    if force_git:
        return "git"
    if force_pip:
        return "pip"
    return "git" if is_git_checkout(PROJECT_ROOT) else "pip"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    strategy = choose_strategy(args.git, args.pip)
    if strategy == "git":
        return update_with_git(dry_run=args.dry_run)
    return update_with_pip(dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "choose_strategy",
    "is_git_checkout",
    "main",
    "parse_args",
    "update_with_git",
    "update_with_pip",
]
