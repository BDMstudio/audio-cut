#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/audit_new_module_contracts.py
# AI-SUMMARY: Audits newly added Python modules for required headers and public docstrings.

from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Violation:
    """One module-contract violation found during audit."""

    path: Path
    message: str


def added_python_files(base_ref: str = "main") -> List[Path]:
    """Return Python files added on the current branch relative to base_ref."""

    result = subprocess.run(
        ["git", "diff", "--name-status", f"{base_ref}...HEAD", "--", "*.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: List[Path] = []
    for raw_line in result.stdout.splitlines():
        parts = raw_line.split("\t")
        if len(parts) < 2 or parts[0] != "A":
            continue
        paths.append(PROJECT_ROOT / parts[1])
    return paths


def audit_paths(paths: Iterable[Path], *, require_public_docstrings: bool) -> List[Violation]:
    """Audit explicit paths for required file headers and public docstrings."""

    violations: List[Violation] = []
    for path in sorted(Path(item) for item in paths):
        text = path.read_text(encoding="utf-8")
        header = text.splitlines()[:6]
        if not any(line.startswith("# File:") for line in header):
            violations.append(Violation(path=path, message="missing # File: header"))
        if not any(line.startswith("# AI-SUMMARY:") for line in header):
            violations.append(Violation(path=path, message="missing # AI-SUMMARY: header"))
        if require_public_docstrings:
            violations.extend(_public_docstring_violations(path, text))
    return violations


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for auditing new module headers and docstrings."""

    parser = argparse.ArgumentParser(description="Audit newly added Python modules.")
    parser.add_argument("--base-ref", default="main", help="Base ref for branch diff.")
    parser.add_argument("paths", nargs="*", help="Optional explicit Python files to audit.")
    args = parser.parse_args(argv)

    paths = [Path(item) for item in args.paths] if args.paths else added_python_files(args.base_ref)
    violations: List[Violation] = []
    for path in paths:
        violations.extend(
            audit_paths(
                [path],
                require_public_docstrings=_requires_public_docstrings(path),
            )
        )

    for violation in violations:
        rel_path = _display_path(violation.path)
        print(f"{rel_path}: {violation.message}")
    return 1 if violations else 0


def _public_docstring_violations(path: Path, text: str) -> List[Violation]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [Violation(path=path, message=f"syntax error: {exc.msg}")]

    violations: List[Violation] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name) and ast.get_docstring(node) is None:
            violations.append(Violation(path=path, message=f"public class {node.name} missing docstring"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            if ast.get_docstring(node) is None:
                violations.append(Violation(path=path, message=f"public function {node.name} missing docstring"))
    return violations


def _requires_public_docstrings(path: Path) -> bool:
    rel_parts = _relative_parts(path)
    if not rel_parts:
        return False
    return rel_parts[0] in {"src", "scripts"}


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_parts(path: Path) -> Sequence[str]:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).parts
    except ValueError:
        return path.parts


if __name__ == "__main__":
    raise SystemExit(main())
