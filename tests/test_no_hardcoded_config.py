"""A literal d or H anywhere but config.py silently pins one grid point and
invalidates the sweep that produces F1."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "wmk"
DEPLOYMENT_PARAMS = {
    "context_depth",
    "tournament_depth",
    "masking_window",
    "temperature",
    "top_p",
    "top_k",
    "n_rounds",
    "d",
}
MODULES = sorted(p for p in SRC.glob("*.py") if p.name != "config.py")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_deployment_parameters_are_never_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in DEPLOYMENT_PARAMS for t in targets) and isinstance(
                node.value, ast.Constant
            ):
                offenders.append(f"line {node.lineno}: {targets} = {node.value.value!r}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
            names = [a.arg for a in args.args + args.posonlyargs + args.kwonlyargs]
            if not set(names) & DEPLOYMENT_PARAMS:
                continue
            positional = args.posonlyargs + args.args
            paired = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
            paired += [
                (a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None
            ]
            for arg, default in paired:
                if arg.arg in DEPLOYMENT_PARAMS and isinstance(default, ast.Constant):
                    offenders.append(
                        f"line {node.lineno}: {node.name}({arg.arg}={default.value!r})"
                    )

    assert not offenders, f"{path.name} hardcodes deployment parameters: {offenders}"
