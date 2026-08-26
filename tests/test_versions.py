"""C7: transformers changes tokenizer behaviour, logits-processor interfaces and
sampling code across minor versions, so a drifted install silently invalidates
byte-identical reruns."""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[A-Za-z0-9._+!-]+)$")


def _pins() -> list[tuple[str, str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    specs = list(data["project"]["dependencies"])
    for extra in data["project"].get("optional-dependencies", {}).values():
        specs.extend(extra)
    out = []
    for spec in specs:
        m = PIN.match(spec.strip())
        if m:
            out.append((m["name"], m["version"]))
    return out


def test_every_dependency_is_exactly_pinned():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    specs = list(data["project"]["dependencies"])
    unpinned = [s for s in specs if not PIN.match(s.strip())]
    assert not unpinned, f"dependencies must be pinned with ==: {unpinned}"


@pytest.mark.parametrize("name,pinned", _pins(), ids=lambda v: str(v))
def test_installed_version_matches_pin(name, pinned):
    try:
        installed = version(name)
    except Exception:
        pytest.skip(f"{name} is not installed (optional extra)")
    # torch publishes CPU builds as 2.9.1+cpu against a 2.9.1 pin
    assert installed.split("+")[0] == pinned, f"{name}: installed {installed}, pinned {pinned}"
