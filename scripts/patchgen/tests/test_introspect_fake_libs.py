# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""``introspect.py`` against small on-disk fake libraries (no network, no venv)."""

from __future__ import annotations

import importlib
import io
import json
import sys
import textwrap
import typing
from collections.abc import Iterator
from pathlib import Path

import pytest
from patchgen import introspect
from patchgen.introspect import _collect_exports, _resolve_annotation, describe


def _write_pkg(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.fixture
def libroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.syspath_prepend(str(tmp_path))
    before = set(sys.modules)
    yield tmp_path
    for name in set(sys.modules) - before:
        del sys.modules[name]
    importlib.invalidate_caches()


# --------------------------------------------------------------------------- exports (P1)


def test_collect_exports_tolerates_lazy_attribute_errors(libroot: Path) -> None:
    _write_pkg(
        libroot,
        {
            "pgx_exp/__init__.py": "from . import lazy, vendored\nfrom .vendored import six\n\nclass Client: ...\n",
            "pgx_exp/lazy.py": """\
                class Thing: ...

                def __dir__():
                    return ["Thing", "gnu"]

                def __getattr__(name):
                    raise ModuleNotFoundError("No module named '_gdbm'")
            """,
            "pgx_exp/vendored/__init__.py": "",
            "pgx_exp/vendored/six.py": """\
                class Vendored: ...

                def __getattr__(name):
                    raise RuntimeError("lazy six.moves attribute")
            """,
        },
    )
    importlib.import_module("pgx_exp")
    assert _collect_exports("pgx_exp") == ["Client", "lazy.Thing"]


def test_main_keeps_targets_when_exports_fail(libroot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_pkg(libroot, {"pgx_main/__init__.py": "def send(x: int) -> int:\n    return x\n"})

    def boom(_top: str) -> list[str]:
        raise ModuleNotFoundError("No module named '_gdbm'")

    monkeypatch.setattr(introspect, "_collect_exports", boom)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"top": "pgx_main", "targets": {"s": "pgx_main:send"}})))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    introspect.main()
    result = json.loads(out.getvalue())
    assert result["exports"] == []
    assert "_gdbm" in result["exports_error"]
    assert result["targets"]["s"]["params"] == [["x", "POSITIONAL_OR_KEYWORD", None, ["int"]]]


# --------------------------------------------------------------------------- unimportable vs missing (L2)


def test_describe_separates_unimportable_from_missing(libroot: Path) -> None:
    _write_pkg(
        libroot,
        {
            "pgx_imp/__init__.py": "class Client:\n    def send(self) -> None: ...\n",
            "pgx_imp/needs_dep.py": "import pgx_not_installed_dependency\n\ndef f() -> None: ...\n",
            "pgx_imp/broken.py": "import collections\ncollections.Mapping\n",
        },
    )
    importlib.import_module("pgx_imp")
    unimportable = describe("pgx_imp", "pgx_imp.needs_dep:f")
    assert "error" not in unimportable
    assert "pgx_not_installed_dependency" in unimportable["unimportable"]
    assert "AttributeError" in describe("pgx_imp", "pgx_imp.broken:f")["unimportable"]
    # A library submodule or attribute that does not exist in this version stays "target not found".
    assert describe("pgx_imp", "pgx_imp.cluster:Cluster.send")["error"].startswith("target not found")
    assert describe("pgx_imp", "pgx_imp.sub.deeper:f")["error"].startswith("target not found")
    assert describe("pgx_imp", "pgx_imp:Client.fetchmany")["error"].startswith("target not found")


# --------------------------------------------------------------------------- forward refs and owner_defined


def test_describe_resolves_runtime_forward_refs_and_falls_back_on_type_checking_only(libroot: Path) -> None:
    _write_pkg(
        libroot,
        {
            "pgx_tc/__init__.py": "from .engine import Engine, URL\nfrom .create import create_engine, plain_strings\n",
            "pgx_tc/engine.py": "class Engine: ...\n\nclass URL: ...\n",
            "pgx_tc/create.py": """\
                from __future__ import annotations

                from typing import TYPE_CHECKING, Union
                from .engine import URL

                if TYPE_CHECKING:
                    from .engine import Engine

                def create_engine(url: Union[str, URL]) -> Engine: ...

                def plain_strings(url: "URL") -> "URL": ...
            """,
        },
    )
    importlib.import_module("pgx_tc")
    described = describe("pgx_tc", "pgx_tc.create:create_engine")
    assert described["return"] == ["Any"]
    assert described["params"][0][3] == ["str", "pgx_tc.URL"]
    assert len(described["notes"]) == 1
    plain = describe("pgx_tc", "pgx_tc.create:plain_strings")
    assert plain["return"] == ["pgx_tc.URL"]
    assert plain["params"][0][3] == ["pgx_tc.URL"]
    assert plain["notes"] == []


def test_resolve_annotation_evaluates_forward_refs() -> None:
    assert _resolve_annotation(typing.ForwardRef("int"), None, "x", {}) is int
    assert _resolve_annotation(typing.ForwardRef("Missing"), None, "x", {}) == "Missing"


def test_describe_records_owner_defined(libroot: Path) -> None:
    _write_pkg(
        libroot,
        {
            "pgx_own/__init__.py": "from .client.base import Client\n",
            "pgx_own/client/__init__.py": "",
            "pgx_own/client/base.py": """\
                from typing import Any, Optional

                class Client:
                    def get(self, key: str, default: Any = None, other: Optional[Any] = None) -> Any: ...

                class _Hidden:
                    def send(self, x: int) -> int: ...

                def func(x: int) -> int: ...
            """,
        },
    )
    importlib.import_module("pgx_own")
    method = describe("pgx_own", "pgx_own.client.base:Client.get")
    assert method["owner"] == "pgx_own.Client"
    assert method["owner_defined"] == "pgx_own.client.base.Client"
    assert "library_any" not in method

    hidden = describe("pgx_own", "pgx_own.client.base:_Hidden.send")
    assert hidden["owner"] is None
    assert hidden["owner_defined"] == "pgx_own.client.base._Hidden"

    func = describe("pgx_own", "pgx_own.client.base:func")
    assert "owner_defined" not in func
