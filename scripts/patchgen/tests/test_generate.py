# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Hermetic tests for patchgen: synthetic specs and locks, no network."""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import cli
import patchgen.generate as generate_mod
import pytest
import yaml
from packaging.version import Version
from patchgen import lock as lock_mod
from patchgen.generate import (
    GenerateError,
    any_report,
    any_report_package,
    build_patch,
    generate,
    input_hash,
    render,
)
from patchgen.lock import LockError, Release, build_lock, select_versions
from patchgen.lockfile import dump_lock
from patchgen.spec import Policy, SpecError, load_spec

P = "POSITIONAL_OR_KEYWORD"
K = "KEYWORD_ONLY"


def _entry(params: list[list[Any]], ret: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "binding": "method",
        "flavor": "sync",
        "owner": "fakelib.Client",
        "params": [["self", P, None, None], *params],
        "return": ret if ret is not None else ["fakelib.Response"],
        "notes": [],
    }
    entry.update(extra)
    return entry


def _lock(versions: dict[str, dict[str, Any]], exports: list[str] | None = None) -> dict[str, Any]:
    names = exports if exports is not None else ["Client", "Request", "Response"]
    return {
        "library": "fakelib",
        "generated_with": "patchgen 0.1",
        "python": "3.12",
        "specifier": ">=1",
        "versions": versions,
        "exports": {v: names for v in versions},
    }


SEND_V1 = [["request", P, None, ["fakelib.Request"]], ["timeout", P, "None", ["float", "None"]]]
SEND_V2 = [*SEND_V1, ["retries", K, "0", ["int"]]]


def _package(tmp_path: Path, spec: str, lock: dict[str, Any] | None = None, name: str = "patches") -> Path:
    pkg = tmp_path / "opentelemetry-instrumentation-fakelib"
    (pkg / "src" / "opentelemetry" / "instrumentation" / "fakelib").mkdir(parents=True, exist_ok=True)
    (pkg / f"{name}.yaml").write_text(textwrap.dedent(spec), encoding="utf-8")
    if lock is not None:
        (pkg / f"{name}.lock.json").write_text(dump_lock(lock), encoding="utf-8")
    return pkg


SPEC = """\
library: fakelib
patches:
  - name: client_send
    target: fakelib:Client.send
    scope: [class, instance]
"""


def _build(tmp_path: Path, spec: str, lock: dict[str, Any]) -> Any:
    pkg = _package(tmp_path, spec, lock)
    loaded = load_spec(pkg / "patches.yaml")
    return build_patch(loaded, loaded.patches[0], lock)


def _source(tmp_path: Path, spec: str, lock: dict[str, Any]) -> str:
    pkg = _package(tmp_path, spec, lock)
    return generate(load_spec(pkg / "patches.yaml"))[1]


# --------------------------------------------------------------------------- spec


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("library: x\npatches: []\n", "non-empty"),
        ("library: x\nbogus: 1\npatches: [{name: a, target: 'x:f'}]\n", "unknown keys"),
        ("library: x\npatches: [{name: A, target: 'x:f'}]\n", "snake_case"),
        ("library: x\npatches: [{name: a, target: 'x.f'}]\n", "module:qualname"),
        ("library: x\npatches: [{name: a, target: 'x:f', scope: [global]}]\n", "scope"),
        ("library: x\npatches: [{name: a, target: 'x:f'}, {name: a, target: 'x:g'}]\n", "unique"),
        ("library: x\npatches: [{name: a, target: 'x:f'}, {name: b, target: 'y:g'}]\n", "one top-level"),
        ("library: x\npatches: [{name: a, target: 'x:f', versions: '>>1'}]\n", "invalid versions"),
        (
            "library: x\npatches: [{name: a, target: 'x:f'}, {name: b, target: 'x:f', versions: '>=2'}]\n",
            "sharing target",
        ),
        ("library: x\npolicy: {max: 0}\npatches: [{name: a, target: 'x:f'}]\n", "policy.max"),
        ("library: x\npolicy: {bogus: 1}\npatches: [{name: a, target: 'x:f'}]\n", "policy: unknown keys"),
        ("library: x\npolicy: {include: ['not-a-ver']}\npatches: [{name: a, target: 'x:f'}]\n", "invalid version"),
        ("library: x\nimports: ['123bad']\npatches: [{name: a, target: 'x:f'}]\n", "imports must be"),
    ],
)
def test_spec_validation_errors(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "patches.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(SpecError, match=message):
        load_spec(path)


def test_spec_lock_path_and_module(tmp_path: Path) -> None:
    pkg = _package(
        tmp_path,
        "library: other\nmodule: _patches_other\npatches: [{name: a, target: 'other:f'}]\n",
        name="patches.other",
    )
    spec = load_spec(pkg / "patches.other.yaml")
    assert spec.lock_path.name == "patches.other.lock.json"
    assert spec.module == "_patches_other"


# --------------------------------------------------------------------------- version selection


def _vs(*versions: str) -> list[Version]:
    return [Version(v) for v in versions]


def test_select_versions_policy() -> None:
    available = _vs("1.0.0", "1.0.1", "1.1.0", "1.1.5", "1.2.0", "2.0.0rc1", "2.0.0", "2.0.3")
    assert select_versions(available, ">=1", Policy()) == _vs("1.0.0", "1.1.0", "1.2.0", "2.0.0", "2.0.3")
    assert select_versions(available, ">=1.1,<2", Policy()) == _vs("1.1.0", "1.2.0")


def test_select_versions_caps_and_keeps_endpoints() -> None:
    available = _vs(*(f"1.{minor}.0" for minor in range(20)))
    chosen = select_versions(available, ">=1", Policy(max=5))
    assert len(chosen) == 5
    assert chosen[0] == Version("1.0.0")
    assert chosen[-1] == Version("1.19.0")


def test_select_versions_pins_major_newest_patch_ranges_and_include() -> None:
    available = _vs(
        "1.0.0",
        "1.1.0",
        "1.2.0",
        "1.2.9",
        "1.3.0",
        "1.3.5",
        "2.0.0",
        "2.1.0",
        "2.1.4",
        "2.2.0",
    )
    chosen = select_versions(
        available,
        ">=1",
        Policy(max=5, include=("1.2.9",)),
        patch_ranges=(">=1.2.0", ">=2.1.0"),
    )
    # Pinned: oldest (1.0.0), newest of major 1 (1.3.5), newest/major 2 (2.2.0),
    # lowest of each patch_range (1.2.0, 2.1.0), and policy.include (1.2.9).
    for expected in ("1.0.0", "1.2.0", "1.2.9", "1.3.5", "2.1.0", "2.2.0"):
        assert Version(expected) in chosen

    with pytest.raises(LockError, match="policy.include"):
        select_versions(available, ">=1", Policy(include=("9.9.9",)))


def test_build_lock_skips_failing_versions_and_warns_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _package(tmp_path, "library: fakelib\nversions: '>=1.0'\n" + SPEC.split("\n", 1)[1])
    spec = load_spec(pkg / "patches.yaml")
    monkeypatch.setattr(
        lock_mod,
        "fetch_releases",
        lambda _lib: [Release(Version(v), (None,)) for v in ("1.0.0", "1.1.0", "2.0.0")],
    )

    def fake_introspect(_spec: Any, version: Version, _python: str) -> dict[str, Any]:
        if version == Version("1.0.0"):
            raise LockError("fakelib==1.0.0: cannot import fakelib")
        return {"targets": {"client_send": _entry(SEND_V1)}, "exports": ["Client", "Request", "Response"]}

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    logs: list[str] = []
    locked = build_lock(spec, python="3.12", log=logs.append)
    assert list(locked["versions"]) == ["1.1.0", "2.0.0"]
    assert any("skipping fakelib==1.0.0" in line for line in logs)
    assert any("skipped 1 version(s): 1.0.0" in line for line in logs)
    assert any('versions: ">=1.1.0"' in line for line in logs)

    monkeypatch.setattr(
        lock_mod,
        "introspect_version",
        lambda _s, v, _p: (_ for _ in ()).throw(LockError(f"boom {v}")),
    )
    with pytest.raises(LockError, match="all selected versions of fakelib failed"):
        build_lock(spec, python="3.12", log=logs.append)


# --------------------------------------------------------------------------- union


def test_union_marks_new_params_missing_with_presence_comment(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V2)}})
    model = _build(tmp_path, SPEC, lock)
    fields = {f.name: f for f in model.fields}
    assert [f.name for f in model.fields] == ["request", "timeout", "retries"]
    assert fields["request"].annotation == "fakelib.Request"
    assert fields["timeout"].annotation == "float | None"
    assert fields["retries"].annotation == "int | Missing"
    assert fields["retries"].comment == "fakelib >= 2.0"
    assert model.instance_type == "fakelib.Client"
    assert model.result_type == "fakelib.Response"


def test_union_removed_param_and_kind_change(tmp_path: Path) -> None:
    v1 = [["request", P, None, ["str"]], ["legacy", P, "None", None]]
    v2 = [["request", K, None, ["bytes"]]]
    lock = _lock({"1.0": {"client_send": _entry(v1)}, "2.0": {"client_send": _entry(v2)}})
    model = _build(tmp_path, SPEC, lock)
    fields = {f.name: f for f in model.fields}
    assert fields["request"].annotation == "bytes | str"
    assert fields["legacy"].annotation == "Any | Missing"
    assert fields["legacy"].comment == "fakelib <= 1.0"
    assert len(model.shapes) == 2


def test_unannotated_old_version_uses_annotated_type_with_note(tmp_path: Path) -> None:
    v1 = [["request", P, None, None], ["timeout", P, "None", None]]
    v2 = SEND_V1
    lock = _lock({"1.0": {"client_send": _entry(v1, ret=None)}, "2.0": {"client_send": _entry(v2)}})
    lock["versions"]["1.0"]["client_send"]["return"] = None
    model = _build(tmp_path, SPEC, lock)
    fields = {f.name: f for f in model.fields}
    assert fields["request"].annotation == "fakelib.Request"
    assert fields["timeout"].annotation == "float | None"
    assert model.result_type == "fakelib.Response"
    assert "client_send.request: unannotated in some locked versions; using fakelib.Request" in model.notes
    assert "client_send return: unannotated in some locked versions; using fakelib.Response" in model.notes


def test_var_args_and_kwargs_become_generic_fields(tmp_path: Path) -> None:
    params = [["args", "VAR_POSITIONAL", None, None], ["kw", "VAR_KEYWORD", None, None]]
    model = _build(tmp_path, SPEC, _lock({"1.0": {"client_send": _entry(params)}}))
    fields = {f.name: f for f in model.fields}
    assert fields["var_args"].annotation == "tuple[Any, ...]"
    assert fields["var_kwargs"].annotation == "dict[str, Any]"


def test_reserved_parameter_name_is_rejected(tmp_path: Path) -> None:
    for reserved in ("raw_args", "_sig", "_wrapped", "call"):
        lock = _lock({"1.0": {"client_send": _entry([[reserved, P, None, None]])}})
        with pytest.raises(GenerateError, match="collides"):
            _build(tmp_path, SPEC, lock)


def test_library_type_missing_from_some_version_becomes_any(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V1)}})
    lock["exports"]["1.0"] = ["Client", "Response"]
    model = _build(tmp_path, SPEC, lock)
    assert {f.name: f for f in model.fields}["request"].annotation == "Any"
    assert any("fakelib.Request not in every locked version" in n for n in model.notes)


def test_types_override_and_unknown_override(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}})
    spec = SPEC + "    types: {timeout: 'float', return: 'object'}\n"
    model = _build(tmp_path, spec, lock)
    assert {f.name: f for f in model.fields}["timeout"].annotation == "float"
    assert model.result_type == "object"
    with pytest.raises(GenerateError, match="unknown parameters"):
        _build(tmp_path, SPEC + "    types: {nope: int}\n", lock)


def test_notes_deduped_and_dropped_for_overridden_types(tmp_path: Path) -> None:
    req_note = "parameter 'request': <class 'custom.Req'> rendered as Any"
    timeout_note = "parameter 'timeout': <class 'custom.Timeout'> rendered as Any"
    ret_note = "return: <class 'custom.Resp'> rendered as Any"
    lock = _lock(
        {
            "1.0": {"client_send": _entry(SEND_V1, notes=[req_note, ret_note])},
            "1.1": {"client_send": _entry(SEND_V1, notes=[req_note, timeout_note, ret_note])},
            "2.0": {"client_send": _entry(SEND_V1, notes=[req_note, timeout_note, ret_note])},
        }
    )
    model = _build(tmp_path, SPEC, lock)
    assert model.notes == (
        req_note,
        ret_note,
        f"fakelib >= 1.1: {timeout_note}",
    )

    overridden = SPEC + "    types: {request: 'fakelib.Request', return: 'fakelib.Response'}\n"
    model_overridden = _build(tmp_path, overridden, lock)
    assert model_overridden.notes == (f"fakelib >= 1.1: {timeout_note}",)


def test_async_detection(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1, flavor="async")}})
    assert _build(tmp_path, SPEC, lock).is_async
    mixed = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V1, flavor="async")}})
    with pytest.raises(GenerateError, match="async"):
        _build(tmp_path, SPEC, mixed)


def test_absent_versions_are_skipped_but_not_all(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": {"absent": True}}, "2.0": {"client_send": _entry(SEND_V1)}})
    assert _build(tmp_path, SPEC, lock).fields
    gone = _lock({"1.0": {"client_send": {"absent": True}}})
    with pytest.raises(GenerateError, match="absent in every"):
        _build(tmp_path, SPEC, gone)


# --------------------------------------------------------------------------- version ranges

RANGED = """\
library: fakelib
patches:
  - name: send_legacy
    target: fakelib:Client.send
    versions: "<2"
  - name: send
    target: fakelib:Client.send
    versions: ">=2"
"""


def _ranged_lock() -> dict[str, Any]:
    legacy = [["method", P, None, ["bytes"]]]
    return _lock(
        {
            "1.0": {"send_legacy": _entry(legacy), "send": _entry(legacy)},
            "2.0": {"send_legacy": _entry(SEND_V2), "send": _entry(SEND_V2)},
        }
    )


def test_ranges_only_use_in_range_versions(tmp_path: Path) -> None:
    pkg = _package(tmp_path, RANGED, _ranged_lock())
    spec = load_spec(pkg / "patches.yaml")
    legacy = build_patch(spec, spec.patches[0], _ranged_lock())
    modern = build_patch(spec, spec.patches[1], _ranged_lock())
    assert [f.name for f in legacy.fields] == ["method"]
    assert [f.name for f in modern.fields] == ["request", "timeout", "retries"]
    assert all(not f.missing for f in modern.fields)
    source = generate(spec)[1]
    assert 'versions="<2"' in source
    assert 'versions=">=2"' in source


def test_overlapping_ranges_are_rejected(tmp_path: Path) -> None:
    pkg = _package(tmp_path, RANGED.replace('"<2"', '"<=2"'), _ranged_lock())
    with pytest.raises((SpecError, GenerateError), match="overlapping"):
        generate(load_spec(pkg / "patches.yaml"))


# --------------------------------------------------------------------------- rendering


def test_render_is_deterministic_and_independent_of_lock_order(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V2)}})
    pkg = _package(tmp_path, SPEC, lock)
    spec = load_spec(pkg / "patches.yaml")
    spec_text = (pkg / "patches.yaml").read_text(encoding="utf-8")
    reordered = dict(reversed(list(lock["versions"].items())))
    first = render(spec, spec_text, dump_lock(lock))
    second = render(spec, spec_text, dump_lock(lock))
    third = render(spec, spec_text, dump_lock({**lock, "versions": reordered}))

    def strip(src: str) -> str:
        return "\n".join(line for line in src.splitlines() if not line.startswith("# Input hash"))

    assert first == second
    assert strip(first) == strip(third)
    assert "# Generated by patchgen" in first
    assert "DO NOT EDIT" in first


def test_input_hash_ignores_comments_and_whitespace() -> None:
    lock_text = "{}"
    with_comments = "# top comment\n\n" + SPEC + "  # trailing comment\n"
    assert input_hash(SPEC, lock_text) == input_hash(with_comments, lock_text)


def test_render_rejects_wrong_library_and_missing_entries(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}})
    pkg = _package(tmp_path, SPEC, lock)
    spec = load_spec(pkg / "patches.yaml")
    with pytest.raises(GenerateError, match="is for"):
        render(spec, "", dump_lock({**lock, "library": "other"}))
    with pytest.raises(GenerateError, match="no entries"):
        render(spec, "", dump_lock(_lock({"1.0": {"other": _entry(SEND_V1)}})))


# --------------------------------------------------------------------------- end to end


class _Request:
    def __init__(self, url: str) -> None:
        self.url = url


class _Response:
    def __init__(self, request: _Request, timeout: float | None, retries: int) -> None:
        self.request = request
        self.timeout = timeout
        self.retries = retries


class _Client:
    def send(self, request: _Request, timeout: float | None = None, *, retries: int = 0) -> _Response:
        return _Response(request, timeout, retries)

    async def send_async(self, request: _Request, timeout: float | None = None, *, retries: int = 0) -> _Response:
        return _Response(request, timeout, retries)

    @classmethod
    def from_url(cls, url: str, *, retries: int = 0) -> _Client:
        inst = cls()
        setattr(inst, "url", f"{cls.__name__}:{url}:{retries}")
        return inst

    @staticmethod
    def normalize(url: str) -> str:
        return url.strip()


class _SubClient(_Client):
    pass


_Client.__name__ = "Client"
_Client.__qualname__ = "Client"
_Client.__module__ = "fakelib"
_SubClient.__name__ = "SubClient"
_SubClient.__qualname__ = "SubClient"
_SubClient.__module__ = "fakelib"


@pytest.fixture(name="fakelib")
def _fakelib() -> Any:
    module = types.ModuleType("fakelib")
    for name, value in (
        ("Client", _Client),
        ("SubClient", _SubClient),
        ("Request", _Request),
        ("Response", _Response),
    ):
        setattr(module, name, value)
    sys.modules["fakelib"] = module
    original = vars(_Client)["send"]
    original_async = vars(_Client)["send_async"]
    original_from_url = vars(_Client)["from_url"]
    original_normalize = vars(_Client)["normalize"]
    yield module
    setattr(_Client, "send", original)
    setattr(_Client, "send_async", original_async)
    setattr(_Client, "from_url", original_from_url)
    setattr(_Client, "normalize", original_normalize)
    del sys.modules["fakelib"]
    sys.modules.pop("fakelib_patches", None)


def _load_generated(source: str) -> Any:
    module = types.ModuleType("fakelib_patches")
    sys.modules[module.__name__] = module
    exec(compile(source, "fakelib_patches.py", "exec"), module.__dict__)
    return module


def test_generated_module_binds_args_and_round_trips(tmp_path: Path, fakelib: Any) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V2)}})
    patches = _load_generated(_source(tmp_path, SPEC, lock))
    seen: list[Any] = []

    def wrapper(wrapped: Any, instance: Any, args: Any) -> Any:
        seen.append((instance, args))
        return wrapped(*args.raw_args, **args.raw_kwargs)

    original = fakelib.Client.send
    patches.instrument(client_send=wrapper)
    try:
        client = fakelib.Client()
        request = fakelib.Request("http://x")
        response = client.send(request, retries=3)
        assert response.retries == 3
        instance, args = seen[-1]
        assert instance is client
        assert args.request is request
        assert args.timeout is None
        assert args.retries == 3
        assert args.raw_args == (request,)
        assert args.raw_kwargs == {"retries": 3}
    finally:
        patches.uninstrument()
    assert fakelib.Client.send is original

    client = fakelib.Client()
    patches.instrument_instance(client, client_send=wrapper)
    assert fakelib.Client().send(fakelib.Request("y")).retries == 0
    count = len(seen)
    client.send(fakelib.Request("z"), 1.5)
    assert len(seen) == count + 1
    assert seen[-1][1].timeout == 1.5
    patches.uninstrument_instance(client)
    assert "send" not in vars(client)


def test_generated_args_call_forwards_original_arguments(tmp_path: Path, fakelib: Any) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V2)}})
    source = _source(tmp_path, SPEC, lock)
    assert (
        "    if TYPE_CHECKING:\n\n"
        "        def call(\n"
        "            self,\n"
        "            *,\n"
        "            request: fakelib.Request | Missing = MISSING,\n"
        "            timeout: float | None | Missing = MISSING,\n"
        "            retries: int | Missing = MISSING,\n"
        "        ) -> fakelib.Response: ...\n"
    ) in source
    assert "_patching.bind_wrapper(" in source
    patches = _load_generated(source)
    seen: list[Any] = []

    def wrapper(wrapped: Any, instance: Any, args: Any) -> Any:
        seen.append(args)
        return args.call()

    patches.instrument(client_send=wrapper)
    try:
        request = fakelib.Request("http://x")
        response = fakelib.Client().send(request, 2.5, retries=3)
        assert (response.request, response.timeout, response.retries) == (request, 2.5, 3)
        assert seen[-1].raw_args == (request, 2.5)
        assert seen[-1].raw_kwargs == {"retries": 3}
    finally:
        patches.uninstrument()

    client = fakelib.Client()
    patches.instrument_instance(client, client_send=wrapper)
    try:
        response = client.send(request=fakelib.Request("y"), timeout=1.0)
        assert (response.request.url, response.timeout, response.retries) == ("y", 1.0, 0)
        assert len(seen) == 2
    finally:
        patches.uninstrument_instance(client)


def test_generated_args_call_overrides_parameters(tmp_path: Path, fakelib: Any) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V2)}})
    patches = _load_generated(_source(tmp_path, SPEC, lock))
    rewritten = fakelib.Request("http://rewritten")
    seen: list[Any] = []

    def wrapper(wrapped: Any, instance: Any, args: Any) -> Any:
        seen.append(args)
        return args.call(request=rewritten, retries=7)

    patches.instrument(client_send=wrapper)
    try:
        client = fakelib.Client()
        # Passed positionally: replaced in place; untouched arguments are kept.
        original = fakelib.Request("http://orig")
        res1 = client.send(original, 1.0)
        assert (res1.request, res1.timeout, res1.retries) == (rewritten, 1.0, 7)
        assert seen[-1].raw_args == (original, 1.0)
        # Passed by keyword:
        res2 = client.send(request=fakelib.Request("http://orig2"), timeout=2.0, retries=1)
        assert (res2.request, res2.timeout, res2.retries) == (rewritten, 2.0, 7)
    finally:
        patches.uninstrument()


ASYNC_SPEC = """\
library: fakelib
patches:
  - name: client_send_async
    target: fakelib:Client.send_async
"""


def test_generated_args_call_async_target(tmp_path: Path, fakelib: Any) -> None:
    lock = _lock(
        {
            "1.0": {"client_send_async": _entry(SEND_V1, flavor="async")},
            "2.0": {"client_send_async": _entry(SEND_V2, flavor="async")},
        }
    )
    source = _source(tmp_path, ASYNC_SPEC, lock)
    assert "        ) -> Awaitable[fakelib.Response]: ...\n" in source
    patches = _load_generated(source)
    overrides: dict[str, Any] = {}

    def wrapper(wrapped: Any, instance: Any, args: Any) -> Any:
        return args.call(**overrides)

    patches.instrument(client_send_async=wrapper)
    try:
        request = fakelib.Request("http://x")
        response = asyncio.run(fakelib.Client().send_async(request, retries=2))
        assert (response.request, response.timeout, response.retries) == (request, None, 2)
        overrides.update(timeout=4.0)
        response = asyncio.run(fakelib.Client().send_async(request, 1.0, retries=2))
        assert (response.request, response.timeout, response.retries) == (request, 4.0, 2)
    finally:
        patches.uninstrument()


def test_generated_args_call_without_parameters_and_shadowing_names(tmp_path: Path) -> None:
    no_params = _source(tmp_path, SPEC, _lock({"1.0": {"client_send": _entry([])}}))
    assert "        def call(self) -> fakelib.Response: ...\n" in no_params

    params = [
        ["updates", P, None, ["int"]],
        ["args", P, None, ["int"]],
        ["kwargs", P, "0", ["int"]],
        ["key", P, "0", ["int"]],
        ["value", P, "0", ["int"]],
    ]
    source = _source(tmp_path / "shadow", SPEC, _lock({"1.0": {"client_send": _entry(params)}}))
    module = _load_generated(source)

    def target(updates: int, args: int, kwargs: int = 0, key: int = 0, value: int = 0) -> tuple[int, ...]:
        return (updates, args, kwargs, key, value)

    parsed = module.ClientSendArgs(
        raw_args=(1, 2),
        raw_kwargs={},
        _wrapped=target,
    )
    assert parsed.call() == (1, 2, 0, 0, 0)
    assert parsed.call(args=5, kwargs=6, value=7) == (1, 5, 6, 0, 7)
    assert parsed.raw_args == (1, 2)


def test_generated_args_call_builtin_names_and_long_types(tmp_path: Path) -> None:
    long_types = ["dict[str, tuple[int, ...]]", "list[tuple[str, tuple[int, ...]]]", "frozenset[tuple[str, int]]"]
    params = [["input", P, None, ["int"]], ["mapping", P, "None", [*long_types, "None"]]]
    pkg = _package(tmp_path, SPEC, _lock({"1.0": {"client_send": _entry(params)}}))
    path, source = generate(load_spec(pkg / "patches.yaml"))
    # The keyword must match the target's parameter, so a builtin name is kept and silenced.
    assert "\n# ruff: noqa: A002\n" in source
    assert "            input: int | Missing = MISSING,\n" in source
    # Formatted by the pinned ruff: the over-long parameter is wrapped, and formatting again
    # (which also re-runs the lint gate) changes nothing.
    one_line = f"            mapping: {' | '.join(long_types)} | None | Missing = MISSING,"
    assert len(one_line) > 120
    assert one_line not in source
    assert "            mapping: " in source
    assert generate_mod.format_source(source, path) == source
    module = _load_generated(source)

    def target(input: int, mapping: Any = None) -> tuple[int, Any]:  # noqa: A002
        return (input, mapping)

    parsed = module.ClientSendArgs(
        raw_args=(1,),
        raw_kwargs={},
        _wrapped=target,
    )
    assert parsed.call(input=2, mapping={"a": (1,)}) == (2, {"a": (1,)})


def test_generate_only_exempts_builtin_names_when_needed(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V2)}})
    assert "ruff: noqa" not in _source(tmp_path, SPEC, lock)


def test_generate_fails_on_lint_violations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _package(tmp_path, SPEC, _lock({"1.0": {"client_send": _entry(SEND_V1)}}))
    monkeypatch.setattr(generate_mod, "render", lambda *_: "import os\n")
    with pytest.raises(GenerateError, match="F401"):
        generate(load_spec(pkg / "patches.yaml"))


def test_generate_requires_pinned_ruff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _package(tmp_path, SPEC, _lock({"1.0": {"client_send": _entry(SEND_V1)}}))
    monkeypatch.setattr(generate_mod, "RUFF_VERSION", "0.0.0")
    generate_mod._require_ruff.cache_clear()
    try:
        with pytest.raises(GenerateError, match=r"needs ruff==0\.0\.0"):
            generate(load_spec(pkg / "patches.yaml"))
    finally:
        generate_mod._require_ruff.cache_clear()


def test_ruff_pin_matches_pre_commit_and_cli() -> None:
    repo = Path(cli.__file__).resolve().parents[2]
    pre_commit = yaml.safe_load((repo / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    revs = [r["rev"] for r in pre_commit["repos"] if r["repo"].endswith("/ruff-pre-commit")]
    assert revs == [f"v{generate_mod.RUFF_VERSION}"]
    assert f'"ruff=={generate_mod.RUFF_VERSION}"' in Path(cli.__file__).read_text(encoding="utf-8")
    assert generate_mod.RUFF_CONFIG == repo / "pyproject.toml"


def test_generated_module_falls_back_on_bind_failure(tmp_path: Path, fakelib: Any) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}})
    patches = _load_generated(_source(tmp_path, SPEC, lock))
    calls: list[Any] = []
    rewritten = fakelib.Request("http://rewritten")

    def wrapper(wrapped: Any, instance: Any, args: Any) -> Any:
        calls.append(args)
        return args.call(request=rewritten)

    patches.instrument(client_send=wrapper)
    try:
        req = fakelib.Request("http://orig")
        for bad_args, bad_kwargs in (
            ((), {}),  # missing required 'request'
            ((req,), {"request": req}),  # duplicate positional + keyword argument
            ((req, 1.0, "extra"), {}),  # excess positional argument
            ((req,), {"unexpected_kw": 1}),  # unexpected keyword argument
        ):
            with pytest.raises(TypeError):
                fakelib.Client().send(*bad_args, **bad_kwargs)
            assert not calls
            assert patches._REGISTRY.active
        res_ok = fakelib.Client().send(req)
        assert len(calls) == 1
        assert res_ok.request is rewritten
    finally:
        patches.uninstrument()

    # 1. Unlocked release that reordered parameters: (self, timeout=None, request=None).
    # Binding against the live signature binds by parameter name, so positional timeout=2.5
    # and keyword request=req still rewrite request cleanly without clobbering timeout.
    def reordered_send(self: Any, timeout: float | None = None, request: Any = None) -> Any:
        return fakelib.Response(request, timeout, 0)

    calls.clear()
    fakelib.Client.send = reordered_send
    assert patches.instrument(client_send=wrapper)
    try:
        req = fakelib.Request("http://orig")
        res = fakelib.Client().send(2.5, request=req)
        assert len(calls) == 1
        assert res.request is rewritten
        assert res.timeout == 2.5
        assert patches._REGISTRY.active
    finally:
        patches.uninstrument()

    # 2. Unlocked release that renamed 'request' -> 'req': instrument() warns and applies,
    # and the first call fails open, runs the original unmodified, and self-removes the patch.
    def renamed_send(self: Any, timeout: float | None = None, req: Any = None) -> Any:
        return fakelib.Response(req, timeout, 0)

    calls.clear()
    fakelib.Client.send = renamed_send
    assert patches.instrument(client_send=wrapper)
    try:
        req = fakelib.Request("http://orig")
        res = fakelib.Client().send(2.5, req=req)
        assert not calls
        assert res.request is req
        assert res.timeout == 2.5
        assert not patches._REGISTRY.active
    finally:
        patches.uninstrument()


# --------------------------------------------------------------------------- stdlib mode

STDLIB_SPEC = "library: fakelib\nstdlib: true\n" + SPEC.split("\n", 1)[1]
_THIS_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


def _stdlib_lock(versions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lock = _lock(versions)
    del lock["python"]
    lock["stdlib"] = True
    return lock


def test_stdlib_render_names_python_and_marks_registry(tmp_path: Path) -> None:
    lock = _stdlib_lock({"3.10": {"client_send": _entry(SEND_V1)}, "3.12": {"client_send": _entry(SEND_V2)}})
    model = _build(tmp_path, STDLIB_SPEC, lock)
    assert {f.name: f for f in model.fields}["retries"].comment == "python >= 3.12"
    source = _source(tmp_path, STDLIB_SPEC, lock)
    assert "Locked python versions: 3.10, 3.12." in source
    assert "_REGISTRY = _patching.Registry(_LIBRARY, stdlib=True)\n" in source
    assert "# python >= 3.12\n" in source


def test_library_render_is_unchanged_by_stdlib_support(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V2)}})
    source = _source(tmp_path, SPEC, lock)
    assert "Locked fakelib versions: 1.0, 2.0." in source
    assert "_REGISTRY = _patching.Registry(_LIBRARY)\n" in source
    assert "stdlib" not in source


def test_render_rejects_lock_with_other_stdlib_mode(tmp_path: Path) -> None:
    library_lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}})
    pkg = _package(tmp_path, STDLIB_SPEC, library_lock)
    with pytest.raises(GenerateError, match="`stdlib:` setting; rerun lock"):
        generate(load_spec(pkg / "patches.yaml"))

    stdlib_lock = _stdlib_lock({"3.12": {"client_send": _entry(SEND_V1)}})
    pkg = _package(tmp_path / "other", SPEC, stdlib_lock)
    with pytest.raises(GenerateError, match="`stdlib:` setting; rerun lock"):
        generate(load_spec(pkg / "patches.yaml"))


def test_stdlib_module_resolves_versions_against_running_python(
    tmp_path: Path, fakelib: Any, caplog: pytest.LogCaptureFixture
) -> None:
    spec = textwrap.dedent(
        """\
        library: fakelib
        stdlib: true
        patches:
          - name: send_legacy
            target: fakelib:Client.send
            versions: "<3.0"
          - name: send
            target: fakelib:Client.send
            versions: ">=3.0"
        """
    )
    lock = _stdlib_lock(
        {
            "2.7": {"send_legacy": _entry(SEND_V1), "send": _entry(SEND_V1)},
            _THIS_PYTHON: {"send_legacy": _entry(SEND_V2), "send": _entry(SEND_V2)},
        }
    )
    patches = _load_generated(_source(tmp_path, spec, lock))
    legacy_calls: list[Any] = []
    modern_calls: list[Any] = []

    def legacy(wrapped: Any, instance: Any, args: Any) -> Any:
        legacy_calls.append(args)
        return wrapped(*args.raw_args, **args.raw_kwargs)

    def modern(wrapped: Any, instance: Any, args: Any) -> Any:
        modern_calls.append(args)
        return wrapped(*args.raw_args, **args.raw_kwargs)

    with caplog.at_level("WARNING", logger="opentelemetry.instrumentation._patching"):
        patches.instrument(send_legacy=legacy, send=modern)
        try:
            fakelib.Client().send(fakelib.Request("x"))
        finally:
            patches.uninstrument()
    # The running Python is in the lock, so no "not in the lock" warning, and the
    # ">=3.0" alternative was chosen even though "fakelib" is not an installed distribution.
    assert not caplog.records
    assert legacy_calls == []
    assert len(modern_calls) == 1


# --------------------------------------------------------------------------- cli


def test_cli_generate_and_check_detects_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}})
    pkg = _package(tmp_path, SPEC, lock)
    assert cli.main(["generate"]) == 1
    assert cli.main(["check", str(pkg)]) == 1
    assert cli.main(["generate", str(pkg)]) == 0
    out = pkg / "src" / "opentelemetry" / "instrumentation" / "fakelib" / "_patches.py"
    assert out.is_file()
    assert cli.main(["check", str(pkg)]) == 0
    # Comment-only edit to patches.yaml does not make generated output stale:
    spec_path = pkg / "patches.yaml"
    spec_path.write_text("# a comment\n" + spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    assert cli.main(["check", str(pkg)]) == 0
    out.write_text(out.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    capsys.readouterr()
    assert cli.main(["check", str(pkg)]) == 1
    assert "STALE" in capsys.readouterr().out
    monkeypatch.setattr(cli, "_all_packages", lambda: [pkg])
    assert cli.main(["generate", "--all"]) == 0
    assert cli.main(["check", "--all"]) == 0

    bad_pkg = tmp_path / "bad_pkg"
    bad_pkg.mkdir()
    (bad_pkg / "patches.yaml").write_text(SPEC, encoding="utf-8")
    monkeypatch.setattr(cli, "_all_packages", lambda: [pkg, bad_pkg])
    capsys.readouterr()
    assert cli.main(["generate", "--all"]) == 1
    assert "FAILED" in capsys.readouterr().err


# --------------------------------------------------------------------------- typed _wrapped


def test_args_wrapped_is_typed_like_the_protocol(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}})
    sync = _source(tmp_path / "sync", SPEC, lock)
    assert "    _wrapped: Callable[..., fakelib.Response]\n" in sync
    assert "        wrapped: Callable[..., fakelib.Response],\n" in sync
    assert "_wrapped: Any" not in sync

    async_lock = _lock({"1.0": {"client_send_async": _entry(SEND_V1, flavor="async")}})
    async_source = _source(tmp_path / "async", ASYNC_SPEC, async_lock)
    assert "    _wrapped: Callable[..., Awaitable[fakelib.Response]]\n" in async_source
    assert "        wrapped: Callable[..., Awaitable[fakelib.Response]],\n" in async_source


def test_args_wrapped_is_required_and_sig_is_optional(tmp_path: Path) -> None:
    module = _load_generated(_source(tmp_path, SPEC, _lock({"1.0": {"client_send": _entry(SEND_V1)}})))
    fields: dict[str, Any] = {"raw_args": (), "raw_kwargs": {}}
    with pytest.raises(TypeError, match="_wrapped"):
        module.ClientSendArgs(**fields)

    def target(request: Any, timeout: Any = None) -> tuple[Any, Any]:
        return (request, timeout)

    parsed = module.ClientSendArgs(raw_args=(1,), raw_kwargs={}, _wrapped=target)
    assert parsed._sig is None
    assert parsed.call() == (1, None)
    assert "_wrapped" not in repr(parsed)


# --------------------------------------------------------------------------- Any report

GAPS_PARAMS = [
    ["request", P, None, ["fakelib.Request"]],
    ["timeout", P, "None", None],
    ["mode", P, "None", ["Any"]],
    ["cookies", P, "None", ["Any", "None"]],
    ["extensions", P, "None", ["dict[Any, Any]", "None"]],
    ["hook", P, "None", ["fakelib.Hook"]],
    ["args", "VAR_POSITIONAL", None, None],
    ["kw", "VAR_KEYWORD", None, None],
]
GAPS_NOTE = "parameter 'cookies': <class 'http.cookies.BaseCookie'> rendered as Any"
GAPS_REPORT = [
    "fakelib: patch client_send: parameter 'timeout' is Any (unannotated in all locked versions); "
    "add types: {timeout: ...} to patches.yaml",
    "fakelib: patch client_send: parameter 'cookies' is Any "
    "(<class 'http.cookies.BaseCookie'> rendered as Any); add types: {cookies: ...} to patches.yaml",
    "fakelib: patch client_send: parameter 'extensions' is dict[Any, Any] | None "
    "(dict[Any, Any] has only Any type arguments); add types: {extensions: ...} to patches.yaml",
    "fakelib: patch client_send: parameter 'hook' is Any (fakelib.Hook not in every locked version); "
    "add types: {hook: ...} to patches.yaml",
    "fakelib: patch client_send: return type is Any (unannotated in all locked versions); "
    "add types: {return: ...} to patches.yaml",
]


def _gaps_lock(owner: str = "fakelib.Client") -> dict[str, Any]:
    entry = _entry(GAPS_PARAMS, notes=[GAPS_NOTE, GAPS_NOTE], owner=owner, **{"return": None})
    lock = _lock({"1.0": {"client_send": entry}, "2.0": {"client_send": entry}})
    lock["exports"]["2.0"] = [*lock["exports"]["2.0"], "Hook"]
    return lock


def test_any_report_lists_each_any_with_reason_and_remedy(tmp_path: Path) -> None:
    pkg = _package(tmp_path, SPEC, _gaps_lock())
    assert any_report(load_spec(pkg / "patches.yaml")) == GAPS_REPORT
    assert any_report_package(pkg) == GAPS_REPORT


def test_any_report_omits_overrides_and_by_design_fields(tmp_path: Path) -> None:
    # Every reportable type is overridden, some of them deliberately as Any, including the instance
    # type (the owner is not exported); var_args/var_kwargs are Any by design.
    overrides = (
        "    instance_type: Any\n"
        "    types: {timeout: Any, mode: str, cookies: Any, extensions: 'dict[str, str] | None', "
        "hook: Any, return: Any}\n"
    )
    pkg = _package(tmp_path, SPEC + overrides, _gaps_lock(owner="fakelib.Hidden"))
    spec = load_spec(pkg / "patches.yaml")
    model = build_patch(spec, spec.patches[0], _gaps_lock(owner="fakelib.Hidden"))
    assert model.instance_type == "Any"
    assert {f.name: f.type for f in model.fields}["var_args"] == "tuple[Any, ...]"
    assert any_report(spec) == []


def test_any_report_scopes_reasons_to_versions(tmp_path: Path) -> None:
    v1 = [["timeout", P, "None", None]]
    v2 = [["timeout", P, "None", ["Any"]]]
    note = "parameter 'timeout': <class 'custom.Timeout'> rendered as Any"
    lock = _lock({"1.0": {"client_send": _entry(v1)}, "2.0": {"client_send": _entry(v2, notes=[note])}})
    pkg = _package(tmp_path, SPEC, lock)
    assert any_report(load_spec(pkg / "patches.yaml")) == [
        "fakelib: patch client_send: parameter 'timeout' is Any "
        "(unannotated in fakelib <= 1.0; <class 'custom.Timeout'> rendered as Any in fakelib >= 2.0); "
        "add types: {timeout: ...} to patches.yaml",
    ]


def test_cli_reports_any_without_changing_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pkg = _package(tmp_path, SPEC, _gaps_lock())
    out = pkg / "src" / "opentelemetry" / "instrumentation" / "fakelib" / "_patches.py"
    expected = "".join(f"{line}\n" for line in GAPS_REPORT)

    # `report` writes nothing and is stable across runs.
    assert cli.main(["report", str(pkg)]) == 0
    first = capsys.readouterr()
    assert (first.out, first.err) == (expected, "")
    assert not out.exists()
    assert cli.main(["report", str(pkg)]) == 0
    assert capsys.readouterr().out == expected

    # `generate` and `check` print the same report to stderr and still succeed.
    assert cli.main(["generate", str(pkg)]) == 0
    generated = capsys.readouterr()
    assert generated.err == expected
    assert generated.out == f"wrote {out}\n"
    source = out.read_text(encoding="utf-8")
    assert f"# note: {GAPS_NOTE}\n" in source
    assert "AnyGap" not in source and "add types:" not in source
    assert cli.main(["check", str(pkg)]) == 0
    checked = capsys.readouterr()
    assert checked.err == expected
    assert checked.out == f"ok {out}\n"


def test_cli_report_without_gaps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pkg = _package(tmp_path, SPEC, _lock({"1.0": {"client_send": _entry(SEND_V1)}}))
    assert cli.main(["report", str(pkg)]) == 0
    assert capsys.readouterr().out == "fakelib: no Any gaps (patches.yaml)\n"
    assert cli.main(["report"]) == 1


def test_submodule_imports_and_type_checking(tmp_path: Path) -> None:
    spec = """\
    library: fakelib
    imports:
      - external_pkg.types
    patches:
      - name: sub_call
        target: fakelib.sub:SubClient.call
        versions: ">=2.0"
        instance_type: fakelib.sub.SubClient
        types:
          extra: fakelib.deep.mod.Config
          mode: typing.Literal["fast", "slow"]
          hook: external_pkg.types.Hook | None
    """
    params = [
        ["req", P, None, ["fakelib.http.SubRequest"]],
        ["extra", P, "None", None],
        ["mode", P, "'fast'", None],
        ["hook", P, "None", None],
    ]
    entry = _entry(params, ret=["fakelib.http.SubResponse"], owner="fakelib.sub.SubClient")
    lock = {
        "library": "fakelib",
        "generated_with": "patchgen 0.1",
        "python": "3.12",
        "specifier": ">=1",
        "versions": {
            "1.0": {"sub_call": {"absent": True}},
            "2.0": {"sub_call": entry},
        },
        "exports": {
            "1.0": ["Client"],
            "2.0": ["Client", "http.SubRequest", "http.SubResponse", "sub.SubClient"],
        },
    }
    source = _source(tmp_path, spec, lock)
    # Stdlib first, then everything else, as isort (ruff `I`) requires.
    assert (
        "if TYPE_CHECKING:\n"
        "    import typing\n"
        "\n"
        "    import external_pkg.types\n"
        "    import fakelib.deep.mod\n"
        "    import fakelib.http\n"
        "    import fakelib.sub\n"
    ) in source
    assert "req: fakelib.http.SubRequest" in source
    assert "extra: fakelib.deep.mod.Config" in source
    assert 'mode: typing.Literal["fast", "slow"]' in source
    assert ") -> fakelib.http.SubResponse: ..." in source

    unused = spec.replace("          hook: external_pkg.types.Hook | None\n", "")
    with pytest.raises(GenerateError, match=r"`imports:` \['external_pkg.types'\] not referenced"):
        _source(tmp_path / "unused", unused, lock)


def test_introspect_renders_submodule_classes_and_py39_union(monkeypatch: pytest.MonkeyPatch) -> None:
    import typing as std_typing  # noqa: PLC0415

    from patchgen import introspect  # noqa: PLC0415

    top_mod = types.ModuleType("subpkg")
    sub_mod = types.ModuleType("subpkg.web")
    priv_mod = types.ModuleType("subpkg._internal")

    class Handler:
        pass

    Handler.__module__ = "subpkg.web"
    Handler.__qualname__ = "Handler"

    class Secret:
        pass

    Secret.__module__ = "subpkg._internal"
    Secret.__qualname__ = "Secret"

    def handle(self: Any, h: Any, s: Any) -> Any:
        return h

    union_cls: Any = std_typing.Union
    handle.__annotations__ = {
        "self": Any,
        "h": union_cls[Handler, None],
        "s": Secret,
        "return": Handler,
    }
    handle.__module__ = "subpkg.web"
    handle.__qualname__ = "Handler.handle"
    setattr(Handler, "handle", handle)
    setattr(sub_mod, "Handler", Handler)
    setattr(priv_mod, "Secret", Secret)
    setattr(top_mod, "web", sub_mod)

    monkeypatch.setitem(sys.modules, "subpkg", top_mod)
    monkeypatch.setitem(sys.modules, "subpkg.web", sub_mod)
    monkeypatch.setitem(sys.modules, "subpkg._internal", priv_mod)
    monkeypatch.setattr(introspect, "_UNION_TYPE", None)

    desc = introspect.describe("subpkg", "subpkg.web:Handler.handle")
    assert desc["owner"] == "subpkg.web.Handler"
    assert desc["params"][1][3] == ["subpkg.web.Handler", "None"]
    assert desc["params"][2][3] == ["Any"]
    assert desc["return"] == ["subpkg.web.Handler"]
    exports = introspect._collect_exports("subpkg")
    assert "web.Handler" in exports
    assert not any("Secret" in e for e in exports)


# --------------------------------------------------------------------------- numeric unions (P6)


def test_numeric_unions_collapse_to_the_widest_type(tmp_path: Path) -> None:
    v1 = [["timeout", P, "None", ["int", "None"]], ["scale", P, "1", ["complex"]]]
    v2 = [["timeout", P, "None", ["float", "None"]], ["scale", P, "1", ["float", "int"]]]
    lock = _lock({"1.0": {"client_send": _entry(v1)}, "2.0": {"client_send": _entry(v2)}})
    model = _build(tmp_path, SPEC, lock)
    fields = {f.name: f.type for f in model.fields}
    assert fields == {"timeout": "float | None", "scale": "complex"}
    # The generated module passes the ruff gate (PYI041).
    assert "timeout: float | None\n" in _source(tmp_path / "gen", SPEC, lock)


# --------------------------------------------------------------------------- partial-range patches (E2)

PARTIAL_SPEC = """\
library: fakelib
patches:
  - name: client_send
    target: fakelib:Client.send
    scope: [class, instance]
  - name: client_send_async
    target: fakelib:Client.send_async
    scope: [class, instance]
"""


def test_partial_range_patches_are_optional(tmp_path: Path) -> None:
    lock = _lock(
        {
            "1.0": {"client_send": _entry(SEND_V1), "client_send_async": {"absent": True}},
            "2.0": {"client_send": _entry(SEND_V1), "client_send_async": _entry(SEND_V1, flavor="async")},
        }
    )
    pkg = _package(tmp_path, PARTIAL_SPEC, lock)
    spec = load_spec(pkg / "patches.yaml")
    assert [build_patch(spec, p, lock).partial for p in spec.patches] == [False, True]
    source = generate(spec)[1]
    instrument = source[source.index("def instrument(") : source.index("def uninstrument(")]
    assert "    client_send: ClientSendWrapper,\n" in instrument
    assert "    client_send_async: ClientSendAsyncWrapper,\n" in instrument
    assert "is not None" not in instrument
    instance = source[source.index("def instrument_instance(") : source.index("def uninstrument_instance(")]
    assert "    client_send: ClientSendWrapper,\n" in instance
    assert "    client_send_async: ClientSendAsyncWrapper | None = None,\n" in instance
    assert "    if client_send_async is not None:\n" in instance

    # Per-patch `versions:` ranges and `required: false` patches are partial in instrument_instance(),
    # while only `required: false` is optional in instrument().
    ranged = load_spec(_package(tmp_path / "ranged", RANGED, _ranged_lock()) / "patches.yaml")
    assert [build_patch(ranged, p, _ranged_lock()).partial for p in ranged.patches] == [True, True]
    optional_spec = load_spec(_package(tmp_path / "opt", SPEC + "    required: false\n", lock) / "patches.yaml")
    assert build_patch(optional_spec, optional_spec.patches[0], lock).partial is True
    opt_source = generate(optional_spec)[1]
    opt_instrument = opt_source[opt_source.index("def instrument(") : opt_source.index("def uninstrument(")]
    assert "    client_send: ClientSendWrapper | None = None,\n" in opt_instrument
    assert "    if client_send is not None:\n" in opt_instrument


def test_partial_range_patch_can_be_omitted(tmp_path: Path, fakelib: Any) -> None:
    spec = textwrap.dedent(
        """\
        library: fakelib
        stdlib: true
        patches:
          - name: send_legacy
            target: fakelib:Client.send
            versions: "<3.0"
            scope: [class, instance]
          - name: send
            target: fakelib:Client.send
            versions: ">=3.0"
            scope: [class, instance]
          - name: send_optional
            target: fakelib:Client.send_async
            required: false
        """
    )
    lock = _stdlib_lock(
        {
            "2.7": {
                "send_legacy": _entry(SEND_V1),
                "send": _entry(SEND_V1),
                "send_optional": _entry(SEND_V1, flavor="async"),
            },
            _THIS_PYTHON: {
                "send_legacy": _entry(SEND_V2),
                "send": _entry(SEND_V2),
                "send_optional": _entry(SEND_V2, flavor="async"),
            },
        }
    )
    patches = _load_generated(_source(tmp_path, spec, lock))
    calls: list[Any] = []

    def legacy(wrapped: Any, instance: Any, args: Any) -> Any:
        raise AssertionError("out-of-range legacy patch must not run")

    def modern(wrapped: Any, instance: Any, args: Any) -> Any:
        calls.append(args)
        return args.call()

    patches.instrument(send_legacy=legacy, send=modern)
    try:
        fakelib.Client().send(fakelib.Request("x"))
    finally:
        patches.uninstrument()
    assert len(calls) == 1

    client = fakelib.Client()
    patches.instrument_instance(client, send=modern)
    try:
        client.send(fakelib.Request("y"))
    finally:
        patches.uninstrument_instance(client)
    assert len(calls) == 2


# --------------------------------------------------------------------------- opt_out scope (E4)


def _opt_out_spec(tmp_path: Path, lock: dict[str, Any], spec_text: str = SPEC) -> tuple[Any, str]:
    pkg = _package(tmp_path, spec_text, lock)
    loaded = load_spec(pkg / "patches.yaml")
    patches = tuple(dataclasses.replace(p, scope=("class", "opt_out")) for p in loaded.patches)
    return dataclasses.replace(loaded, patches=patches), (pkg / "patches.yaml").read_text(encoding="utf-8")


def test_opt_out_scope_only_generates_uninstrument_instance(tmp_path: Path, fakelib: Any) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}})
    spec, spec_text = _opt_out_spec(tmp_path, lock)
    path = generate_mod.output_path(spec)
    source = generate_mod.format_source(render(spec, spec_text, dump_lock(lock)), path)
    assert "instrument_instance(" not in source.replace("uninstrument_instance(", "")
    assert '"instrument_instance"' not in source
    assert '"uninstrument_instance"' in source
    assert "def uninstrument_instance(obj: fakelib.Client) -> None:\n" in source
    patches = _load_generated(source)
    calls: list[Any] = []

    def wrapper(wrapped: Any, instance: Any, args: Any) -> Any:
        calls.append(args)
        return args.call()

    patches.instrument(client_send=wrapper)
    try:
        pinned = fakelib.Client()
        patches.uninstrument_instance(pinned)
        pinned.send(fakelib.Request("x"))
        assert calls == []
        fakelib.Client().send(fakelib.Request("y"))
        assert len(calls) == 1
    finally:
        patches.uninstrument()


def test_opt_out_scope_requires_a_method(tmp_path: Path) -> None:
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1, binding="function")}})
    spec, _ = _opt_out_spec(tmp_path, lock)
    with pytest.raises(GenerateError, match="opt_out"):
        build_patch(spec, spec.patches[0], lock)


# --------------------------------------------------------------------------- instance type (T3, P9)

_SPELLING_EXPORTS = ["Client", "Request", "Response", "base.Client"]


def test_instance_type_tries_each_owner_spelling(tmp_path: Path) -> None:
    v1 = _entry(SEND_V1, owner="fakelib.base.Client")
    v2 = _entry(SEND_V1, owner="fakelib.Client")
    lock = _lock({"1.0": {"client_send": v1}, "2.0": {"client_send": v2}}, exports=_SPELLING_EXPORTS)
    lock["exports"]["1.0"] = ["Request", "Response", "base.Client"]
    model = _build(tmp_path, SPEC, lock)
    assert model.instance_type == "fakelib.base.Client"
    assert model.any_gaps == ()


def test_instance_type_falls_back_to_defining_module(tmp_path: Path) -> None:
    hidden = _entry(SEND_V1, owner=None, owner_defined="fakelib.base.Client")
    lock = _lock({"1.0": {"client_send": hidden}, "2.0": {"client_send": hidden}}, exports=_SPELLING_EXPORTS)
    model = _build(tmp_path, SPEC, lock)
    assert model.instance_type == "fakelib.base.Client"
    assert model.any_gaps == ()


def test_unresolvable_instance_type_is_reported(tmp_path: Path) -> None:
    entry = _entry(SEND_V1, owner=None, owner_defined="fakelib._impl.Client")
    lock = _lock({"1.0": {"client_send": entry}})
    pkg = _package(tmp_path, SPEC, lock)
    spec = load_spec(pkg / "patches.yaml")
    model = build_patch(spec, spec.patches[0], lock)
    assert model.instance_type == "Any"
    reasons = "owner not publicly exported in any locked version; defined as fakelib._impl.Client, which is not public"
    assert any_report(spec) == [
        f"fakelib: patch client_send: instance type is Any ({reasons} in every locked version); "
        "add instance_type: ... to patches.yaml"
    ]
    assert f"client_send instance: {reasons} in every locked version; using Any" in model.notes

    # An old lock (no owner_defined) whose owner is not exported in every version.
    old = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(SEND_V1)}})
    old["exports"]["1.0"] = ["Request", "Response"]
    model = _build(tmp_path / "old", SPEC, old)
    assert model.instance_type == "Any"
    assert model.any_gaps[-1].reasons == ("owner fakelib.Client not in every locked version",)

    # `instance_type:` silences it.
    overridden = _package(tmp_path / "override", SPEC + "    instance_type: object\n", lock)
    assert any_report(load_spec(overridden / "patches.yaml")) == []


def _taskgroup_lock() -> dict[str, Any]:
    entry = _entry(SEND_V1, owner="fakelib.Group")
    lock = _stdlib_lock(
        {
            "3.10": {"client_send": {"absent": True}},
            "3.11": {"client_send": entry},
            "3.12": {"client_send": entry},
        }
    )
    lock["exports"] = {
        "3.10": ["Request", "Response"],
        "3.11": ["Group", "Request", "Response"],
        "3.12": ["Group", "Request", "Response"],
    }
    return lock


def test_stdlib_owner_missing_on_typecheck_python_becomes_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(generate_mod._typecheck_python(), str)
    spec = STDLIB_SPEC + '    versions: ">=3.11"\n'
    monkeypatch.setattr(generate_mod, "_typecheck_python", lambda: "3.10")
    model = _build(tmp_path, spec, _taskgroup_lock())
    assert model.instance_type == "object"
    assert any("fakelib.Group not in python 3.10" in n for n in model.notes)
    assert model.any_gaps == ()
    source = _source(tmp_path / "gen", spec, _taskgroup_lock())
    assert "instance: object," in source

    monkeypatch.setattr(generate_mod, "_typecheck_python", lambda: "3.11")
    assert _build(tmp_path / "newer", spec, _taskgroup_lock()).instance_type == "fakelib.Group"

    # Library specs: `versions:` are library versions, so the type-checking Python is irrelevant.
    library_lock = _lock({"1.0": {"client_send": {"absent": True}}, "2.0": {"client_send": _entry(SEND_V1)}})
    library_lock["exports"]["1.0"] = ["Request", "Response"]
    monkeypatch.setattr(generate_mod, "_typecheck_python", lambda: "3.10")
    assert _build(tmp_path / "lib", SPEC, library_lock).instance_type == "fakelib.Client"


# --------------------------------------------------------------------------- library-declared Any (T4)


def test_library_declared_any_is_not_gap(tmp_path: Path) -> None:
    params = [["mode", P, "None", ["Any"]], ["timeout", P, "None", ["Any"]]]
    note = "parameter 'timeout': <class 'custom.Timeout'> rendered as Any"
    entry = _entry(params, ret=["Any"], notes=[note])
    pkg = _package(tmp_path, SPEC, _lock({"1.0": {"client_send": entry}}))
    spec = load_spec(pkg / "patches.yaml")
    gap = (
        "fakelib: patch client_send: parameter 'timeout' is Any (<class 'custom.Timeout'> rendered as Any); "
        "add types: {timeout: ...} to patches.yaml"
    )
    assert any_report(spec) == [gap]
    assert any_report_package(pkg) == [gap]


# --------------------------------------------------------------------------- call(var_args=...) (P8)


def test_generated_call_replaces_var_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    params = [["first", P, None, ["int"]], ["args", "VAR_POSITIONAL", None, ["int"]]]
    source = _source(tmp_path, SPEC, _lock({"1.0": {"client_send": _entry(params)}}))
    assert "            var_args: tuple[Any, ...] | Missing = MISSING,\n" in source
    module = _load_generated(source)
    seen: list[dict[str, Any]] = []

    def fake_replace(
        sig: Any, args: tuple[Any, ...], kwargs: dict[str, Any], updates: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        seen.append(updates)
        return (args[0], *updates["var_args"]), dict(kwargs)

    monkeypatch.setattr(module._patching, "replace_raw_args", fake_replace)

    def target(first: int, *args: int) -> tuple[int, ...]:
        return (first, *args)

    parsed = module.ClientSendArgs(
        raw_args=(1, 2, 3),
        raw_kwargs={},
        _wrapped=target,
    )
    assert parsed.call() == (1, 2, 3)
    assert parsed.call(var_args=(9,)) == (1, 9)
    assert seen == [{"var_args": (9,)}]
    monkeypatch.undo()
    # End to end through the runtime's replace_raw_args (contract C5).
    assert parsed.call(first=5, var_args=(7, 8)) == (5, 7, 8)

    only = _source(tmp_path / "only", SPEC, _lock({"1.0": {"client_send": _entry([params[1]])}}))
    assert (
        "        def call(\n            self,\n            *,\n            var_args: tuple[Any, ...] | Missing = MISSING,\n        )"
        in only
    )
    no_var_args = _source(tmp_path / "none", SPEC, _lock({"1.0": {"client_send": _entry(SEND_V1)}}))
    assert "var_args" not in no_var_args


# --------------------------------------------------------------------------- locked_shapes (P4)


def test_locked_shapes_record_has_default_per_parameter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v2 = [*SEND_V2, ["hook", K, "<function default_hook at 0x1>", None]]
    lock = _lock({"1.0": {"client_send": _entry(SEND_V1)}, "2.0": {"client_send": _entry(v2)}})
    model = _build(tmp_path, SPEC, lock)
    assert model.shapes == (
        (
            ("self", P, False),
            ("request", P, False),
            ("timeout", P, True),
        ),
        (
            ("self", P, False),
            ("request", P, False),
            ("timeout", P, True),
            ("retries", K, True),
            ("hook", K, True),
        ),
    )
    module = _load_generated(_source(tmp_path / "gen", SPEC, lock))
    captured: list[Any] = []
    monkeypatch.setattr(module._patching, "apply_all", lambda _registry, **kw: captured.extend(kw["patches"]))
    monkeypatch.setattr(module._patching, "apply_instance", lambda _obj, patches, **_kw: captured.extend(patches))
    module.instrument(client_send=lambda wrapped, instance, args: args.call())
    module.instrument_instance(object(), client_send=lambda wrapped, instance, args: args.call())
    assert len(captured) == 2
    for patch_def in captured:
        assert patch_def.locked_shapes == frozenset(model.shapes)


# --------------------------------------------------------------------------- N2, N3, N4, N5


def test_optional_any_and_mixed_unannotated_declared_any_are_not_gaps(tmp_path: Path) -> None:
    v1_params = [
        ["value", P, None, None],
        ["default", P, "None", None],
    ]
    v2_params = [
        ["value", P, None, ["Any"]],
        ["default", P, "None", ["Any", "None"]],
    ]
    lock = _lock(
        {
            "1.0": {"client_send": _entry(v1_params, ret=None, **{"return": None})},
            "2.0": {"client_send": _entry(v2_params, ret=["Any"])},
        }
    )
    pkg = _package(tmp_path, SPEC, lock)
    spec = load_spec(pkg / "patches.yaml")
    assert any_report(spec) == []


def test_types_module_imports_from_lock_entries(tmp_path: Path) -> None:
    entry = _entry(
        [["source", P, None, ["str"]]],
        ret=["str", "types.CodeType"],
        imports=["types"],
    )
    lock = _lock({"1.0": {"client_send": entry}})
    pkg = _package(tmp_path, SPEC, lock)
    spec = load_spec(pkg / "patches.yaml")
    assert any_report(spec) == []
    source = generate(spec)[1]
    assert "if TYPE_CHECKING:\n    import types\n" in source
    assert ") -> str | types.CodeType: ..." in source


# --------------------------------------------------------------------------- classmethod and staticmethod (P7)


def test_classmethod_and_staticmethod_targets(tmp_path: Path, fakelib: Any) -> None:
    from patchgen import introspect  # noqa: PLC0415

    cls_desc = introspect.describe("fakelib", "fakelib:Client.from_url")
    assert cls_desc["binding"] == "classmethod"
    assert cls_desc["owner"] == "fakelib.Client"
    assert cls_desc["owner_defined"] == "fakelib.Client"
    assert cls_desc["params"] == [
        ["cls", P, None, None],
        ["url", P, None, ["str"]],
        ["retries", K, "0", ["int"]],
    ]

    static_desc = introspect.describe("fakelib", "fakelib:Client.normalize")
    assert static_desc["binding"] == "staticmethod"
    assert static_desc["owner"] is None
    assert "owner_defined" not in static_desc
    assert static_desc["params"] == [["url", P, None, ["str"]]]
    assert static_desc["return"] == ["str"]

    spec_text = """\
    library: fakelib
    patches:
      - name: client_from_url
        target: fakelib:Client.from_url
      - name: client_normalize
        target: fakelib:Client.normalize
    """
    cls_entry = dict(cls_desc, **{"return": ["fakelib.Client"]})
    lock = _lock(
        {"1.0": {"client_from_url": cls_entry, "client_normalize": static_desc}},
        exports=["Client", "SubClient", "Request", "Response"],
    )
    pkg = _package(tmp_path, spec_text, lock)
    spec = load_spec(pkg / "patches.yaml")
    cls_model = build_patch(spec, spec.patches[0], lock)
    static_model = build_patch(spec, spec.patches[1], lock)

    assert cls_model.method is True
    assert cls_model.instance_type == "type[fakelib.Client]"
    assert [f.name for f in cls_model.fields] == ["url", "retries"]
    assert static_model.method is False
    assert static_model.instance_type == "object"
    assert [f.name for f in static_model.fields] == ["url"]

    # Neither classmethod nor staticmethod may use instance or opt_out scope.
    for bad_binding in ("classmethod", "staticmethod"):
        bad_lock = _lock({"1.0": {"client_send": _entry(SEND_V1, binding=bad_binding)}})
        with pytest.raises(GenerateError, match="require a method target"):
            _build(tmp_path / f"bad_{bad_binding}", SPEC, bad_lock)

    source = generate(spec)[1]
    assert "instance: type[fakelib.Client]," in source
    assert "instance: object," in source
    patches = _load_generated(source)
    seen_cls: list[tuple[Any, str, int]] = []
    seen_static: list[tuple[Any, str]] = []

    def wrap_from_url(wrapped: Any, instance: Any, args: Any) -> Any:
        seen_cls.append((instance, args.url, args.retries))
        return args.call(retries=args.retries + 1)

    def wrap_normalize(wrapped: Any, instance: Any, args: Any) -> Any:
        seen_static.append((instance, args.url))
        return args.call()

    patches.instrument(client_from_url=wrap_from_url, client_normalize=wrap_normalize)
    try:
        c1 = fakelib.Client.from_url("http://a", retries=2)
        assert type(c1) is fakelib.Client
        assert c1.url == "Client:http://a:3"
        c2 = fakelib.SubClient.from_url("http://b")
        assert type(c2) is fakelib.SubClient
        assert c2.url == "SubClient:http://b:1"
        assert seen_cls == [
            (fakelib.Client, "http://a", 2),
            (fakelib.SubClient, "http://b", 0),
        ]

        assert fakelib.Client.normalize("  hi ") == "hi"
        assert fakelib.Client().normalize("  there ") == "there"
        assert [u for _, u in seen_static] == ["  hi ", "  there "]
    finally:
        patches.uninstrument()


# --------------------------------------------------------------------------- disjoint signatures warning


def test_disjoint_signatures_across_versions_emits_split_warning(tmp_path: Path) -> None:
    v1 = [["method", P, None, ["str"]], ["url", P, None, ["str"]]]
    v2 = [["request", P, None, ["fakelib.Request"]]]
    lock = _lock({"1.0": {"client_send": _entry(v1)}, "2.0": {"client_send": _entry(v2)}})
    pkg = _package(tmp_path, SPEC, lock)
    spec = load_spec(pkg / "patches.yaml")
    model = build_patch(spec, spec.patches[0], lock)

    assert all(f.missing for f in model.fields)
    assert any(
        "client_send: shares no parameters across <2.0 and >=2.0; consider splitting with versions:" in n
        for n in model.notes
    )
    assert any_report(spec) == [
        "fakelib: patch client_send: shares no parameters across <2.0 and >=2.0; "
        'consider splitting into two patches with versions: "<2.0" and versions: ">=2.0" in patches.yaml'
    ]

    # When at least one parameter is shared across all versions, no split warning is emitted.
    shared_v1 = [*v1, ["timeout", P, "None", ["float", "None"]]]
    shared_v2 = [*v2, ["timeout", P, "None", ["float", "None"]]]
    shared_pkg = _package(
        tmp_path / "shared",
        SPEC,
        _lock({"1.0": {"client_send": _entry(shared_v1)}, "2.0": {"client_send": _entry(shared_v2)}}),
    )
    assert any_report(load_spec(shared_pkg / "patches.yaml")) == []

    # Splitting into two patches with `versions:` ranges eliminates the warning.
    split_pkg = _package(tmp_path / "split", RANGED, _ranged_lock())
    assert any_report(load_spec(split_pkg / "patches.yaml")) == []

    # Adding parameters to an initially parameterless signature is not a disjoint split.
    added_pkg = _package(
        tmp_path / "added",
        SPEC,
        _lock({"1.0": {"client_send": _entry([])}, "2.0": {"client_send": _entry(SEND_V1)}}),
    )
    assert any_report(load_spec(added_pkg / "patches.yaml")) == []
