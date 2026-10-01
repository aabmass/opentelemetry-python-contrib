# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Hermetic tests for ``patchgen lock``: stdlib mode and library extras (no network, no uv)."""

from __future__ import annotations

import json
import sys
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from packaging.version import Version
from patchgen import lock as lock_mod
from patchgen.lock import (
    LockError,
    Release,
    _floor_suggestion,
    _lock_selected,
    _run,
    _summarize,
    build_lock,
    fetch_pythons,
    introspect_version,
    select_pythons,
)
from patchgen.lockfile import dump_lock
from patchgen.spec import load_spec

P = "POSITIONAL_OR_KEYWORD"

ENTRY: dict[str, Any] = {
    "binding": "method",
    "flavor": "sync",
    "owner": "urllib.request.OpenerDirector",
    "params": [["self", P, None, None], ["fullurl", P, None, None], ["data", P, "None", None]],
    "return": None,
    "notes": [],
}

STDLIB_SPEC = """\
library: urllib
stdlib: true
patches:
  - name: opener_open
    target: urllib.request:OpenerDirector.open
"""

EXTRAS_SPEC = """\
library: fakelib
extras: [asyncpg, mysql]
versions: ">=1.0"
patches:
  - name: client_send
    target: fakelib:Client.send
"""


def _vs(*versions: str) -> list[Version]:
    return [Version(v) for v in versions]


def _spec(tmp_path: Path, body: str, requires_python: str | None = ">=3.10") -> Any:
    pkg = tmp_path / "opentelemetry-instrumentation-fake"
    pkg.mkdir()
    (pkg / "patches.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    pyproject = '[project]\nname = "fake"\n'
    if requires_python is not None:
        pyproject += f'requires-python = "{requires_python}"\n'
    (pkg / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return load_spec(pkg / "patches.yaml")


class _FakeRun:
    """Stand-in for ``lock._run`` that records commands and answers the introspect call."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.commands: list[list[str]] = []
        self.envs: list[Mapping[str, str] | None] = []
        self.constraints: str | None = None

    def __call__(self, cmd: Sequence[str], *, stdin: str | None = None, env: Mapping[str, str] | None = None) -> str:
        self.commands.append(list(cmd))
        self.envs.append(env)
        if "--constraint" in cmd:
            self.constraints = Path(cmd[list(cmd).index("--constraint") + 1]).read_text(encoding="utf-8")
        if stdin is not None:
            return json.dumps(self.result)
        return ""


# --------------------------------------------------------------------------- Python enumeration


def test_select_pythons_one_per_minor_and_prerelease_handling() -> None:
    available = _vs("3.9.25", "3.10.3", "3.10.19", "3.11.14", "3.12.12", "3.13.10", "3.14.1", "3.15.0b2")
    assert select_pythons(available, ">=3.10") == _vs("3.10", "3.11", "3.12", "3.13", "3.14")
    assert select_pythons(available, ">=3.11,<3.13") == _vs("3.11", "3.12")
    with pytest.raises(LockError, match="no Python versions match"):
        select_pythons(available, ">=4")


def test_fetch_pythons_filters_implementations_and_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [
        {"version": "3.12.12", "implementation": "cpython", "variant": "default"},
        {"version": "3.12.12", "implementation": "cpython", "variant": "default"},  # duplicate (symlink)
        {"version": "3.13.9", "implementation": "cpython", "variant": "debug"},
        {"version": "3.14.1", "implementation": "cpython", "variant": "freethreaded"},
        {"version": "3.10.16", "implementation": "pypy", "variant": "default"},
        {"version": "3.11.14", "implementation": "cpython", "variant": "default"},
        {"version": "not-a-version", "implementation": "cpython", "variant": "default"},
    ]
    seen: list[list[str]] = []

    def fake_run(cmd: Sequence[str], *, stdin: str | None = None) -> str:
        seen.append(list(cmd))
        return json.dumps(listing)

    monkeypatch.setattr(lock_mod, "_run", fake_run)
    assert fetch_pythons() == _vs("3.11.14", "3.12.12")
    assert seen == [["uv", "python", "list", "--output-format", "json"]]


# --------------------------------------------------------------------------- introspection commands


def test_introspect_version_installs_library_with_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path, EXTRAS_SPEC)
    run = _FakeRun({"targets": {"client_send": ENTRY}, "exports": ["Client"]})
    monkeypatch.setattr(lock_mod, "_run", run)

    result = introspect_version(spec, Version("1.2.0"), "3.12")

    assert result["targets"]["client_send"] == ENTRY
    assert [c[:2] for c in run.commands] == [["uv", "venv"], ["uv", "pip"], run.commands[2][:2]]
    assert run.commands[0][2:5] == ["--quiet", "--python", "3.12"]
    assert run.commands[1][-1] == "fakelib[asyncpg,mysql]==1.2.0"
    assert run.commands[2][1].endswith("introspect.py")


def test_introspect_version_stdlib_skips_install_and_uses_that_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, STDLIB_SPEC)
    run = _FakeRun({"targets": {"opener_open": ENTRY}, "exports": ["request.OpenerDirector"]})
    monkeypatch.setattr(lock_mod, "_run", run)

    introspect_version(spec, Version("3.11"), "3.11")

    assert [c[:2] for c in run.commands] == [["uv", "venv"], run.commands[1][:2]]
    assert run.commands[0][2:5] == ["--quiet", "--python", "3.11"]
    assert not any(c[:2] == ["uv", "pip"] for c in run.commands)


def test_introspect_version_error_label_names_python_for_stdlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, STDLIB_SPEC)
    monkeypatch.setattr(lock_mod, "_run", _FakeRun({"error": "cannot import urllib: boom"}))
    with pytest.raises(LockError, match=r"^python 3\.11: cannot import urllib"):
        introspect_version(spec, Version("3.11"), "3.11")


# --------------------------------------------------------------------------- build_lock


def test_build_lock_stdlib_locks_every_supported_minor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path, STDLIB_SPEC, requires_python=">=3.11")
    monkeypatch.setattr(lock_mod, "fetch_pythons", lambda: _vs("3.10.19", "3.11.14", "3.12.12", "3.13.0rc1"))
    monkeypatch.setattr(lock_mod, "fetch_releases", lambda _lib: pytest.fail("PyPI must not be queried"))
    calls: list[tuple[Version, str]] = []

    def fake_introspect(_spec: Any, version: Version, python: str) -> dict[str, Any]:
        calls.append((version, python))
        return {"targets": {"opener_open": ENTRY}, "exports": ["request.OpenerDirector"]}

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    logs: list[str] = []
    locked = build_lock(spec, python="3.12", log=logs.append)

    # requires-python floor applied, 3.13 skipped (only a prerelease), each minor on its own interpreter.
    assert calls == [(Version("3.11"), "3.11"), (Version("3.12"), "3.12")]
    assert list(locked["versions"]) == ["3.11", "3.12"]
    assert locked["stdlib"] is True
    assert locked["specifier"] == ">=3.11"
    assert locked["library"] == "urllib"
    assert "python" not in locked
    assert locked["exports"]["3.11"] == ["request.OpenerDirector"]
    assert any("--python is ignored with stdlib: true" in line for line in logs)
    assert any("python >=3.11 -> 3.11, 3.12 (stdlib)" in line for line in logs)


def test_build_lock_stdlib_explicit_versions_and_floor_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path, STDLIB_SPEC.replace("stdlib: true\n", 'stdlib: true\nversions: ">=3.10"\n'))
    monkeypatch.setattr(lock_mod, "fetch_pythons", lambda: _vs("3.10.19", "3.11.14"))

    def fake_introspect(_spec: Any, version: Version, _python: str) -> dict[str, Any]:
        if version == Version("3.10"):
            raise LockError("python 3.10: cannot import urllib")
        return {"targets": {"opener_open": ENTRY}, "exports": []}

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    logs: list[str] = []
    locked = build_lock(spec, log=logs.append)
    assert list(locked["versions"]) == ["3.11"]
    assert any("skipping python 3.10" in line for line in logs)
    assert any("spec/requires-python floor is below 3.11" in line for line in logs)


def test_build_lock_records_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path, EXTRAS_SPEC)
    monkeypatch.setattr(lock_mod, "fetch_releases", lambda _lib: [Release(Version("1.0.0"), (None,))])
    monkeypatch.setattr(lock_mod, "fetch_pythons", lambda: pytest.fail("uv python must not be queried"))
    monkeypatch.setattr(
        lock_mod,
        "introspect_version",
        lambda _s, _v, _p: {"targets": {"client_send": ENTRY}, "exports": ["Client"]},
    )
    logs: list[str] = []
    locked = build_lock(spec, python="3.12", log=logs.append)
    assert locked["extras"] == ["asyncpg", "mysql"]
    assert locked["python"] == "3.12"
    assert "stdlib" not in locked
    assert any(line.startswith("fakelib[asyncpg,mysql]: >=1.0 -> 1.0.0") for line in logs)


def test_build_lock_without_extras_keeps_lock_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path, EXTRAS_SPEC.replace("extras: [asyncpg, mysql]\n", ""))
    monkeypatch.setattr(lock_mod, "fetch_releases", lambda _lib: [Release(Version("1.0.0"), (None,))])
    monkeypatch.setattr(
        lock_mod,
        "introspect_version",
        lambda _s, _v, _p: {"targets": {"client_send": ENTRY}, "exports": ["Client"]},
    )
    locked = build_lock(spec, python="3.12", log=lambda _line: None)
    assert list(locked) == ["library", "generated_with", "python", "specifier", "versions", "exports"]


# --------------------------------------------------------------------------- requires-python metadata (P2)


def test_release_supports_treats_invalid_requires_python_as_unsupported() -> None:
    assert Release(Version("1.0"), (">3.4.*",)).supports("3.12") is False
    assert Release(Version("1.0"), (">3.4.*", ">=3.8")).supports("3.12") is True
    assert Release(Version("1.0"), (">3.4.*", None)).supports("3.12") is True
    assert Release(Version("1.0"), ("<3.8",)).supports("3.12") is False


# --------------------------------------------------------------------------- with / constraints (P3)

WITH_SPEC = """\
library: fakelib
versions: ">=1.0"
with: ["anyio>=3", "setuptools<81"]
constraints: ["kafka-python<2.3"]
patches:
  - name: client_send
    target: fakelib:Client.send
"""


def test_introspect_version_installs_with_requirements_under_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, WITH_SPEC)
    run = _FakeRun({"targets": {"client_send": ENTRY}, "exports": ["Client"]})
    monkeypatch.setattr(lock_mod, "_run", run)
    monkeypatch.setenv("UV_CONSTRAINT", "/somewhere/constraints.txt")
    monkeypatch.setenv("UV_OVERRIDE", "/somewhere/overrides.txt")

    introspect_version(spec, Version("1.2.0"), "3.12")

    install = run.commands[1]
    assert install[:3] == ["uv", "pip", "install"]
    reqs = install[install.index("fakelib==1.2.0") :]
    assert reqs[:3] == ["fakelib==1.2.0", "anyio>=3", "setuptools<81"]
    assert reqs[3] == "--constraint"
    assert run.constraints == "kafka-python<2.3\n"
    env = run.envs[1]
    assert env is not None
    assert "UV_CONSTRAINT" not in env
    assert "UV_OVERRIDE" not in env


def test_build_lock_records_with_and_constraints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path, WITH_SPEC)
    monkeypatch.setattr(lock_mod, "fetch_releases", lambda _lib: [Release(Version("1.0.0"), (None,))])
    monkeypatch.setattr(
        lock_mod,
        "introspect_version",
        lambda _s, _v, _p: {"targets": {"client_send": ENTRY}, "exports": ["Client"]},
    )
    locked = build_lock(spec, python="3.12", log=lambda _line: None)
    assert locked["with"] == ["anyio>=3", "setuptools<81"]
    assert locked["constraints"] == ["kafka-python<2.3"]


# --------------------------------------------------------------------------- readable failures (L1)


def test_summarize_prefers_uv_headline_then_last_line() -> None:
    uv_err = "\n".join(
        [
            "  × Failed to build `asyncpg==0.22.0`",
            "  ├─▶ The build backend returned an error",
            "  ╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)",
            "      [stderr]",
            *[f"      gcc noise {i}" for i in range(300)],
        ]
    )
    assert _summarize(uv_err) == (
        "Failed to build `asyncpg==0.22.0`: The build backend returned an error: "
        "Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)"
    )
    traceback = 'Traceback (most recent call last):\n  File "x.py", line 1\nAttributeError: module has no Mapping\n'
    assert _summarize(traceback) == "AttributeError: module has no Mapping"
    assert _summarize("") == "(no error output)"
    assert len(_summarize("x" * 1000)) <= 300


def test_run_error_is_one_line_with_full_details() -> None:
    script = "import sys; sys.stderr.write('noise\\n' * 200 + 'error: boom happened\\n'); sys.exit(3)"
    with pytest.raises(LockError) as info:
        _run([sys.executable, "-c", script])
    assert "\n" not in str(info.value)
    assert "failed (3): error: boom happened" in str(info.value)
    assert info.value.details is not None
    assert info.value.details.count("noise\n") == 200


def test_lock_selected_logs_one_line_per_skip_and_keeps_full_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, EXTRAS_SPEC.replace("extras: [asyncpg, mysql]\n", ""))

    def fake_introspect(_spec: Any, version: Version, _python: str) -> dict[str, Any]:
        if version == Version("1.0"):
            raise LockError("uv pip install failed (1): Failed to build `fakelib==1.0`", details="BUILD LOG\n" * 50)
        return {"targets": {"client_send": ENTRY}, "exports": ["Client"]}

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    logs: list[str] = []
    versions, _ = _lock_selected(spec, _vs("1.0", "1.1"), python_for=lambda _v: "3.12", log=logs.append)
    assert list(versions) == ["1.1"]
    assert "  skipping fakelib==1.0: uv pip install failed (1): Failed to build `fakelib==1.0`" in logs
    assert "  skipped 1 version(s): 1.0" in logs
    assert "    fakelib==1.0: uv pip install failed (1): Failed to build `fakelib==1.0`" in logs
    assert not any("BUILD LOG" in line for line in logs)
    log_line = next(line for line in logs if line.startswith("  full error output: "))
    full = Path(log_line.removeprefix("  full error output: "))
    try:
        assert full.read_text(encoding="utf-8").count("BUILD LOG") == 50
    finally:
        full.unlink()


def test_lock_selected_all_failed_lists_reasons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path, EXTRAS_SPEC.replace("extras: [asyncpg, mysql]\n", ""))

    def fake_introspect(_spec: Any, version: Version, _python: str) -> dict[str, Any]:
        raise LockError(f"fakelib=={version}: cannot import fakelib: no module named six")

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    logs: list[str] = []
    with pytest.raises(LockError, match=r"(?s)all selected versions of fakelib failed to lock:.*1\.0: .*no module"):
        _lock_selected(spec, _vs("1.0"), python_for=lambda _v: "3.12", log=logs.append)
    assert "  skipped 1 version(s): 1.0" in logs


# --------------------------------------------------------------------------- unimportable versions (L2)

TWO_TARGET_SPEC = """\
library: fakelib
versions: ">=1.0"
patches:
  - name: client_send
    target: fakelib:Client.send
  - name: extra_send
    target: fakelib.extra:Extra.send
    required: false
"""


def test_lock_selected_skips_version_with_unimportable_required_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, TWO_TARGET_SPEC)

    def fake_introspect(_spec: Any, version: Version, _python: str) -> dict[str, Any]:
        if version == Version("1.0"):
            bad = {"unimportable": "cannot import fakelib: ModuleNotFoundError: No module named 'pkg_resources'"}
            return {"targets": {"client_send": bad, "extra_send": bad}, "exports": []}
        missing = {"unimportable": "cannot import fakelib.extra: ModuleNotFoundError: No module named 'anyio'"}
        return {"targets": {"client_send": ENTRY, "extra_send": missing}, "exports": ["Client"]}

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    logs: list[str] = []
    versions, _ = _lock_selected(spec, _vs("1.0", "1.1"), python_for=lambda _v: "3.12", log=logs.append)
    # 1.0 is skipped with the reason (not an abort); in 1.1 the optional target is merely absent.
    assert list(versions) == ["1.1"]
    assert any(line.startswith("  skipping fakelib==1.0: unimportable: fakelib:Client.send") for line in logs)
    assert versions["1.1"]["extra_send"]["absent"] is True
    assert versions["1.1"]["client_send"] == ENTRY


def test_lock_selected_missing_required_target_still_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path, TWO_TARGET_SPEC)
    missing = {"error": "target not found: module 'fakelib' has no attribute 'Client'"}
    monkeypatch.setattr(
        lock_mod,
        "introspect_version",
        lambda _s, _v, _p: {"targets": {"client_send": missing, "extra_send": ENTRY}, "exports": []},
    )
    with pytest.raises(LockError, match="target not found"):
        _lock_selected(spec, _vs("1.0"), python_for=lambda _v: "3.12", log=lambda _line: None)


# --------------------------------------------------------------------------- exports failures (P1)


def test_lock_selected_keeps_version_when_exports_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path, EXTRAS_SPEC.replace("extras: [asyncpg, mysql]\n", ""))
    monkeypatch.setattr(
        lock_mod,
        "introspect_version",
        lambda _s, _v, _p: {
            "targets": {"client_send": ENTRY},
            "exports": [],
            "exports_error": "ModuleNotFoundError: No module named '_gdbm'",
        },
    )
    logs: list[str] = []
    versions, exports = _lock_selected(spec, _vs("1.0"), python_for=lambda _v: "3.12", log=logs.append)
    assert list(versions) == ["1.0"]
    assert exports == {"1.0": []}
    assert any("exports collection failed" in line and "_gdbm" in line for line in logs)


# --------------------------------------------------------------------------- floor suggestion (L3)


@pytest.mark.parametrize(
    ("specifier", "expected"),
    [
        (">=0.5, <1.0", ">=0.9.0, <1.0"),
        (">=0.5", ">=0.9.0"),
        (None, ">=0.9.0"),
        ("~=0.5", ">=0.9.0, <1"),
        ("~=0.5.1", ">=0.9.0, <0.6"),
        (">=0.1, !=0.9.5, <=2.0", ">=0.9.0, !=0.9.5, <=2.0"),
        ("==0.*", ">=0.9.0, ==0.*"),
    ],
)
def test_floor_suggestion_keeps_upper_bound(specifier: str | None, expected: str) -> None:
    assert _floor_suggestion(Version("0.9.0"), specifier) == expected


def test_build_lock_floor_warning_keeps_instruments_upper_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, EXTRAS_SPEC.replace("extras: [asyncpg, mysql]\n", "").replace('">=1.0"', '">=1.0, <2.0"'))
    monkeypatch.setattr(
        lock_mod,
        "fetch_releases",
        lambda _lib: [Release(Version("1.0.0"), (None,)), Release(Version("1.5.0"), (None,))],
    )

    def fake_introspect(_spec: Any, version: Version, _python: str) -> dict[str, Any]:
        if version == Version("1.0.0"):
            raise LockError("fakelib==1.0.0: cannot import fakelib")
        return {"targets": {"client_send": ENTRY}, "exports": ["Client"]}

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    logs: list[str] = []
    build_lock(spec, python="3.12", log=logs.append)
    assert any('consider setting `versions: ">=1.5.0, <2.0"`' in line for line in logs)


# --------------------------------------------------------------------------- re-lock safety (N1)


def test_build_lock_reuses_python_from_existing_lock_and_warns_on_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, EXTRAS_SPEC.replace("extras: [asyncpg, mysql]\n", ""))
    spec.lock_path.write_text(
        dump_lock({"python": "3.10", "versions": {"1.0.0": {"client_send": ENTRY}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(lock_mod, "fetch_releases", lambda _lib: [Release(Version("1.0.0"), ("<3.12",))])
    seen_pythons: list[str] = []

    def fake_introspect(_spec: Any, _version: Version, python: str) -> dict[str, Any]:
        seen_pythons.append(python)
        return {"targets": {"client_send": ENTRY}, "exports": ["Client"]}

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    logs: list[str] = []
    locked = build_lock(spec, log=logs.append)
    assert locked["python"] == "3.10"
    assert seen_pythons == ["3.10"]
    assert "fakelib: using python 3.10 from patches.lock.json" in logs

    # Explicit --python differing from the existing lock warns:
    monkeypatch.setattr(lock_mod, "fetch_releases", lambda _lib: [Release(Version("1.0.0"), (None,))])
    logs.clear()
    locked2 = build_lock(spec, python="3.12", log=logs.append)
    assert locked2["python"] == "3.12"
    assert "  warning: --python 3.12 differs from patches.lock.json python 3.10" in logs


def test_build_lock_errors_when_relock_drops_previously_locked_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, EXTRAS_SPEC.replace("extras: [asyncpg, mysql]\n", ""))
    spec.lock_path.write_text(
        dump_lock(
            {
                "python": "3.10",
                "versions": {"1.0.0": {"client_send": ENTRY}, "1.5.0": {"client_send": ENTRY}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lock_mod,
        "fetch_releases",
        lambda _lib: [Release(Version("1.0.0"), (None,)), Release(Version("1.5.0"), (None,))],
    )

    def fake_introspect(_spec: Any, version: Version, _python: str) -> dict[str, Any]:
        if version == Version("1.0.0"):
            raise LockError("fakelib==1.0.0: cannot import fakelib")
        return {"targets": {"client_send": ENTRY}, "exports": ["Client"]}

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    with pytest.raises(LockError, match=r"^re-lock dropped previously locked version\(s\): 1\.0\.0$"):
        build_lock(spec, log=lambda _line: None)

    # Narrowing `versions:` in patches.yaml allows dropping versions outside the new specifier:
    narrowed_body = EXTRAS_SPEC.replace("extras: [asyncpg, mysql]\n", "").replace('">=1.0"', '">=1.5"')
    (spec.package_dir / "patches.yaml").write_text(textwrap.dedent(narrowed_body), encoding="utf-8")
    narrowed_spec = load_spec(spec.package_dir / "patches.yaml")
    locked_narrowed = build_lock(narrowed_spec, log=lambda _line: None)
    assert list(locked_narrowed["versions"]) == ["1.5.0"]

    # A newer patch release in the same minor replacing the previous newest-of-major is allowed:
    monkeypatch.setattr(
        lock_mod,
        "introspect_version",
        lambda _s, _v, _p: {"targets": {"client_send": ENTRY}, "exports": ["Client"]},
    )
    spec.lock_path.write_text(
        dump_lock(
            {
                "python": "3.10",
                "versions": {
                    "1.0.0": {"client_send": ENTRY},
                    "1.5.0": {"client_send": ENTRY},
                    "1.5.1": {"client_send": ENTRY},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lock_mod,
        "fetch_releases",
        lambda _lib: [
            Release(Version("1.0.0"), (None,)),
            Release(Version("1.5.0"), (None,)),
            Release(Version("1.5.1"), (None,)),
            Release(Version("1.5.2"), (None,)),
        ],
    )
    locked_progressed = build_lock(spec, log=lambda _line: None)
    assert list(locked_progressed["versions"]) == ["1.0.0", "1.5.0", "1.5.2"]


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('{"versions": {}}', r"lock format None is not supported .*; delete it and rerun `patchgen lock`"),
        ("not json", r"not valid JSON"),
    ],
)
def test_build_lock_errors_on_unreadable_existing_lock(tmp_path: Path, text: str, match: str) -> None:
    # An existing lock that can't be read must not be treated as "no lock": that would silently
    # drop the recorded python and the guard against losing previously locked versions.
    spec = _spec(tmp_path, EXTRAS_SPEC.replace("extras: [asyncpg, mysql]\n", ""))
    spec.lock_path.write_text(text, encoding="utf-8")
    with pytest.raises(LockError, match=rf"^cannot read existing lock .*patches\.lock\.json: .*{match}"):
        build_lock(spec, log=lambda _line: None)


def test_build_lock_trims_exports_to_referenced_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = """\
    library: fakelib
    versions: ">=1.0"
    patches:
      - name: client_send
        target: fakelib:Client.send
        types:
          extra: fakelib.Config
    """
    spec = _spec(tmp_path, body)
    monkeypatch.setattr(
        lock_mod,
        "fetch_releases",
        lambda _lib: [Release(Version("1.0.0"), (None,)), Release(Version("2.0.0"), (None,))],
    )
    v1_entry = {
        "binding": "method",
        "flavor": "sync",
        "owner": "fakelib.Client",
        "params": [["self", P, None, None], ["req", P, None, None], ["extra", P, "None", None]],
        "return": None,
        "notes": [],
    }
    v2_entry = {
        "binding": "method",
        "flavor": "sync",
        "owner": "fakelib.Client",
        "params": [["self", P, None, None], ["req", P, None, ["fakelib.Request"]], ["extra", P, "None", None]],
        "return": ["fakelib.Response"],
        "notes": [],
    }

    def fake_introspect(_spec: Any, version: Version, _python: str) -> dict[str, Any]:
        entry = v1_entry if version == Version("1.0.0") else v2_entry
        return {
            "targets": {"client_send": entry},
            "exports": ["Client", "Config", "Request", "Response", "UnusedA", "sub.UnusedB"],
        }

    monkeypatch.setattr(lock_mod, "introspect_version", fake_introspect)
    locked = build_lock(spec, python="3.12", log=lambda _line: None)
    assert locked["exports"] == {
        "1.0.0": ["Client", "Config", "Request", "Response"],
        "2.0.0": ["Client", "Config", "Request", "Response"],
    }
