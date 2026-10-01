# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""``patchgen lock``: resolve versions, record signatures, write the lock.

This is the only patchgen step with side inputs (network, PyPI state,
temporary environments).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

from . import GENERATED_WITH
from .lockfile import LockFormatError, dump_lock, load_lock
from .spec import Policy, Spec, default_python_versions, default_versions

INTROSPECT = Path(__file__).with_name("introspect.py")
PYPI_SIMPLE = "https://pypi.org/simple/{name}/"
# Environment variables that would make ``uv pip install`` non-hermetic; constraints
# belong in the spec (``constraints:``) so they are recorded in the lock.
_SCRUBBED_ENV = ("UV_CONSTRAINT", "UV_OVERRIDE")
_SUMMARY_MAX = 300


class LockError(RuntimeError):
    """Raised when a lock cannot be produced.

    ``str(exc)`` is a short, readable reason; ``details`` keeps the full output
    (e.g. a build log) when there is more to it.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class Release:
    version: Version
    requires_python: tuple[str | None, ...]

    def supports(self, python: str) -> bool:
        target = Version(python)
        for rp in self.requires_python:
            if rp is None:
                return True
            try:
                spec = SpecifierSet(rp)
            except InvalidSpecifier:
                # Invalid metadata (e.g. '>3.4.*') on old releases: treat that file as unsupported.
                continue
            if spec.contains(target, prereleases=True):
                return True
        return False


def _version_from_filename(filename: str) -> Version | None:
    try:
        if filename.endswith(".whl"):
            return parse_wheel_filename(filename)[1]
        if filename.endswith((".tar.gz", ".zip")):
            return parse_sdist_filename(filename)[1]
    except (InvalidWheelFilename, InvalidSdistFilename, InvalidVersion):
        return None
    return None


def fetch_releases(library: str) -> list[Release]:
    """Non-yanked releases of ``library`` from PyPI's JSON simple API (PEP 691)."""
    request = urllib.request.Request(
        PYPI_SIMPLE.format(name=canonicalize_name(library)),
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload: dict[str, Any] = json.load(response)
    by_version: dict[Version, list[str | None]] = {}
    for file in payload.get("files", []):
        if file.get("yanked"):
            continue
        version = _version_from_filename(file["filename"])
        if version is None:
            continue
        by_version.setdefault(version, []).append(file.get("requires-python"))
    return [Release(v, tuple(rps)) for v, rps in sorted(by_version.items())]


def fetch_pythons() -> list[Version]:
    """CPython versions ``uv`` can provide (installed or downloadable), default variant only."""
    entries: list[dict[str, Any]] = json.loads(_run(["uv", "python", "list", "--output-format", "json"]))
    found: set[Version] = set()
    for entry in entries:
        if entry.get("implementation") != "cpython" or entry.get("variant", "default") != "default":
            continue
        try:
            found.add(Version(str(entry.get("version"))))
        except InvalidVersion:
            continue
    return sorted(found)


def select_pythons(
    available: Iterable[Version],
    specifier: str,
) -> list[Version]:
    """One ``major.minor`` per Python minor whose newest release matches ``specifier``.

    A minor with only pre-releases (e.g. the next beta) is skipped. Every
    matching minor is locked; there is no sampling.
    """
    newest_by_minor: dict[tuple[int, int], Version] = {}
    for version in sorted(available):
        if version.is_prerelease or version.is_devrelease:
            continue
        newest_by_minor[(version.major, version.minor)] = version
    spec = SpecifierSet(specifier)
    candidates = [
        Version(f"{major}.{minor}")
        for (major, minor), newest in sorted(newest_by_minor.items())
        if spec.contains(newest, prereleases=True)
    ]
    if not candidates:
        raise LockError(f"no Python versions match {specifier!r}")
    return candidates


def select_versions(
    available: Iterable[Version],
    specifier: str,
    policy: Policy,
    *,
    patch_ranges: Iterable[str] = (),
) -> list[Version]:
    """Apply the spec's version policy to the available releases.

    ``oldest``, ``newest``, newest-of-each-major, lowest release of each
    per-patch ``versions:`` range, and ``policy.include`` are always kept and
    pinned; if the policy yields more than ``policy.max`` versions, the rest
    are sampled evenly.
    """
    spec = SpecifierSet(specifier)
    candidates = sorted(v for v in available if not v.is_prerelease and not v.is_devrelease and spec.contains(v))
    if not candidates:
        raise LockError(f"no releases match {specifier!r}")

    pinned_set: set[Version] = {candidates[0], candidates[-1]}
    newest_by_major: dict[int, Version] = {}
    for version in candidates:
        newest_by_major[version.major] = version
    pinned_set.update(newest_by_major.values())
    for patch_range in patch_ranges:
        range_spec = SpecifierSet(patch_range)
        for version in candidates:
            if range_spec.contains(version):
                pinned_set.add(version)
                break
    candidate_set = set(candidates)
    for raw_ver in policy.include:
        ver = Version(raw_ver)
        if ver not in candidate_set:
            raise LockError(f"policy.include version {raw_ver!r} is not in matching releases")
        pinned_set.add(ver)

    chosen: set[Version] = set(pinned_set)
    firsts: dict[tuple[int, int], Version] = {}
    for version in candidates:
        firsts.setdefault((version.major, version.minor), version)
    chosen.update(firsts.values())
    ordered = sorted(chosen)
    if len(ordered) <= policy.max:
        return ordered

    pinned = sorted(pinned_set)
    rest = [v for v in ordered if v not in pinned_set]
    slots = max(policy.max - len(pinned), 0)
    sampled: list[Version] = []
    if slots == 1 and rest:
        sampled = [rest[len(rest) // 2]]
    elif slots > 1 and rest:
        step = (len(rest) - 1) / (slots - 1)
        sampled = [rest[round(i * step)] for i in range(slots)]
    return sorted(set(pinned + sampled))


def _summarize(text: str) -> str:
    """One readable line out of a (possibly huge) command error output.

    Prefers uv's headline (``× ...`` / ``error: ...``, plus the ``╰─▶`` cause
    line under it), otherwise the last non-empty line (e.g. a traceback's
    exception line).
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "(no error output)"
    summary = lines[-1]
    for i, line in enumerate(lines):
        if line.startswith(("×", "error:", "ERROR")):
            summary = line.removeprefix("×").strip()
            for follow in lines[i + 1 : i + 3]:
                if not follow.startswith(("╰─▶", "├─▶")):
                    break
                summary += f": {follow[3:].strip()}"
            break
    if len(summary) > _SUMMARY_MAX:
        summary = summary[: _SUMMARY_MAX - 3] + "..."
    return summary


def _short_command(cmd: Sequence[str]) -> str:
    return " ".join(Path(c).name if "/" in c else c for c in cmd[:3])


def _run(cmd: Sequence[str], *, stdin: str | None = None, env: Mapping[str, str] | None = None) -> str:
    proc = subprocess.run(
        list(cmd),
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=None if env is None else dict(env),
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise LockError(
            f"{_short_command(cmd)} failed ({proc.returncode}): {_summarize(stderr)}",
            details=f"$ {' '.join(cmd)}\n{stderr}",
        )
    return proc.stdout


def _label(spec: Spec, version: Version) -> str:
    return f"python {version}" if spec.stdlib else f"{spec.library}=={version}"


def introspect_version(spec: Spec, version: Version, python: str) -> dict[str, Any]:
    """Record the targets in a throwaway venv on ``python``.

    ``library[extras]==version`` (plus the spec's ``with:`` requirements, under
    its ``constraints:``) is installed first unless the spec is
    ``stdlib: true``, in which case ``version`` is the Python version itself and
    the targets are read from that interpreter's standard library.
    """
    targets = {p.name: p.target for p in spec.patches}
    with tempfile.TemporaryDirectory(prefix="patchgen-") as tmp:
        venv = Path(tmp) / "venv"
        _run(["uv", "venv", "--quiet", "--python", python, str(venv)])
        venv_python = str(venv / "bin" / "python")
        if not spec.stdlib:
            install = ["uv", "pip", "install", "--quiet", "--python", venv_python]
            install += [spec.install_requirement(str(version)), *spec.with_requirements]
            if spec.constraints:
                constraints = Path(tmp) / "constraints.txt"
                constraints.write_text("\n".join(spec.constraints) + "\n", encoding="utf-8")
                install += ["--constraint", str(constraints)]
            env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV}
            _run(install, env=env)
        output = _run(
            [venv_python, str(INTROSPECT)],
            stdin=json.dumps({"top": spec.top_module, "targets": targets}),
        )
    result: dict[str, Any] = json.loads(output)
    if "error" in result:
        raise LockError(f"{_label(spec, version)}: {result['error']}")
    return result


def _upper_clause(clause: Specifier) -> str | None:
    """The part of ``clause`` that bounds versions from above (or excludes some), if any."""
    if clause.operator in ("<", "<=", "!="):
        return str(clause)
    if clause.operator == "==" and clause.version.endswith(".*"):
        return str(clause)
    if clause.operator == "~=":
        bound = list(Version(clause.version).release[:-1])
        bound[-1] += 1
        return "<" + ".".join(map(str, bound))
    return None


def _floor_suggestion(min_locked: Version, specifier: str | None) -> str:
    """``>=min_locked`` plus the upper bounds/exclusions of ``specifier``."""
    clauses = [f">={min_locked}"]
    try:
        original = SpecifierSet(specifier or "")
    except InvalidSpecifier:
        original = SpecifierSet()
    for clause in sorted(original, key=str):
        upper = _upper_clause(clause)
        if upper is not None and upper not in clauses:
            clauses.append(upper)
    return ", ".join(clauses)


def _write_details(spec: Spec, failures: Sequence[tuple[Version, str]]) -> Path | None:
    """Keep the full error output of skipped versions in a temp log; returns its path."""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            prefix=f"patchgen-lock-{canonicalize_name(spec.library)}-",
            suffix=".log",
            delete=False,
            encoding="utf-8",
        ) as fh:
            for version, text in failures:
                fh.write(f"===== {_label(spec, version)} =====\n{text}\n\n")
            return Path(fh.name)
    except OSError:
        return None


def _lock_selected(
    spec: Spec,
    selected: Sequence[Version],
    *,
    python_for: Callable[[Version], str],
    log: Any,
    specifier: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Introspect every selected version; returns ``(versions, exports)`` lock sections."""
    versions: dict[str, Any] = {}
    exports: dict[str, list[str]] = {}
    errors: list[str] = []
    skipped: list[tuple[Version, LockError]] = []
    for version in selected:
        label = _label(spec, version)
        log(f"  locking {label}")
        try:
            result = introspect_version(spec, version, python_for(version))
        except LockError as exc:
            log(f"  skipping {label}: {exc}")
            skipped.append((version, exc))
            continue
        # A required target whose module exists but cannot be imported (missing
        # non-library dependency, removed stdlib API, ...) makes the version
        # unusable; that is a skip, not a "target not found" error.
        unimportable = [
            f"{patch.target}: {result['targets'][patch.name]['unimportable']}"
            for patch in spec.patches
            if patch.required and patch.in_range(str(version)) and "unimportable" in result["targets"][patch.name]
        ]
        if unimportable:
            exc = LockError(f"unimportable: {unimportable[0]}")
            log(f"  skipping {label}: {exc}")
            skipped.append((version, exc))
            continue
        if result.get("exports_error"):
            log(f"  warning: {label}: exports collection failed ({result['exports_error']}); version kept")
        exports[str(version)] = result.get("exports", [])
        entries: dict[str, Any] = {}
        for patch in spec.patches:
            entry: dict[str, Any] = result["targets"][patch.name]
            if "unimportable" in entry:
                entries[patch.name] = {"absent": True, "error": entry["unimportable"]}
                continue
            if "error" in entry:
                if patch.required and patch.in_range(str(version)):
                    errors.append(f"{patch.target} ({spec.subject} {version}): {entry['error']}")
                entries[patch.name] = {"absent": True, "error": entry["error"]}
                continue
            entries[patch.name] = entry
        versions[str(version)] = entries
    if skipped:
        log(f"  skipped {len(skipped)} version(s): {', '.join(str(v) for v, _ in skipped)}")
        for version, exc in skipped:
            log(f"    {_label(spec, version)}: {exc}")
        details = [(v, exc.details) for v, exc in skipped if exc.details]
        if details and (log_path := _write_details(spec, details)) is not None:
            log(f"  full error output: {log_path}")
    if not versions:
        reasons = "\n".join(f"  {v}: {exc}" for v, exc in skipped)
        raise LockError(f"all selected versions of {spec.subject} failed to lock:\n{reasons}")
    if errors:
        raise LockError("\n".join(errors))
    if skipped:
        min_locked = min(Version(v) for v in versions)
        if any(v < min_locked for v, _ in skipped):
            floor = "requires-python" if spec.stdlib else "instruments"
            log(
                f"  warning: spec/{floor} floor is below {min_locked}; "
                f'consider setting `versions: "{_floor_suggestion(min_locked, specifier)}"` in patches.yaml'
            )
    return versions, exports


def _trim_exports(spec: Spec, versions: Mapping[str, Any], exports: Mapping[str, list[str]]) -> dict[str, list[str]]:
    """Filter each version's ``exports`` to symbols actually referenced by the lock or spec."""
    ref_re = re.compile(rf"(?<![\w.]){re.escape(spec.top_module)}\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
    referenced: set[str] = set()
    for patch in spec.patches:
        if patch.instance_type:
            referenced.update(ref_re.findall(patch.instance_type))
        for tp in patch.types.values():
            referenced.update(ref_re.findall(tp))
    for entries in versions.values():
        for entry in entries.values():
            if not isinstance(entry, Mapping):
                continue
            for key in ("owner", "owner_defined"):
                val = entry.get(key)
                if isinstance(val, str):
                    referenced.update(ref_re.findall(val))
            for param in entry.get("params") or ():
                for member in param[3] or ():
                    referenced.update(ref_re.findall(member))
            for member in entry.get("return") or ():
                referenced.update(ref_re.findall(member))
    return {v: [s for s in syms if s in referenced] for v, syms in exports.items()}


def _load_existing_lock(path: Path) -> dict[str, Any]:
    """The existing lock at ``path``, or ``{}`` if there is none.

    An unreadable or unsupported lock is an error rather than "no lock": silently ignoring it
    would drop the recorded ``python`` and the guard against losing locked versions.
    """
    if not path.is_file():
        return {}
    try:
        return load_lock(path.read_text(encoding="utf-8"))
    except (OSError, LockFormatError) as exc:
        raise LockError(f"cannot read existing lock {path}: {exc}") from exc


def build_lock(
    spec: Spec,
    *,
    python: str | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Build the lock dictionary for ``spec``.

    ``python`` selects the interpreter for the throwaway environments (default:
    the ``python`` recorded in an existing lock, or the current interpreter).
    With ``stdlib: true`` it is ignored: every supported Python minor is locked
    on its own interpreter.
    """
    existing_lock = _load_existing_lock(spec.lock_path)
    lock: dict[str, Any] = {"library": spec.library, "generated_with": GENERATED_WITH}
    if spec.stdlib:
        if python is not None:
            log(f"{spec.library}: --python is ignored with stdlib: true")
        specifier = spec.versions or default_python_versions(spec.package_dir)
        selected = select_pythons(fetch_pythons(), specifier)
        log(f"{spec.library}: python {specifier} -> {', '.join(map(str, selected))} (stdlib)")
        versions, exports = _lock_selected(spec, selected, python_for=str, log=log, specifier=specifier)
        lock["stdlib"] = True
    else:
        existing_python = existing_lock.get("python")
        if not isinstance(existing_python, str) or not existing_python:
            existing_python = None
        if python is None and existing_python is not None:
            python = existing_python
            log(f"{spec.library}: using python {python} from {spec.lock_path.name}")
        elif python is not None and existing_python is not None and python != existing_python:
            log(f"  warning: --python {python} differs from {spec.lock_path.name} python {existing_python}")
        python = python or f"{sys.version_info.major}.{sys.version_info.minor}"
        specifier = spec.versions or default_versions(spec.package_dir, spec.library)
        releases = [r for r in fetch_releases(spec.library) if r.supports(python)]
        selected = select_versions(
            (r.version for r in releases),
            specifier,
            spec.policy,
            patch_ranges=[p.versions for p in spec.patches if p.versions is not None],
        )
        extras_str = f"[{','.join(spec.extras)}]" if spec.extras else ""
        log(f"{spec.library}{extras_str}: {specifier} -> {', '.join(map(str, selected))} (python {python})")
        if spec.with_requirements or spec.constraints:
            log(f"  with {list(spec.with_requirements)}, constraints {list(spec.constraints)}")
        versions, exports = _lock_selected(spec, selected, python_for=lambda _v: python, log=log, specifier=specifier)
        lock["python"] = python
        if spec.extras:
            lock["extras"] = list(spec.extras)
        if spec.with_requirements:
            lock["with"] = list(spec.with_requirements)
        if spec.constraints:
            lock["constraints"] = list(spec.constraints)

    existing_versions = existing_lock.get("versions")
    if isinstance(existing_versions, Mapping):
        spec_set = SpecifierSet(specifier)
        selected_set = set(selected)
        locked_parsed = [Version(v) for v in versions]
        dropped = sorted(
            old_v
            for v in existing_versions
            if (old_v := Version(v)) not in locked_parsed
            and spec_set.contains(old_v, prereleases=True)
            and (
                old_v in selected_set
                or not any((lv.major, lv.minor) == (old_v.major, old_v.minor) and lv > old_v for lv in locked_parsed)
            )
        )
        if dropped:
            raise LockError(f"re-lock dropped previously locked version(s): {', '.join(map(str, dropped))}")

    log(f"  {len(spec.patches)} targets")
    lock.update({"specifier": specifier, "versions": versions, "exports": _trim_exports(spec, versions, exports)})
    return lock


def write_lock(
    spec: Spec,
    *,
    python: str | None = None,
) -> Path:
    lock = build_lock(spec, python=python)
    lock_path = spec.lock_path
    lock_path.write_text(dump_lock(lock), encoding="utf-8")
    return lock_path
