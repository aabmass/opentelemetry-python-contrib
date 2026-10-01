# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Loading and validating ``patches.yaml``."""

from __future__ import annotations

import builtins
import keyword
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

SCOPES = ("class", "instance", "opt_out")
RESERVED_FIELDS = ("raw_args", "raw_kwargs", "var_args", "var_kwargs", "_wrapped", "_sig", "call")
_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_TARGET_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
_EXTRA_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")


class SpecError(ValueError):
    """Raised for invalid ``patches.yaml`` content."""


@dataclass(frozen=True)
class Policy:
    max: int = 8
    include: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatchSpec:
    name: str
    target: str
    scope: tuple[str, ...] = ("class",)
    types: dict[str, str] = field(default_factory=dict)
    required: bool = True
    instance_type: str | None = None
    versions: str | None = None

    def in_range(self, version: str) -> bool:
        """Whether ``version`` is covered by this patch's ``versions:`` range."""
        if self.versions is None:
            return True
        return SpecifierSet(self.versions).contains(Version(version), prereleases=True)

    @property
    def module(self) -> str:
        return self.target.partition(":")[0]

    @property
    def qualname(self) -> str:
        return self.target.partition(":")[2]

    @property
    def attr(self) -> str:
        return self.qualname.rsplit(".", 1)[-1]

    @property
    def class_name(self) -> str:
        return "".join(part.capitalize() for part in self.name.split("_") if part)


@dataclass(frozen=True)
class Spec:
    path: Path
    library: str
    versions: str | None
    policy: Policy
    module: str
    patches: tuple[PatchSpec, ...]
    imports: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()
    stdlib: bool = False
    # Lock-time only: extra requirements installed alongside the library, and
    # uv constraints (``with:`` / ``constraints:`` in patches.yaml).
    with_requirements: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()

    @property
    def package_dir(self) -> Path:
        return self.path.parent

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name.removesuffix(".yaml") + ".lock.json")

    @property
    def top_module(self) -> str:
        return self.patches[0].module.split(".")[0]

    @property
    def subject(self) -> str:
        """What the locked version numbers refer to: the library, or ``python`` for stdlib specs."""
        return "python" if self.stdlib else self.library

    def install_requirement(self, version: str) -> str:
        """``library[extras]==version`` for ``uv pip install``."""
        extras = f"[{','.join(self.extras)}]" if self.extras else ""
        return f"{self.library}{extras}=={version}"


def find_specs(package_dir: Path) -> list[Path]:
    """``patches.yaml`` first, then any ``patches.<name>.yaml`` in name order."""
    found: list[Path] = []
    main = package_dir / "patches.yaml"
    if main.is_file():
        found.append(main)
    found.extend(sorted(p for p in package_dir.glob("patches.*.yaml") if p.is_file()))
    return found


def _expect(cond: bool, path: Path, message: str) -> None:
    if not cond:
        raise SpecError(f"{path}: {message}")


def _parse_patch(path: Path, raw: Any, index: int) -> PatchSpec:
    _expect(isinstance(raw, dict), path, f"patches[{index}] must be a mapping")
    data: dict[str, Any] = raw
    known = {"name", "target", "scope", "types", "required", "instance_type", "versions"}
    unknown = sorted(set(data) - known)
    _expect(not unknown, path, f"patches[{index}]: unknown keys {unknown}")

    name = data.get("name")
    _expect(isinstance(name, str) and bool(_NAME_RE.match(name)), path, f"patches[{index}].name must be snake_case")
    assert isinstance(name, str)
    if hasattr(builtins, name) and not name.startswith("_"):
        raise SpecError(
            f"{path}: patch {name}: name shadows a Python built-in; choose a more specific name (e.g. prefix with the target class name)"
        )

    target = data.get("target")
    _expect(
        isinstance(target, str) and bool(_TARGET_RE.match(target)),
        path,
        f"patch {name}: target must be 'module:qualname'",
    )
    assert isinstance(target, str)

    scope_raw = data.get("scope", ["class"])
    if isinstance(scope_raw, str):
        scope_raw = [scope_raw]
    _expect(
        isinstance(scope_raw, list) and bool(scope_raw) and all(s in SCOPES for s in scope_raw),
        path,
        f"patch {name}: scope must be a non-empty list of {SCOPES}",
    )
    scope = tuple(s for s in SCOPES if s in scope_raw)
    if "opt_out" in scope:
        _expect("class" in scope, path, f"patch {name}: scope opt_out requires class")
        _expect("instance" not in scope, path, f"patch {name}: scope opt_out and instance are mutually exclusive")

    types_raw = data.get("types") or {}
    _expect(
        isinstance(types_raw, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in types_raw.items()),
        path,
        f"patch {name}: types must map parameter names to type expressions",
    )

    required = data.get("required", True)
    _expect(isinstance(required, bool), path, f"patch {name}: required must be a boolean")

    instance_type = data.get("instance_type")
    _expect(
        instance_type is None or isinstance(instance_type, str),
        path,
        f"patch {name}: instance_type must be a type expression",
    )

    versions = data.get("versions")
    if versions is not None:
        _expect(isinstance(versions, str), path, f"patch {name}: versions must be a specifier string")
        try:
            SpecifierSet(versions)
        except InvalidSpecifier as exc:
            raise SpecError(f"{path}: patch {name}: invalid versions {versions!r}") from exc

    return PatchSpec(
        name=name,
        target=target,
        scope=scope,
        types=dict(types_raw),
        required=required,
        instance_type=instance_type,
        versions=versions,
    )


def _requirement_list(path: Path, data: dict[str, Any], key: str) -> tuple[str, ...]:
    """Validate an optional top-level list of PEP 508 requirement strings."""
    raw = data.get(key)
    if raw is None:
        return ()
    _expect(
        isinstance(raw, list) and all(isinstance(r, str) and bool(r.strip()) for r in raw),
        path,
        f"{key} must be a list of requirement strings",
    )
    for req in raw:
        try:
            Requirement(req)
        except InvalidRequirement as exc:
            raise SpecError(f"{path}: {key}: invalid requirement {req!r}") from exc
    return tuple(raw)


def _probe_versions(specs: list[SpecifierSet]) -> set[Version]:
    probes: set[Version] = {Version("0.0.1")}
    for spec in specs:
        for clause in spec:
            raw = clause.version.removesuffix(".*")
            try:
                v = Version(raw)
            except InvalidVersion:
                continue
            probes.update(
                {
                    v,
                    Version(f"{v.major}.{v.minor}.{v.micro + 1}"),
                    Version(f"{v.major}.{v.minor + 1}.0"),
                    Version(f"{v.major + 1}.0.0"),
                }
            )
            if v.micro > 0:
                probes.add(Version(f"{v.major}.{v.minor}.{v.micro - 1}"))
            if v.minor > 0:
                probes.add(Version(f"{v.major}.{v.minor - 1}.0"))
            if v.major > 0:
                probes.add(Version(f"{v.major - 1}.0.0"))
    return probes


def load_spec(path: Path) -> Spec:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    _expect(isinstance(raw, dict), path, "must be a mapping")
    data: dict[str, Any] = raw
    unknown = sorted(
        set(data)
        - {"library", "stdlib", "extras", "with", "constraints", "versions", "policy", "module", "imports", "patches"}
    )
    _expect(not unknown, path, f"unknown keys {unknown}")

    library = data.get("library")
    _expect(isinstance(library, str) and bool(library), path, "library is required")
    assert isinstance(library, str)

    stdlib = data.get("stdlib", False)
    _expect(isinstance(stdlib, bool), path, "stdlib must be a boolean")

    extras_raw = data.get("extras")
    if extras_raw is None:
        extras_raw = []
    _expect(
        isinstance(extras_raw, list) and all(isinstance(e, str) and bool(_EXTRA_RE.match(e)) for e in extras_raw),
        path,
        "extras must be a list of extra names (PEP 685)",
    )
    extras = tuple(extras_raw)
    _expect(not (stdlib and extras), path, "extras cannot be combined with stdlib: true (nothing is installed)")

    with_requirements = _requirement_list(path, data, "with")
    constraints = _requirement_list(path, data, "constraints")
    _expect(
        not (stdlib and (with_requirements or constraints)),
        path,
        "with/constraints cannot be combined with stdlib: true (nothing is installed)",
    )

    versions = data.get("versions")
    if versions is not None:
        _expect(isinstance(versions, str), path, "versions must be a specifier string")
        try:
            SpecifierSet(versions)
        except InvalidSpecifier as exc:
            raise SpecError(f"{path}: invalid versions {versions!r}") from exc

    policy_raw = data.get("policy") or {}
    _expect(isinstance(policy_raw, dict), path, "policy must be a mapping")
    _expect(
        not (stdlib and policy_raw), path, "policy is not supported with stdlib: true (every Python minor is locked)"
    )
    unknown_policy = sorted(set(policy_raw) - {"max", "include"})
    _expect(not unknown_policy, path, f"policy: unknown keys {unknown_policy}")
    max_versions = policy_raw.get("max", 8)
    _expect(isinstance(max_versions, int) and max_versions >= 1, path, "policy.max must be a positive integer")
    include_raw = policy_raw.get("include")
    if include_raw is None:
        include_raw = []
    _expect(
        isinstance(include_raw, list) and all(isinstance(v, str) for v in include_raw),
        path,
        "policy.include must be a list of version strings",
    )
    for v in include_raw:
        try:
            Version(v)
        except InvalidVersion as exc:
            raise SpecError(f"{path}: policy.include: invalid version {v!r}") from exc
    include = tuple(include_raw)

    module = data.get("module", "_patches")
    _expect(
        isinstance(module, str) and bool(_NAME_RE.match(module)) and not keyword.iskeyword(module),
        path,
        "module must be a Python module name",
    )

    imports_raw = data.get("imports")
    if imports_raw is None:
        imports_raw = []
    _expect(
        isinstance(imports_raw, list)
        and all(
            isinstance(m, str)
            and bool(m)
            and all(part.isidentifier() and not keyword.iskeyword(part) for part in m.split("."))
            for m in imports_raw
        ),
        path,
        "imports must be a list of Python module names",
    )
    imports = tuple(imports_raw)

    patches_raw = data.get("patches")
    _expect(isinstance(patches_raw, list) and bool(patches_raw), path, "patches must be a non-empty list")
    assert isinstance(patches_raw, list)
    patches = tuple(_parse_patch(path, p, i) for i, p in enumerate(patches_raw))
    names = [p.name for p in patches]
    _expect(len(set(names)) == len(names), path, "patch names must be unique")
    tops = {p.module.split(".")[0] for p in patches}
    _expect(len(tops) == 1, path, f"all targets must live in one top-level module, got {sorted(tops)}")
    by_target: dict[str, list[PatchSpec]] = {}
    for patch in patches:
        by_target.setdefault(patch.target, []).append(patch)
    for target, group in by_target.items():
        if len(group) == 1:
            continue
        _expect(
            all(p.versions is not None for p in group),
            path,
            f"patches sharing target {target} must each declare non-overlapping `versions:`",
        )
        spec_sets = [SpecifierSet(p.versions or "") for p in group]
        overlap = any(
            sum(1 for s in spec_sets if s.contains(v, prereleases=True)) > 1 for v in _probe_versions(spec_sets)
        )
        _expect(
            not overlap,
            path,
            f"patches sharing target {target} have overlapping `versions:` ranges",
        )

    return Spec(
        path=path,
        library=library,
        versions=versions,
        policy=Policy(max=max_versions, include=include),
        module=module,
        patches=patches,
        imports=imports,
        extras=extras,
        stdlib=stdlib,
        with_requirements=with_requirements,
        constraints=constraints,
    )


def default_versions(package_dir: Path, library: str) -> str:
    """Specifier for ``library`` from the ``instruments`` (or ``instruments-any``) extra."""
    pyproject = tomllib.loads((package_dir / "pyproject.toml").read_text(encoding="utf-8"))
    extras: dict[str, list[str]] = pyproject.get("project", {}).get("optional-dependencies", {})
    wanted = canonicalize_name(library)
    for extra in ("instruments", "instruments-any"):
        for line in extras.get(extra, []):
            req = Requirement(line)
            if canonicalize_name(req.name) == wanted:
                return str(req.specifier)
    raise SpecError(f"{package_dir}: no `versions` in spec and {library} is not in the instruments extras")


def default_python_versions(package_dir: Path) -> str:
    """Specifier for ``stdlib: true`` specs from ``project.requires-python``."""
    pyproject = tomllib.loads((package_dir / "pyproject.toml").read_text(encoding="utf-8"))
    requires_python = pyproject.get("project", {}).get("requires-python")
    if not isinstance(requires_python, str) or not requires_python:
        raise SpecError(f"{package_dir}: no `versions` in spec and pyproject.toml has no `requires-python`")
    return requires_python


def output_path(spec: Spec) -> Path:
    """Where the generated module for ``spec`` lives."""
    base = spec.package_dir / "src" / "opentelemetry" / "instrumentation"
    candidates = sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))
    if len(candidates) != 1:
        raise SpecError(f"{base}: expected exactly one instrumentation package, found {[p.name for p in candidates]}")
    return candidates[0] / f"{spec.module}.py"
