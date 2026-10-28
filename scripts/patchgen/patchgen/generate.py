# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""``patchgen generate``: a pure function of ``patches.yaml`` + ``patches.lock.json``.

No network, no installed library, deterministic output. The rendered module is formatted and
linted with the pinned ruff (``RUFF_VERSION``) so it is byte-identical to what pre-commit expects.
"""

from __future__ import annotations

import builtins
import functools
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import tomllib
import yaml
from packaging.version import Version

from . import GENERATED_WITH
from .lockfile import LockFormatError, canonical_lock_text, load_lock
from .spec import RESERVED_FIELDS, PatchSpec, Spec, find_specs, load_spec, output_path

TEMPLATES = Path(__file__).with_name("templates")
_VAR_KINDS = {"VAR_POSITIONAL": "var_args", "VAR_KEYWORD": "var_kwargs"}
_VAR_TYPES = {"var_args": "tuple[Any, ...]", "var_kwargs": "dict[str, Any]"}
_BUILTIN_NAMES = frozenset(dir(builtins))

# Generated modules are formatted and linted with exactly the ruff that pre-commit runs, using the
# repo's configuration, so pre-commit never rewrites them and `generate` stays idempotent. Keep in
# sync with the `ruff-pre-commit` rev in .pre-commit-config.yaml and the pin in cli.py (both tested).
RUFF_VERSION = "0.16.1"
RUFF_CONFIG = Path(__file__).resolve().parents[3] / "pyproject.toml"


class GenerateError(ValueError):
    """Raised when the spec and lock cannot be turned into a module."""


@dataclass(frozen=True)
class FieldModel:
    name: str
    type: str
    missing: bool
    comment: str | None
    source: str  # "param", "var_args" or "var_kwargs"

    @property
    def annotation(self) -> str:
        return f"{self.type} | Missing" if self.missing else self.type


@dataclass(frozen=True)
class AnyGap:
    """A generated parameter or return type that is (or contains) ``Any`` without a ``types:`` override."""

    patch: str
    name: str  # parameter name, "return", or "instance" (the wrapper's instance type)
    type: str
    reasons: tuple[str, ...]

    def _what(self) -> str:
        if self.name == "return":
            return "return type"
        if self.name == "instance":
            return "instance type"
        return f"parameter {self.name!r}"

    def describe(self, spec: Spec) -> str:
        remedy = "instance_type: ..." if self.name == "instance" else f"types: {{{self.name}: ...}}"
        return (
            f"{spec.library}: patch {self.patch}: {self._what()} is {self.type} ({'; '.join(self.reasons)}); "
            f"add {remedy} to {spec.path.name}"
        )


@dataclass(frozen=True)
class PatchModel:
    spec: PatchSpec
    fields: tuple[FieldModel, ...]
    is_async: bool
    method: bool
    result_type: str
    instance_type: str
    shapes: tuple[tuple[tuple[str, str, bool], ...], ...]
    notes: tuple[str, ...]
    any_gaps: tuple[AnyGap, ...] = ()
    # Present in only part of the locked range or optional (required: false): | None = None in instrument_instance().
    partial: bool = False
    # Modules named by typeshed-filled lock annotations (lock `imports`), imported if still referenced.
    type_modules: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def args_class(self) -> str:
        return f"{self.spec.class_name}Args"

    @property
    def wrapper_class(self) -> str:
        return f"{self.spec.class_name}Wrapper"

    @property
    def shapes_const(self) -> str:
        return f"_SHAPES_{self.spec.name.upper()}"

    @property
    def wrapped_type(self) -> str:
        if self.is_async:
            return f"Callable[..., Awaitable[{self.result_type}]]"
        return f"Callable[..., {self.result_type}]"

    @property
    def return_type(self) -> str:
        return f"Awaitable[{self.result_type}]" if self.is_async else self.result_type

    @property
    def has_var_args(self) -> bool:
        return any(f.source == "var_args" for f in self.fields)

    @property
    def call_params(self) -> tuple[FieldModel, ...]:
        """Named target parameters accepted as keyword overrides by ``args.call()``."""
        return tuple(f for f in self.fields if f.source == "param")


@dataclass(frozen=True)
class InstanceGroup:
    instance_type: str
    patches: tuple[PatchModel, ...]


# --------------------------------------------------------------------------- union


def _sorted_versions(lock: dict[str, Any]) -> list[str]:
    return sorted(lock["versions"], key=Version)


def _presence(present: list[str], versions: list[str], library: str) -> str:
    idx = [versions.index(v) for v in present]
    contiguous = idx == list(range(idx[0], idx[-1] + 1))
    if contiguous and idx[-1] == len(versions) - 1:
        return f"{library} >= {present[0]}"
    if contiguous and idx[0] == 0:
        return f"{library} <= {present[-1]}"
    return f"{library} " + ", ".join(present)


# PEP 484 numeric promotion: `float` accepts `int`, `complex` accepts both. ruff (PYI041) rejects
# the redundant members, so they are dropped from generated unions.
_NUMERIC_SUPERSEDES = {"float": ("int",), "complex": ("float", "int")}


def _collapse_numeric(members: list[str]) -> list[str]:
    """``members`` without ``int`` next to ``float``, and without ``float``/``int`` next to ``complex``."""
    drop = {narrow for wide, narrows in _NUMERIC_SUPERSEDES.items() if wide in members for narrow in narrows}
    return [m for m in members if m not in drop]


class _TypeChecker:
    """Keeps library references only if they exist in every locked version."""

    def __init__(
        self,
        top: str,
        exports: dict[str, list[str]],
        versions: list[str],
    ) -> None:
        self.top = top
        self.common: set[str] = set.intersection(*(set(exports.get(v, [])) for v in versions)) if versions else set()
        self.ref_re = re.compile(rf"(?<![\w.]){re.escape(top)}\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
        self.notes: list[str] = []

    def missing(self, member: str) -> list[str]:
        """Library references in ``member`` that are not exported in every locked version."""
        return sorted({f"{self.top}.{m}" for m in self.ref_re.findall(member) if m not in self.common})

    def check(self, member: str, where: str) -> str:
        missing = self.missing(member)
        if missing:
            self.notes.append(f"{where}: {', '.join(missing)} not in every locked version; using Any")
            return "Any"
        return member

    def union(self, member_lists: list[list[str] | None], where: str) -> str:
        annotated = [m for m in member_lists if m is not None]
        if not annotated:
            return "Any"
        out: list[str] = []
        for members in annotated:
            for member in members:
                checked = self.check(member, where)
                if checked not in out:
                    out.append(checked)
        if "Any" in out:
            return "Any"
        result = " | ".join(_collapse_numeric(out))
        if len(annotated) < len(member_lists):
            self.notes.append(f"{where}: unannotated in some locked versions; using {result}")
        return result


# A generic whose type arguments are all Any / `...`: what `lock` renders for an unparameterized
# annotation such as bare `dict`, `tuple`, `Callable` or a generic library class.
_UNPARAMETERIZED_RE = re.compile(r"^[A-Za-z_][\w.]*\[(?:Any|\.\.\.)(?:, (?:Any|\.\.\.))*\]$")
_RENDERED_AS_ANY = " rendered as Any"
_USING_ANY = "; using Any"


def _field_gap_reasons(
    spec: Spec,
    checker: _TypeChecker,
    annotations: dict[str, list[str] | None],
    entry_notes: dict[str, list[str]],
    where: str,
    check_where: str,
    rendered: str,
) -> tuple[str, ...]:
    """Why ``rendered`` (for a parameter or return without a ``types:`` override) is a gap."""
    if all(a is None for a in annotations.values()):
        return ("unannotated in all locked versions",)

    versions = sorted(annotations, key=Version)
    reasons: list[str] = []

    prefix = f"{where}: "
    by_note: dict[str, list[str]] = {}
    for version in versions:
        for n in entry_notes.get(version, []):
            if n.startswith(prefix) and n.endswith((_RENDERED_AS_ANY, _USING_ANY)):
                msg = n[len(prefix) :].removesuffix(_USING_ANY)
                seen = by_note.setdefault(msg, [])
                if version not in seen:
                    seen.append(version)

    if by_note:
        unannotated = [v for v in versions if annotations[v] is None]
        if unannotated:
            reasons.append(f"unannotated in {_presence(unannotated, versions, spec.subject)}")
        for msg, in_versions in by_note.items():
            scope = "" if in_versions == versions else f" in {_presence(in_versions, versions, spec.subject)}"
            reasons.append(f"{msg}{scope}")

    check_prefix = f"{check_where}: "
    for n in checker.notes:
        if n.startswith(check_prefix) and n.endswith(_USING_ANY):
            reasons.append(n[len(check_prefix) : -len(_USING_ANY)])

    if rendered != "Any" and not spec.stdlib:
        for version in versions:
            for member in annotations[version] or ():
                if _UNPARAMETERIZED_RE.match(member):
                    reasons.append(f"{member} has only Any type arguments")

    return tuple(dict.fromkeys(reasons))


def _entries(spec: Spec, patch: PatchSpec, lock: dict[str, Any], versions: list[str]) -> dict[str, dict[str, Any]]:
    """Per-version lock entries for ``patch``, newest first, skipping absent versions."""
    out: dict[str, dict[str, Any]] = {}
    for version in reversed(versions):
        if not patch.in_range(version):
            continue
        entry = lock["versions"][version].get(patch.name)
        if entry is None:
            raise GenerateError(f"{spec.lock_path.name}: no entry for {patch.name} at {version}; rerun lock")
        if entry.get("absent"):
            continue
        out[version] = entry
    if not out:
        raise GenerateError(f"{patch.name}: target is absent in every locked version")
    return out


def _is_public_path(path: str, top: str) -> bool:
    return path.startswith(f"{top}.") and not any(part.startswith("_") for part in path.split("."))


def _owner_type(checker: _TypeChecker, entries: dict[str, dict[str, Any]]) -> tuple[str, tuple[str, ...]]:
    """The instance type of a method patch, from its lock entries (newest first).

    Uses the first public ``owner`` spelling that type-checks in every locked version, else the
    owner's defining path (lock ``owner_defined``) if it is public and type-checks. Otherwise
    returns ``("Any", reasons)``.
    """
    owners = list(dict.fromkeys(e.get("owner") for e in entries.values()))
    public = [o for o in owners if o]
    for owner in public:
        if not checker.missing(owner):
            return owner, ()
    defined = list(dict.fromkeys(e.get("owner_defined") for e in entries.values()))
    candidate = defined[0] if len(defined) == 1 else None
    if candidate and _is_public_path(candidate, checker.top) and not checker.missing(candidate):
        return candidate, ()

    reasons: list[str] = []
    if public:
        reasons.append(f"owner {', '.join(public)} not in every locked version")
    if None in owners:
        where = "any" if not public else "some"
        reasons.append(f"owner not publicly exported in {where} locked version{'s' if public else ''}")
    if candidate:
        reasons.append(f"defined as {candidate}, which is not public in every locked version")
    elif len(defined) > 1 and all(defined):
        reasons.append(f"defined as {', '.join(str(d) for d in defined)} across locked versions")
    return "Any", tuple(reasons)


@functools.cache
def _typecheck_python() -> str | None:
    """``tool.pyright.pythonVersion`` of the repo: the Python generated modules are type-checked on."""
    try:
        config = tomllib.loads(RUFF_CONFIG.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    value = config.get("tool", {}).get("pyright", {}).get("pythonVersion")
    return value if isinstance(value, str) else None


def _typecheck_python_type(spec: Spec, lock: dict[str, Any], type_: str, where: str, notes: list[str]) -> str:
    """``type_`` if it exists on the type-checking Python, else ``object`` (stdlib specs only).

    A stdlib patch limited by ``versions:`` to newer Pythons can name a class (e.g.
    ``asyncio.TaskGroup``) that pyright's ``pythonVersion`` does not have. That Python's exports are
    taken from the newest locked version not above it (or the oldest locked version).
    """
    floor = _typecheck_python()
    exports: dict[str, list[str]] = lock.get("exports", {})
    if floor is None:
        return type_
    versions = _sorted_versions(lock)
    older = [v for v in versions if Version(v) <= Version(floor)]
    target = older[-1] if older else versions[0]
    if target not in exports:
        return type_
    missing = _TypeChecker(spec.top_module, exports, [target]).missing(type_)
    if not missing:
        return type_
    notes.append(f"{where}: {', '.join(missing)} not in python {floor} (the type-checking target); using object")
    return "object"


def build_patch(spec: Spec, patch: PatchSpec, lock: dict[str, Any]) -> PatchModel:
    versions = _sorted_versions(lock)
    entries = _entries(spec, patch, lock, versions)
    present_versions = sorted(entries, key=Version)
    checker = _TypeChecker(spec.top_module, lock.get("exports", {}), present_versions)

    bindings = {e["binding"] for e in entries.values()}
    if len(bindings) != 1:
        raise GenerateError(f"{patch.name}: target is a method in some versions and a function in others")
    binding = bindings.pop()
    method = binding in ("method", "classmethod")
    if binding != "method" and ("instance" in patch.scope or "opt_out" in patch.scope):
        raise GenerateError(f"{patch.name}: instance and opt_out scopes require a method target")

    # Ordered union: newest version's order first, then older-only params.
    order: list[str] = []
    per_field: dict[str, dict[str, list[str] | None]] = {}
    for version, entry in entries.items():
        params: list[list[Any]] = entry["params"][1:] if method else entry["params"]
        for name, kind, _default, annotation in params:
            key = _VAR_KINDS.get(kind, name)
            if key in RESERVED_FIELDS and kind not in _VAR_KINDS:
                raise GenerateError(f"{patch.name}: parameter {name!r} collides with a generated field name")
            if key not in per_field:
                order.append(key)
                per_field[key] = {}
            per_field[key][version] = annotation

    entry_notes: dict[str, list[str]] = {v: e.get("notes", []) for v, e in entries.items()}
    gaps: list[AnyGap] = []

    def gap(name: str, rendered: str, annotations: dict[str, list[str] | None]) -> None:
        where = "return" if name == "return" else f"parameter {name!r}"
        check_where = f"{patch.name} return" if name == "return" else f"{patch.name}.{name}"
        reasons = _field_gap_reasons(spec, checker, annotations, entry_notes, where, check_where, rendered)
        if reasons:
            gaps.append(AnyGap(patch=patch.name, name=name, type=rendered, reasons=reasons))

    fields: list[FieldModel] = []
    for key in order:
        seen = per_field[key]
        present = sorted(seen, key=Version)
        missing = len(present) != len(present_versions)
        if key in _VAR_TYPES:
            type_ = _VAR_TYPES[key]
            source = key
        else:
            source = "param"
            if key in patch.types:
                type_ = patch.types[key]
            else:
                type_ = checker.union([seen[v] for v in entries if v in seen], f"{patch.name}.{key}")
                gap(key, type_, seen)
        comment = _presence(present, present_versions, spec.subject) if missing else None
        fields.append(FieldModel(name=key, type=type_, missing=missing, comment=comment, source=source))

    unknown_overrides = sorted(set(patch.types) - set(order) - {"return"})
    if unknown_overrides:
        raise GenerateError(f"{patch.name}: types override unknown parameters {unknown_overrides}")

    if "return" in patch.types:
        result_type = patch.types["return"]
    else:
        result_type = checker.union([e["return"] for e in entries.values()], f"{patch.name} return")
        gap("return", result_type, {v: e["return"] for v, e in entries.items()})

    flavors = {e["flavor"] for e in entries.values()}
    if flavors == {"async"}:
        is_async = True
    elif "async" in flavors:
        raise GenerateError(f"{patch.name}: async in some versions and sync in others")
    else:
        is_async = False

    if patch.instance_type is not None:
        instance_type = patch.instance_type
    elif binding == "function":
        instance_type = "None"
    elif binding == "staticmethod":
        instance_type = "object"
    else:
        owner_type, owner_reasons = _owner_type(checker, entries)
        if not owner_reasons and spec.stdlib:
            owner_type = _typecheck_python_type(spec, lock, owner_type, f"{patch.name} instance", checker.notes)
        instance_type = f"type[{owner_type}]" if binding == "classmethod" else owner_type
        if owner_reasons:
            checker.notes.append(f"{patch.name} instance: {'; '.join(owner_reasons)}; using {instance_type}")
            gaps.append(AnyGap(patch=patch.name, name="instance", type=instance_type, reasons=owner_reasons))

    shapes = tuple(sorted({tuple((p[0], p[1], p[2] is not None) for p in e["params"]) for e in entries.values()}))

    ignored_note_prefixes = tuple("return:" if k == "return" else f"parameter {k!r}:" for k in patch.types)
    note_versions: dict[str, list[str]] = {}
    for version in present_versions:
        for n in entries[version].get("notes", []):
            if ignored_note_prefixes and n.startswith(ignored_note_prefixes):
                continue
            vecs = note_versions.setdefault(n, [])
            if version not in vecs:
                vecs.append(version)

    notes: list[str] = []
    for n, n_versions in note_versions.items():
        if n_versions == present_versions:
            notes.append(n)
        else:
            notes.append(f"{_presence(n_versions, present_versions, spec.subject)}: {n}")
    notes.extend(checker.notes)

    warnings: list[str] = []
    param_fields = [f for f in fields if f.source == "param"]
    if len(param_fields) >= 2 and all(f.missing for f in param_fields):
        names_by_v = {
            v: {
                name
                for name, kind, _default, _ann in (entries[v]["params"][1:] if method else entries[v]["params"])
                if kind not in _VAR_KINDS
            }
            for v in present_versions
        }
        boundary: str | None = None
        for split_idx in range(1, len(present_versions)):
            before: set[str] = set[str]().union(*(names_by_v[v] for v in present_versions[:split_idx]))
            after: set[str] = set[str]().union(*(names_by_v[v] for v in present_versions[split_idx:]))
            if before and after and before.isdisjoint(after):
                boundary = present_versions[split_idx]
                break
        if boundary is not None:
            notes.append(
                f"{patch.name}: shares no parameters across <{boundary} and >={boundary}; "
                "consider splitting with versions:"
            )
            warnings.append(
                f"{spec.library}: patch {patch.name}: shares no parameters across <{boundary} and >={boundary}; "
                f'consider splitting into two patches with versions: "<{boundary}" and versions: ">={boundary}" '
                f"in {spec.path.name}"
            )

    return PatchModel(
        spec=patch,
        fields=tuple(fields),
        is_async=is_async,
        method=method,
        result_type=result_type,
        instance_type=instance_type,
        shapes=shapes,
        notes=tuple(dict.fromkeys(notes)),
        any_gaps=tuple(gaps),
        partial=len(present_versions) < len(versions) or not patch.required,
        type_modules=tuple(dict.fromkeys(m for v in present_versions for m in entries[v].get("imports", []))),
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- rendering


def _environment() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
        autoescape=False,
    )
    env.filters["pystr"] = lambda value: json.dumps(value)
    return env


def _sorted_exports(names: list[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        if name.isupper():
            return (0, name)
        if name[:1].isupper():
            return (1, name)
        return (2, name)

    return sorted(set(names), key=key)


def input_hash(spec_text: str, lock_text: str) -> str:
    try:
        parsed_spec = yaml.safe_load(spec_text)
        normalized_spec = json.dumps(parsed_spec, sort_keys=True)
    except Exception:
        normalized_spec = spec_text
    digest = hashlib.sha256()
    digest.update(normalized_spec.encode())
    digest.update(b"\0")
    digest.update(lock_text.encode())
    return f"sha256:{digest.hexdigest()}"


def _references(all_types: str, module: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(module)}\.", all_types) is not None


# Modules the template already imports at runtime; never re-imported under ``if TYPE_CHECKING:``.
_TEMPLATE_IMPORTS = frozenset({"collections.abc"})


def _type_imports(
    top: str, all_types: str, extra_imports: Iterable[str] = (), lock_imports: Iterable[str] = ()
) -> list[list[str]]:
    """Modules to import under ``if TYPE_CHECKING:``, grouped stdlib-first as isort orders them.

    ``extra_imports`` is the spec's ``imports:`` (each must be referenced); ``lock_imports`` are the
    lock entries' ``imports`` (modules typeshed annotations name), kept only if still referenced.
    """
    pattern = re.compile(rf"(?<![\w.])({re.escape(top)}(?:\.[A-Za-z_]\w*)*)\.[A-Za-z_]\w*")
    found = set(pattern.findall(all_types))
    unused = [m for m in extra_imports if not _references(all_types, m)]
    if unused:
        raise GenerateError(f"`imports:` {unused} not referenced by any generated type; remove them")
    found.update(extra_imports)
    # References through a module the template already imports (`collections.abc.X`) need nothing more.
    template_refs = "|".join(re.escape(m) for m in sorted(_TEMPLATE_IMPORTS))
    residual = re.sub(rf"(?<![\w.])(?:{template_refs})\.", "", all_types)
    found.update(m for m in lock_imports if m not in _TEMPLATE_IMPORTS and _references(residual, m))
    if "typing." in all_types:
        found.add("typing")
    stdlib = sorted(m for m in found if m.partition(".")[0] in sys.stdlib_module_names)
    others = sorted(found.difference(stdlib))
    return [group for group in (stdlib, others) if group]


def _build_patches(spec: Spec, lock_text: str) -> tuple[dict[str, Any], list[PatchModel]]:
    """Validate ``lock_text`` against ``spec`` and build every patch model."""
    try:
        lock = load_lock(lock_text)
    except LockFormatError as exc:
        raise GenerateError(f"{spec.lock_path.name}: {exc}") from exc
    if lock.get("library") != spec.library:
        raise GenerateError(f"{spec.lock_path.name} is for {lock.get('library')!r}, spec is for {spec.library!r}")
    if bool(lock.get("stdlib", False)) != spec.stdlib:
        raise GenerateError(f"{spec.lock_path.name} does not match the spec's `stdlib:` setting; rerun lock")
    missing = sorted({p.name for p in spec.patches} - {n for v in lock["versions"].values() for n in v})
    if missing:
        raise GenerateError(f"{spec.lock_path.name} has no entries for {missing}; rerun lock")

    patches = [build_patch(spec, p, lock) for p in spec.patches]
    for version in _sorted_versions(lock):
        targets = [p.target for p in spec.patches if p.versions is not None and p.in_range(version)]
        overlapping = sorted({t for t in targets if targets.count(t) > 1})
        if overlapping:
            raise GenerateError(f"overlapping `versions:` ranges for {overlapping} at {spec.subject} {version}")
    return lock, patches


def render(spec: Spec, spec_text: str, lock_text: str) -> str:
    lock, patches = _build_patches(spec, lock_text)
    class_patches = [p for p in patches if "class" in p.spec.scope]
    groups: dict[str, list[PatchModel]] = {}
    for patch in patches:
        if "instance" in patch.spec.scope:
            groups.setdefault(patch.instance_type, []).append(patch)
    instance_groups = [InstanceGroup(t, tuple(ps)) for t, ps in groups.items()]
    instance_patches = [p for g in instance_groups for p in g.patches]
    # `opt_out` patches can only be pinned to the original (uninstrument_instance), never instrumented.
    uninstrument_patches = instance_patches + [p for p in patches if "opt_out" in p.spec.scope]
    uninstrument_types = list(dict.fromkeys(p.instance_type for p in uninstrument_patches))

    all_types = " ".join(
        [p.result_type for p in patches]
        + [p.instance_type for p in patches]
        + [f.type for p in patches for f in p.fields]
    )
    lock_imports = dict.fromkeys(m for p in patches for m in p.type_modules)
    type_imports = _type_imports(spec.top_module, all_types, spec.imports, lock_imports)
    package_dir = spec.package_dir.resolve()
    context = {
        "generated_with": GENERATED_WITH,
        "spec_name": spec.path.name,
        "lock_name": spec.lock_path.name,
        "package_rel": f"{package_dir.parent.name}/{package_dir.name}",
        "input_hash": input_hash(spec_text, canonical_lock_text(lock)),
        "library": spec.library,
        "subject": spec.subject,
        "stdlib": spec.stdlib,
        "type_imports": type_imports,
        "versions": _sorted_versions(lock),
        "patches": patches,
        "exports": _sorted_exports(
            ["MISSING", "Manual", "Missing", "instrument", "uninstrument"]
            + [name for p in patches for name in (p.args_class, p.wrapper_class)]
            + (["instrument_instance"] if instance_groups else [])
            + (["uninstrument_instance"] if uninstrument_patches else [])
        ),
        "class_patches": class_patches,
        "instance_groups": instance_groups,
        "instance_patches": instance_patches,
        "uninstrument_patches": uninstrument_patches,
        "uninstrument_types": uninstrument_types,
        "uses_abc": "collections.abc." in all_types,
        "uses_any": bool(re.search(r"(?<![\w.])Any\b", all_types)) or any(p.has_var_args for p in patches),
        "uses_awaitable": any(p.is_async for p in patches),
        "uses_overload": len(instance_groups) > 1,
        "shadows_builtins": any(f.name in _BUILTIN_NAMES for p in patches for f in p.call_params),
    }
    return _environment().get_template("patches.py.j2").render(**context)


def _ruff(*args: str, source: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )


@functools.cache
def _require_ruff() -> None:
    result = _ruff("--version")
    found = result.stdout.strip() if result.returncode == 0 else "no ruff"
    if found != f"ruff {RUFF_VERSION}":
        raise GenerateError(
            f"generate needs ruff=={RUFF_VERSION} (found {found}); run it via `uv run scripts/patchgen/cli.py`"
        )


def format_source(source: str, path: Path) -> str:
    """Format ``source`` (to be written at ``path``) with the pinned ruff, then require it to lint clean.

    Lint findings are errors rather than fixes: they mean the template emitted bad code.
    """
    _require_ruff()
    where = ("--config", str(RUFF_CONFIG), "--stdin-filename", str(path), "-")
    formatted = _ruff("format", *where, source=source)
    if formatted.returncode != 0:
        raise GenerateError(f"ruff format failed on generated {path}:\n{formatted.stderr}")
    linted = _ruff("check", "--no-fix", "--output-format", "concise", *where, source=formatted.stdout)
    if linted.returncode != 0:
        raise GenerateError(f"generated {path} fails ruff check:\n{linted.stdout}{linted.stderr}")
    return formatted.stdout


def _read_lock(spec: Spec) -> str:
    if not spec.lock_path.is_file():
        raise GenerateError(f"{spec.lock_path} not found; run `patchgen lock` first")
    return spec.lock_path.read_text(encoding="utf-8")


def generate(spec: Spec) -> tuple[Path, str]:
    """Return ``(output path, generated source)`` for ``spec``."""
    spec_text = spec.path.read_text(encoding="utf-8")
    lock_text = _read_lock(spec)
    path = output_path(spec)
    return path, format_source(render(spec, spec_text, lock_text), path)


def generate_package(package_dir: Path) -> list[tuple[Path, str]]:
    specs = find_specs(package_dir)
    if not specs:
        raise GenerateError(f"{package_dir}: no patches.yaml found")
    return [generate(load_spec(path)) for path in specs]


def any_report(spec: Spec) -> list[str]:
    """One line per generated parameter, return or instance type that is ``Any`` without an override.

    ``var_args``/``var_kwargs``/``raw_*`` are ``Any`` by design and never listed.
    Also reports patches that merge completely disjoint signatures across locked versions.
    """
    _, patches = _build_patches(spec, _read_lock(spec))
    return [
        *(gap.describe(spec) for patch in patches for gap in patch.any_gaps),
        *(warning for patch in patches for warning in patch.warnings),
    ]


def any_report_package(package_dir: Path) -> list[str]:
    specs = find_specs(package_dir)
    if not specs:
        raise GenerateError(f"{package_dir}: no patches.yaml found")
    return [line for path in specs for line in any_report(load_spec(path))]
