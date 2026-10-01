# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Record target signatures inside a throwaway environment.

Run by ``patchgen lock`` as ``<venv python> introspect.py`` with a JSON request
on stdin: ``{"top": "httpx", "targets": {"name": "module:qualname", ...}}``.
Writes a JSON result to stdout. Must only depend on the standard library,
because it runs in an environment that contains nothing but the target
library and its dependencies.

Annotations are rendered to type expressions that are valid in the generated
module: builtins, ``collections.abc`` generics, ``Any``, ``None`` and classes
re-exported at the top level of the library (``httpx.Request``). Anything else
becomes ``Any`` and a note is recorded.
"""

from __future__ import annotations

import collections.abc
import contextlib
import importlib
import inspect
import json
import re
import sys
import types
import typing
from typing import Any

_BUILTIN_ARITY: dict[type, int] = {list: 1, set: 1, frozenset: 1, dict: 2, type: 1}
_PLAIN_BUILTINS = {int, str, bytes, bool, float, complex, bytearray, memoryview, object}
_ABC_ARITY: dict[Any, int] = {
    collections.abc.Mapping: 2,
    collections.abc.MutableMapping: 2,
    collections.abc.Sequence: 1,
    collections.abc.MutableSequence: 1,
    collections.abc.Iterable: 1,
    collections.abc.Iterator: 1,
    collections.abc.AsyncIterable: 1,
    collections.abc.AsyncIterator: 1,
    collections.abc.Awaitable: 1,
    collections.abc.Collection: 1,
    collections.abc.Generator: 3,
    collections.abc.AsyncGenerator: 2,
    collections.abc.Coroutine: 3,
}
_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")
_UNION_TYPE = getattr(types, "UnionType", None)
_TYPES_BUILTINS: dict[type, str] = {
    tp: f"types.{name}"
    for name in (
        "AsyncGeneratorType",
        "BuiltinFunctionType",
        "CellType",
        "ClassMethodDescriptorType",
        "CodeType",
        "CoroutineType",
        "FrameType",
        "FunctionType",
        "GeneratorType",
        "GetSetDescriptorType",
        "MappingProxyType",
        "MemberDescriptorType",
        "MethodDescriptorType",
        "MethodType",
        "MethodWrapperType",
        "ModuleType",
        "SimpleNamespace",
        "TracebackType",
        "WrapperDescriptorType",
    )
    if isinstance((tp := getattr(types, name, None)), type)
}


class _Renderer:
    def __init__(self, top: str, owner: type | None = None) -> None:
        self.top = top
        self.top_module = sys.modules.get(top)
        self.owner = owner
        self.notes: list[str] = []
        self.imports: list[str] = []

    def any(self, what: object, where: str) -> str:
        self.notes.append(f"{where}: {_safe_repr(what)} rendered as Any")
        return "Any"

    def members(self, tp: object, where: str) -> list[str]:
        origin = typing.get_origin(tp)
        if origin is typing.Union or (_UNION_TYPE is not None and origin is _UNION_TYPE):
            out: list[str] = []
            for arg in typing.get_args(tp):
                for member in self.members(arg, where):
                    if member not in out:
                        out.append(member)
            return out
        return [self.render(tp, where)]

    def render(self, tp: object, where: str) -> str:
        if tp is Any:
            return "Any"
        if tp is None or tp is type(None):
            return "None"
        if tp is Ellipsis:
            return "..."
        if self.owner is not None and _safe_repr(tp) in ("typing.Self", "typing_extensions.Self"):
            return self.render(self.owner, where)
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)
        if origin is typing.Union or (_UNION_TYPE is not None and origin is _UNION_TYPE):
            return " | ".join(self.members(tp, where))
        if origin is not None:
            if origin is tuple:
                if not args:
                    return "tuple[Any, ...]"
                return "tuple[" + ", ".join(self.render(a, where) for a in args) + "]"
            if origin in _BUILTIN_ARITY:
                if not args:
                    return self.render(origin, where)
                return f"{origin.__name__}[" + ", ".join(self.render(a, where) for a in args) + "]"
            if origin is collections.abc.Callable:
                if not args:
                    return "collections.abc.Callable[..., Any]"
                params, ret = args[0], args[-1]
                if params is Ellipsis:
                    rendered_params = "..."
                else:
                    rendered_params = "[" + ", ".join(self.render(a, where) for a in params) + "]"
                return f"collections.abc.Callable[{rendered_params}, {self.render(ret, where)}]"
            if origin in _ABC_ARITY:
                if not args:
                    return self.render(origin, where)
                return f"collections.abc.{origin.__name__}[" + ", ".join(self.render(a, where) for a in args) + "]"
            if isinstance(origin, type) and (ref := self.public_ref(origin)) is not None and args:
                return f"{ref}[" + ", ".join(self.render(a, where) for a in args) + "]"
            return self.any(tp, where)
        if isinstance(tp, type):
            if tp in _PLAIN_BUILTINS:
                return tp.__name__
            if (types_ref := _TYPES_BUILTINS.get(tp)) is not None:
                if "types" not in self.imports:
                    self.imports.append("types")
                return types_ref
            if tp is tuple:
                return "tuple[Any, ...]"
            if tp in _BUILTIN_ARITY:
                return f"{tp.__name__}[" + ", ".join(["Any"] * _BUILTIN_ARITY[tp]) + "]"
            if tp in _ABC_ARITY:
                return f"collections.abc.{tp.__name__}[" + ", ".join(["Any"] * _ABC_ARITY[tp]) + "]"
            if tp is collections.abc.Callable:
                return "collections.abc.Callable[..., Any]"
            ref = self.public_ref(tp)
            if ref is not None:
                params = getattr(tp, "__parameters__", ())
                if params:
                    return f"{ref}[" + ", ".join(["Any"] * len(params)) + "]"
                return ref
        return self.any(tp, where)

    def _submodule_ref(self, tp: type, name: str) -> str | None:
        mod_name = getattr(tp, "__module__", "")
        if mod_name != self.top and not mod_name.startswith(f"{self.top}."):
            return None
        parts = mod_name.split(".")
        for i in range(len(parts), 1, -1):
            prefix_parts = parts[:i]
            if any(p.startswith("_") for p in prefix_parts):
                continue
            candidate = ".".join(prefix_parts)
            try:
                mod = importlib.import_module(candidate)
            except ImportError:
                continue
            if getattr(mod, name, None) is tp:
                return f"{candidate}.{name}"
        return None

    def public_ref(self, tp: type) -> str | None:
        name = getattr(tp, "__name__", "")
        if not name or name.startswith("_") or getattr(tp, "__qualname__", None) != name:
            return None
        if self.top_module is not None and getattr(self.top_module, name, None) is tp:
            return f"{self.top}.{name}"
        return self._submodule_ref(tp, name)


def _safe_repr(obj: object) -> str:
    return _ADDRESS_RE.sub("", repr(obj))


def _signature(fn: Any) -> inspect.Signature:
    if sys.version_info >= (3, 14):
        import annotationlib  # noqa: PLC0415

        return inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)
    return inspect.signature(fn)


def _resolve_annotation(ann: object, hints: dict[str, Any] | None, name: str, globalns: dict[str, Any]) -> object:
    if hints and name in hints:
        return hints[name]
    # ``typing.ForwardRef`` / ``annotationlib.ForwardRef`` (3.14 FORWARDREF format).
    forward_arg = getattr(ann, "__forward_arg__", None)
    if isinstance(forward_arg, str):
        ann = forward_arg
    if isinstance(ann, str):
        try:
            return eval(ann, globalns, {})
        except Exception:
            return ann
    return ann


def _flavor(fn: Any) -> str:
    unwrapped = inspect.unwrap(fn)
    if inspect.iscoroutinefunction(unwrapped):
        return "async"
    if inspect.isasyncgenfunction(unwrapped):
        return "asyncgen"
    if inspect.isgeneratorfunction(unwrapped):
        return "generator"
    return "sync"


def _target_module_missing(exc: ImportError, module_name: str) -> bool:
    """Whether ``exc`` means the target module itself (or a parent package) does not exist."""
    missing = getattr(exc, "name", None)
    return (
        isinstance(exc, ModuleNotFoundError)
        and isinstance(missing, str)
        and (module_name == missing or module_name.startswith(f"{missing}."))
    )


def describe(top: str, target: str) -> dict[str, Any]:
    module_name, _, qualname = target.partition(":")
    parts = qualname.split(".")
    try:
        owner: Any = importlib.import_module(module_name)
    except ImportError as exc:
        if _target_module_missing(exc, module_name):
            return {"error": f"target not found: {exc}"}
        # The module exists but importing it fails (a missing non-library
        # dependency, a removed stdlib API, ...): the version is unusable.
        return {"unimportable": f"cannot import {module_name}: {type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"unimportable": f"cannot import {module_name}: {type(exc).__name__}: {exc}"}
    try:
        for part in parts[:-1]:
            owner = getattr(owner, part)
        raw: Any = inspect.getattr_static(owner, parts[-1])
        fn: Any = getattr(owner, parts[-1])
    except (ImportError, AttributeError) as exc:
        return {"error": f"target not found: {exc}"}

    owner_ref: str | None = None
    owner_defined: str | None = None
    if inspect.isclass(owner):
        if type(raw) is classmethod:
            binding = "classmethod"
            fn = raw.__func__
            owner_ref = _Renderer(top).public_ref(owner)
            owner_defined = f"{owner.__module__}.{owner.__qualname__}"
        elif type(raw) is staticmethod:
            binding = "staticmethod"
            fn = raw.__func__
        else:
            binding = "method"
            owner_ref = _Renderer(top).public_ref(owner)
            owner_defined = f"{owner.__module__}.{owner.__qualname__}"
    else:
        binding = "function"

    try:
        sig = _signature(fn)
    except (TypeError, ValueError):
        kind = "builtin" if inspect.isbuiltin(fn) or inspect.ismethoddescriptor(fn) else "not introspectable"
        return {"error": f"no signature available ({kind})"}

    unwrapped = inspect.unwrap(fn)
    globalns: dict[str, Any] = getattr(unwrapped, "__globals__", {})
    try:
        hints: dict[str, Any] = typing.get_type_hints(unwrapped)
    except Exception:
        hints = {}

    renderer = _Renderer(top, owner if inspect.isclass(owner) and binding in ("method", "classmethod") else None)
    params: list[list[Any]] = []
    for param in sig.parameters.values():
        default = None if param.default is inspect.Parameter.empty else _safe_repr(param.default)
        if param.annotation is inspect.Parameter.empty:
            annotation = None
        else:
            resolved = _resolve_annotation(param.annotation, hints, param.name, globalns)
            annotation = renderer.members(resolved, f"parameter {param.name!r}")
        params.append([param.name, param.kind.name, default, annotation])

    if sig.return_annotation is inspect.Signature.empty:
        returns = None
    else:
        resolved = _resolve_annotation(sig.return_annotation, hints, "return", globalns)
        returns = renderer.members(resolved, "return")

    described: dict[str, Any] = {
        "binding": binding,
        "flavor": _flavor(fn),
        "module": getattr(unwrapped, "__module__", None),
        "qualname": getattr(unwrapped, "__qualname__", None),
        "owner": owner_ref,
        "params": params,
        "return": returns,
        "notes": renderer.notes,
    }
    if owner_defined is not None:
        described["owner_defined"] = owner_defined
    if renderer.imports:
        described["imports"] = sorted(set(renderer.imports))
    return described


def _skip_exports_module(mod_name: str) -> bool:
    """Vendored copies (and ``six.moves``) have lazy attributes that fail on access."""
    parts = mod_name.split(".")
    if any(part.startswith("_") or part in ("vendor", "vendored") for part in parts):
        return True
    return "six.moves" in mod_name


def _collect_exports(top: str) -> list[str]:
    exports: set[str] = set()
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or (mod_name != top and not mod_name.startswith(f"{top}.")):
            continue
        if _skip_exports_module(mod_name):
            continue
        is_top = mod_name == top
        prefix = "" if is_top else f"{mod_name[len(top) + 1 :]}."
        names: list[str] = []
        with contextlib.suppress(Exception):
            names = dir(mod)
        for name in names:
            if name.startswith("_"):
                continue
            # Lazy attributes (six.moves, vendored modules) may raise anything on access.
            with contextlib.suppress(Exception):
                val = getattr(mod, name, None)
                if inspect.isclass(val) and (is_top or getattr(val, "__module__", "").startswith(mod_name)):
                    exports.add(f"{prefix}{name}")
    return sorted(exports)


def main() -> None:
    request = json.load(sys.stdin)
    top: str = request["top"]
    targets: dict[str, str] = request["targets"]
    result: dict[str, Any] = {"targets": {}}
    try:
        importlib.import_module(top)
    except ImportError as exc:
        result["error"] = f"cannot import {top}: {exc}"
    except Exception as exc:
        result["error"] = f"cannot import {top}: {type(exc).__name__}: {exc}"
    for name, target in targets.items():
        result["targets"][name] = describe(top, target)
    try:
        result["exports"] = _collect_exports(top)
    except Exception as exc:
        # Never drop the target signatures because export collection failed.
        result["exports"] = []
        result["exports_error"] = f"{type(exc).__name__}: {exc}"
    json.dump(result, sys.stdout, indent=None, sort_keys=True)


if __name__ == "__main__":
    main()
