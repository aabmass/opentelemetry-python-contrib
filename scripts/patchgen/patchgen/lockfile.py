# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""On-disk layout of ``patches.lock.json``.

In memory (``build_lock``, ``generate``) a lock is *expanded*: ``versions`` maps every locked
version to its ``{patch name: entry}`` and ``exports`` maps every version to its symbol list. Most
of that repeats from version to version, so on disk (format ``LOCK_FORMAT``) it is grouped:

- ``versions``: the locked versions, in lock order;
- ``targets``: per patch name, one entry per distinct signature, each with the ``versions`` that
  share it;
- ``exports``: one ``{"versions": [...], "symbols": [...]}`` per distinct symbol list.

``dump_lock`` / ``load_lock`` convert between the two losslessly. ``canonical_lock_text`` is the
expanded lock as indented JSON; the generated module's input hash is computed over it, so the hash
depends on what was locked, not on how the file is laid out.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from typing import Any

LOCK_FORMAT = 2
_WIDTH = 100
_INDENT = "  "


class LockFormatError(ValueError):
    """Raised when a lock file is not in the current format."""


def _grouped(items: Iterable[tuple[str, Any]], key: str | None) -> list[dict[str, Any]]:
    """Versions sharing an identical value, in first-seen order.

    Values are compared by their JSON text, so ``1`` and ``True`` (equal in Python) stay apart.
    With ``key`` the value is stored under it; otherwise the value (a dict) is inlined after
    ``versions``.
    """
    groups: dict[str, tuple[Any, list[str]]] = {}
    for version, value in items:
        text = json.dumps(value)
        if text in groups:
            groups[text][1].append(version)
        else:
            groups[text] = (value, [version])
    if key is not None:
        return [{"versions": versions, key: value} for value, versions in groups.values()]
    return [{"versions": versions, **value} for value, versions in groups.values()]


def compact_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """The on-disk form of an expanded ``lock``."""
    out: dict[str, Any] = {"format": LOCK_FORMAT}
    for key, value in lock.items():
        if key == "versions":
            names = list(dict.fromkeys(name for entries in value.values() for name in entries))
            out["versions"] = list(value)
            out["targets"] = {
                name: _grouped(((v, e[name]) for v, e in value.items() if name in e), None) for name in names
            }
        elif key == "exports":
            out["exports"] = _grouped(value.items(), "symbols")
        else:
            out[key] = value
    return out


def expand_lock(data: Mapping[str, Any]) -> dict[str, Any]:
    """The expanded form of an on-disk lock (``compact_lock`` inverted)."""
    if data.get("format") != LOCK_FORMAT:
        raise LockFormatError(
            f"lock format {data.get('format')!r} is not supported (expected {LOCK_FORMAT}); "
            "delete it and rerun `patchgen lock` (re-locking from scratch won't reuse the recorded python)"
        )
    versions: list[str] = list(data.get("versions", []))
    order = {v: i for i, v in enumerate(versions)}
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in ("format", "targets"):
            continue
        if key == "versions":
            expanded: dict[str, dict[str, Any]] = {v: {} for v in versions}
            for name, groups in data.get("targets", {}).items():
                for group in groups:
                    entry = {k: v for k, v in group.items() if k != "versions"}
                    for version in group["versions"]:
                        expanded.setdefault(version, {})[name] = copy.deepcopy(entry)
            out["versions"] = expanded
        elif key == "exports":
            pairs = [(v, list(group["symbols"])) for group in value for v in group["versions"]]
            pairs.sort(key=lambda pair: order.get(pair[0], len(order)))
            out["exports"] = dict(pairs)
        else:
            out[key] = value
    return out


def _is_container(value: Any) -> bool:
    return isinstance(value, (dict, list))


def _format(value: Any, level: int, *, keyed: bool = False) -> str:
    """Indented JSON, compact enough to review.

    Objects get one member per line. An object member that is a list of containers (``params``,
    ``targets``) gets one element per line; a list inside a list (one parameter) stays on one line
    unless it holds an object. A member that is a list of scalars (``versions``, ``notes``) stays on
    one line, or is packed over several lines if it does not fit in ``_WIDTH``.
    """
    pad = _INDENT * (level + 1)
    end = _INDENT * level
    if isinstance(value, dict) and value:
        items = [f"{pad}{json.dumps(k)}: {_format(v, level + 1, keyed=True)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{end}}}"
    if not isinstance(value, list) or not value:
        return json.dumps(value)
    if (keyed and any(_is_container(v) for v in value)) or any(isinstance(v, dict) for v in value):
        return "[\n" + ",\n".join(f"{pad}{_format(v, level + 1)}" for v in value) + f"\n{end}]"
    inline = json.dumps(value)
    if not keyed or len(pad) + len(inline) <= _WIDTH:
        return inline
    lines: list[str] = []
    line = ""
    for item in (json.dumps(v) for v in value):
        if line and len(pad) + len(line) + len(item) + 2 > _WIDTH:
            lines.append(line)
            line = ""
        line = f"{line} {item}," if line else f"{item},"
    lines.append(line.removesuffix(","))
    return "[\n" + "\n".join(f"{pad}{line}" for line in lines) + f"\n{end}]"


def dump_lock(lock: Mapping[str, Any]) -> str:
    """Serialize an expanded ``lock`` in the on-disk format."""
    return _format(compact_lock(lock), 0) + "\n"


def load_lock(text: str) -> dict[str, Any]:
    """Parse on-disk lock ``text`` into the expanded form; raises ``LockFormatError``."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LockFormatError(f"lock is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LockFormatError("lock is not a JSON object")
    return expand_lock(data)


def canonical_lock_text(lock: Mapping[str, Any]) -> str:
    """The expanded ``lock`` as indented JSON, independent of the on-disk layout."""
    return json.dumps(lock, indent=2) + "\n"
