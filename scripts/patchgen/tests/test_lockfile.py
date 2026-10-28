# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the on-disk lock layout (``patchgen.lockfile``)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from patchgen.lockfile import (
    LOCK_FORMAT,
    LockFormatError,
    canonical_lock_text,
    dump_lock,
    load_lock,
)

V1 = {
    "binding": "method",
    "flavor": "sync",
    "params": [["self", "POSITIONAL_OR_KEYWORD", None, None], ["url", "POSITIONAL_OR_KEYWORD", None, ["str"]]],
    "return": None,
}
V2 = {**V1, "return": ["fakelib.Response"]}
ABSENT = {"absent": True, "error": "target not found"}


def _expanded() -> dict[str, Any]:
    return {
        "library": "fakelib",
        "generated_with": "patchgen 0.1",
        "python": "3.12",
        "specifier": ">=1.0",
        "versions": {
            "1.0": {"send": V1, "extra": ABSENT},
            "1.1": {"send": V1, "extra": ABSENT},
            "2.0": {"send": V2, "extra": V1},
        },
        "exports": {"1.0": ["Client"], "1.1": ["Client"], "2.0": ["Client", "Response"]},
    }


def test_dump_groups_versions_sharing_a_signature() -> None:
    data = json.loads(dump_lock(_expanded()))
    assert data["format"] == LOCK_FORMAT
    assert list(data) == [
        "format",
        "library",
        "generated_with",
        "python",
        "specifier",
        "versions",
        "targets",
        "exports",
    ]
    assert data["versions"] == ["1.0", "1.1", "2.0"]
    assert data["targets"] == {
        "send": [{"versions": ["1.0", "1.1"], **V1}, {"versions": ["2.0"], **V2}],
        "extra": [{"versions": ["1.0", "1.1"], **ABSENT}, {"versions": ["2.0"], **V1}],
    }
    assert data["exports"] == [
        {"versions": ["1.0", "1.1"], "symbols": ["Client"]},
        {"versions": ["2.0"], "symbols": ["Client", "Response"]},
    ]


def test_round_trip_is_lossless_including_key_order() -> None:
    expanded = _expanded()
    loaded = load_lock(dump_lock(expanded))
    assert loaded == expanded
    assert canonical_lock_text(loaded) == canonical_lock_text(expanded)


def test_loaded_entries_are_independent_copies() -> None:
    loaded = load_lock(dump_lock(_expanded()))
    loaded["versions"]["1.0"]["send"]["return"] = ["int"]
    assert loaded["versions"]["1.1"]["send"]["return"] is None
    loaded["exports"]["1.0"].append("Extra")
    assert loaded["exports"]["1.1"] == ["Client"]


def test_grouping_keeps_values_that_are_only_equal_in_python_apart() -> None:
    expanded = _expanded()
    expanded["versions"]["1.0"]["send"] = {**V1, "return": 1}
    expanded["versions"]["1.1"]["send"] = {**V1, "return": True}
    data = json.loads(dump_lock(expanded))
    assert [group["versions"] for group in data["targets"]["send"]] == [["1.0"], ["1.1"], ["2.0"]]
    loaded = load_lock(dump_lock(expanded))
    assert canonical_lock_text(loaded) == canonical_lock_text(expanded)


def test_dump_puts_one_parameter_per_line_and_wraps_long_version_lists() -> None:
    versions = {f"1.{i}": {"send": V1} for i in range(40)}
    text = dump_lock({"library": "fakelib", "versions": versions, "exports": {}})
    assert '          ["self", "POSITIONAL_OR_KEYWORD", null, null],\n' in text
    assert '          ["url", "POSITIONAL_OR_KEYWORD", null, ["str"]]\n' in text
    assert max(len(line) for line in text.splitlines()) <= 100
    assert load_lock(text)["versions"] == versions


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (json.dumps({"library": "x", "versions": {}}), "lock format None is not supported"),
        (json.dumps({"format": LOCK_FORMAT + 1}), f"lock format {LOCK_FORMAT + 1} is not supported"),
        ("[]", "not a JSON object"),
        ("{", "not valid JSON"),
    ],
)
def test_load_rejects_other_formats(text: str, match: str) -> None:
    with pytest.raises(LockFormatError, match=match):
        load_lock(text)
