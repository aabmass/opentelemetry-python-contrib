# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from typing import Any, Generic, Optional, TypeVar, Union

import pytest
from patchgen.introspect import _Renderer, describe

T = TypeVar("T")
U = TypeVar("U")


class CustomGeneric(Generic[T]):
    pass


class MultiGeneric(Generic[T, U]):
    pass


def test_render_unparameterized_generic_classes() -> None:
    renderer = _Renderer("tests.test_introspect")
    renderer.top_module = sys.modules[__name__]

    assert renderer.render(CustomGeneric, "test") == "tests.test_introspect.CustomGeneric[Any]"


def test_render_parameterized_custom_generics() -> None:
    renderer = _Renderer("tests.test_introspect")
    renderer.top_module = sys.modules[__name__]

    tp = CustomGeneric[int]
    assert renderer.render(tp, "test") == "tests.test_introspect.CustomGeneric[int]"


def test_render_multiple_parameters() -> None:
    renderer = _Renderer("tests.test_introspect")
    renderer.top_module = sys.modules[__name__]

    assert renderer.render(MultiGeneric, "test") == "tests.test_introspect.MultiGeneric[Any, Any]"
    assert renderer.render(MultiGeneric[int, str], "test") == "tests.test_introspect.MultiGeneric[int, str]"


def test_render_typing_self_uses_owner() -> None:
    import typing as std_typing  # noqa: PLC0415

    renderer = _Renderer("tests.test_introspect", owner=CustomGeneric)
    renderer.top_module = sys.modules[__name__]
    self_tp = getattr(std_typing, "Self", None)
    if self_tp is not None:
        assert renderer.render(self_tp, "return") == "tests.test_introspect.CustomGeneric[Any]"
        assert renderer.render(list[self_tp], "return") == "list[tests.test_introspect.CustomGeneric[Any]]"


def test_describe_renders_top_reexport_and_types_builtins(monkeypatch: pytest.MonkeyPatch) -> None:
    optional_cls: Any = Optional
    union_cls: Any = Union
    top_mod = types.ModuleType("fakepkg")
    sub_mod = types.ModuleType("fakepkg.engine")
    url_mod = types.ModuleType("fakepkg.engine.url")

    class URL:
        pass

    URL.__module__ = "fakepkg.engine.url"
    URL.__qualname__ = "URL"

    def compile_target(
        url: str | URL,
        transaction: Any | None = None,
        shard_hint: Any | None = None,
    ) -> str | types.CodeType:
        return ""

    compile_target.__module__ = "fakepkg.engine"
    compile_target.__qualname__ = "compile_target"
    compile_target.__annotations__ = {
        "url": union_cls[str, URL],
        "transaction": optional_cls[Any],
        "shard_hint": union_cls[Any, None],
        "return": union_cls[str, types.CodeType],
    }

    setattr(url_mod, "URL", URL)
    setattr(sub_mod, "url", url_mod)
    setattr(sub_mod, "compile_target", compile_target)
    setattr(top_mod, "engine", sub_mod)
    setattr(top_mod, "URL", URL)

    monkeypatch.setitem(sys.modules, "fakepkg", top_mod)
    monkeypatch.setitem(sys.modules, "fakepkg.engine", sub_mod)
    monkeypatch.setitem(sys.modules, "fakepkg.engine.url", url_mod)

    desc = describe("fakepkg", "fakepkg.engine:compile_target")
    assert desc["params"] == [
        ["url", "POSITIONAL_OR_KEYWORD", None, ["str", "fakepkg.URL"]],
        ["transaction", "POSITIONAL_OR_KEYWORD", "None", ["Any", "None"]],
        ["shard_hint", "POSITIONAL_OR_KEYWORD", "None", ["Any", "None"]],
    ]
    assert desc["return"] == ["str", "types.CodeType"]
    assert desc["imports"] == ["types"]
    assert "library_any" not in desc
    assert "canonical" not in desc
    assert desc["notes"] == []
