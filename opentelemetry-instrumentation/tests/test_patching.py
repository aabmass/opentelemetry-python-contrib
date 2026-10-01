# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access,too-few-public-methods,no-self-use,invalid-name,no-member

from __future__ import annotations

import asyncio
import functools
import inspect
import sys
import types
from collections.abc import Callable
from typing import Any
from unittest import TestCase, mock

from opentelemetry.instrumentation import _patching
from opentelemetry.instrumentation._patching import (
    MISSING,
    InstancePatchDef,
    Manual,
    Missing,
    PatchDef,
    PatchMismatchError,
    Registry,
    apply_all,
    apply_instance,
    bind_wrapper,
    remove_instance,
    unwind,
)

_MODULE = "_patching_testlib"
_REAL_INSTALLED_VERSION = _patching._installed_version


class _Base:
    def inherited(self, x: int) -> int:
        return x


def _make_library() -> types.ModuleType:
    module = types.ModuleType(_MODULE)

    class Client(_Base):
        def send(self, request: str, *, timeout: float = 1.0) -> str:
            return f"sent {request} {timeout}"

        def other(self, value: int) -> int:
            return value * 2

        @classmethod
        def from_url(cls, url: str, *, timeout: float = 1.0) -> Client:
            inst = cls()
            setattr(inst, "url", f"{cls.__name__}:{url}:{timeout}")
            return inst

        @staticmethod
        def parse(raw: str, *, strict: bool = False) -> str:
            return f"parsed:{raw}:{strict}"

    class SubClient(Client):
        def send(self, request: str, *, timeout: float = 1.0) -> str:
            return super().send(request, timeout=timeout)

    class Slotted:
        __slots__ = ()

        def send(self, request: str, *, timeout: float = 1.0) -> str:
            return request

    setattr(module, "Client", Client)
    setattr(module, "SubClient", SubClient)
    setattr(module, "Slotted", Slotted)
    return module


def _recording_factory(calls: list[tuple[Any, ...]]) -> _patching.WrapperFactory:
    def factory(sig: inspect.Signature) -> _patching.WraptWrapper:
        def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            calls.append((instance, dict(bound.arguments)))
            return wrapped(*args, **kwargs)

        return wrapper

    return factory


def _shape(*names: str) -> _patching.ParamShape:
    return tuple((name, "POSITIONAL_OR_KEYWORD", False) for name in names)


SEND_SHAPE = (
    ("self", "POSITIONAL_OR_KEYWORD", False),
    ("request", "POSITIONAL_OR_KEYWORD", False),
    ("timeout", "KEYWORD_ONLY", True),
)


class PatchingTestBase(TestCase):
    def setUp(self) -> None:
        self.module = _make_library()
        sys.modules[_MODULE] = self.module
        self.Client: Any = self.module.Client
        self.registry = Registry("patching-testlib")
        self.calls: list[tuple[Any, ...]] = []
        _patching._limiter.reset()
        version_patch = mock.patch.object(_patching, "_installed_version", return_value="1.0.0")
        version_patch.start()
        self.addCleanup(version_patch.stop)
        self.addCleanup(sys.modules.pop, _MODULE, None)
        self.addCleanup(unwind, self.registry)

    def patch_def(self, target: str = f"{_MODULE}:Client.send", **kwargs: Any) -> PatchDef:
        values: dict[str, Any] = {
            "name": "client_send",
            "target": target,
            "required": True,
            "method": True,
            "locked_shapes": frozenset({SEND_SHAPE}),
            "make_wrapper": _recording_factory(self.calls),
        }
        values.update(kwargs)
        return PatchDef(**values)

    def apply(self, *patches: PatchDef, extra: tuple[Manual, ...] = ()) -> bool:
        return apply_all(self.registry, locked_versions=frozenset({"1.0.0"}), patches=patches, extra=extra)


class TestMissing(TestCase):
    def test_sentinel(self) -> None:
        self.assertIs(MISSING, _patching._MissingType.MISSING)
        self.assertEqual(repr(MISSING), "MISSING")


class TestApplyAll(PatchingTestBase):
    def test_apply_binds_and_unwind_restores_identity(self) -> None:
        original = vars(self.Client)["send"]
        self.assertTrue(self.apply(self.patch_def()))
        client = self.Client()
        self.assertEqual(client.send("r"), "sent r 1.0")
        self.assertEqual(len(self.calls), 1)
        instance, arguments = self.calls[0]
        self.assertIs(instance, client)
        self.assertEqual(arguments, {"request": "r", "timeout": 1.0})
        self.assertIsNot(vars(self.Client)["send"], original)

        unwind(self.registry)
        self.assertIs(vars(self.Client)["send"], original)
        self.assertFalse(self.registry.active)

    def test_inherited_target_is_removed_not_overwritten(self) -> None:
        patch = self.patch_def(
            f"{_MODULE}:Client.inherited",
            locked_shapes=frozenset({_shape("self", "x")}),
        )
        self.assertNotIn("inherited", vars(self.Client))
        self.assertTrue(self.apply(patch))
        self.assertIn("inherited", vars(self.Client))
        self.assertEqual(self.Client().inherited(3), 3)
        unwind(self.registry)
        self.assertNotIn("inherited", vars(self.Client))
        self.assertIs(self.Client.inherited, _Base.inherited)

    def test_classmethod_preserves_calling_class_and_unwinds_descriptor(self) -> None:
        original = vars(self.Client)["from_url"]
        self.assertIsInstance(original, classmethod)
        shape = (
            ("cls", "POSITIONAL_OR_KEYWORD", False),
            ("url", "POSITIONAL_OR_KEYWORD", False),
            ("timeout", "KEYWORD_ONLY", True),
        )
        patch = self.patch_def(
            f"{_MODULE}:Client.from_url",
            name="client_from_url",
            method=True,
            locked_shapes=frozenset({shape}),
        )
        self.assertTrue(self.apply(patch))

        c1 = self.Client.from_url("http://a")
        self.assertIs(type(c1), self.Client)
        self.assertEqual(c1.url, "Client:http://a:1.0")
        self.assertEqual(self.calls[0], (self.Client, {"url": "http://a", "timeout": 1.0}))

        sub = self.module.SubClient.from_url("http://b", timeout=2.5)
        self.assertIs(type(sub), self.module.SubClient)
        self.assertEqual(sub.url, "SubClient:http://b:2.5")
        self.assertEqual(self.calls[1], (self.module.SubClient, {"url": "http://b", "timeout": 2.5}))

        unwind(self.registry)
        self.assertIs(vars(self.Client)["from_url"], original)

    def test_staticmethod_binds_all_positional_args_and_unwinds_descriptor(self) -> None:
        original = vars(self.Client)["parse"]
        self.assertIsInstance(original, staticmethod)
        shape = (
            ("raw", "POSITIONAL_OR_KEYWORD", False),
            ("strict", "KEYWORD_ONLY", True),
        )
        patch = self.patch_def(
            f"{_MODULE}:Client.parse",
            name="client_parse",
            method=False,
            locked_shapes=frozenset({shape}),
        )
        self.assertTrue(self.apply(patch))

        self.assertEqual(self.Client.parse("abc"), "parsed:abc:False")
        self.assertEqual(self.calls[0], (None, {"raw": "abc", "strict": False}))

        client = self.Client()
        self.assertEqual(client.parse("xyz", strict=True), "parsed:xyz:True")
        self.assertEqual(self.calls[1][1], {"raw": "xyz", "strict": True})

        unwind(self.registry)
        self.assertIs(vars(self.Client)["parse"], original)

    def test_missing_required_target_patches_nothing(self) -> None:
        original = vars(self.Client)["send"]
        with self.assertLogs(_patching._logger, level="ERROR"):
            applied = self.apply(self.patch_def(), self.patch_def(f"{_MODULE}:Client.nope", name="nope"))
        self.assertFalse(applied)
        self.assertIs(vars(self.Client)["send"], original)
        self.assertFalse(self.registry.active)

    def test_missing_required_target_raises_in_strict_mode(self) -> None:
        with mock.patch.dict("os.environ", {"OTEL_PYTHON_PATCH_STRICT": "true"}):
            with self.assertRaises(PatchMismatchError):
                self.apply(self.patch_def(f"{_MODULE}:Client.nope"))

    def test_missing_optional_target_is_skipped(self) -> None:
        self.assertTrue(self.apply(self.patch_def(), self.patch_def(f"{_MODULE}:Nope.send", required=False)))
        self.Client().send("r")
        self.assertEqual(len(self.calls), 1)

    def test_unlocked_version_outside_range_with_locked_shapes_logs_debug_only(self) -> None:
        with mock.patch.object(_patching, "_installed_version", return_value="9.9.9"):
            with mock.patch.dict("os.environ", {"OTEL_PYTHON_PATCH_STRICT": "true"}):
                with self.assertLogs(_patching._logger, level="DEBUG") as logs:
                    self.assertTrue(self.apply(self.patch_def()))
        self.assertEqual([record.levelname for record in logs.records], ["DEBUG"])
        self.assertIn("9.9.9 was not in the lock", logs.output[0])

    def test_unlocked_version_outside_range_with_shape_mismatch_warns_or_raises(self) -> None:
        patch = self.patch_def(locked_shapes=frozenset({_shape("self", "request")}))
        with mock.patch.object(_patching, "_installed_version", return_value="9.9.9"):
            with self.assertLogs(_patching._logger, level="WARNING") as logs:
                self.assertTrue(self.apply(patch))
            self.assertIn("9.9.9 was not in the lock", logs.output[0])
            self.assertIn("differs from every locked version", logs.output[1])
            unwind(self.registry)
            with mock.patch.dict("os.environ", {"OTEL_PYTHON_PATCH_STRICT": "true"}):
                with self.assertRaises(PatchMismatchError):
                    self.apply(patch)

    def test_unlocked_version_outside_range_without_shapes_warns_or_raises(self) -> None:
        # A patch with no locked shapes gives no evidence that the version is compatible.
        unverified = self.patch_def(f"{_MODULE}:Client.other", name="other", locked_shapes=frozenset())
        with mock.patch.object(_patching, "_installed_version", return_value="9.9.9"):
            with self.assertLogs(_patching._logger, level="WARNING") as logs:
                self.assertTrue(self.apply(self.patch_def(), unverified))
            self.assertEqual(len(logs.records), 1)
            self.assertIn("9.9.9 was not in the lock", logs.output[0])
            unwind(self.registry)
            with mock.patch.dict("os.environ", {"OTEL_PYTHON_PATCH_STRICT": "true"}):
                with self.assertRaises(PatchMismatchError):
                    self.apply(self.patch_def(), unverified)

    def test_unsampled_version_in_locked_range_logs_debug_only(self) -> None:
        with mock.patch.object(_patching, "_installed_version", return_value="1.0.1"):
            with mock.patch.dict("os.environ", {"OTEL_PYTHON_PATCH_STRICT": "true"}):
                with self.assertLogs(_patching._logger, level="DEBUG") as logs:
                    self.assertTrue(
                        apply_all(
                            self.registry,
                            locked_versions=frozenset({"1.0.0", "1.1.0"}),
                            patches=(self.patch_def(),),
                        )
                    )
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelname, "DEBUG")
        self.assertIn("1.0.1 was not in the lock", logs.output[0])

    def test_signature_mismatch_warns_or_raises(self) -> None:
        patch = self.patch_def(locked_shapes=frozenset({_shape("self", "something_else")}))
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertTrue(self.apply(patch))
        self.assertIn("differs from every locked version", logs.output[0])
        unwind(self.registry)
        with mock.patch.dict("os.environ", {"OTEL_PYTHON_PATCH_STRICT": "TRUE"}):
            with self.assertRaises(PatchMismatchError):
                self.apply(patch)
        self.assertFalse(self.registry.active)

    def test_failure_unwinds_applied_patches_and_returns_false(self) -> None:
        original = vars(self.Client)["send"]
        error = RuntimeError("boom")

        def broken_factory(sig: inspect.Signature) -> _patching.WraptWrapper:
            raise error

        second = self.patch_def(
            f"{_MODULE}:Client.other",
            name="other",
            locked_shapes=frozenset({_shape("self", "value")}),
            make_wrapper=broken_factory,
        )
        with self.assertLogs(_patching._logger, level="ERROR") as logs:
            self.assertFalse(self.apply(self.patch_def(), second))
        self.assertIn("failed to apply patches; unwinding", logs.output[0])
        self.assertIs(vars(self.Client)["send"], original)
        self.assertFalse(self.registry.active)

    def test_manual_apply_failure_unwinds_and_returns_false(self) -> None:
        original = vars(self.Client)["send"]
        events: list[str] = []
        manuals = (
            Manual(lambda: events.append("apply a"), lambda: events.append("remove a")),
            Manual(lambda: (_ for _ in ()).throw(RuntimeError("manual boom")), lambda: events.append("remove b")),
        )
        with self.assertLogs(_patching._logger, level="ERROR"):
            self.assertFalse(self.apply(self.patch_def(), extra=manuals))
        self.assertEqual(events, ["apply a", "remove a"])
        self.assertIs(vars(self.Client)["send"], original)
        self.assertFalse(self.registry.active)

    def test_unwind_continues_when_one_entry_raises(self) -> None:
        original = vars(self.Client)["send"]
        events: list[str] = []
        manuals = (
            Manual(lambda: events.append("apply a"), lambda: events.append("remove a")),
            Manual(
                lambda: events.append("apply b"),
                lambda: (_ for _ in ()).throw(RuntimeError("unwind boom")),
            ),
        )
        self.assertTrue(self.apply(self.patch_def(), extra=manuals))
        with self.assertLogs(_patching._logger, level="ERROR") as logs:
            unwind(self.registry)
        self.assertIn("failed to unwind patch", logs.output[0])
        self.assertEqual(events, ["apply a", "apply b", "remove a"])
        self.assertIs(vars(self.Client)["send"], original)
        self.assertFalse(self.registry.active)

    def test_resolve_unexpected_exception_fails_open(self) -> None:
        class BrokenDescriptor:
            def __get__(self, obj: Any, owner: Any = None) -> Any:
                raise RuntimeError("descriptor boom")

        setattr(self.Client, "broken", BrokenDescriptor())
        with self.assertLogs(_patching._logger, level="ERROR"):
            self.assertFalse(self.apply(self.patch_def(f"{_MODULE}:Client.broken", name="broken")))
        self.assertFalse(self.registry.active)

    def test_below_locked_floor_refuses_to_patch_and_logs_warning(self) -> None:
        original = vars(self.Client)["send"]
        with mock.patch.object(_patching, "_installed_version", return_value="0.9.0"):
            with self.assertLogs(_patching._logger, level="WARNING") as logs:
                applied = apply_all(
                    self.registry,
                    locked_versions=frozenset({"1.0.0", "2.0.0"}),
                    patches=(self.patch_def(),),
                )
            self.assertFalse(applied)
            self.assertIn("0.9.0 is below the minimum locked version 1.0.0", logs.output[0])
            self.assertIs(vars(self.Client)["send"], original)
            self.assertFalse(self.registry.active)

            with mock.patch.dict("os.environ", {"OTEL_PYTHON_PATCH_STRICT": "true"}):
                with self.assertRaises(PatchMismatchError):
                    apply_all(
                        self.registry,
                        locked_versions=frozenset({"1.0.0", "2.0.0"}),
                        patches=(self.patch_def(),),
                    )
            self.assertIs(vars(self.Client)["send"], original)
            self.assertFalse(self.registry.active)

    def test_no_patches_in_version_range_returns_false(self) -> None:
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertFalse(self.apply(self.patch_def(versions=">=2.0")))
        self.assertIn("no patches match installed version 1.0.0", logs.output[0])
        self.assertFalse(self.registry.active)
        with mock.patch.dict("os.environ", {"OTEL_PYTHON_PATCH_STRICT": "true"}):
            with self.assertRaises(PatchMismatchError):
                self.apply(self.patch_def(versions=">=2.0"))

    def test_manual_patches_apply_in_order_and_unwind_in_reverse(self) -> None:
        events: list[str] = []
        manuals = (
            Manual(lambda: events.append("apply a"), lambda: events.append("remove a")),
            Manual(lambda: events.append("apply b"), lambda: events.append("remove b")),
        )
        self.assertTrue(self.apply(self.patch_def(), extra=manuals))
        unwind(self.registry)
        self.assertEqual(events, ["apply a", "apply b", "remove b", "remove a"])

    def test_double_apply_warns_and_is_noop(self) -> None:
        self.assertTrue(self.apply(self.patch_def()))
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertFalse(self.apply(self.patch_def()))
        self.Client().send("r")
        self.assertEqual(len(self.calls), 1)

    def test_version_ranges_select_one_alternative(self) -> None:
        legacy_calls: list[tuple[Any, ...]] = []
        legacy = self.patch_def(name="legacy", versions="<1.0", make_wrapper=_recording_factory(legacy_calls))
        modern = self.patch_def(name="modern", versions=">=1.0")
        self.assertTrue(self.apply(legacy, modern))
        self.Client().send("r")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(legacy_calls, [])

    def test_version_ranges_skipped_when_version_unknown(self) -> None:
        with mock.patch.object(_patching, "_installed_version", return_value=None):
            with self.assertLogs(_patching._logger, level="ERROR"):
                self.apply(self.patch_def(versions=">=1.0"))
        self.Client().send("r")
        self.assertEqual(self.calls, [])

    def test_unpatchable_owner_reports_error_and_patches_nothing(self) -> None:
        setattr(self.module, "slotted_instance", self.module.Slotted())
        original = vars(self.Client)["send"]
        with self.assertLogs(_patching._logger, level="ERROR") as logs:
            applied = self.apply(
                self.patch_def(),
                self.patch_def(f"{_MODULE}:slotted_instance.send", name="slotted_send"),
            )
        self.assertFalse(applied)
        self.assertIn("owner does not support attribute assignment", logs.output[0])
        self.assertIs(vars(self.Client)["send"], original)
        self.assertFalse(self.registry.active)

    def test_unwind_preserves_attribute_replaced_after_instrument(self) -> None:
        self.assertTrue(self.apply(self.patch_def()))
        wrapper = vars(self.Client)["send"]
        with mock.patch.object(self.Client, "send") as replacement:
            with self.assertLogs(_patching._logger, level="DEBUG") as logs:
                unwind(self.registry)
            self.assertIs(vars(self.Client)["send"], replacement)
            self.assertIn("was replaced after instrumentation", logs.output[0])
            self.assertFalse(self.registry.active)
        # mock restores what it replaced, which was our wrapper.
        self.assertIs(vars(self.Client)["send"], wrapper)

    def test_reinstrument_after_wrappers_were_replaced(self) -> None:
        self.assertTrue(self.apply(self.patch_def()))

        def replaced(self: Any, request: str, *, timeout: float = 1.0) -> str:
            return f"replaced {request}"

        self.Client.send = replaced
        self.assertFalse(self.registry.active)
        self.assertTrue(self.apply(self.patch_def()))
        self.assertTrue(self.registry.active)
        self.assertEqual(self.Client().send("r"), "replaced r")
        self.assertEqual(len(self.calls), 1)
        unwind(self.registry)
        self.assertIs(vars(self.Client)["send"], replaced)

    def test_registry_with_manual_stays_active(self) -> None:
        manual = Manual(lambda: None, lambda: None)
        self.assertTrue(self.apply(self.patch_def(), extra=(manual,)))
        self.Client.send = lambda obj, request: request
        self.assertTrue(self.registry.active)

    def test_unlocked_shape_binds_live_signature_and_fails_open_on_incompatible_call(self) -> None:
        sentinel = object()

        def newer_release(
            self: Any,
            request: str,
            *,
            timeout: object = sentinel,
            new_opt: bool = False,
        ) -> str:
            return f"newer {request} {new_opt}"

        setattr(self.Client, "send", newer_release)
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertTrue(self.apply(self.patch_def()))
        self.assertIn("differs from every locked version", logs.output[0])

        # Passing the newly added keyword parameter succeeds and preserves live sentinel default.
        self.assertEqual(self.Client().send("r", new_opt=True), "newer r True")
        self.assertEqual(
            self.calls[0][1],
            {"request": "r", "timeout": sentinel, "new_opt": True},
        )
        unwind(self.registry)

        # Breaking signature change (renamed parameter): instrument() warns and applies,
        # and the first call fails open, calls the original unmodified, and self-removes.
        def renamed(self: Any, req: str, *, timeout: float = 1.0) -> str:
            return f"renamed {req}"

        _patching._limiter.reset()
        setattr(self.Client, "send", renamed)
        patch = self.patch_def(
            make_wrapper=bind_wrapper(
                "patching-testlib",
                "client_send",
                _SendArgs,
                lambda wrapped, instance, args: args.call(),
            ),
        )
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertTrue(self.apply(patch))
        self.assertIn("differs from every locked version", logs.output[0])

        _patching._limiter.reset()
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(self.Client().send("r"), "renamed r")
        self.assertIn("could not bind arguments for patch 'client_send'", logs.output[0])
        self.assertFalse(self.registry.active)
        self.assertIs(vars(self.Client)["send"], renamed)


class _SendArgs(_patching.BoundArgs):
    __slots__ = ()

    request: str
    timeout: float
    optional_flag: bool | Missing = MISSING
    _wrapped: Callable[..., Any]


class _VarArgs(_patching.BoundArgs):
    __slots__ = ()

    head: str
    var_args: tuple[Any, ...]
    var_kwargs: dict[str, Any]
    _wrapped: Callable[..., Any]


class TestBindWrapper(PatchingTestBase):
    def test_bind_wrapper_populates_required_and_optional_fields(self) -> None:
        seen: list[_SendArgs] = []

        def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            seen.append(args)
            return args.call()

        patch = self.patch_def(make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl))
        self.assertTrue(self.apply(patch))
        self.assertEqual(self.Client().send("hello", timeout=5.0), "sent hello 5.0")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].request, "hello")
        self.assertEqual(seen[0].timeout, 5.0)
        self.assertIs(seen[0].optional_flag, MISSING)
        self.assertEqual(seen[0].raw_args, ("hello",))
        self.assertEqual(seen[0].raw_kwargs, {"timeout": 5.0})
        self.assertNotIn("_wrapped", repr(seen[0]))

    def test_bind_wrapper_populates_var_args_and_var_kwargs(self) -> None:
        seen: list[_VarArgs] = []

        def fn(self: Any, head: str, *items: int, **opts: Any) -> str:
            return f"{head}:{items}:{opts}"

        setattr(self.Client, "send", fn)
        shape = (
            ("self", "POSITIONAL_OR_KEYWORD", False),
            ("head", "POSITIONAL_OR_KEYWORD", False),
            ("items", "VAR_POSITIONAL", False),
            ("opts", "VAR_KEYWORD", False),
        )

        def impl(wrapped: Callable[..., Any], instance: Any, args: _VarArgs) -> Any:
            seen.append(args)
            return args.call()

        patch = self.patch_def(
            locked_shapes=frozenset({shape}),
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _VarArgs, impl),
        )
        self.assertTrue(self.apply(patch))
        self.assertEqual(self.Client().send("h", 1, 2, flag=True), "h:(1, 2):{'flag': True}")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].head, "h")
        self.assertEqual(seen[0].var_args, (1, 2))
        self.assertEqual(seen[0].var_kwargs, {"flag": True})

    def test_bind_wrapper_positional_only_and_keyword_only_and_defaults(self) -> None:
        class _ComplexArgs(_patching.BoundArgs):
            __slots__ = ()

            pos_only: str
            pos_or_kw: int
            opt_pos: str
            kw_req: bool
            kw_opt: int

        seen: list[_ComplexArgs] = []

        def fn(
            self: Any, pos_only: str, /, pos_or_kw: int, opt_pos: str = "def", *, kw_req: bool, kw_opt: int = 9
        ) -> str:
            return f"{pos_only}:{pos_or_kw}:{opt_pos}:{kw_req}:{kw_opt}"

        setattr(self.Client, "send", fn)
        shape = (
            ("self", "POSITIONAL_ONLY", False),
            ("pos_only", "POSITIONAL_ONLY", False),
            ("pos_or_kw", "POSITIONAL_OR_KEYWORD", False),
            ("opt_pos", "POSITIONAL_OR_KEYWORD", True),
            ("kw_req", "KEYWORD_ONLY", False),
            ("kw_opt", "KEYWORD_ONLY", True),
        )

        def impl(wrapped: Callable[..., Any], instance: Any, args: _ComplexArgs) -> Any:
            seen.append(args)
            return args.call()

        patch = self.patch_def(
            locked_shapes=frozenset({shape}),
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _ComplexArgs, impl),
        )
        self.assertTrue(self.apply(patch))
        self.assertEqual(self.Client().send("p", pos_or_kw=3, kw_req=True), "p:3:def:True:9")
        self.assertEqual(seen[0].pos_only, "p")
        self.assertEqual(seen[0].pos_or_kw, 3)
        self.assertEqual(seen[0].opt_pos, "def")
        self.assertIs(seen[0].kw_req, True)
        self.assertEqual(seen[0].kw_opt, 9)

    def test_bind_wrapper_logs_and_calls_through_on_bind_failure(self) -> None:
        def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            raise AssertionError("should not be reached on bind failure")

        # Advertised __signature__ expects (request, timeout) while underlying function accepts *args, **kwargs.
        def facade_send(self: Any, *args: Any, **kwargs: Any) -> str:
            return "fallback ok"

        setattr(
            facade_send,
            "__signature__",
            inspect.Signature(
                [
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("timeout", inspect.Parameter.KEYWORD_ONLY, default=1.0),
                ]
            ),
        )
        setattr(self.Client, "send", facade_send)
        patch = self.patch_def(
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl),
        )
        self.assertTrue(self.apply(patch))

        for bad_args, bad_kwargs in (
            ((), {}),  # missing required 'request'
            (("r",), {"unexpected_kw": 1}),  # unexpected keyword argument
            (("r", 2, 3), {}),  # excess positional arguments
            (("r",), {"request": "dup"}),  # duplicate positional + keyword argument
        ):
            with self.subTest(args=bad_args, kwargs=bad_kwargs):
                _patching._limiter.reset()
                if not self.registry.active:
                    self.assertTrue(self.apply(patch))
                with self.assertLogs(_patching._logger, level="WARNING") as logs:
                    self.assertEqual(self.Client().send(*bad_args, **bad_kwargs), "fallback ok")
                self.assertIn("could not bind arguments for patch 'client_send'", logs.output[0])
                self.assertFalse(self.registry.active)

        unwind(self.registry)

        # Also verify the has_var_kw=True branch in bind_wrapper.
        def facade_varkw(self: Any, *args: Any, **kwargs: Any) -> str:
            return "fallback ok"

        setattr(
            facade_varkw,
            "__signature__",
            inspect.Signature(
                [
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("timeout", inspect.Parameter.KEYWORD_ONLY, default=1.0),
                    inspect.Parameter("extra", inspect.Parameter.VAR_KEYWORD),
                ]
            ),
        )
        setattr(self.Client, "send", facade_varkw)
        patch_varkw = self.patch_def(
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl),
        )
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertTrue(self.apply(patch_varkw))

        for bad_args, bad_kwargs in (
            ((), {"extra_opt": 1}),  # missing required 'request' with kwargs
            (("r", 2), {}),  # excess positional argument
            (("r",), {"request": "dup"}),  # duplicate positional + keyword argument
        ):
            with self.subTest(varkw=True, args=bad_args, kwargs=bad_kwargs):
                _patching._limiter.reset()
                if not self.registry.active:
                    with self.assertLogs(_patching._logger, level="WARNING"):
                        self.assertTrue(self.apply(patch_varkw))
                    _patching._limiter.reset()
                with self.assertLogs(_patching._logger, level="WARNING") as logs:
                    self.assertEqual(self.Client().send(*bad_args, **bad_kwargs), "fallback ok")
                self.assertIn("could not bind arguments for patch 'client_send'", logs.output[0])
                self.assertFalse(self.registry.active)

    def test_call_time_failures_self_remove_and_preserve_target_exceptions(self) -> None:
        original = vars(self.Client)["send"]

        # 1. Pre-wrapped exception in impl logs warning, self-removes, and calls original.
        impl_calls = 0

        def broken_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            nonlocal impl_calls
            impl_calls += 1
            raise RuntimeError("wrapper setup bug")

        patch = self.patch_def(
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, broken_impl),
        )
        self.assertTrue(self.apply(patch))
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(self.Client().send("r", timeout=2.0), "sent r 2.0")
        self.assertIn("could not bind arguments for patch 'client_send'", logs.output[0])
        self.assertEqual(impl_calls, 1)
        self.assertFalse(self.registry.active)
        self.assertIs(vars(self.Client)["send"], original)
        # Subsequent call goes directly to original without invoking broken_impl.
        self.assertEqual(self.Client().send("r2"), "sent r2 1.0")
        self.assertEqual(impl_calls, 1)

        # 2. Invalid replacement in args.call(**updates) logs warning, self-removes, and falls back to raw args.
        _patching._limiter.reset()
        replace_calls = 0

        def bad_replace_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            nonlocal replace_calls
            replace_calls += 1
            bad_updates: dict[str, Any] = {"nonexistent_param": "val"}
            return args.call(**bad_updates)

        self.assertTrue(
            self.apply(
                self.patch_def(
                    make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, bad_replace_impl),
                )
            )
        )
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(self.Client().send("r", timeout=3.0), "sent r 3.0")
        self.assertIn("could not bind arguments for patch 'client_send'", logs.output[0])
        self.assertEqual(replace_calls, 1)
        self.assertFalse(self.registry.active)
        self.assertIs(vars(self.Client)["send"], original)
        self.assertEqual(self.Client().send("r2"), "sent r2 1.0")
        self.assertEqual(replace_calls, 1)

        # 3. Inherited method self-removal deletes shadowing attribute from subclass.
        _patching._limiter.reset()
        inh_patch = self.patch_def(
            f"{_MODULE}:Client.inherited",
            name="client_inherited",
            locked_shapes=frozenset({_shape("self", "x")}),
            make_wrapper=bind_wrapper("patching-testlib", "client_inherited", _SendArgs, broken_impl),
        )
        self.assertTrue(self.apply(inh_patch))
        self.assertIn("inherited", vars(self.Client))
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertEqual(self.Client().inherited(7), 7)
        self.assertNotIn("inherited", vars(self.Client))
        self.assertFalse(self.registry.active)

        # 4. Exception raised inside the wrapped target propagates unmodified and keeps the patch active.
        target_calls = 0

        def raising_send(self: Any, request: str, *, timeout: float = 1.0) -> str:
            nonlocal target_calls
            target_calls += 1
            raise ValueError(f"target rejected {request}")

        setattr(self.Client, "send", raising_send)
        passthrough_patch = self.patch_def(
            make_wrapper=bind_wrapper(
                "patching-testlib",
                "client_send",
                _SendArgs,
                lambda wrapped, instance, args: args.call(),
            ),
        )
        self.assertTrue(self.apply(passthrough_patch))
        with self.assertRaisesRegex(ValueError, "target rejected r"):
            self.Client().send("r")
        self.assertEqual(target_calls, 1)
        self.assertTrue(self.registry.active)

    def test_caller_invalid_args_raises_typeerror_without_self_removing(self) -> None:
        seen: list[_SendArgs] = []

        def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            seen.append(args)
            return args.call()

        patch = self.patch_def(
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl),
        )
        self.assertTrue(self.apply(patch))
        with self.assertNoLogs(_patching._logger, level="WARNING"):
            for bad_args, bad_kwargs in (
                ((), {}),
                (("r",), {"unknown_kw": 1}),
                (("r", "extra"), {}),
                (("r",), {"request": "dup"}),
            ):
                with self.assertRaises(TypeError):
                    self.Client().send(*bad_args, **bad_kwargs)
                self.assertTrue(self.registry.active)
                self.assertEqual(seen, [])
        # Subsequent valid call is still instrumented.
        self.assertEqual(self.Client().send("ok", timeout=4.0), "sent ok 4.0")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].request, "ok")

    def test_direct_wrapped_call_with_mock_and_partial_preserves_target_exception(self) -> None:
        # 1. MagicMock target called directly by impl (not via args.call()).
        mock_send = mock.MagicMock(side_effect=ValueError("mock boom"))
        mock_send.__signature__ = inspect.Signature(
            [
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("timeout", inspect.Parameter.KEYWORD_ONLY, default=1.0),
            ]
        )
        setattr(self.Client, "send", mock_send)
        patch = self.patch_def(
            make_wrapper=bind_wrapper(
                "patching-testlib",
                "client_send",
                _SendArgs,
                lambda wrapped, instance, args: wrapped(*args.raw_args, **args.raw_kwargs),
            ),
        )
        self.assertTrue(self.apply(patch))
        with self.assertRaisesRegex(ValueError, "mock boom"):
            self.Client().send("r")
        self.assertEqual(mock_send.call_count, 1)
        self.assertTrue(self.registry.active)
        unwind(self.registry)

        # 2. functools.partial target called directly by impl.
        partial_calls = 0

        def raw_target(prefix: str, request: str, *, timeout: float = 1.0) -> str:
            nonlocal partial_calls
            partial_calls += 1
            raise ValueError(f"{prefix}:{request}:{timeout}")

        partial_fn = functools.partial(raw_target, "P")
        setattr(self.module, "partial_fn", partial_fn)
        partial_patch = self.patch_def(
            f"{_MODULE}:partial_fn",
            name="partial_fn",
            method=False,
            locked_shapes=frozenset({(("request", "POSITIONAL_OR_KEYWORD", False), ("timeout", "KEYWORD_ONLY", True))}),
            make_wrapper=bind_wrapper(
                "patching-testlib",
                "partial_fn",
                _SendArgs,
                lambda wrapped, instance, args: wrapped(*args.raw_args, **args.raw_kwargs),
            ),
        )
        self.assertTrue(self.apply(partial_patch))
        with self.assertRaisesRegex(ValueError, "P:r:1.0"):
            self.module.partial_fn("r")
        self.assertEqual(partial_calls, 1)
        self.assertTrue(self.registry.active)
        unwind(self.registry)

        # 3. types.MethodType wrapping a functools.partial and AsyncMock target called directly by impl.
        def method_raw_target(prefix: str, self_obj: Any, request: str, *, timeout: float = 1.0) -> str:
            nonlocal partial_calls
            partial_calls += 1
            raise ValueError(f"method:{prefix}:{request}:{timeout}")

        client = self.Client()
        bound_partial = types.MethodType(functools.partial(method_raw_target, "M"), client)
        try:
            bound_partial("r")
        except ValueError as exc:
            self.assertTrue(_patching._in_traceback(exc.__traceback__, bound_partial))

        async_mock = mock.AsyncMock(side_effect=ValueError("async mock boom"))
        try:
            asyncio.run(async_mock("r"))
        except ValueError as exc:
            self.assertTrue(_patching._in_traceback(exc.__traceback__, async_mock))

        # 4. C built-in function (no __code__) preserves target TypeError without detaching.
        class _LenArgs(_patching.BoundArgs):
            __slots__ = ()

            obj: Any

        setattr(self.module, "c_len", len)
        builtin_patch = self.patch_def(
            f"{_MODULE}:c_len",
            name="c_len",
            method=False,
            locked_shapes=frozenset({(("obj", "POSITIONAL_ONLY", False),)}),
            make_wrapper=bind_wrapper(
                "patching-testlib",
                "c_len",
                _LenArgs,
                lambda wrapped, instance, args: wrapped(*args.raw_args, **args.raw_kwargs),
            ),
        )
        self.assertTrue(self.apply(builtin_patch))
        with self.assertNoLogs(_patching._logger, level="WARNING"):
            with self.assertRaises(TypeError):
                self.module.c_len(42)
            with self.assertRaises(TypeError):
                self.module.c_len()
        self.assertTrue(self.registry.active)
        self.assertEqual(self.module.c_len("abc"), 3)
        unwind(self.registry)

    def test_call_updates_boundary_typeerror_falls_back_and_self_removes(self) -> None:
        impl_calls = 0

        def rewrite_timeout_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            nonlocal impl_calls
            impl_calls += 1
            return args.call(timeout=5.0)

        # 1. Target advertises 'timeout' via __signature__ while actual callable rejects it:
        #    _replace_raw_args succeeds, _wrapped(*args, **kwargs) raises boundary TypeError,
        #    which logs, self-removes, and falls back to raw args.
        def mismatched_send(self: Any, request: str, *, hdrs: float = 1.0) -> str:
            return f"sent {request} {hdrs}"

        setattr(
            mismatched_send,
            "__signature__",
            inspect.signature(vars(self.Client)["send"]),
        )
        setattr(self.Client, "send", mismatched_send)
        patch = self.patch_def(
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, rewrite_timeout_impl),
        )
        self.assertTrue(self.apply(patch))
        _patching._limiter.reset()
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(self.Client().send("r"), "sent r 1.0")
        self.assertIn("could not bind arguments for patch 'client_send'", logs.output[0])
        self.assertEqual(impl_calls, 1)
        self.assertFalse(self.registry.active)
        self.assertIs(vars(self.Client)["send"], mismatched_send)
        self.assertEqual(self.Client().send("r2"), "sent r2 1.0")
        self.assertEqual(impl_calls, 1)

        # 2. TypeError raised inside target body after args.call(timeout=5.0) propagates without detaching.
        def body_typeerror_send(self: Any, request: str, *, timeout: float = 1.0) -> str:
            raise TypeError(f"internal type error on {request} {timeout}")

        setattr(self.Client, "send", body_typeerror_send)
        self.assertTrue(
            self.apply(
                self.patch_def(
                    make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, rewrite_timeout_impl),
                )
            )
        )
        with self.assertNoLogs(_patching._logger, level="WARNING"):
            with self.assertRaisesRegex(TypeError, "internal type error on r 5.0"):
                self.Client().send("r")
        self.assertTrue(self.registry.active)

    def test_async_target_and_impl_call_time_failures(self) -> None:
        async def async_send(self: Any, request: str, *, timeout: float = 1.0) -> str:
            return f"async {request} {timeout}"

        setattr(self.Client, "send", async_send)
        seen: list[_SendArgs] = []

        async def async_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            seen.append(args)
            return await args.call()

        patch = self.patch_def(
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, async_impl),
        )
        self.assertTrue(self.apply(patch))

        # 1. Caller TypeError on missing required arg does not detach the patch.
        with self.assertRaises(TypeError):
            asyncio.run(self.Client().send())
        self.assertTrue(self.registry.active)
        self.assertEqual(asyncio.run(self.Client().send("ok")), "async ok 1.0")
        self.assertEqual(len(seen), 1)
        unwind(self.registry)

        # 2. Invalid args.call(nonexistent_param=1) inside async_impl logs, self-removes, and awaits raw args.
        _patching._limiter.reset()

        async def bad_replace_async_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            bad_updates: dict[str, Any] = {"nonexistent_param": 1}
            return await args.call(**bad_updates)

        self.assertTrue(
            self.apply(
                self.patch_def(
                    make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, bad_replace_async_impl),
                )
            )
        )
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(asyncio.run(self.Client().send("r", timeout=2.5)), "async r 2.5")
        self.assertIn("could not bind arguments for patch 'client_send'", logs.output[0])
        self.assertFalse(self.registry.active)

        # 3. Call-time bind mismatch against advertised __signature__ on async facade logs, self-removes, and awaits fallback.
        _patching._limiter.reset()

        async def async_facade(self: Any, *args: Any, **kwargs: Any) -> str:
            return f"facade {args} {kwargs}"

        setattr(async_facade, "__signature__", inspect.signature(async_send))
        setattr(self.Client, "send", async_facade)
        self.assertTrue(
            self.apply(
                self.patch_def(
                    make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, async_impl),
                )
            )
        )
        _patching._limiter.reset()
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(asyncio.run(self.Client().send(unexpected_kw=1)), "facade () {'unexpected_kw': 1}")
        self.assertIn("could not bind arguments for patch 'client_send'", logs.output[0])
        self.assertFalse(self.registry.active)

    def test_bind_wrapper_fails_open_when_required_args_cls_field_absent_from_sig(self) -> None:
        class _ExtraReqArgs(_patching.BoundArgs):
            __slots__ = ()

            request: str
            nonexistent_required: str

        def impl(wrapped: Callable[..., Any], instance: Any, args: _ExtraReqArgs) -> Any:
            raise AssertionError("should not be reached when required field is missing from sig")

        patch = self.patch_def(
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _ExtraReqArgs, impl),
        )
        self.assertTrue(self.apply(patch))
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(self.Client().send("r"), "sent r 1.0")
        self.assertIn("could not bind arguments for patch 'client_send'", logs.output[0])

    def test_bind_wrapper_absent_var_args_and_var_kwargs_defaults(self) -> None:
        class _RequiredVarFieldsArgs(_patching.BoundArgs):
            __slots__ = ()

            request: str
            var_args: tuple[Any, ...]
            var_kwargs: dict[str, Any]

        class _OptionalVarFieldsArgs(_patching.BoundArgs):
            __slots__ = ()

            request: str
            var_args: tuple[Any, ...] | Missing = MISSING
            var_kwargs: dict[str, Any] | Missing = MISSING

        seen_req: list[_RequiredVarFieldsArgs] = []
        seen_opt: list[_OptionalVarFieldsArgs] = []

        def impl_req(wrapped: Callable[..., Any], instance: Any, args: _RequiredVarFieldsArgs) -> Any:
            seen_req.append(args)
            return args.call()

        def impl_opt(wrapped: Callable[..., Any], instance: Any, args: _OptionalVarFieldsArgs) -> Any:
            seen_opt.append(args)
            return args.call()

        self.assertTrue(
            self.apply(
                self.patch_def(
                    make_wrapper=bind_wrapper("patching-testlib", "client_send", _RequiredVarFieldsArgs, impl_req)
                )
            )
        )
        self.assertEqual(self.Client().send("r"), "sent r 1.0")
        self.assertEqual(seen_req[0].var_args, ())
        d1 = seen_req[0].var_kwargs
        d2 = seen_req[0].var_kwargs
        self.assertEqual(d1, {})
        self.assertIsNot(d1, d2)

        unwind(self.registry)
        self.assertTrue(
            self.apply(
                self.patch_def(
                    make_wrapper=bind_wrapper("patching-testlib", "client_send", _OptionalVarFieldsArgs, impl_opt)
                )
            )
        )
        self.assertEqual(self.Client().send("r"), "sent r 1.0")
        self.assertIs(seen_opt[0].var_args, MISSING)
        self.assertIs(seen_opt[0].var_kwargs, MISSING)

    def test_bind_wrapper_positional_only_varkw_collision_and_reserved_param_names(self) -> None:
        # 1. PEP 570: POSITIONAL_ONLY parameter name also passed inside **kwargs.
        seen_var: list[_VarArgs] = []

        def fn_pos_only(self: Any, head: str, /, *items: int, **opts: Any) -> str:
            return f"{head}:{items}:{opts}"

        setattr(self.Client, "send", fn_pos_only)
        pos_only_shape = (
            ("self", "POSITIONAL_ONLY", False),
            ("head", "POSITIONAL_ONLY", False),
            ("items", "VAR_POSITIONAL", False),
            ("opts", "VAR_KEYWORD", False),
        )

        def impl_var(wrapped: Callable[..., Any], instance: Any, args: _VarArgs) -> Any:
            seen_var.append(args)
            return args.call()

        self.assertTrue(
            self.apply(
                self.patch_def(
                    locked_shapes=frozenset({pos_only_shape}),
                    make_wrapper=bind_wrapper("patching-testlib", "client_send", _VarArgs, impl_var),
                )
            )
        )
        self.assertEqual(
            self.Client().send("pos_h", 1, 2, head="kw_h", flag=True),
            "pos_h:(1, 2):{'head': 'kw_h', 'flag': True}",
        )
        self.assertEqual(seen_var[0].head, "pos_h")
        self.assertEqual(seen_var[0].var_args, (1, 2))
        self.assertEqual(seen_var[0].var_kwargs, {"head": "kw_h", "flag": True})

        unwind(self.registry)

        # 2. Unlocked library version adding a parameter named 'call' must not clobber BoundArgs.call().
        def fn_with_call_param(self: Any, request: str, timeout: float = 1.0, call: Any = None) -> str:
            return f"sent {request} {timeout} call={call}"

        setattr(self.Client, "send", fn_with_call_param)

        def impl_rewrite(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            return args.call(request=f"{args.request}!")

        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertTrue(
                self.apply(
                    self.patch_def(
                        make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl_rewrite),
                    )
                )
            )
        self.assertEqual(self.Client().send("r", call="cb"), "sent r! 1.0 call=cb")


class TestWrapperFrames(PatchingTestBase):
    def test_wrapper_adds_documented_number_of_frames(self) -> None:
        frames: list[types.FrameType] = []

        def impl(wrapped: Callable[..., Any], instance: Any, parsed: _SendArgs) -> Any:
            frames.append(sys._getframe())
            return wrapped(parsed.request, timeout=parsed.timeout)

        make_wrapper = bind_wrapper("patching-testlib", "client_send", _SendArgs, impl)
        self.assertTrue(self.apply(self.patch_def(make_wrapper=make_wrapper)))
        client = self.Client()

        def caller() -> str:
            return client.send("r")

        self.assertEqual(caller(), "sent r 1.0")
        frame: types.FrameType | None = frames[0]
        for _ in range(_patching.WRAPPER_FRAMES + 1):
            assert frame is not None
            frame = frame.f_back
        assert frame is not None
        self.assertIs(frame.f_code, caller.__code__)

    def test_coroutine_impl_adds_documented_number_of_frames(self) -> None:
        # The wrapper returns before the coroutine runs; the fail-open guard the caller awaits
        # is the only generated frame left, so async wrappers see the same depth as sync ones.
        codes: list[types.CodeType | None] = []

        async def impl(wrapped: Callable[..., Any], instance: Any, parsed: _SendArgs) -> Any:
            # Walk while the frames are live: once asyncio.run returns, finished
            # coroutine frames may no longer link to their callers (CPython 3.13).
            frame: types.FrameType | None = sys._getframe()
            for _ in range(_patching.WRAPPER_FRAMES + 1):
                frame = frame.f_back if frame is not None else None
            codes.append(frame.f_code if frame is not None else None)
            return wrapped(parsed.request)

        make_wrapper = bind_wrapper("patching-testlib", "client_send", _SendArgs, impl)
        self.assertTrue(self.apply(self.patch_def(make_wrapper=make_wrapper)))
        client = self.Client()

        async def caller() -> str:
            return await client.send("r")

        self.assertEqual(asyncio.run(caller()), "sent r 1.0")
        self.assertIs(codes[0], caller.__code__)


_THIS_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


class TestCallOutcome(PatchingTestBase):
    """Fail-open decisions based on how far ``args.call()`` got with the original."""

    def setUp(self) -> None:
        super().setUp()
        self.target_calls = 0

    def _install(self, impl: Callable[..., Any], *, async_target: bool = False, raises: bool = False) -> None:
        test = self

        def send(self: Any, request: str, *, timeout: float = 1.0) -> str:
            test.target_calls += 1
            if raises:
                raise ValueError(f"target rejected {request}")
            return f"sent {request} {timeout}"

        async def async_send(self: Any, request: str, *, timeout: float = 1.0) -> str:
            return send(self, request, timeout=timeout)

        setattr(self.Client, "send", async_send if async_target else send)
        patch = self.patch_def(make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl))
        self.assertTrue(self.apply(patch))

    def _assert_failed_open(self, logs: Any, message: str) -> None:
        self.assertEqual(len(logs.records), 1)
        self.assertIn(message, logs.output[0])
        self.assertIsNotNone(logs.records[0].exc_info)
        self.assertEqual(self.target_calls, 1)
        self.assertFalse(self.registry.active)

    def test_async_wrapper_failure_before_call_falls_back(self) -> None:
        async def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            raise RuntimeError("span setup bug")

        self._install(impl, async_target=True)
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(asyncio.run(self.Client().send("r", timeout=2.0)), "sent r 2.0")
        self._assert_failed_open(logs, "could not bind arguments for patch 'client_send'")
        # Detached: the next call goes straight to the original.
        self.assertEqual(asyncio.run(self.Client().send("r2")), "sent r2 1.0")
        self.assertEqual(self.target_calls, 2)

    def test_coroutines_from_sync_wrappers_are_not_guarded(self) -> None:
        # Only ``async def`` wrappers are guarded. A sync wrapper's coroutine may be the
        # original's own (``return wrapped(...)``), which can't be told apart from a helper,
        # so failures inside it propagate unchanged and the original is never re-run.
        test = self

        async def _library_impl(request: str) -> str:
            test.target_calls += 1
            raise ValueError(f"library rejected {request}")

        def send(self: Any, request: str, *, timeout: float = 1.0) -> Any:
            # A plain ``def`` returning another coroutine: no ``__wrapped__`` link to follow.
            return _library_impl(request)

        async def traced(args: _SendArgs) -> Any:
            raise RuntimeError("helper bug")

        def passthrough(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            return wrapped(*args.raw_args, **args.raw_kwargs)

        def helper(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            return traced(args)

        for case, impl, expected in (
            ("direct passthrough", passthrough, "library rejected r"),
            ("helper coroutine (documented limitation)", helper, "helper bug"),
        ):
            with self.subTest(case):
                unwind(self.registry)
                self.target_calls = 0
                setattr(self.Client, "send", send)
                patch = self.patch_def(make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl))
                self.assertTrue(self.apply(patch))
                with self.assertNoLogs(_patching._logger, level="WARNING"):
                    with self.assertRaisesRegex(Exception, expected):
                        asyncio.run(self.Client().send("r"))
                self.assertEqual(self.target_calls, 1 if impl is passthrough else 0)
                self.assertTrue(self.registry.active)

    def test_async_wrapper_sync_target_retry_then_reraise_propagates(self) -> None:
        test = self

        def flaky_send(self: Any, request: str, *, timeout: float = 1.0) -> str:
            test.target_calls += 1
            if test.target_calls == 1:
                raise ValueError("first attempt failed")
            return f"sent {request} {timeout}"

        async def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            try:
                return args.call()
            except ValueError as exc:
                args.call()  # the retry returns...
                raise exc  # ...but the wrapper re-raises the first failure

        setattr(self.Client, "send", flaky_send)
        patch = self.patch_def(make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl))
        self.assertTrue(self.apply(patch))
        with self.assertNoLogs(_patching._logger, level="WARNING"):
            with self.assertRaisesRegex(ValueError, "first attempt failed"):
                asyncio.run(self.Client().send("r"))
        self.assertEqual(self.target_calls, 2)
        self.assertTrue(self.registry.active)

    def test_async_original_exception_propagates_unchanged(self) -> None:
        async def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            return await args.call(timeout=3.0)

        async def direct_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            return await wrapped(*args.raw_args, **args.raw_kwargs)

        async def rewrapping_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            try:
                return await args.call()
            except ValueError as exc:
                raise RuntimeError("wrapped by the wrapper") from exc

        for case, wrapper, expected in (
            ("args.call", impl, ValueError),
            ("direct wrapped call", direct_impl, ValueError),
            ("wrapper re-raises after the original raised", rewrapping_impl, RuntimeError),
        ):
            with self.subTest(case):
                unwind(self.registry)
                self.target_calls = 0
                self._install(wrapper, async_target=True, raises=True)
                with self.assertNoLogs(_patching._logger, level="WARNING"):
                    with self.assertRaises(expected):
                        asyncio.run(self.Client().send("r"))
                self.assertEqual(self.target_calls, 1)
                self.assertTrue(self.registry.active)

    def test_async_wrapper_failure_after_original_returned_returns_result(self) -> None:
        async def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            result = await args.call(timeout=4.0)
            raise RuntimeError(f"telemetry bug after {result}")

        self._install(impl, async_target=True)
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(asyncio.run(self.Client().send("r")), "sent r 4.0")
        self._assert_failed_open(logs, "raised after the original returned")

    def test_sync_wrapper_failure_after_original_returned_returns_result(self) -> None:
        def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            result = args.call(timeout=4.0)
            raise RuntimeError(f"telemetry bug after {result}")

        self._install(impl)
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(self.Client().send("r"), "sent r 4.0")
        self._assert_failed_open(logs, "raised after the original returned")

    def test_sync_wrapper_failure_after_call_on_async_target_returns_pending_coroutine(self) -> None:
        # A sync wrapper for a coroutine target: the original runs once, when the caller awaits.
        def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            args.call()
            raise RuntimeError("telemetry bug")

        self._install(impl, async_target=True)
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertEqual(asyncio.run(self.Client().send("r")), "sent r 1.0")
        self._assert_failed_open(logs, "raised after the original returned")

    def test_sync_original_exception_propagates_unchanged(self) -> None:
        def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            try:
                return args.call()
            except ValueError as exc:
                raise RuntimeError("wrapped by the wrapper") from exc

        self._install(impl, raises=True)
        with self.assertNoLogs(_patching._logger, level="WARNING"):
            with self.assertRaisesRegex(RuntimeError, "wrapped by the wrapper"):
                self.Client().send("r")
        self.assertEqual(self.target_calls, 1)
        self.assertTrue(self.registry.active)

    def test_async_wrapper_awaiting_non_coroutine_awaitable_propagates_later_failures(self) -> None:
        # When the original's work finishes is unknown for other awaitables, so the guard
        # conservatively treats the original as still running.
        class _Pending:
            def __await__(self) -> Any:
                yield from asyncio.sleep(0).__await__()
                return "pending done"

        setattr(self.Client, "send", lambda self, request, *, timeout=1.0: _Pending())

        async def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            result = await args.call()
            raise RuntimeError(f"telemetry bug after {result}")

        patch = self.patch_def(make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl))
        self.assertTrue(self.apply(patch))
        with self.assertRaisesRegex(RuntimeError, "telemetry bug after pending done"):
            asyncio.run(self.Client().send("r"))
        self.assertTrue(self.registry.active)

    def test_async_retry_after_original_raised(self) -> None:
        # Every args.call() inside an async wrapper is tracked, not just the first.
        attempts: list[int] = []
        test = self

        async def flaky_send(self: Any, request: str, *, timeout: float = 1.0) -> str:
            test.target_calls += 1
            attempts.append(test.target_calls)
            if request == "always" or test.target_calls == 1:
                raise ValueError(f"attempt {test.target_calls} failed")
            return f"sent {request} {timeout}"

        async def retrying_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            try:
                return await args.call()
            except ValueError:
                return await args.call()

        setattr(self.Client, "send", flaky_send)
        patch = self.patch_def(make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, retrying_impl))
        self.assertTrue(self.apply(patch))

        with self.assertNoLogs(_patching._logger, level="WARNING"):
            self.assertEqual(asyncio.run(self.Client().send("r")), "sent r 1.0")
        self.assertEqual(self.target_calls, 2)

        self.target_calls = 0
        with self.assertNoLogs(_patching._logger, level="WARNING"):
            with self.assertRaisesRegex(ValueError, "attempt 2 failed"):
                asyncio.run(self.Client().send("always"))
        self.assertEqual(self.target_calls, 2)
        self.assertTrue(self.registry.active)

    def test_async_concurrent_calls_one_failing_propagates(self) -> None:
        # Concurrent originals share one BoundArgs: a success must not hide another's failure.
        test = self

        async def send(self: Any, request: str, *, timeout: float = 1.0) -> str:
            test.target_calls += 1
            await asyncio.sleep(0 if request == "good" else 0.01)
            if request == "bad":
                raise ValueError("library rejected bad")
            return f"sent {request} {timeout}"

        async def gathering_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            results = await asyncio.gather(args.call(request="good"), args.call(request="bad"))
            return " | ".join(results)

        async def gathering_after_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            try:
                await asyncio.gather(args.call(request="bad"), args.call(request="good"))
            finally:
                await asyncio.sleep(0.02)  # let the slower original settle either way

        for case, impl in (("good first", gathering_impl), ("bad first", gathering_after_impl)):
            with self.subTest(case):
                unwind(self.registry)
                self.target_calls = 0
                setattr(self.Client, "send", send)
                patch = self.patch_def(make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl))
                self.assertTrue(self.apply(patch))
                with self.assertNoLogs(_patching._logger, level="WARNING"):
                    with self.assertRaisesRegex(ValueError, "library rejected bad"):
                        asyncio.run(self.Client().send("r"))
                self.assertEqual(self.target_calls, 2)
                self.assertTrue(self.registry.active)

    def test_async_concurrent_calls_both_succeeding(self) -> None:
        async def gathering_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            results = await asyncio.gather(args.call(timeout=2.0), args.call(timeout=3.0))
            return " | ".join(results)

        self._install(gathering_impl, async_target=True)
        with self.assertNoLogs(_patching._logger, level="WARNING"):
            self.assertEqual(asyncio.run(self.Client().send("r")), "sent r 2.0 | sent r 3.0")
        self.assertEqual(self.target_calls, 2)
        self.assertTrue(self.registry.active)

    def test_async_wrapper_failure_while_original_pending_propagates(self) -> None:
        # One original finished, another is still running: not "after the original".
        async def send(self: Any, request: str, *, timeout: float = 1.0) -> str:
            if request == "slow":
                await asyncio.sleep(10)
            return f"sent {request} {timeout}"

        async def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            await args.call()
            slow = asyncio.ensure_future(args.call(request="slow"))
            await asyncio.sleep(0)  # let the slow original start
            assert not slow.done()
            raise RuntimeError("telemetry bug")

        setattr(self.Client, "send", send)
        patch = self.patch_def(make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl))
        self.assertTrue(self.apply(patch))
        with self.assertNoLogs(_patching._logger, level="WARNING"):
            with self.assertRaisesRegex(RuntimeError, "telemetry bug"):
                asyncio.run(self.Client().send("r"))
        self.assertTrue(self.registry.active)

    def test_sync_retry_after_original_raised(self) -> None:
        def retrying_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            try:
                return args.call()
            except ValueError:
                return args.call()

        self._install(retrying_impl, raises=True)
        with self.assertNoLogs(_patching._logger, level="WARNING"):
            with self.assertRaisesRegex(ValueError, "target rejected r"):
                self.Client().send("r")
        self.assertEqual(self.target_calls, 2)
        self.assertTrue(self.registry.active)

    def test_failure_after_return_reraises_in_strict_mode(self) -> None:
        def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            args.call()
            raise RuntimeError("telemetry bug")

        async def async_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            await args.call()
            raise RuntimeError("telemetry bug")

        with mock.patch.dict("os.environ", {"OTEL_PYTHON_PATCH_STRICT": "true"}):
            for case, wrapper, async_target in (("sync", impl, False), ("async", async_impl, True)):
                with self.subTest(case):
                    unwind(self.registry)
                    self.target_calls = 0
                    self._install(wrapper, async_target=async_target)
                    with self.assertNoLogs(_patching._logger, level="WARNING"):
                        with self.assertRaisesRegex(RuntimeError, "telemetry bug"):
                            if async_target:
                                asyncio.run(self.Client().send("r"))
                            else:
                                self.Client().send("r")
                    self.assertEqual(self.target_calls, 1)
                    self.assertTrue(self.registry.active)

    def test_direct_wrapped_call_then_failure_is_not_tracked(self) -> None:
        # Pins a documented limitation: calling ``wrapped`` directly bypasses outcome
        # recording, so a later wrapper failure looks like a failure before the original
        # and the fallback runs the original a second time. Use ``args.call()``.
        def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            wrapped(*args.raw_args, **args.raw_kwargs)
            raise RuntimeError("telemetry bug")

        async def async_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            await wrapped(*args.raw_args, **args.raw_kwargs)
            raise RuntimeError("telemetry bug")

        for case, wrapper, async_target in (("sync", impl, False), ("async", async_impl, True)):
            with self.subTest(case):
                unwind(self.registry)
                _patching._limiter.reset()
                self.target_calls = 0
                self._install(wrapper, async_target=async_target)
                with self.assertLogs(_patching._logger, level="WARNING") as logs:
                    if async_target:
                        self.assertEqual(asyncio.run(self.Client().send("r")), "sent r 1.0")
                    else:
                        self.assertEqual(self.Client().send("r"), "sent r 1.0")
                self.assertIn("could not bind arguments", logs.output[0])
                self.assertEqual(self.target_calls, 2)
                self.assertFalse(self.registry.active)

    def test_failure_after_return_message_without_detach(self) -> None:
        def target(x: int) -> int:
            return x

        sample = _SendArgs(raw_args=(3,), raw_kwargs={}, _wrapped=target)
        sample.call()
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            try:
                raise RuntimeError("telemetry bug")
            except RuntimeError:
                self.assertEqual(_patching._fail_open_after_return(sample), 3)
        self.assertNotIn("removing the patch", logs.output[0])

    def test_call_records_outcome(self) -> None:
        def target(x: int) -> int:
            if x < 0:
                raise ValueError("negative")
            return x * 2

        sample = _SendArgs(raw_args=(3,), raw_kwargs={}, _wrapped=target)
        self.assertEqual(sample._call_state, _patching._NOT_ENTERED)
        self.assertEqual(sample.call(), 6)
        self.assertEqual(sample._call_state, _patching._RETURNED)
        self.assertEqual(sample._call_result, 6)
        failing = _SendArgs(raw_args=(-1,), raw_kwargs={}, _wrapped=target)
        with self.assertRaises(ValueError):
            failing.call()
        self.assertEqual(failing._call_state, _patching._ENTERED)


class TestCallUpdatesThroughBindWrapper(PatchingTestBase):
    """``args.call(**updates)`` via the replacement plan precomputed at ``instrument()`` time."""

    def test_updates_use_precomputed_plan(self) -> None:
        class _AllKindsArgs(_patching.BoundArgs):
            __slots__ = ()

            a: int
            b: int
            c: int
            var_args: tuple[Any, ...]
            d: int
            var_kwargs: dict[str, Any]

        def fn(self: Any, a: int, /, b: int, c: int = 3, *items: int, d: int = 4, **extra: Any) -> tuple[Any, ...]:
            return (a, b, c, items, d, extra)

        setattr(self.Client, "send", fn)
        shape = (
            ("self", "POSITIONAL_ONLY", False),
            ("a", "POSITIONAL_ONLY", False),
            ("b", "POSITIONAL_OR_KEYWORD", False),
            ("c", "POSITIONAL_OR_KEYWORD", True),
            ("items", "VAR_POSITIONAL", False),
            ("d", "KEYWORD_ONLY", True),
            ("extra", "VAR_KEYWORD", False),
        )
        updates: dict[str, Any] = {}

        def impl(wrapped: Callable[..., Any], instance: Any, args: _AllKindsArgs) -> Any:
            return args.call(**updates)

        patch = self.patch_def(
            locked_shapes=frozenset({shape}),
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _AllKindsArgs, impl),
        )
        self.assertTrue(self.apply(patch))
        client = self.Client()
        # The plan is built once at instrument() time; calls must not rebuild it.
        with mock.patch.object(_patching, "_ReplacePlan", side_effect=AssertionError("plan rebuilt")):
            for case, new, call_args, call_kwargs, expected in (
                ("positional-only in place", {"a": 10}, (1, 2), {}, (10, 2, 3, (), 4, {})),
                ("positional-or-keyword passed by keyword", {"b": 20}, (1,), {"b": 2}, (1, 20, 3, (), 4, {})),
                ("omitted positional-or-keyword", {"c": 30}, (1, 2), {}, (1, 2, 30, (), 4, {})),
                ("keyword-only", {"d": 40}, (1, 2), {"d": 5}, (1, 2, 3, (), 40, {})),
                ("var-keyword", {"z": 1}, (1, 2), {}, (1, 2, 3, (), 4, {"z": 1})),
                ("var_args", {"var_args": (7, 8)}, (1, 2, 3, 9), {}, (1, 2, 3, (7, 8), 4, {})),
                ("var_args moves keyword positionals", {"var_args": (7,)}, (1,), {"b": 2}, (1, 2, 3, (7,), 4, {})),
                ("MISSING is ignored", {"b": MISSING}, (1, 2), {}, (1, 2, 3, (), 4, {})),
            ):
                with self.subTest(case):
                    updates.clear()
                    updates.update(new)
                    with self.assertNoLogs(_patching._logger, level="WARNING"):
                        self.assertEqual(client.send(*call_args, **call_kwargs), expected)
                    self.assertTrue(self.registry.active)
        self.assertTrue(self.registry.active)

    def test_updates_unknown_parameter_falls_back_with_precomputed_plan(self) -> None:
        def impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            bad: dict[str, Any] = {"nonexistent": 1}
            return args.call(**bad)

        patch = self.patch_def(make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, impl))
        self.assertTrue(self.apply(patch))
        with mock.patch.object(_patching, "_ReplacePlan", side_effect=AssertionError("plan rebuilt")):
            with self.assertLogs(_patching._logger, level="WARNING") as logs:
                self.assertEqual(self.Client().send("r", timeout=2.0), "sent r 2.0")
        self.assertEqual(len(logs.records), 1)
        self.assertFalse(self.registry.active)


class TestStdlibRegistry(PatchingTestBase):
    """``Registry(stdlib=True)``: the lock is keyed by Python version, not a distribution's."""

    def setUp(self) -> None:
        super().setUp()
        self.registry = Registry("patching-testlib", stdlib=True)
        self.addCleanup(unwind, self.registry)

    def test_installed_version_is_running_python(self) -> None:
        # _installed_version is mocked to "1.0.0" by the base class; stdlib must ignore it.
        self.assertEqual(self.registry.installed_version(), _THIS_PYTHON)
        self.assertEqual(self.registry.subject, "python")
        self.assertEqual(Registry("x").subject, "x")
        self.assertEqual(Registry("x").installed_version(), "1.0.0")

    def test_running_python_in_lock_is_silent(self) -> None:
        with self.assertNoLogs(_patching._logger, level="DEBUG"):
            self.assertTrue(
                apply_all(self.registry, locked_versions=frozenset({"3.0", _THIS_PYTHON}), patches=(self.patch_def(),))
            )

    def test_running_python_outside_lock_names_python(self) -> None:
        patch = self.patch_def(locked_shapes=frozenset({_shape("self", "something_else")}))
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertTrue(apply_all(self.registry, locked_versions=frozenset({"2.7"}), patches=(patch,)))
        self.assertIn(f"python {_THIS_PYTHON} was not in the lock", logs.output[0])

    def test_running_python_outside_lock_with_locked_shapes_logs_debug(self) -> None:
        with self.assertLogs(_patching._logger, level="DEBUG") as logs:
            self.assertTrue(apply_all(self.registry, locked_versions=frozenset({"2.7"}), patches=(self.patch_def(),)))
        self.assertEqual([record.levelname for record in logs.records], ["DEBUG"])
        self.assertIn(f"python {_THIS_PYTHON} was not in the lock", logs.output[0])

    def test_version_ranges_use_python_version(self) -> None:
        legacy_calls: list[tuple[Any, ...]] = []
        legacy = self.patch_def(name="legacy", versions="<3.0", make_wrapper=_recording_factory(legacy_calls))
        modern = self.patch_def(name="modern", versions=">=3.0")
        self.assertTrue(apply_all(self.registry, locked_versions=frozenset({_THIS_PYTHON}), patches=(legacy, modern)))
        self.Client().send("r")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(legacy_calls, [])

    def test_instance_scope_uses_python_version(self) -> None:
        patch = InstancePatchDef(
            name="client_send",
            attr="send",
            locked_shapes=frozenset({SEND_SHAPE}),
            make_wrapper=_recording_factory(self.calls),
            versions=">=3.0",
        )
        client = self.Client()
        self.assertTrue(apply_instance(client, (patch,), class_registry=self.registry))
        client.send("r")
        self.assertEqual(len(self.calls), 1)
        remove_instance(client, ("send",), class_registry=self.registry)


class TestLogging(TestCase):
    def setUp(self) -> None:
        _patching._limiter.reset()

    def test_bind_failure_is_rate_limited(self) -> None:
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            try:
                raise ValueError("bad")
            except ValueError:
                _patching.log_bind_failure("lib", "patch")
                _patching.log_bind_failure("lib", "patch")
            _patching.log_bind_failure("lib", "other")
        self.assertEqual(len(logs.records), 2)
        self.assertIsNotNone(logs.records[0].exc_info)

    def test_bind_failure_logs_again_after_interval(self) -> None:
        with mock.patch.object(_patching.time, "monotonic", side_effect=[0.0, 1.0, 1000.0]):
            with self.assertLogs(_patching._logger, level="WARNING") as logs:
                for _ in range(3):
                    _patching.log_bind_failure("lib", "patch", exc_info=False)
        self.assertEqual(len(logs.records), 2)
        self.assertIn("repeated", logs.output[1])


class TestInstanceScope(PatchingTestBase):
    def instance_def(self, **kwargs: Any) -> InstancePatchDef:
        values: dict[str, Any] = {
            "name": "client_send",
            "attr": "send",
            "locked_shapes": frozenset({SEND_SHAPE}),
            "make_wrapper": _recording_factory(self.calls),
        }
        values.update(kwargs)
        return InstancePatchDef(**values)

    def test_instance_isolation_and_exact_restore(self) -> None:
        client, other = self.Client(), self.Client()
        before = dict(vars(client))
        class_send = vars(self.Client)["send"]
        self.assertTrue(apply_instance(client, [self.instance_def()], class_registry=self.registry))

        self.assertEqual(client.send("r", timeout=2.0), "sent r 2.0")
        other.send("r")
        self.assertEqual(len(self.calls), 1)
        self.assertIs(self.calls[0][0], client)
        self.assertEqual(self.calls[0][1], {"request": "r", "timeout": 2.0})
        self.assertIs(vars(self.Client)["send"], class_send)
        self.assertEqual(vars(other), {})

        remove_instance(client, ["send"], class_registry=self.registry)
        self.assertEqual(vars(client), before)

    def test_double_instance_instrument_warns(self) -> None:
        client = self.Client()
        apply_instance(client, [self.instance_def()], class_registry=self.registry)
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertFalse(apply_instance(client, [self.instance_def()], class_registry=self.registry))
        client.send("r")
        self.assertEqual(len(self.calls), 1)

    def test_slotted_object_is_left_alone(self) -> None:
        obj = self.module.Slotted()
        with self.assertLogs(_patching._logger, level="ERROR"):
            self.assertFalse(apply_instance(obj, [self.instance_def()], class_registry=self.registry))

    def test_class_patched_instance_runs_wrapper_once(self) -> None:
        self.apply(self.patch_def())
        client = self.Client()
        self.assertFalse(apply_instance(client, [self.instance_def()], class_registry=self.registry))
        client.send("r")
        self.assertEqual(len(self.calls), 1)

    def test_class_patched_subclass_overriding_method_runs_wrapper_once(self) -> None:
        self.apply(self.patch_def())
        sub = self.module.SubClient()
        self.assertFalse(apply_instance(sub, [self.instance_def()], class_registry=self.registry))
        self.assertEqual(sub.send("r"), "sent r 1.0")
        self.assertEqual(len(self.calls), 1)

    def test_remove_instance_keeps_object_uninstrumented_while_class_patched(self) -> None:
        self.apply(self.patch_def())
        client, other = self.Client(), self.Client()
        remove_instance(client, ["send"], class_registry=self.registry)
        self.assertEqual(client.send("r"), "sent r 1.0")
        self.assertEqual(self.calls, [])
        other.send("r")
        self.assertEqual(len(self.calls), 1)

        # Instrumenting the object again replaces the opt-out.
        unwind(self.registry)
        self.assertTrue(apply_instance(client, [self.instance_def()], class_registry=self.registry))
        client.send("r")
        self.assertEqual(len(self.calls), 2)

    def test_instance_version_ranges(self) -> None:
        legacy_calls: list[tuple[Any, ...]] = []
        client = self.Client()
        patches = [
            self.instance_def(name="legacy", versions="<1.0", make_wrapper=_recording_factory(legacy_calls)),
            self.instance_def(name="modern", versions=">=1.0"),
        ]
        self.assertTrue(apply_instance(client, patches, class_registry=self.registry))
        client.send("r")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(legacy_calls, [])

    def test_unlocked_instance_shape_binds_live_signature(self) -> None:
        def newer(obj: Any, request: str, *, timeout: float = 1.0, extra: bool = False) -> str:
            return f"newer {request} {timeout} {extra}"

        setattr(self.Client, "send", newer)
        client = self.Client()
        with self.assertLogs(_patching._logger, level="WARNING") as logs:
            self.assertTrue(apply_instance(client, [self.instance_def()], class_registry=self.registry))
        self.assertIn("differs from every locked version", logs.output[0])
        self.assertEqual(client.send("r", timeout=3.0, extra=True), "newer r 3.0 True")
        self.assertEqual(self.calls[0][1], {"request": "r", "timeout": 3.0, "extra": True})
        remove_instance(client, ["send"], class_registry=self.registry)

    def test_apply_instance_caches_installed_version_on_registry(self) -> None:
        _REAL_INSTALLED_VERSION.cache_clear()
        self.addCleanup(_REAL_INSTALLED_VERSION.cache_clear)
        with (
            mock.patch.object(_patching, "_installed_version", _REAL_INSTALLED_VERSION),
            mock.patch.object(_patching, "version", return_value="1.0.0") as mock_ver,
        ):
            for _ in range(5):
                client = self.Client()
                self.assertTrue(
                    apply_instance(
                        client,
                        [self.instance_def()],
                        class_registry=self.registry,
                        locked_versions=frozenset({"1.0.0"}),
                    )
                )
            self.assertEqual(mock_ver.call_count, 1)

    def test_instance_below_locked_floor_refuses_to_patch(self) -> None:
        client = self.Client()
        with mock.patch.object(_patching, "_installed_version", return_value="0.5.0"):
            with self.assertLogs(_patching._logger, level="WARNING") as logs:
                applied = apply_instance(
                    client,
                    [self.instance_def()],
                    class_registry=self.registry,
                    locked_versions=frozenset({"1.0.0"}),
                )
        self.assertFalse(applied)
        self.assertIn("0.5.0 is below the minimum locked version 1.0.0", logs.output[0])
        self.assertEqual(vars(client), {})

    def test_instance_mid_apply_exception_unwinds_and_returns_false(self) -> None:
        self.apply(self.patch_def())
        client = self.Client()
        remove_instance(client, ["send"], class_registry=self.registry)
        opt_out = vars(client)["send"]
        unwind(self.registry)

        def broken_factory(sig: inspect.Signature) -> _patching.WraptWrapper:
            raise RuntimeError("instance factory boom")

        patches = [
            self.instance_def(),
            self.instance_def(
                name="client_other",
                attr="other",
                locked_shapes=frozenset({_shape("self", "value")}),
                make_wrapper=broken_factory,
            ),
        ]
        with self.assertLogs(_patching._logger, level="ERROR") as logs:
            self.assertFalse(apply_instance(client, patches, class_registry=self.registry))
        self.assertIn("failed to instrument", logs.output[0])
        self.assertEqual(vars(client), {"send": opt_out})

    def test_instance_self_removes_on_call_time_failure(self) -> None:
        client = self.Client()
        impl_calls = 0

        def broken_impl(wrapped: Callable[..., Any], instance: Any, args: _SendArgs) -> Any:
            nonlocal impl_calls
            impl_calls += 1
            raise RuntimeError("instance wrapper bug")

        patch = self.instance_def(
            make_wrapper=bind_wrapper("patching-testlib", "client_send", _SendArgs, broken_impl),
        )
        self.assertTrue(apply_instance(client, [patch], class_registry=self.registry))
        self.assertIn("send", vars(client))
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertEqual(client.send("r"), "sent r 1.0")
        self.assertEqual(impl_calls, 1)
        self.assertNotIn("send", vars(client))
        self.assertEqual(client.send("r2"), "sent r2 1.0")
        self.assertEqual(impl_calls, 1)


class TestReplaceRawArgs(TestCase):
    def setUp(self) -> None:
        _patching._limiter.reset()

    def test_replace_positional_and_keyword_args(self) -> None:
        def fn(a: str, /, b: int = 1, *, c: float = 2.0) -> None:
            pass

        sig = inspect.signature(fn)
        raw_args = ("orig_a",)
        raw_kwargs = {"c": 3.0}
        new_args, new_kwargs = _patching.replace_raw_args(sig, raw_args, raw_kwargs, {"a": "new_a", "b": 10, "c": 4.0})
        self.assertEqual(new_args, ("new_a",))
        self.assertEqual(new_kwargs, {"b": 10, "c": 4.0})
        self.assertEqual(raw_args, ("orig_a",))
        self.assertEqual(raw_kwargs, {"c": 3.0})

    def test_replace_omitted_positional_only_parameter(self) -> None:
        def fn_no_varkw(a: str = "a0", b: int = 1, /, c: int = 2) -> None:
            pass

        def fn_with_varkw(a: str = "a0", b: int = 1, /, **kwargs: Any) -> None:
            pass

        for fn in (fn_no_varkw, fn_with_varkw):
            with self.subTest(fn.__name__):
                sig = inspect.signature(fn)
                new_args, new_kwargs = _patching.replace_raw_args(sig, ("x",), {}, {"b": 99})
                self.assertEqual(new_args, ("x", 99))
                self.assertEqual(new_kwargs, {})

                # Preceding omitted positional-only parameter filled from its default.
                new_args, new_kwargs = _patching.replace_raw_args(sig, (), {}, {"b": 99})
                self.assertEqual(new_args, ("a0", 99))
                self.assertEqual(new_kwargs, {})

        def fn_required_preceding(a: str, b: int = 1, /) -> None:
            pass

        with self.assertRaises(TypeError):
            _patching._replace_raw_args(inspect.signature(fn_required_preceding), (), {}, {"b": 99})
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertEqual(
                _patching.replace_raw_args(inspect.signature(fn_required_preceding), (), {}, {"b": 99}),
                ((), {}),
            )

    def test_replace_var_keyword_and_unknown_parameter(self) -> None:
        def fn_strict(a: int) -> None:
            pass

        def fn_varkw(a: int, **kwargs: Any) -> None:
            pass

        with self.assertRaises(KeyError):
            _patching._replace_raw_args(inspect.signature(fn_strict), (1,), {}, {"extra": 2})
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertEqual(
                _patching.replace_raw_args(inspect.signature(fn_strict), (1,), {}, {"extra": 2}),
                ((1,), {}),
            )

        new_args, new_kwargs = _patching.replace_raw_args(inspect.signature(fn_varkw), (1,), {}, {"a": 9, "extra": 2})
        self.assertEqual(new_args, (9,))
        self.assertEqual(new_kwargs, {"extra": 2})

    def test_replace_when_sig_is_none(self) -> None:
        new_args, new_kwargs = _patching.replace_raw_args(None, (1,), {"x": 2}, {"y": 3})
        self.assertEqual(new_args, (1,))
        self.assertEqual(new_kwargs, {"x": 2, "y": 3})

    def test_replace_var_args_splices_after_leading_positionals(self) -> None:
        def fn(a: int, /, b: int, *items: int, c: int = 0, **kwargs: Any) -> None:
            pass

        sig = inspect.signature(fn)
        for key in ("var_args", "items"):
            with self.subTest(key):
                new_args, new_kwargs = _patching.replace_raw_args(sig, (1, 2, 3, 4), {"c": 5}, {key: (9, 8, 7)})
                self.assertEqual(new_args, (1, 2, 9, 8, 7))
                self.assertEqual(new_kwargs, {"c": 5})

        new_args, new_kwargs = _patching.replace_raw_args(sig, (1, 2, 3), {}, {"var_args": (), "b": 20})
        self.assertEqual(new_args, (1, 20))
        self.assertEqual(new_kwargs, {})

    def test_replace_var_args_moves_keyword_positionals(self) -> None:
        def fn(a: int, b: int = 5, *items: int) -> None:
            pass

        sig = inspect.signature(fn)
        new_args, new_kwargs = _patching.replace_raw_args(sig, (), {"a": 1}, {"var_args": [7, 8]})
        self.assertEqual(new_args, (1, 5, 7, 8))
        self.assertEqual(new_kwargs, {})

        new_args, new_kwargs = _patching.replace_raw_args(sig, (), {"a": 1}, {"var_args": (7,), "b": 6})
        self.assertEqual(new_args, (1, 6, 7))
        self.assertEqual(new_kwargs, {})

        with self.assertRaises(TypeError):
            _patching._replace_raw_args(sig, (), {}, {"var_args": (7,)})
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertEqual(_patching.replace_raw_args(sig, (), {}, {"var_args": (7,)}), ((), {}))

    def test_replace_var_args_without_var_positional(self) -> None:
        def fn_varkw(a: int, **kwargs: Any) -> None:
            pass

        with self.assertRaises(KeyError):
            _patching._replace_raw_args(inspect.signature(fn_varkw), (1,), {}, {"var_args": (2,)})
        with self.assertRaises(KeyError):
            _patching._replace_raw_args(None, (1,), {}, {"var_args": (2,)})
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertEqual(
                _patching.replace_raw_args(inspect.signature(fn_varkw), (1,), {}, {"var_args": (2,)}),
                ((1,), {}),
            )
        _patching._limiter.reset()
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertEqual(
                _patching.replace_raw_args(None, (1,), {}, {"var_args": (2,)}),
                ((1,), {}),
            )

    def test_replace_duplicate_positional_and_keyword_raises_and_positional_only_preserves_varkw(self) -> None:
        def fn_pos_or_kw(a: str, b: int = 1) -> None:
            pass

        with self.assertRaises(TypeError):
            _patching._replace_raw_args(inspect.signature(fn_pos_or_kw), ("pos",), {"a": "dup"}, {"a": "new"})
        with self.assertLogs(_patching._logger, level="WARNING"):
            self.assertEqual(
                _patching.replace_raw_args(inspect.signature(fn_pos_or_kw), ("pos",), {"a": "dup"}, {"a": "new"}),
                (("pos",), {"a": "dup"}),
            )

        def fn_pos_only_varkw(a: str, /, **kwargs: Any) -> None:
            pass

        new_args, new_kwargs = _patching.replace_raw_args(
            inspect.signature(fn_pos_only_varkw), ("pos_a",), {"a": "varkw_a"}, {"a": "new_pos_a"}
        )
        self.assertEqual(new_args, ("new_pos_a",))
        self.assertEqual(new_kwargs, {"a": "varkw_a"})

    def test_bound_args_direct_construction_and_call(self) -> None:
        class SampleArgs(_patching.BoundArgs):
            __slots__ = ()
            topic: str
            opt: int | _patching.Missing = _patching.MISSING
            var_args: tuple[Any, ...]
            var_kwargs: dict[str, Any]

        called_with: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def wrapped(topic: str, opt: int = 10, **kwargs: Any) -> str:
            called_with.append(((topic, opt), kwargs))
            return "ok"

        sample = SampleArgs(
            raw_args=("orders",),
            raw_kwargs={"partition": 0},
            _wrapped=wrapped,
        )
        self.assertIsInstance(sample, SampleArgs)
        self.assertIsNone(sample._sig)
        self.assertEqual(sample.call(), "ok")
        self.assertEqual(sample.call(topic=MISSING), "ok")
        self.assertEqual(sample.call(topic="events", opt=MISSING), "ok")
        self.assertEqual(
            called_with,
            [
                (("orders", 10), {"partition": 0}),
                (("orders", 10), {"partition": 0}),
                (("events", 10), {"partition": 0}),
            ],
        )
