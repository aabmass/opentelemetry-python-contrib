# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

OTEL_PYTHON_DISABLED_INSTRUMENTATIONS = "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"
"""
.. envvar:: OTEL_PYTHON_DISABLED_INSTRUMENTATIONS
"""

OTEL_PYTHON_DISTRO = "OTEL_PYTHON_DISTRO"
"""
.. envvar:: OTEL_PYTHON_DISTRO
"""

OTEL_PYTHON_CONFIGURATOR = "OTEL_PYTHON_CONFIGURATOR"
"""
.. envvar:: OTEL_PYTHON_CONFIGURATOR
"""

OTEL_PYTHON_AUTO_INSTRUMENTATION_EXPERIMENTAL_GEVENT_PATCH = (
    "OTEL_PYTHON_AUTO_INSTRUMENTATION_EXPERIMENTAL_GEVENT_PATCH"
)
"""
.. envvar:: OTEL_PYTHON_AUTO_INSTRUMENTATION_EXPERIMENTAL_GEVENT_PATCH
"""

OTEL_SEMCONV_STABILITY_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"
"""
.. envvar:: OTEL_SEMCONV_STABILITY_OPT_IN

Opt-in to stable semantic convention signals.
Comma-separated list of signals, e.g. ``http``, ``http/dup``.
"""

OTEL_PYTHON_PATCH_STRICT = "OTEL_PYTHON_PATCH_STRICT"
"""
.. envvar:: OTEL_PYTHON_PATCH_STRICT

When ``true``, patchgen-generated instrumentations raise instead of logging a
warning when the installed library version or a target signature does not
match the lock the instrumentation was generated against. Intended for CI.
"""
