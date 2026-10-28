# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jinja2>=3.1",
#   "packaging>=23",
#   "pyyaml>=6",
#   "ruff==0.16.1",
# ]
# ///

"""patchgen command line: ``lock``, ``generate``, ``check`` and ``report``.

Usage::

    uv run scripts/patchgen/cli.py lock instrumentation/opentelemetry-instrumentation-httpx
    uv run scripts/patchgen/cli.py generate instrumentation/opentelemetry-instrumentation-httpx
    uv run scripts/patchgen/cli.py check --all
    uv run scripts/patchgen/cli.py report instrumentation/opentelemetry-instrumentation-httpx

``generate`` and ``check`` also print the ``Any`` report to stderr; ``report`` prints only that,
to stdout, and writes nothing.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patchgen.generate import GenerateError, any_report, any_report_package, generate_package
from patchgen.lock import LockError, write_lock
from patchgen.spec import SpecError, find_specs, load_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_GLOBS = ("instrumentation/*", "instrumentation-genai/*", "exporter/*", "propagator/*", "util/*")


def _failure_reason(exc: BaseException) -> str:
    known = (SpecError, GenerateError, LockError)
    return str(exc) if isinstance(exc, known) else f"{type(exc).__name__}: {exc}"


def _print_failed(subject: object, exc: BaseException, suffix: str = "") -> None:
    """Print the full reason, then one final ``FAILED <subject>: <first line><suffix>`` line."""
    lines = _failure_reason(exc).splitlines() or [type(exc).__name__]
    for line in lines[1:]:
        print(line, file=sys.stderr)
    print(f"FAILED {subject}: {lines[0].rstrip(':')}{suffix}", file=sys.stderr)


def _lock(args: argparse.Namespace) -> int:
    failed = False
    for package in args.packages:
        specs = find_specs(Path(package))
        if not specs:
            print(f"FAILED {package}: no patches.yaml found; lock not written", file=sys.stderr)
            failed = True
            continue
        for spec_path in specs:
            try:
                spec = load_spec(spec_path)
                path = write_lock(
                    spec,
                    python=args.python,
                )
            except Exception as exc:  # report every spec, keep going, fail at the end
                _print_failed(spec_path, exc, "; lock not written")
                failed = True
                continue
            print(f"wrote {path}")
    return 1 if failed else 0


def _print_any_report(package: Path) -> None:
    for line in any_report_package(package):
        print(line, file=sys.stderr)


def _generate(args: argparse.Namespace) -> int:
    packages = _selected_packages(args)
    if not packages:
        print("no packages to generate", file=sys.stderr)
        return 0 if args.all else 1
    errored = 0
    for package in packages:
        try:
            for path, source in generate_package(package):
                path.write_text(source, encoding="utf-8")
                print(f"wrote {path}")
            _print_any_report(package)
        except (SpecError, GenerateError, LockError) as exc:
            errored += 1
            _print_failed(package, exc)
    if errored:
        print(f"{len(packages) - errored} ok, {errored} failed", file=sys.stderr)
    return 1 if errored else 0


def _all_packages() -> list[Path]:
    found: set[Path] = set()
    for pattern in PACKAGE_GLOBS:
        for package in REPO_ROOT.glob(pattern):
            if find_specs(package):
                found.add(package)
    return sorted(found)


def _selected_packages(args: argparse.Namespace) -> list[Path]:
    return _all_packages() if args.all else [Path(p) for p in args.packages]


def _check(args: argparse.Namespace) -> int:
    packages = _selected_packages(args)
    if not packages:
        print("no packages to check", file=sys.stderr)
        return 0 if args.all else 1
    ok = stale = errored = 0
    with tempfile.TemporaryDirectory(prefix="patchgen-check-") as tmp:
        for package in packages:
            try:
                for path, source in generate_package(package):
                    fresh = Path(tmp) / f"{package.name}-{path.name}"
                    fresh.write_text(source, encoding="utf-8")
                    current = path.read_text(encoding="utf-8") if path.is_file() else ""
                    if current == fresh.read_text(encoding="utf-8"):
                        ok += 1
                        print(f"ok {path}")
                        continue
                    stale += 1
                    print(f"STALE {path}: regenerate with `uv run scripts/patchgen/cli.py generate {package}`")
                    sys.stdout.writelines(
                        difflib.unified_diff(
                            current.splitlines(keepends=True),
                            source.splitlines(keepends=True),
                            fromfile=str(path),
                            tofile="generated",
                        )
                    )
                _print_any_report(package)
            except (SpecError, GenerateError, LockError) as exc:
                errored += 1
                _print_failed(package, exc)
    if len(packages) > 1 or errored:
        print(f"{ok} ok, {stale} stale, {errored} failed", file=sys.stderr)
    return 1 if stale or errored else 0


def _report(args: argparse.Namespace) -> int:
    packages = _selected_packages(args)
    if not packages:
        print("no packages to report on", file=sys.stderr)
        return 0 if args.all else 1
    errored = 0
    for package in packages:
        try:
            specs = find_specs(package)
            if not specs:
                raise GenerateError(f"{package}: no patches.yaml found")
            for spec_path in specs:
                spec = load_spec(spec_path)
                gaps = any_report(spec)
                if not gaps:
                    print(f"{spec.library}: no Any gaps ({spec_path.name})")
                for line in gaps:
                    print(line)
        except (SpecError, GenerateError, LockError) as exc:
            errored += 1
            _print_failed(package, exc)
    if errored:
        print(f"{len(packages) - errored} ok, {errored} failed", file=sys.stderr)
    return 1 if errored else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="patchgen", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    lock = sub.add_parser("lock", help="resolve versions and record target signatures (network)")
    lock.add_argument("packages", nargs="+")
    lock.add_argument("--python", help="Python version for the throwaway environments (default: current)")
    lock.set_defaults(func=_lock)

    generate = sub.add_parser("generate", help="generate the patch module from spec + lock (hermetic)")
    generate.add_argument("packages", nargs="*")
    generate.add_argument("--all", action="store_true", help="generate every package with a patches.yaml")
    generate.set_defaults(func=_generate)

    check = sub.add_parser("check", help="fail if a generated module is out of date")
    check.add_argument("packages", nargs="*")
    check.add_argument("--all", action="store_true", help="check every package with a patches.yaml")
    check.set_defaults(func=_check)

    report = sub.add_parser("report", help="list generated types that are Any without a types: override")
    report.add_argument("packages", nargs="*")
    report.add_argument("--all", action="store_true", help="report on every package with a patches.yaml")
    report.set_defaults(func=_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SpecError, GenerateError, LockError) as exc:
        print(f"patchgen: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
