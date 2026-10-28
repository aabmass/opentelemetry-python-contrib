# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI failure handling for ``lock`` (L1) and ``check``/``report --all`` (L5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cli
import pytest
from patchgen.generate import GenerateError
from patchgen.lock import LockError

SPEC = "library: fakelib\nversions: '>=1.0'\npatches: [{name: a, target: 'fakelib:f'}]\n"


def _pkg(root: Path, name: str, spec: str | None = SPEC) -> Path:
    pkg = root / name
    pkg.mkdir()
    if spec is not None:
        (pkg / "patches.yaml").write_text(spec, encoding="utf-8")
    return pkg


def test_lock_reports_every_failed_spec_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = _pkg(tmp_path, "missing", spec=None)
    broken = _pkg(tmp_path, "broken")
    crashing = _pkg(tmp_path, "crashing")
    good = _pkg(tmp_path, "good")

    def fake_write_lock(spec: Any, **_kwargs: Any) -> Path:
        name = spec.package_dir.name
        if name == "broken":
            raise LockError("all selected versions of fakelib failed to lock:\n  1.0: cannot import fakelib")
        if name == "crashing":
            raise ValueError("unexpected")
        return spec.lock_path

    monkeypatch.setattr(cli, "write_lock", fake_write_lock)
    assert cli.main(["lock", str(missing), str(broken), str(crashing), str(good)]) == 1
    out, err = capsys.readouterr()
    assert f"FAILED {missing}: no patches.yaml found; lock not written" in err
    assert "  1.0: cannot import fakelib" in err
    assert f"FAILED {broken / 'patches.yaml'}: all selected versions of fakelib failed to lock; lock not written" in err
    assert f"FAILED {crashing / 'patches.yaml'}: ValueError: unexpected; lock not written" in err
    assert err.rstrip().splitlines()[-1].startswith(f"FAILED {crashing}")
    assert f"wrote {good / 'patches.lock.json'}" in out


def test_lock_success_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = _pkg(tmp_path, "good")
    monkeypatch.setattr(cli, "write_lock", lambda spec, **_kw: spec.lock_path)
    assert cli.main(["lock", str(good)]) == 0


def test_check_all_keeps_going_and_fails_at_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _pkg(tmp_path, "a-bad")
    good = _pkg(tmp_path, "b-good")
    module = good / "_patches.py"
    module.write_text("generated\n", encoding="utf-8")

    def fake_generate(package: Path) -> list[tuple[Path, str]]:
        if package == bad:
            raise GenerateError(f"{package}: patches.lock.json missing")
        return [(module, "generated\n")]

    monkeypatch.setattr(cli, "_all_packages", lambda: [bad, good])
    monkeypatch.setattr(cli, "generate_package", fake_generate)
    monkeypatch.setattr(cli, "_print_any_report", lambda _package: None)
    assert cli.main(["check", "--all"]) == 1
    out, err = capsys.readouterr()
    assert f"FAILED {bad}: {bad}: patches.lock.json missing" in err
    assert f"ok {module}" in out
    assert "1 ok, 0 stale, 1 failed" in err


def test_report_all_keeps_going_and_fails_at_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _pkg(tmp_path, "a-bad", spec=None)
    good = _pkg(tmp_path, "b-good")
    monkeypatch.setattr(cli, "_all_packages", lambda: [bad, good])
    monkeypatch.setattr(cli, "any_report", lambda _spec, info=False: [])
    assert cli.main(["report", "--all"]) == 1
    out, err = capsys.readouterr()
    assert f"FAILED {bad}: {bad}: no patches.yaml found" in err
    assert "fakelib: no Any gaps (patches.yaml)" in out
