from pathlib import Path

import pytest
from patchgen.spec import SCOPES, SpecError, _parse_patch, default_python_versions, default_versions, load_spec


def test_parse_patch_rejects_shadowing_builtins():
    path = Path("patches.yaml")

    valid_spec = {
        "name": "my_compile",
        "target": "module:target",
    }
    _parse_patch(path, valid_spec, 0)

    for builtin_name in ["compile", "open", "print", "id"]:
        invalid_spec = {
            "name": builtin_name,
            "target": "module:target",
        }
        with pytest.raises(SpecError, match="name shadows a Python built-in"):
            _parse_patch(path, invalid_spec, 0)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "patches.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_spec_defaults_are_not_stdlib_and_have_no_extras(tmp_path: Path) -> None:
    spec = load_spec(_write(tmp_path, "library: fakelib\npatches: [{name: a, target: 'fakelib:f'}]\n"))
    assert spec.stdlib is False
    assert spec.extras == ()
    assert spec.subject == "fakelib"
    assert spec.install_requirement("1.2.3") == "fakelib==1.2.3"


def test_spec_extras_are_parsed_into_install_requirement(tmp_path: Path) -> None:
    body = "library: tortoise-orm\nextras: [asyncpg, aiomysql]\npatches: [{name: a, target: 'tortoise:f'}]\n"
    spec = load_spec(_write(tmp_path, body))
    assert spec.extras == ("asyncpg", "aiomysql")
    assert spec.install_requirement("0.20.0") == "tortoise-orm[asyncpg,aiomysql]==0.20.0"


def test_spec_stdlib_mode(tmp_path: Path) -> None:
    body = "library: urllib\nstdlib: true\npatches: [{name: a, target: 'urllib.request:OpenerDirector.open'}]\n"
    spec = load_spec(_write(tmp_path, body))
    assert spec.stdlib is True
    assert spec.subject == "python"
    assert spec.top_module == "urllib"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("library: x\nstdlib: yes please\npatches: [{name: a, target: 'x:f'}]\n", "stdlib must be a boolean"),
        ("library: x\nextras: asyncpg\npatches: [{name: a, target: 'x:f'}]\n", "extras must be a list"),
        ("library: x\nextras: ['bad extra!']\npatches: [{name: a, target: 'x:f'}]\n", "extras must be a list"),
        ("library: x\nextras: ['']\npatches: [{name: a, target: 'x:f'}]\n", "extras must be a list"),
        ("library: x\nstdlib: true\nextras: [a]\npatches: [{name: a, target: 'x:f'}]\n", "extras cannot be combined"),
        (
            "library: x\nstdlib: true\npolicy: {max: 2}\npatches: [{name: a, target: 'x:f'}]\n",
            "policy is not supported",
        ),
    ],
)
def test_spec_stdlib_and_extras_validation(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        load_spec(_write(tmp_path, body))


def test_default_python_versions_reads_requires_python(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nrequires-python = ">=3.10"\n', encoding="utf-8")
    assert default_python_versions(tmp_path) == ">=3.10"

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(SpecError, match="requires-python"):
        default_python_versions(tmp_path)


# --------------------------------------------------------------------------- with / constraints (P3)


def test_spec_with_and_constraints_are_parsed(tmp_path: Path) -> None:
    body = "library: x\nwith: ['anyio>=3', setuptools]\nconstraints: ['setuptools<81']\npatches: [{name: a, target: 'x:f'}]\n"
    spec = load_spec(_write(tmp_path, body))
    assert spec.with_requirements == ("anyio>=3", "setuptools")
    assert spec.constraints == ("setuptools<81",)


def test_spec_without_with_and_constraints_defaults_empty(tmp_path: Path) -> None:
    spec = load_spec(_write(tmp_path, "library: x\npatches: [{name: a, target: 'x:f'}]\n"))
    assert spec.with_requirements == ()
    assert spec.constraints == ()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("library: x\nwith: anyio\npatches: [{name: a, target: 'x:f'}]\n", "with must be a list"),
        ("library: x\nwith: ['not a req!!']\npatches: [{name: a, target: 'x:f'}]\n", "with: invalid requirement"),
        ("library: x\nconstraints: [1]\npatches: [{name: a, target: 'x:f'}]\n", "constraints must be a list"),
        (
            "library: x\nconstraints: ['setuptools<<81']\npatches: [{name: a, target: 'x:f'}]\n",
            "constraints: invalid requirement",
        ),
        (
            "library: x\nstdlib: true\nwith: [anyio]\npatches: [{name: a, target: 'x:f'}]\n",
            "with/constraints cannot be combined with stdlib",
        ),
        (
            "library: x\nstdlib: true\nconstraints: [anyio]\npatches: [{name: a, target: 'x:f'}]\n",
            "with/constraints cannot be combined with stdlib",
        ),
    ],
)
def test_spec_with_and_constraints_validation(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        load_spec(_write(tmp_path, body))


# --------------------------------------------------------------------------- opt_out scope (E4)


def test_parse_patch_opt_out_scope() -> None:
    path = Path("patches.yaml")
    patch = _parse_patch(path, {"name": "send", "target": "m:C.send", "scope": ["opt_out", "class"]}, 0)
    assert patch.scope == ("class", "opt_out")
    assert "opt_out" in SCOPES

    with pytest.raises(SpecError, match="opt_out requires class"):
        _parse_patch(path, {"name": "send", "target": "m:C.send", "scope": ["opt_out"]}, 0)
    with pytest.raises(SpecError, match="mutually exclusive"):
        _parse_patch(path, {"name": "send", "target": "m:C.send", "scope": ["class", "instance", "opt_out"]}, 0)


# --------------------------------------------------------------------------- name canonicalization (E7)


@pytest.mark.parametrize(
    ("library", "extra_line"),
    [
        ("aio-pika", "aio_pika >= 7.2.0, < 10.0.0"),
        ("aio_pika", "aio-pika >= 7.2.0, < 10.0.0"),
        ("Aio.Pika", "aio_pika >= 7.2.0, < 10.0.0"),
    ],
)
def test_default_versions_canonicalizes_names(tmp_path: Path, library: str, extra_line: str) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\n[project.optional-dependencies]\ninstruments = ["{extra_line}"]\n',
        encoding="utf-8",
    )
    assert default_versions(tmp_path, library) == "<10.0.0,>=7.2.0"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("library: x\nversions: '>>1'\npatches: [{name: a, target: 'x:f'}]\n", "invalid versions"),
        ("library: x\npolicy: {select: [oldest]}\npatches: [{name: a, target: 'x:f'}]\n", "policy: unknown keys"),
        ("library: x\npatches: [{name: a, target: 'x:f', async: true}]\n", "unknown keys"),
        ("library: x\npatches: [{name: a, target: 'x:f', signature: '()'}]\n", "unknown keys"),
        (
            "library: x\npatches: [{name: a, target: 'x:f', versions: '<=2'}, {name: b, target: 'x:f', versions: '>=2'}]\n",
            "overlapping",
        ),
        (
            "library: x\npatches: [{name: a, target: 'x:f', versions: '>1,<3'}, {name: b, target: 'x:f', versions: '>2,<4'}]\n",
            "overlapping",
        ),
    ],
)
def test_spec_rejects_invalid_versions_overlapping_ranges_and_removed_keys(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(SpecError, match=message):
        load_spec(_write(tmp_path, body))
