"""Local parser registration and conflict validation."""

from dataclasses import replace
from enum import Enum

import pytest
from qcdata import CalcType

from qccodec import CodecDefinition
from qccodec.exceptions import RegistryError


class FileType(str, Enum):
    """Example program-local artifact identifiers."""

    stdout = "stdout"
    directory = "directory"


def energy(contents: str) -> float:
    """Parse a directly callable scalar value."""
    return float(contents)


def test_registration_is_local_and_preserves_function() -> None:
    """Decorators only affect their owning definition and preserve direct calls."""
    first, second = CodecDefinition("first"), CodecDefinition("second")
    decorated = first.parser(
        filetype=FileType.stdout, target="energy", calctypes=[CalcType.energy]
    )(energy)
    assert decorated is energy and decorated("-1") == -1
    assert len(first.parsers) == 1 and second.parsers == []
    spec = first.parsers[0]
    assert spec.filetype == "stdout" and spec.calctypes == [CalcType.energy]


def test_conflicting_targets_are_rejected() -> None:
    """Two parsers cannot produce the same target for overlapping calculations."""
    codec = CodecDefinition("local")
    codec.parser(filetype="stdout", target="energy", calctypes=[CalcType.energy])(
        energy
    )
    with pytest.raises(RegistryError, match="Duplicate parser target"):
        codec.parser(
            filetype="another-file",
            target="energy",
            calctypes=[CalcType.energy, CalcType.gradient],
        )(energy)
    codec.parser(filetype="stdout", target="energy", calctypes=[CalcType.gradient])(
        energy
    )
    assert len(codec.parsers) == 2


def test_target_validation_and_optional_metadata() -> None:
    """Validate parser targets and retain optional-parser semantics."""
    codec = CodecDefinition("local")
    with pytest.raises(RegistryError, match="must have a target"):
        codec.parser(filetype="stdout")(energy)
    codec.parser(filetype=FileType.directory)(energy)
    codec.parser(
        filetype="stdout", target=("provenance", "program_version"), required=False
    )(energy)
    assert codec.parsers[0].target is None
    assert codec.parsers[1].required is False
    assert codec.parsers[1].explicit_calctypes is False
    assert set(codec.parsers[1].calctypes) == set(CalcType)


def test_replaced_definition_uses_selected_parsers() -> None:
    """Changing a copied definition's parsers changes its composed decoder."""
    codec = CodecDefinition(
        "local", iter_files=lambda stdout, directory: [("stdout", stdout)]
    )
    codec.parser(filetype="stdout", target="energy")(energy)
    copied = replace(codec, parsers=[])
    assert copied.decode_artifacts("energy", stdout="not an energy", as_dict=True) == {
        "provenance": {"program": "local"}
    }
    assert codec.decode_artifacts("energy", stdout="-1", as_dict=True)["energy"] == -1
