"""Built-in definitions own their parser specifications without a global registry."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from qcdata import ConformerSearchData, ProgramInput, SinglePointData

from qccodec import SubprocessOutput, get_codec, supported_programs
from qccodec.registry import ParserSpec


@pytest.mark.parametrize("program", ["orca", "terachem", "crest"])
def test_builtin_parser_ownership(
    program: str, prog_input_factory: Callable[..., ProgramInput], tmp_path: Path
) -> None:
    """All file-codec decoders respect parser lists on copied definitions."""
    original = get_codec(program)
    assert original.parsers and all(
        isinstance(spec, ParserSpec) for spec in original.parsers
    )
    assert any(spec.filetype == "stdout" for spec in original.parsers)
    if program == "terachem":
        (tmp_path / "scr.geometry").mkdir()
    copied = replace(original, parsers=[])
    data = copied.decode(
        prog_input_factory("energy", program),
        SubprocessOutput(
            stdout="No scientific fields",
            directory=tmp_path if program != "orca" else None,
            returncode=0,
        ),
    )
    assert isinstance(data, SinglePointData) and data.energy is None
    assert original.parsers


def test_builtin_discovery_is_independent_of_parsers() -> None:
    """Python integrations remain discoverable even without file parsers."""
    assert {"orca", "terachem", "crest", "xtb", "geometric"} <= set(
        supported_programs()
    )
    assert get_codec("xtb").parsers == []


def test_crest_options_do_not_modify_standalone_parsing(
    prog_input_factory: Callable[..., ProgramInput],
    test_data_dir: Path,
) -> None:
    """Execution options control rotamers while standalone artifact decoding keeps them."""
    from qccodec.parsers.crest import parse_rotamers

    inp = prog_input_factory("conformer_search", "crest")
    native = SubprocessOutput(directory=test_data_dir / "crest", returncode=0)
    for enabled in [False, True]:
        codec = get_codec("crest", collect_rotamers=enabled)
        codec = replace(
            codec,
            parsers=[spec for spec in codec.parsers if spec.parser is parse_rotamers],
        )
        result = codec.decode(inp, native)
        assert isinstance(result, ConformerSearchData)
        assert bool(result.rotamers) is enabled
        standalone = codec.decode_artifacts(
            inp.calctype, directory=native.directory, input_data=inp
        )
        assert isinstance(standalone, ConformerSearchData) and standalone.rotamers
