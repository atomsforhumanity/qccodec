import inspect
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from qcdata import CalcType, ProgramInput, ProgramOutput
from qcdata.utils import water

from qccodec import Codec, Context, ParserSpec
from qccodec.exceptions import MatchNotFoundError


@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory Path"""
    return Path(__file__).parent / "data"


@pytest.fixture
def terachem_file(test_data_dir):
    """Return a function that reads a file from the 'terachem' subdirectory."""

    def _read(filename: str) -> str:
        return (test_data_dir / "terachem" / filename).read_text()

    return _read


@pytest.fixture
def crest_file(test_data_dir):
    """Return a function that reads a file from the 'crest' subdirectory."""

    def _read(filename: str) -> str:
        return (test_data_dir / "crest" / filename).read_text()

    return _read


@pytest.fixture(scope="session")
def prog_input_factory():
    """Create a function that returns a ProgramInput object with a specified calculation type."""

    def create_input(calctype, program="terachem"):
        return ProgramInput(
            program=program,
            structure=water,
            calctype=calctype,
            # Tests depend up this model; do not change
            model={"method": "hf", "basis": "sto-3g"},
        )

    return create_input


@dataclass
class ParserTestCase:
    """Test case for a parser function.

    Attributes:
        name: A human-readable name for the test case.
        parser: The parser function to be tested.
        calctype: The calculation type for the test case.
        success: Boolean indicating if the parser should succeed on the contents.
        stdout: The stdout file contents as a string, if needed
        decode_exc: Boolean indicating if an exception is expected for MatchNotFound
            errors during decode. Default is True. required=False parsers should not
            raise an error.
        answer: The expected output from the parser.
        clear_registry: Boolean indicating if the codec should be restricted to only
            the parser under test before running the decode integration test.
        extra_files: Optional list of additional files to be copied to the test directory.
        extra_files_names: Optional list of names for the extra files, these names will
            be used as the filenames in the test directory (subdirectories are created
            as needed). If not provided, the filenames will be the same as in the
            extra_files list.
        program_input: Optional ProgramInput to be used instead of the default one created
            from the calctype.
        codec: Optional codec owning the parser; defaults to the ``codec`` of the
            parser's module.
        options: Decode options passed to the codec for this case.
    """

    name: str
    parser: Callable
    calctype: CalcType
    success: bool
    stdout: str | Path | None = None
    decode_exc: bool = True
    answer: Any | None = None
    clear_registry: bool = True
    extra_files: list[str] | None = None
    extra_files_names: list[str] | None = None
    program_input: ProgramInput | None = None
    codec: Codec | None = None
    options: dict[str, Any] | None = None


def _load_stdout(directory, stdout):
    """Load the stdout for a TestCase."""
    if isinstance(stdout, Path):
        return (directory / stdout).read_text()
    return stdout


def get_target_value(results, target):
    """Lookup the value of a target in the results dictionary.
    Args:
        results: The results dictionary.
        target: The target key or tuple of keys to look up.
    """
    keys = target if isinstance(target, tuple) else (target,)
    d = results
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    return d.get(keys[-1], None)


def _test_parser_direct(tc: ParserTestCase, ctx: Context, spec: ParserSpec):
    """Call the parser directly with the contents the codec would hand it."""
    if tc.success:
        parsed = tc.parser(ctx.contents(spec))
        assert parsed == tc.answer, (
            f"{tc.name}: Direct parsing returned {parsed} instead of expected {tc.answer}"
        )
    else:
        with pytest.raises(MatchNotFoundError):
            tc.parser(ctx.contents(spec))


def _test_decode_integration(
    tc: ParserTestCase, ctx: Context, codec: Codec, spec: ParserSpec
):
    """Run the codec's decode pipeline and check the parser's contribution."""
    stdout, directory, input_data = ctx.stdout, ctx.directory, ctx.input_data
    if tc.success:
        result = codec.parse(
            tc.calctype,
            stdout=stdout,
            directory=directory,
            input_data=input_data,
            options=tc.options,
            as_dict=True,
        )
        if spec.target is not None:
            final_value = get_target_value(result, spec.target)
        else:
            assert tc.answer is not None
            final_value = {key: result[key] for key in tc.answer}
            assert result["provenance"]["program"] == codec.program
        assert final_value == tc.answer, (
            f"{tc.name}: decode() returned {final_value} for target '{spec.target}' "
            f"instead of expected {tc.answer}"
        )
    elif tc.decode_exc and spec.required:
        with pytest.raises(MatchNotFoundError):
            codec.parse(
                tc.calctype,
                stdout=stdout,
                directory=directory,
                input_data=input_data,
                options=tc.options,
            )
    else:
        result = codec.parse(
            tc.calctype,
            stdout=stdout,
            directory=directory,
            input_data=input_data,
            options=tc.options,
            as_dict=True,
        )
        final_value = get_target_value(result, spec.target)
        assert final_value in (None, {}), (
            f"{tc.name}: decode() returned non-empty value {final_value} when an empty result was expected."
        )


def run_test_harness(test_data_dir, input_factory, tmp_path, tc: ParserTestCase):
    """
    Args:
        test_data_dir: The base directory containing test data files.
        input_factory: A function that creates a ProgramInput given a CalcType.
        tmp_path: A temporary directory Path for the test.
        tc: The ParserTestCase object containing the test parameters.

    Notes:
        Runs the full test harness:
            1. Load the contents.
            2. Run the parser directly.
            3. Optionally restrict the codec to only the parser under test,
                and run the decode() integration test.
    """
    module = inspect.getmodule(tc.parser)
    assert module is not None
    program = module.__name__.split(".")[-1]
    codec: Codec = tc.codec or module.codec
    spec = next(spec for spec in codec.parsers if spec.parser is tc.parser)

    stdout = _load_stdout(test_data_dir / program, tc.stdout)
    proginput = tc.program_input or input_factory(tc.calctype, program=program)

    # Copy over extra files if provided; stdout-only cases need no directory.
    if tc.extra_files:
        names = tc.extra_files_names or tc.extra_files
        for extra_file, name in zip(tc.extra_files, names):
            destination = tmp_path / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(test_data_dir / program / extra_file, destination)
        directory = tmp_path
    else:
        directory = None

    if (
        isinstance(tc.answer, list)
        and tc.answer
        and isinstance(tc.answer[0], ProgramOutput)
    ):
        tc = replace(
            tc,
            answer=[
                ProgramOutput.model_validate(
                    {
                        **entry.model_dump(),
                        "execution": {
                            **entry.execution.model_dump(),
                            "scratch_dir": directory,
                        },
                    }
                )
                for entry in tc.answer
            ],
        )

    ctx = Context(
        codec=codec,
        calctype=tc.calctype,
        stdout=stdout,
        directory=directory,
        input_data=proginput,
        options={**codec.options, **(tc.options or {})},
    )
    _test_parser_direct(tc, ctx, spec)

    selected = (
        Codec(codec.program, codec.calctypes, options=codec.options, parsers=[spec])
        if tc.clear_registry
        else codec
    )
    _test_decode_integration(tc, ctx, selected, spec)
