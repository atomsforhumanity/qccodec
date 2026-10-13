"""Encoding and standalone artifact decoding."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, overload

from qcdata import CalcType, ProgramInput, StructuredData, get_data_type

from .definition import CodecDefinition
from .exceptions import DecoderError, MatchNotFoundError, ParserError
from .models import DataCollector
from .native import NativeInput

logger = logging.getLogger(__name__)


def decode_artifacts(
    codec: CodecDefinition,
    calctype: CalcType | str,
    *,
    stdout: str | None = None,
    directory: str | Path | None = None,
    input_data: ProgramInput | None = None,
    as_dict: bool = False,
    failed: bool = False,
) -> StructuredData | dict[str, Any]:
    """Decode the output of a quantum chemistry program into a standardized output.

    Args:
        codec: The program definition containing artifact discovery and parsers.
        calctype: The type of calculation that was run.
        stdout: The stdout file contents as a string.
        directory: The directory containing the output files.
        input_data: The input data used for the calculation.
            This is used to provide additional context for the parsers.
        failed: Recover available data from a failed calculation. Missing values and
            artifacts are tolerated; malformed recovered values still fail validation.
        as_dict: If True, return the results as a dictionary instead of a
            StructuredData object. Used mostly for testing purposes to enable
            returning parsed data that isn't a fully valid StructuredData object.

    Returns:
        A StructuredData object containing the parsed data.

    Raises:
        DecoderError: If neither stdout nor directory is provided or if the program
            or calctype is not supported.
        MatchNotFoundError: If a required parser fails to find a match.
    """
    program = codec.program
    if codec.iter_files is None:
        raise DecoderError(
            f"{program} has no file-artifact decoder; pass its native output to codec.decode()."
        )
    logger.info("Starting decode for program: %s with calctype: %s", program, calctype)
    if not stdout and not directory and not failed:
        raise ValueError("Either stdout, directory, or both must be provided.")

    calctype = CalcType(calctype)
    data_type = get_data_type(calctype)

    # Create a list from stdout (if provided) and all parsable files in directory.
    # File discovery stays program-local and dumb; decode enforces the parser registry
    # contract after it knows which filetypes are present.
    try:
        files = list(codec.iter_files(stdout, directory))
    except ParserError as exc:
        if not failed:
            exc.data = data_type.model_validate({"provenance": {"program": program}})
            raise
        files = [("stdout", stdout)] if stdout is not None else []
    seen_filetypes = {filetype for filetype, _ in files}

    # Now iterate over the combined generator of all parsable files
    data_collector = DataCollector(provenance={"program": program})
    for filetype, contents in files:
        # Look up the parsers for the given program, filetype, and calctype
        logger.debug("Processing file with filetype: %s", filetype)
        parser_specs = [
            s
            for s in codec.parsers
            if s.filetype == filetype and calctype in s.calctypes
        ]
        logger.info(
            "Found %d parser(s) for program '%s', filetype '%s', calctype '%s'",
            len(parser_specs),
            program,
            filetype,
            calctype,
        )  # noqa: E501

        for spec in sorted(
            parser_specs,
            key=lambda item: item.target != ("provenance", "program_version"),
        ):
            logger.debug(
                "Running parser '%s' for target '%s'", spec.parser.__name__, spec.target
            )  # noqa: E501
            # Parse the contents using the parser
            try:
                if spec.filetype == "directory":
                    parsed_value: Any = spec.parser(
                        directory, stdout, input_data, failed=failed
                    )
                else:
                    parsed_value = spec.parser(contents)
                logger.info(
                    "Parser '%s' succeeded; returned value: %s",
                    spec.parser.__name__,
                    parsed_value,
                )  # noqa: E501
            # Raised if the parser can't find its data
            except MatchNotFoundError as e:
                if spec.required and not failed:
                    logger.error(
                        "Required parser '%s' failed; raising exception",
                        spec.parser.__name__,
                    )  # noqa: E501
                    e.data = data_type(**data_collector)
                    raise
                else:
                    logger.info(
                        "Parser '%s' did not find a match but is not required.",
                        spec.parser.__name__,
                    )  # noqa: E501
            except FileNotFoundError as e:
                if not failed:
                    raise ParserError(str(e), data=data_type(**data_collector)) from e
            except ParserError as e:
                if not isinstance(e.data, data_type):
                    e.data = data_type(**data_collector)
                raise
            # Place the parsed value into the data collector
            else:
                # If the parser returns a dictionary, assign each key-value pair to the data collector
                if spec.target is None and isinstance(parsed_value, dict):
                    for key, value in parsed_value.items():
                        data_collector.add_data(key, value)
                        logger.debug(
                            "Assigned parsed value to target '%s' on data_collector",
                            (spec.target, key),
                        )
                # Otherwise, assign the parsed value to the specified target
                else:
                    assert spec.target is not None, (
                        "Target must be specified for non-dictionary parsed values."
                    )  # for mypy
                    data_collector.add_data(spec.target, parsed_value)
                logger.debug(
                    "Assigned parsed value to target '%s' on data_collector",
                    spec.target,
                )  # noqa: E501

    logger.info(
        "Completed processing files; final data_collector state: %s", data_collector
    )  # noqa: E501
    required_specs = [
        spec
        for spec in [s for s in codec.parsers if calctype in s.calctypes]
        if spec.required
        and spec.explicit_calctypes
        and spec.filetype not in seen_filetypes
    ]
    if required_specs and not failed:
        missing = sorted(
            {
                str(getattr(spec.filetype, "value", str(spec.filetype)))
                for spec in required_specs
            }
        )
        data = data_type(**data_collector)
        raise ParserError(
            "Missing required output artifact(s): " + ", ".join(missing),
            data=data,
        )

    # Finally, construct and return the StructuredData using the collected data.
    if as_dict:
        return dict(data_collector)
    return data_type(**data_collector)


@overload
def decode(
    program: str, calctype: CalcType | str, *, as_dict: Literal[True], **kwargs: Any
) -> dict[str, Any]:
    """Select dictionary or validated result output from the requested mode."""
    ...


@overload
def decode(
    program: str,
    calctype: CalcType | str,
    *,
    as_dict: Literal[False] = False,
    **kwargs: Any,
) -> StructuredData:
    """Select dictionary or validated result output from the requested mode."""
    ...


@overload
def decode(
    program: str, calctype: CalcType | str, *, as_dict: bool, **kwargs: Any
) -> StructuredData | dict[str, Any]:
    """Select dictionary or validated result output from the requested mode."""
    ...


def decode(
    program: str, calctype: CalcType | str, **kwargs: Any
) -> StructuredData | dict[str, Any]:
    """Decode stored artifacts without running a scientific program."""
    from .codecs import get_codec

    return get_codec(program).decode_artifacts(calctype, **kwargs)


def encode(input_data: ProgramInput) -> NativeInput:
    """Prepare a complete native invocation without executing it."""
    from .codecs import get_codec

    return get_codec(input_data.program).encode(input_data)
