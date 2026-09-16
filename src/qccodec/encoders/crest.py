import copy
import os
from typing import Any

import tomli_w
from qcdata import CalcType, ProgramInput

from qccodec.exceptions import EncoderError
from qccodec.models import NativeInput

SUPPORTED_CALCTYPES = {
    CalcType.conformer_search,
    CalcType.optimization,
    CalcType.energy,
    CalcType.gradient,
    CalcType.hessian,
}


def encode(program_input: ProgramInput) -> NativeInput:
    """Translate a ProgramInput into CREST input files.

    Args:
        program_input: The qcdata ProgramInput object for a computation.

    Returns:
        NativeInput with .input_files being a crest.toml file and .geometry_file the
            Structure's xyz file.
    """
    validate_input(program_input)
    struct_filename = "structure.xyz"

    return NativeInput(
        input_file=tomli_w.dumps(_to_toml_dict(program_input, struct_filename)),
        geometry_file=program_input.structure.to_xyz(),
        geometry_filename=struct_filename,
    )


def validate_input(program_input: ProgramInput):
    """Validate the input for CREST.

    Args:
        program_input: The qcdata ProgramInput object for a computation.

    Raises:
        EncoderError: If the input is invalid.
    """
    if program_input.model is None:
        raise EncoderError("crest input requires a scientific model.")

    # These values come from other parts of the ProgramInput and should not be set
    # in the keywords.
    non_allowed_keywords = ["charge", "uhf"]
    for keyword in non_allowed_keywords:
        if keyword in program_input.keywords:
            raise EncoderError(
                f"{keyword} should not be set in keywords for CREST. It is already set "
                "on the Structure or ProgramInput elsewhere.",
            )
    if "runtype" in program_input.keywords:
        _validate_runtype_calctype(
            program_input.keywords["runtype"],
            program_input.calctype,
        )


def _validate_runtype_calctype(runtype: str, calctype: CalcType):
    """Validate that the runtype is supported for the calctype."""
    invalid_runtype = False
    valid_runtypes = set()

    if calctype == CalcType.conformer_search:
        valid_runtypes = {"imtd-gc", "imtd-smtd", "entropy", "nci", "nci-mtd"}
        if runtype not in valid_runtypes:
            invalid_runtype = True

    elif calctype == CalcType.optimization:
        valid_runtypes = {"optimize", "ancopt"}
        if runtype not in valid_runtypes:
            invalid_runtype = True

    elif calctype in {CalcType.energy, CalcType.gradient}:
        valid_runtypes = {"singlepoint"}
        if runtype not in valid_runtypes:
            invalid_runtype = True

    elif calctype == CalcType.hessian:
        valid_runtypes = {"numhess"}
        if runtype not in valid_runtypes:
            invalid_runtype = True

    if invalid_runtype:
        raise EncoderError(
            f"Unsupported runtype {runtype} for calctype {calctype}. Valid runtypes "
            f"are: {valid_runtypes}.",
        )


def _to_toml_dict(program_input: ProgramInput, struct_filename: str) -> dict[str, Any]:
    """Convert a ProgramInput object to a dictionary in the CREST format of TOML.

    This function makes it easier to test for the correct TOML structure.
    """
    if program_input.model is None:
        raise EncoderError("crest input requires a scientific model.")

    # Start with existing keywords
    toml_dict = copy.deepcopy(program_input.keywords)

    # Top level keywords
    # Logical cores was 10% faster than physical cores, so not using psutil
    toml_dict.setdefault("threads", min(os.cpu_count() or 16, 16))
    toml_dict["input"] = struct_filename

    # Set default runtype if not already set
    if "runtype" not in program_input.keywords:
        if program_input.calctype == CalcType.conformer_search:
            toml_dict["runtype"] = "imtd-gc"
        elif program_input.calctype == CalcType.optimization:
            toml_dict["runtype"] = "optimize"
        elif program_input.calctype in {CalcType.energy, CalcType.gradient}:
            toml_dict["runtype"] = "singlepoint"
        elif program_input.calctype == CalcType.hessian:
            toml_dict["runtype"] = "numhess"
        else:
            raise EncoderError(
                f"Unsupported calctype {program_input.calctype} for CREST encoder.",
            )

    # Calculation level keywords
    calculation = toml_dict.pop("calculation", {})
    calculation_level = calculation.pop("level", [])
    if len(calculation_level) == 0:
        calculation_level.append({})
    for level_dict in calculation_level:
        level_dict["method"] = program_input.model.method
        level_dict["charge"] = program_input.structure.charge
        level_dict["uhf"] = program_input.structure.multiplicity - 1

    calculation["level"] = calculation_level
    toml_dict["calculation"] = calculation

    return toml_dict
