from collections.abc import Mapping
from typing import Any

from qcdata import CalcType, ProgramInput

from qccodec.exceptions import EncoderError
from qccodec.models import NativeInput

SUPPORTED_CALCTYPES = {
    CalcType.energy,
    CalcType.gradient,
    CalcType.hessian,
    CalcType.optimization,
    CalcType.transition_state,
}
XYZ_FILENAME = "geometry.xyz"
PADDING = 20  # padding between keyword and value


def _validate_keywords(keywords: dict[str, Any]) -> None:
    """Validate keywords for ORCA encoder. Expecting all lowercase keys."""

    _NON_BLOCKS = {  # disallowed top-level blocks → where to set instead
        "coords": ".structure",
    }

    _NON_BLOCK_KEYWORDS = {  # disallowed keys inside allowed blocks → where to set instead
        "method": {"method": ".model.method", "runtyp": ".calctype"},
        "basis": {"basis": ".model.basis"},
    }

    # 1) Blocks that should not appear as top-level keywords
    for block, where in _NON_BLOCKS.items():
        if block in keywords:
            raise EncoderError(
                f"Block '{keywords[block]}' should not be set as a keyword. "
                f"Set its data on '{where}'."
            )

    # 2) Disallowed keys inside allowed blocks
    for block, disallowed in _NON_BLOCK_KEYWORDS.items():
        for variable, where in disallowed.items():
            if keywords.get(block, {}).get(variable) is not None:
                raise EncoderError(
                    f"Keyword '{variable}' in block '{block}' should not be set directly. It "
                    f"should be set at '{where}'",
                )


def _fmt(key: str, value: Any) -> Any:
    """Format a value for ORCA input."""
    keywords_needing_quotes = {"auxc", "auxj", "auxjk"}
    if key.casefold() in keywords_needing_quotes:
        return f'"{value}"'  # ORCA needs quotes for certain keywords
    if isinstance(value, bool):
        return str(value).lower()  # ORCA expects 'true'/'false' for booleans
    return value


def encode(program_input: ProgramInput) -> NativeInput:
    """Translate a ProgramInput into ORCA input files.

    Args:
        program_input: The qcdata ProgramInput object for a computation.

    Returns:
        NativeInput with .input being an orca.inp file and .geometry an xyz file.

    Notes:
        - ORCA keywords are case-insensitive. This encoder will preserve the
            casing of keywords as provided in program_input.keywords.
        - ORCA requires passing `numgrad` for numerical gradients. To activate a
            numerical gradient for a single point or optimization pass
            `{"numgrad": "true"} in the keywords or `{"numgrad": {...}}` with a
            dictionary of numgrad block keywords. An empty dictionary will also work
            (equivalent to `{"numgrad": "true"}`).
    """

    if program_input.model is None:
        raise EncoderError("orca input requires a scientific model.")

    # Handle ORCA's case-insensitive keywords by doing caseless lookups
    kw_lower = {k.casefold(): v for k, v in program_input.keywords.items()}
    _validate_keywords(kw_lower)

    # Collect lines for input file
    inp_lines = []

    # maxcore
    if "maxcore" in kw_lower:
        inp_lines.append(f"%maxcore {kw_lower['maxcore']}")

    # pal
    if "pal" in kw_lower:
        inp_lines.append(f"%pal nprocs {kw_lower['pal']} end")

    # Add an empty line if writing maxcore/pal
    if inp_lines:
        inp_lines.append("")

    # Method and Basis
    model_line = f"! {program_input.model.method}"
    # ORCA will not recognize 'None' when using composite methods
    if program_input.model.basis:
        model_line += f" {program_input.model.basis}"

    inp_lines.append(model_line)

    # NumGrad. May be used for gradients or optimizations
    if "numgrad" in kw_lower:
        inp_lines.append(f"! numgrad")

    # Set ORCA runtyp based on calctype
    if program_input.calctype == CalcType.energy:
        runtyp = "energy"
    elif program_input.calctype == CalcType.gradient:
        runtyp = "engrad"
    elif program_input.calctype == CalcType.hessian:
        if "numfreq" in kw_lower:
            runtyp = "numfreq"
        else:
            runtyp = "freq"
    elif program_input.calctype == CalcType.optimization:
        runtyp = "opt"
    elif program_input.calctype == CalcType.transition_state:
        runtyp = "optts"

    inp_lines.append(f"! {runtyp}\n")

    # Input Blocks
    for block, kwargs in kw_lower.items():
        if isinstance(kwargs, Mapping):  # Skip non-block keywords
            inp_lines.append(f"%{block}")
            for key, value in kwargs.items():
                inp_lines.append(f"    {key:<{PADDING}} {_fmt(key, value)}")
            inp_lines.append("end")

    # Write a new line if there were any blocks
    if any(isinstance(v, Mapping) for v in kw_lower.values()):
        inp_lines.append("")

    # Structure
    inp_lines.append(
        f"* xyzfile {program_input.structure.charge} {program_input.structure.multiplicity} {XYZ_FILENAME}"
    )

    return NativeInput(
        input_file="\n".join(inp_lines) + "\n",
        geometry_file=program_input.structure.to_xyz(),
        geometry_filename=XYZ_FILENAME,
    )
