from qcdata import CalcType, ProgramInput

from qccodec.codecs._files import invocation
from qccodec.exceptions import EncoderError
from qccodec.native import SubprocessInput

SUPPORTED_CALCTYPES = {
    CalcType.energy,
    CalcType.gradient,
    CalcType.hessian,
    CalcType.optimization,
    CalcType.transition_state,
}
XYZ_FILENAME = "geometry.xyz"
PADDING = 20  # padding between keyword and value in tc.in


def encode(program_input: ProgramInput) -> SubprocessInput:
    """Translate a ProgramInput into TeraChem input files.

    Args:
        program_input: The qcdata ProgramInput object for a computation.

    Returns:
        A complete command and native input files.
    """

    if program_input.model is None:
        raise EncoderError("terachem input requires a scientific model.")

    # calctype
    if program_input.calctype == CalcType.hessian:
        calctype = "frequencies"
    elif program_input.calctype == CalcType.optimization:
        calctype = "minimize"
        if not program_input.keywords.get("new_minimizer", "no") == "yes":
            raise EncoderError(
                "Only the new_minimizer is supported for optimizations. Add "
                "'new_minimizer': 'yes' to the keywords."
            )
    elif program_input.calctype == CalcType.transition_state:
        calctype = "ts"
    else:
        calctype = program_input.calctype.value

    # Collect lines for input file
    inp_lines = []
    inp_lines.append(f"{'run':<{PADDING}} {calctype}")
    # Structure
    inp_lines.append(f"{'coordinates':<{PADDING}} {XYZ_FILENAME}")
    inp_lines.append(f"{'charge':<{PADDING}} {program_input.structure.charge}")
    inp_lines.append(f"{'spinmult':<{PADDING}} {program_input.structure.multiplicity}")
    # Model
    inp_lines.append(f"{'method':<{PADDING}} {program_input.model.method}")
    inp_lines.append(f"{'basis':<{PADDING}} {program_input.model.basis}")

    # Keywords
    non_keywords = {
        "charge": ".structure.charge",
        "spinmult": ".structure.multiplicity",
        "run": ".calctype",
        "basis": ".model.basis",
        "method": ".model.method",
    }
    for key, value in program_input.keywords.items():
        # Check for keywords that should be passed as structured data
        if key in non_keywords:
            raise EncoderError(
                f"Keyword '{key}' should not be set as a keyword. It "
                f"should be set at '{non_keywords[key]}'",
            )
        # Lowercase booleans
        inp_lines.append(f"{key:<{PADDING}} {str(value).lower()}")
    return invocation(
        program_input,
        {
            "tc.in": "\n".join(inp_lines) + "\n",
            XYZ_FILENAME: program_input.structure.to_xyz(),
        },
        "tc.in",
        wavefunction_files=("scr*/c0", "scr*/ca0", "scr*/cb0"),
    )
