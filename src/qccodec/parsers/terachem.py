"""Parsers for TeraChem output files."""

import re
from collections.abc import Generator
from enum import Enum
from pathlib import Path

from qcconst import constants
from qcdata import (
    CalcType,
    ExecutionInfo,
    ProgramInput,
    ProgramOutput,
    SinglePointData,
    Structure,
)

from qccodec.exceptions import MatchNotFoundError, ParserError

from ..registry import register
from .utils import re_finditer, re_search


class TeraChemFileType(str, Enum):
    """TeraChem filetypes."""

    STDOUT = "stdout"
    DIRECTORY = "directory"


def iter_files(
    stdout: str | None, directory: Path | str | None
) -> Generator[tuple[TeraChemFileType, str | bytes | Path], None, None]:
    """
    Iterate over the files in a TeraChem output directory.

    If stdout is provided, yields a tuple for it.

    If directory is provided, iterates over the directory to yield files according to
    program-specific logic.

    Args:
        stdout: The contents of the TeraChem stdout file.
        directory: The path to the directory containing the TeraChem output files.

    Yields:
        (FileType, contents) tuples for a program's output.
    """
    if stdout is not None:
        yield TeraChemFileType.STDOUT, stdout

    if directory is not None:
        directory = Path(directory)
        # Check if the directory exists and is a directory
        if not directory.exists() or not directory.is_dir():
            raise ParserError(
                f"Directory {directory} does not exist or is not a directory."
            )
        yield TeraChemFileType.DIRECTORY, directory


@register(
    filetype=TeraChemFileType.STDOUT,
    calctypes=[CalcType.energy, CalcType.gradient, CalcType.hessian],
    target="energy",
)
def parse_energy(contents: str) -> float:
    """Parse the final energy from TeraChem stdout.

    NOTE:
        - Works on frequency files containing many energy values because re.search()
          returns the first result.
    """
    regex = r"FINAL ENERGY: (-?\d+(?:\.\d+)?)"
    return float(re_search(regex, contents).group(1))


@register(
    filetype=TeraChemFileType.STDOUT,
    calctypes=[CalcType.gradient, CalcType.hessian],
    target="gradient",
)
def parse_gradient(contents: str) -> list[list[float]]:
    """Parse the gradient from TeraChem stdout.

    Returns:
        The gradient as a list of 3-element lists.

    Raises:
        MatchNotFoundError: If no gradient data is found.

    Notes:
        - This works for exciting state gradients as well because TeraChem prints out
            the targeted gradient as the "regular" gradient.
    """
    regex = (
        r"(?<=dE\/dX\s{12}dE\/dY\s{12}dE\/dZ\n)"  # Just after the header line
        r"[\d\.\-\s]"  # the gradient block itself
        r"+(?=\n(?:--|-=|\Z))"  # top at -- (grad) or -= (opt) or end of file (when I manually split opt logs)
    )
    match = re_search(regex, contents)
    # Convert the found numbers to floats.
    values = [float(val) for val in match.group(0).split()]
    # Group the values into chunks of 3 (for x, y, z).
    gradient = [values[i : i + 3] for i in range(0, len(values), 3)]
    return gradient


@register(
    filetype=TeraChemFileType.STDOUT, calctypes=[CalcType.hessian], target="hessian"
)
def parse_hessian(contents: str) -> list[list[float]]:
    """Parse Hessian Matrix from TeraChem stdout in one pass.

    Args:
        contents: The contents of the TeraChem stdout file.

    Returns:
        A square Hessian matrix as a list of lists of floats.

    Raises:
        MatchNotFoundError: If no Hessian data is found.
        ParserError: If the extracted numbers cannot form a proper square matrix.
    """
    regex = r"\s+(?P<atom_number>\d+)\s(?P<vals>(?:\s-?\d.\d{15}e[+-]\d{2})+)"
    hessian: list[list[float]] = []

    matches = re_finditer(regex, contents)
    # Iterate over all matches and populate the Hessian matrix
    for match in matches:
        atom_index = int(match.group("atom_number")) - 1  # Convert to zero-based index
        # Check if the Hessian matrix has enough rows
        if len(hessian) <= atom_index:
            # Add empty rows if necessary
            hessian.extend([[] for _ in range(atom_index - len(hessian) + 1)])
        # Extract the values and convert them to floats
        vals = [float(val) for val in match.group("vals").split()]
        # Add values to the corresponding row
        hessian[atom_index].extend(vals)

    # Verify that the Hessian is a square matrix.
    for i, row in enumerate(hessian):
        if len(row) != len(hessian):
            raise ParserError(
                f"Hessian matrix is not square: row {i} has {len(row)} elements, expected {len(hessian)}."
            )
    return hessian


@register(
    filetype=TeraChemFileType.STDOUT,
    target=("provenance", "program_version"),
    required=False,
)
def parse_version(contents: str) -> str:
    """Parse version contents plus git commit from TeraChem stdout.

    Matches format of 'terachem --version' on command line.

    Example:
        'v1.9-2022.03-dev [4daa16dd21e78d64be5415f7663c3d7c2785203c]'
    """
    return f"{parse_terachem_version(contents)} [{parse_version_control_details(contents)}]"


@register(filetype=TeraChemFileType.STDOUT, target="calcinfo_natoms")
def parse_natoms(contents: str) -> int:
    """Parse number of atoms value from TeraChem stdout.

    Returns:
        The number of atoms as an integer.

    Raises:
        MatchNotFoundError: If the regex does not match.
    """
    regex = r"Total atoms:\s*(\d+)"
    match = re_search(regex, contents)
    return int(match.group(1))


@register(filetype=TeraChemFileType.STDOUT, target="calcinfo_nmo")
def parse_nmo(contents: str) -> int:
    """Parse the number of molecular orbitals from TeraChem stdout.

    Returns:
        The number of molecular orbitals as an integer.

    Raises:
        MatchNotFoundError: If the regex does not match.
    """
    regex = r"Total orbitals:\s*(\d+)"
    match = re_search(regex, contents)
    return int(match.group(1))


@register(
    filetype=TeraChemFileType.DIRECTORY,
    calctypes=[CalcType.optimization],
    target="trajectory",
)
def parse_trajectory(
    directory: Path | str,
    stdout: str,
    input_data: ProgramInput,
    *,
    failed: bool = False,
) -> list[ProgramOutput]:
    """Parse the output directory of a TeraChem optimization calculation into a trajectory.

    Args:
        directory: Path to the directory containing the TeraChem output files.
        stdout: The contents of the TeraChem stdout file.
        input_data: The input object used for the calculation.

    Returns:
        A list of ProgramOutput objects.
    """
    if input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    if not stdout and failed:
        return []
    directory = Path(directory)

    # Parse the structures
    structures = Structure.open_multi(
        directory / "optim.xyz",
        charge=input_data.structure.charge,
        multiplicity=input_data.structure.multiplicity,
    )

    # Capture initialization stdout
    regex = r"""^(.*?)                # group-1 = everything before the banner
        (?=                       # look-ahead, do NOT consume banner
            ^-{55}\s*\r?\n        # 55 dashes
            0\ additional\ frames\ found\ in\ \S+\.xyz\s*\r?\n
            -{55}                 # 55 dashes
        )
    """
    match = re.search(
        regex,
        stdout,
        flags=re.MULTILINE | re.VERBOSE | re.DOTALL,
    )
    if not match:
        raise MatchNotFoundError(regex, stdout)
    initialization_stdout = match.group(1)

    # Capture the stdout for each gradient calculation
    regex = r"-=#=-\s+\(We'll Be Right Back\)\s+-=#=-\n(.*?)-=#=-\s+Now Returning to Optimizer\s+-=#=-"
    per_gradient_stdout = re.findall(regex, stdout, flags=re.DOTALL)
    if not per_gradient_stdout:
        raise MatchNotFoundError(regex, stdout)

    # Parse the gradient values from the stdout file
    from qccodec import decode

    # Create the trajectory
    trajectory: list[ProgramOutput] = []

    for structure, grad_stdout in zip(structures, per_gradient_stdout):
        # Create the results object for each structure and gradient in the trajectory.
        full_grad_stdout = initialization_stdout + grad_stdout
        parsed_results = decode(
            "terachem", CalcType.gradient, stdout=full_grad_stdout, failed=failed
        )
        assert isinstance(parsed_results, SinglePointData)  # for mypy

        spr_data = parsed_results.model_dump()
        spr_data["energy"] = structure.extras[Structure._xyz_comment_key][0]
        results_obj = SinglePointData(**spr_data)
        # Create input data object for each structure and gradient in the trajectory.
        input_data_obj = ProgramInput.model_validate(
            {
                **input_data.model_dump(),
                "calctype": CalcType.gradient
                if results_obj.gradient is not None
                else CalcType.energy,
                "structure": structure,
            }
        )
        # Create the ProgramOutput object for each structure and gradient in the trajectory.
        traj_entry: ProgramOutput = ProgramOutput(
            input_data=input_data_obj,
            success=True,
            results=results_obj,
            execution=ExecutionInfo(scratch_dir=directory),
            logs=full_grad_stdout,
        )
        trajectory.append(traj_entry)

    return trajectory


def parse_version_control_details(contents: str) -> str:
    """Parse TeraChem git commit or Hg version from TeraChem stdout."""
    regex = r"(Git|Hg) Version: (\S*)"
    return re_search(regex, contents).group(2)


def parse_terachem_version(contents: str) -> str:
    """Parse TeraChem version from TeraChem stdout."""
    regex = r"TeraChem (v\S*)"
    return re_search(regex, contents).group(1)


def calculation_succeeded(contents: str) -> bool:
    """Determine from TeraChem stdout if a calculation completed successfully."""
    regex = r"Job finished:"
    if re.search(regex, contents):
        # If any match for a failure regex is found, the calculation failed
        return True
    return False


def parse_calctype(contents: str) -> CalcType:
    """Parse the calctype from TeraChem stdout."""
    calctypes = {
        r"SINGLE POINT ENERGY CALCULATIONS": CalcType.energy,
        r"SINGLE POINT GRADIENT CALCULATIONS": CalcType.gradient,
        r"FREQUENCY ANALYSIS": CalcType.hessian,
    }
    for regex, calctype in calctypes.items():
        match = re.search(regex, contents)
        if match:
            return calctype
    raise MatchNotFoundError(regex, contents)


@register(
    filetype=TeraChemFileType.STDOUT,
    calctypes=[CalcType.energy, CalcType.gradient],
    required=False,
    target=("extras", "excited_states"),
)
def parse_excited_states(contents: str) -> list[dict]:
    """Parse the excited state information from a TDDFT TeraChem stdout.

    Args:
        contents: The contents of the TeraChem TDDFT stdout file.
    Returns:
        A list of dictionaries containing the excited state information.
    Raises:
        MatchNotFoundError: If no excited state data is found.
    Notes:
        Converts the excitation energy from eV to Hartree.
    """
    regex = (
        r"^\s*(?:\d+)\s+(?P<energy>-?\d+\.\d+)\s+"
        r"(?P<exc_energy>-?\d+\.\d+)\s+"
        r"(?P<osc_strength>-?\d+\.\d+)\s+"
        r"(?P<s_squared>-?\d+\.\d+)\s+"
        r"(?P<max_ci_coeff>-?\d+\.\d+)\s+"
        r"(?P<excitation>\d+\s+->\s+\d+\s+:\s+\w+\s+->\s+\w+)$"
    )
    matches = re_finditer(regex, contents, re.MULTILINE)

    # Create a list of dictionaries for each excited state
    excited_states = []
    for match in matches:
        excited_state = match.groupdict()
        # Convert numeric values to floats
        for key, value in excited_state.items():
            if key != "excitation":  # Keep the excitation string as is
                excited_state[key] = float(value)

        # Convert the excitation energy to Hartree
        excited_state["exc_energy"] *= constants.EV_TO_HARTREE  # type: ignore
        excited_states.append(excited_state)

    return excited_states
