"""Parsers for Orca output files."""

import itertools
import re
from enum import Enum
from pathlib import Path
from typing import Generator

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
from .utils import re_search


class OrcaFileType(str, Enum):
    """Orca filetypes.

    Maps file types to their suffixes as written in the Orca output directory
    (except for STDOUT and DIRECTORY).
    """

    STDOUT = "stdout"
    DIRECTORY = "directory"
    HESS = ".hess"  # basename.hess


def iter_files(
    stdout: str | None, directory: Path | str | None
) -> Generator[tuple[OrcaFileType, str | bytes | Path], None, None]:
    """
    Iterate over the files in a Orca output directory.

    If stdout is provided, yields a tuple for it.

    If directory is provided, iterates over the directory to yield files according to
    program-specific logic.

    Args:
        stdout: The contents of the Orca stdout file.
        directory: The path to the directory containing the Orca output files.

    Yields:
        (FileType, contents) tuples for a program's output.
    """
    if stdout is not None:
        yield OrcaFileType.STDOUT, stdout

    if directory is not None:
        directory = Path(directory)
        # Check if the directory exists and is a directory
        if not directory.exists() or not directory.is_dir():
            raise ParserError(
                f"Directory {directory} does not exist or is not a directory."
            )
        yield OrcaFileType.DIRECTORY, directory

        # Read the basename from STDOUT
        if stdout is not None:
            basename = parse_basename(stdout)

            # Iterate over the files in the directory and yield their contents
            for filetype in OrcaFileType:
                # Ignore STDOUT and DIRECTORY as they are handled above
                if filetype not in (OrcaFileType.STDOUT, OrcaFileType.DIRECTORY):
                    # Get suffix from FileType value
                    file_suffix = filetype.value
                    file_path = directory / f"{basename}{file_suffix}"
                    if file_path.exists():
                        yield filetype, file_path.read_text()


@register(
    filetype=OrcaFileType.STDOUT,
    calctypes=[CalcType.energy, CalcType.gradient, CalcType.hessian],
    target="energy",
)
def parse_energy(contents: str) -> float:
    """Parse the final energy from Orca stdout."""
    regex = r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)"
    return float(re_search(regex, contents).group(1))


@register(
    filetype=OrcaFileType.STDOUT,
    calctypes=[CalcType.gradient],
    target="gradient",
)
def parse_gradient(contents: str) -> list[list[float]]:
    """Parse the gradient from Orca stdout.

    Returns:
        The gradient as a list of 3-element lists.

    Raises:
        MatchNotFoundError: If no gradient data is found.
    """
    # Extract the gradient block lines
    header_regex = r"CARTESIAN GRADIENT.*?(?=\d)"  # non-greedy lookahead to first digit
    header_match = re_search(header_regex, contents, flags=re.DOTALL)

    block_start = header_match.end()
    block_lines = itertools.takewhile(
        lambda line: re.search(r"\d\s*$", line), contents[block_start:].splitlines()
    )

    # Parse values from block lines
    line_regex = r".*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
    line_matches = [re_search(line_regex, line) for line in block_lines]
    gradient = [list(map(float, match.groups())) for match in line_matches]
    return gradient


@register(
    filetype=OrcaFileType.STDOUT,
    calctypes=[CalcType.hessian],
    target="gradient",
    required=False,
)
def parse_gradient_hessian(contents: str) -> list[list[float]]:
    """Parse the gradient from Orca stdout for a hessian calculation.

    Analytic Hessian jobs don't print CARTESIAN GRADIENT block.
    """
    return parse_gradient(contents)


@register(
    filetype=OrcaFileType.HESS,
    calctypes=[CalcType.hessian],
    target="hessian",
)
def parse_hessian(contents: str) -> list[list[float]]:
    """Parse hessian from .hess file."""
    # Find hessian entry in basename.hess file
    entry = next(
        (block for block in contents.split("$") if block.startswith("hess")),
        None,
    )
    if entry is None:
        raise ParserError("Failed to find hessian block in Hessian file.")

    dim = int(entry.splitlines()[1])

    # Split the hessian entry into blocks on lines of the form '  0  1  2  3 ...'
    # (the final block may have only a single column index, e.g. '  5')
    split_result = re.split(r"^\s*\d+(?:\s+\d+)*\s*$", entry, flags=re.MULTILINE)
    if not len(split_result) > 1:
        raise ParserError(f"Failed to parse blocks in hessian entry: {entry}")

    # Get the text for each block and
    blocks = [block.strip() for block in split_result[1:]]
    hessian: list[list[float]] = [[] for _ in range(dim)]
    for block in blocks:
        lines = block.splitlines()
        if not len(lines) == dim:
            raise ParserError(
                f"Block line count {len(lines)} does not match dimension {dim}: {block}"
            )

        for i, line in enumerate(block.splitlines()):
            row = list(map(float, line.split()[1:]))
            hessian[i].extend(row)

    return hessian


@register(
    filetype=OrcaFileType.DIRECTORY,
    calctypes=[CalcType.optimization, CalcType.transition_state],
    target="trajectory",
)
def parse_trajectory(
    directory: Path | str,
    stdout: str,
    input_data: ProgramInput,
    *,
    failed: bool = False,
) -> list[ProgramOutput]:
    """Parse the output directory of a Orca optimization calculation into a trajectory.

    Args:
        directory: Path to the directory containing the Orca output files.
        stdout: The contents of the Orca stdout file.
        input_data: The input object used for the calculation.

    Returns:
        A list of ProgramOutput objects.
    """
    if input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    if not stdout and failed:
        return []
    basename = parse_basename(stdout)
    directory = Path(directory)
    file = directory / f"{basename}_trj.xyz"
    if not file.exists():
        if failed:
            return []
        raise ParserError(f"Trajectory file does not exist: {file}")

    # Parse the structures, energies, and gradients
    # NOTE: trj_xyz carries no (charge, multiplicity), so it will
    # silently default to (0, 1) unless supplied from input_data
    structures = Structure.open_multi(
        file,
        charge=input_data.structure.charge,
        multiplicity=input_data.structure.multiplicity,
    )

    # Capture initialization stdout
    regex = r"^(.*?\*\*\*\*END\s+OF\s+INPUT\*\*\*\*\s*\n\s*=*)"
    match = re_search(regex, stdout, flags=re.MULTILINE | re.VERBOSE | re.DOTALL)
    if not match:
        raise MatchNotFoundError(regex, stdout)
    initialization_stdout = match.group(1)

    # Capture the stdout for each gradient calculation
    regex = (
        r"GEOMETRY\s*OPTIMIZATION\s*CYCLE\s*\d+.*?"  # Match cycle line
        r"\*+\n"  # Match the line of stars
        r"(.*?)"  # Body (Captured)
        r"-+\s*\n\s*ORCA\s+GEOMETRY\s+RELAXATION\s+STEP"  # Footer
    )
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
            "orca", CalcType.gradient, stdout=full_grad_stdout, failed=failed
        )
        assert isinstance(parsed_results, SinglePointData)  # for mypy

        spr_data = parsed_results.model_dump()
        spr_data["energy"] = structure.extras[Structure._xyz_comment_key][-1]
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
        )
        trajectory.append(traj_entry)

    return trajectory


@register(
    filetype=OrcaFileType.STDOUT,
    target=("provenance", "program_version"),
    required=False,
)
def parse_version(contents: str) -> str:
    """Parse version string from Orca stdout."""
    regex = r"Program Version (\d+\.\d+\.\d+)"
    match = re_search(regex, contents)
    return match.group(1)


def parse_basename(contents: str) -> str:
    """Parse the file basename from Orca stdout."""
    regex = r"NAME\s+=\s+(.*)"
    match = re_search(regex, contents)
    return Path(match.group(1)).stem


@register(filetype=OrcaFileType.STDOUT, target="calcinfo_natoms", required=False)
def parse_natoms(contents: str) -> int:
    """Parse number of atoms value from Orca stdout.

    Returns:
        The number of atoms as an integer.

    Raises:
        MatchNotFoundError: If the regex does not match.
    """
    regex = r"Number of atoms\s*...\s*(\d+)"
    match = re_search(regex, contents)
    return int(match.group(1))
