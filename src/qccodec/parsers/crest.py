import math
import re
from collections.abc import Generator
from enum import Enum
from pathlib import Path
from typing import Any

from qcconst import constants
from qcdata import (
    CalcType,
    ExecutionInfo,
    ProgramInput,
    ProgramOutput,
    Provenance,
    SinglePointData,
    Structure,
)

from qccodec.exceptions import MatchNotFoundError, ParserError

from ..registry import register
from .utils import re_finditer, re_search


class CrestFileType(str, Enum):
    """CREST filetypes.

    Maps file types to their names as written in the CREST output directory (except
    for STDOUT and DIRECTORY).
    """

    STDOUT = "stdout"
    DIRECTORY = "directory"
    NUMHESS = "numhess1"
    G98 = "g98.out"
    ENGRAD = "crest.engrad"
    OPTLOG = "crestopt.log"


def iter_files(
    stdout: str | None, directory: Path | str | None
) -> Generator[tuple[CrestFileType, str | bytes | Path], None, None]:
    """
    Iterate over the files in a CREST output directory.

    If stdout is provided, yields a tuple for it.

    If directory is provided, iterates over the directory to yield files according to
    program-specific logic.

    Args:
        stdout: The contents of the CREST stdout file.
        directory: The path to the directory containing the CREST output files.

    Yields:
        (FileType, contents) tuples for a program's output.
    """
    directory = Path(directory) if directory else None

    if stdout is not None:
        yield CrestFileType.STDOUT, stdout

    if directory is not None:
        # Check if the directory exists and is a directory
        if not directory.exists() or not directory.is_dir():
            raise ParserError(
                f"Directory {directory} does not exist or is not a directory."
            )
        yield CrestFileType.DIRECTORY, directory

        # Iterate over the files in the directory and yield their contents
        for filetype in CrestFileType:
            # Ignore STDOUT and DIRECTORY as they are handled above
            if filetype not in (CrestFileType.STDOUT, CrestFileType.DIRECTORY):
                # Check if the file exists in the directory
                file_path = directory / filetype.value
                if file_path.exists():
                    yield filetype, file_path.read_text()


@register(
    filetype=CrestFileType.STDOUT,
    target=("provenance", "program_version"),
    required=False,
)
def parse_version(string: str) -> str:
    """Parse version string from CREST stdout.

    Matches format of 'crest --version' on command line.
    """
    regex = r"Version (\d+\.\d+\.\d+),"
    match = re_search(regex, string)
    return match.group(1)


@register(filetype=CrestFileType.DIRECTORY, calctypes=[CalcType.conformer_search])
def parse_conformers(
    directory: Path | str,
    stdout: str | None,
    input_data: ProgramInput,
    *,
    failed: bool = False,
) -> dict[str, Any]:
    """Parse the conformers from the output directory of a CREST conformer search calculation.

    Args:
        stdout: The contents of the CREST stdout file (not used).
        directory: Path to the directory containing the CREST output files.
        input_data: The input object used for the calculation

    Returns:
        The parsed conformers and their energies as a dictionary.
    """
    directory = Path(directory)
    conformers = Structure.open_multi(
        directory / "crest_conformers.xyz",
        charge=input_data.structure.charge,
        multiplicity=input_data.structure.multiplicity,
    )

    # CREST places the energy as the only value in the comment line
    conf_energies = [
        float(conf.extras[Structure._xyz_comment_key][0]) for conf in conformers
    ]

    # Add identifiers to the conformers if topo is unchanged
    _add_identifiers_to_structures(conformers, input_data)

    return {
        "conformers": conformers,
        "conformer_energies": conf_energies,
    }


@register(filetype=CrestFileType.DIRECTORY, calctypes=[CalcType.conformer_search])
def parse_rotamers(
    directory: Path | str,
    stdout: str | None,
    input_data: ProgramInput,
    *,
    failed: bool = False,
) -> dict[str, Any]:
    """Parse the rotamers from the output directory of a CREST conformer search calculation.

    Args:
        stdout: The contents of the CREST stdout file (not used).
        directory: Path to the directory containing the CREST output files.
        input_data: The input object used for the calculation

    Returns:
        The parsed rotamers and their energies as a dictionary.
    """
    directory = Path(directory)
    rotamers = Structure.open_multi(
        directory / "crest_rotamers.xyz",
        charge=input_data.structure.charge,
        multiplicity=input_data.structure.multiplicity,
    )

    # CREST places the energy as the only value in the comment line
    conf_energies = [
        float(conf.extras[Structure._xyz_comment_key][0]) for conf in rotamers
    ]

    # Add identifiers to the rotamers if topo is unchanged
    _add_identifiers_to_structures(rotamers, input_data)

    return {
        "rotamers": rotamers,
        "rotamer_energies": conf_energies,
    }


def _add_identifiers_to_structures(
    structures: list[Structure], input_data: ProgramInput
) -> None:
    """Add identifiers to a list of Structure objects if the 'topo' is unchanged."""
    if input_data.keywords.get("topo", True):
        ids = input_data.structure.identifiers.model_dump()
        for struct in structures:
            struct.add_identifiers(**ids)


@register(
    filetype=CrestFileType.ENGRAD,
    calctypes=[CalcType.energy, CalcType.gradient],
    target="energy",
)
def parse_energy(contents: str) -> float:
    """Parse the output of a CREST energy and gradient calculation.

    Args:
        contents: The text of the output file.

    Returns:
        The parsed energy as a float.
    """
    energy_regex = r"# Energy \( Eh \)\n#*\n\s*([-\d.]+)"
    return float(re_search(energy_regex, contents).group(1))


@register(
    filetype=CrestFileType.ENGRAD,
    calctypes=[CalcType.energy, CalcType.gradient],
    target="gradient",
)
def parse_gradient(contents: str) -> list[list[float]]:
    """Parse the output of a CREST energy and gradient calculation.

    Args:
        contents: The text of the output file.

    Returns:
        The parsed gradient as a Nx3 list of lists of floats.
    """
    gradient_regex = r"# Gradient \( Eh/a0 \)\n#\s*\n((?:\s*[-\d.]+\n)+)"
    vals = [float(x) for x in re_search(gradient_regex, contents).group(1).split()]
    # Group the values into chunks of 3 (for x, y, z).
    return [vals[i : i + 3] for i in range(0, len(vals), 3)]


@register(filetype=CrestFileType.STDOUT, calctypes=[CalcType.hessian], target="energy")
def parse_energy_numhess(contents: str) -> float:
    """Parse the initial singlepoint calculation energy from stdout.

    Args:
        contents: The text of the stdout file.

    Returns:
        The parsed energy.
    """
    energy_regex = r"Energy\s=\s*([-+]?\d+\.\d+)\s*Eh"
    return float(re_search(energy_regex, contents).group(1))


@register(
    filetype=CrestFileType.NUMHESS, calctypes=[CalcType.hessian], target="hessian"
)
def parse_numhess1(contents: str) -> list[list[float]]:
    """Parse the numerical Hessian from the CREST numhess1 file.

    Args:
        contents: The text of the numhess1 file.

    Returns:
        The parsed Hessian as a list of lists of floats.
    """
    float_regex = r"[-]?\d*\.\d+|\d+"
    numbers = [float(match.group()) for match in re_finditer(float_regex, contents)]
    sqrt_n = int(math.sqrt(len(numbers)))
    if sqrt_n * sqrt_n != len(numbers):
        raise ParserError(
            f"Expected a square matrix, but found {len(numbers)} elements."
        )
    # Reshape the flat list into a square matrix
    return [[numbers[i * sqrt_n + j] for j in range(sqrt_n)] for i in range(sqrt_n)]


@register(
    filetype=CrestFileType.G98, calctypes=[CalcType.hessian], target="freqs_wavenumber"
)
def parse_g98_freqs(contents: str) -> list[float]:
    """Parse the frequencies (wavenumbers) from G98 output text.

    Args:
        contents: The text of the g98 output file.

    Returns:
        A list of frequencies (wavenumbers) as floats.
    """
    regex = r"Frequencies\s+--\s+(?P<floats>(?:-?\d+\.\d+\s*)+)"
    # matches ['-335.2821                75.3406                87.4971\n ', ...]
    matches = re_finditer(regex, contents)
    return [float(freq) for match in matches for freq in match.group("floats").split()]


@register(
    filetype=CrestFileType.G98,
    calctypes=[CalcType.hessian],
    target="normal_modes_cartesian",
)
def parse_g98_normal_modes(contents: str) -> list[list[list[float]]]:
    """Parse the normal mode displacements from G98 output text.

    Args:
        contents: The text of the G98 output file.

    Returns:
        An (n_modes, n_atoms, 3) NumPy array representing the normal mode displacements
        with cartesian coordinates in Bohr.
    """
    block_re = re.compile(r"(Frequencies\s+--.*?)(?=Frequencies\s+--|$)", re.DOTALL)
    floats_re = re.compile(r"[-+]?\d*\.\d+")
    normal_modes_cartesian = []

    # Find all blocks of frequencies and displacements
    for block in block_re.findall(contents):
        # Determine the number of frequencies in this block from freqs line.
        n_freqs = len(block.splitlines()[0].split()[2:])
        # Match all floats in the block after "Atom AN"
        displacements = floats_re.findall(block.split("Atom AN")[1])
        # Initialize an empty list for each mode.
        mode_disp: list[list[list[float]]] = [[] for _ in range(n_freqs)]
        n_atoms = len(displacements) // (3 * n_freqs)
        for i in range(n_atoms):
            for j in range(n_freqs):
                index = i * (3 * n_freqs) + j * 3
                # Convert the displacement values to floats and scale to Bohr
                coords = [
                    float(val) * constants.ANGSTROM_TO_BOHR
                    for val in displacements[index : index + 3]
                ]
                mode_disp[j].append(coords)
        normal_modes_cartesian.extend(mode_disp)

    return normal_modes_cartesian


@register(
    filetype=CrestFileType.DIRECTORY,
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
    """Parse the output directory of a CREST optimization calculation.

    Args:
        directory: Path to the directory containing the CREST output files.
        stdout: The stdout from CREST.
        input_data: The input object used for the calculation.

    Returns:
        The parsed optimization results as a list of ProgramOutput objects.
    """
    if input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    # Read in the xyz file containing the trajectory
    directory = Path(directory)
    xyz_text = (directory / "crestopt.log").read_text()

    # Parse structures and energies from the xyz file
    structures = Structure.from_xyz_multi(
        xyz_text,
        charge=input_data.structure.charge,
        multiplicity=input_data.structure.multiplicity,
    )
    energies = [
        float(struct.extras[Structure._xyz_comment_key][1]) for struct in structures
    ]

    # Intermediate CREST steps expose energies, not gradients. Record only observed data.
    try:
        version = parse_version(stdout) if stdout else None
    except MatchNotFoundError:
        version = None
    provenance = Provenance(program="crest", program_version=version)
    trajectory: list[ProgramOutput] = [
        ProgramOutput(
            input_data=ProgramInput.model_validate(
                {
                    **input_data.model_dump(),
                    "calctype": CalcType.energy,
                    "structure": struct,
                }
            ),
            success=True,
            results=SinglePointData(provenance=provenance, energy=energy),
            execution=ExecutionInfo(scratch_dir=directory),
        )
        for struct, energy in zip(structures, energies)
    ]
    if not trajectory:
        if failed:
            return []
        raise ParserError("CREST optimization trajectory is empty.")

    final = trajectory[-1].model_dump()
    enegrad = directory / CrestFileType.ENGRAD.value
    if enegrad.exists():
        try:
            gradient = parse_gradient(enegrad.read_text())
        except MatchNotFoundError:
            if not failed:
                raise
        else:
            final["input_data"]["calctype"] = CalcType.gradient
            final["results"]["gradient"] = gradient
    trajectory[-1] = ProgramOutput.model_validate(final)

    return trajectory
