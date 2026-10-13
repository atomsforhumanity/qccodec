"""ORCA: input file encoding and stdout/.hess/trajectory parsing."""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qcdata import (
    CalcType,
    ExecutionInfo,
    ProgramInput,
    ProgramOutput,
    SinglePointData,
    Structure,
)

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ..utils import re_search
from ._common import invocation

codec = Codec(
    "orca",
    calctypes={
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
        CalcType.transition_state,
    },
)

XYZ_FILENAME = "geometry.xyz"
PADDING = 20  # padding between keyword and value


# ------------------------------------------------------------------ encoding


def _validate_keywords(keywords: dict[str, Any]) -> None:
    """Reject keywords that belong on structured qcdata fields. Expects lowercase keys."""
    non_blocks = {"coords": ".structure"}
    non_block_keywords = {
        "method": {"method": ".model.method", "runtyp": ".calctype"},
        "basis": {"basis": ".model.basis"},
    }
    for block, where in non_blocks.items():
        if block in keywords:
            raise EncoderError(
                f"Block '{keywords[block]}' should not be set as a keyword. "
                f"Set its data on '{where}'."
            )
    for block, disallowed in non_block_keywords.items():
        for variable, where in disallowed.items():
            if keywords.get(block, {}).get(variable) is not None:
                raise EncoderError(
                    f"Keyword '{variable}' in block '{block}' should not be set "
                    f"directly. It should be set at '{where}'",
                )


def _fmt(key: str, value: Any) -> Any:
    """Format a value for an ORCA input block."""
    if key.casefold() in {"auxc", "auxj", "auxjk"}:
        return f'"{value}"'  # ORCA needs quotes for certain keywords
    if isinstance(value, bool):
        return str(value).lower()
    return value


@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    """Write ``orca.inp`` and the geometry file for an ORCA calculation.

    Notes:
        - ORCA keywords are case-insensitive; the casing in ``keywords`` is kept.
        - Pass ``{"numgrad": True}`` (or a block dictionary) for numerical gradients.
    """
    if inp.model is None:
        raise EncoderError("orca input requires a scientific model.")

    kw = {k.casefold(): v for k, v in inp.keywords.items()}
    _validate_keywords(kw)

    lines: list[str] = []
    if "maxcore" in kw:
        lines.append(f"%maxcore {kw['maxcore']}")
    if "pal" in kw:
        lines.append(f"%pal nprocs {kw['pal']} end")
    if lines:
        lines.append("")

    model_line = f"! {inp.model.method}"
    if inp.model.basis:  # composite methods carry no basis
        model_line += f" {inp.model.basis}"
    lines.append(model_line)
    if "numgrad" in kw:
        lines.append("! numgrad")

    runtyp = {
        CalcType.energy: "energy",
        CalcType.gradient: "engrad",
        CalcType.hessian: "numfreq" if "numfreq" in kw else "freq",
        CalcType.optimization: "opt",
        CalcType.transition_state: "optts",
    }[inp.calctype]
    lines.append(f"! {runtyp}\n")

    blocks = {k: v for k, v in kw.items() if isinstance(v, Mapping)}
    for block, values in blocks.items():
        lines.append(f"%{block}")
        for key, value in values.items():
            lines.append(f"    {key:<{PADDING}} {_fmt(key, value)}")
        lines.append("end")
    if blocks:
        lines.append("")

    lines.append(
        f"* xyzfile {inp.structure.charge} {inp.structure.multiplicity} {XYZ_FILENAME}"
    )
    return invocation(
        inp,
        {"orca.inp": "\n".join(lines) + "\n", XYZ_FILENAME: inp.structure.to_xyz()},
        "orca.inp",
    )


# ------------------------------------------------------------------ decoding


@codec.parser(
    target="energy", calctypes=[CalcType.energy, CalcType.gradient, CalcType.hessian]
)
def parse_energy(stdout: str) -> float:
    """Parse the final single point energy from ORCA stdout."""
    return float(
        re_search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", stdout).group(1)
    )


@codec.parser(target="gradient", calctypes=[CalcType.gradient])
def parse_gradient(stdout: str) -> list[list[float]]:
    """Parse the Cartesian gradient block from ORCA stdout as an N x 3 list."""
    header = re_search(r"CARTESIAN GRADIENT.*?(?=\d)", stdout, flags=re.DOTALL)
    block_lines = itertools.takewhile(
        lambda line: re.search(r"\d\s*$", line), stdout[header.end() :].splitlines()
    )
    line_regex = r".*:\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
    return [
        list(map(float, re_search(line_regex, line).groups())) for line in block_lines
    ]


@codec.parser(target="gradient", calctypes=[CalcType.hessian], required=False)
def parse_gradient_hessian(stdout: str) -> list[list[float]]:
    """Parse the gradient from a Hessian job; analytic Hessian jobs omit it."""
    return parse_gradient(stdout)


@codec.parser(
    file=lambda ctx: f"{parse_basename(ctx.stdout or '')}.hess",
    target="hessian",
    calctypes=[CalcType.hessian],
)
def parse_hessian(contents: str) -> list[list[float]]:
    """Parse the Hessian from ORCA's ``<basename>.hess`` file."""
    entry = next((b for b in contents.split("$") if b.startswith("hess")), None)
    if entry is None:
        raise ParserError("Failed to find hessian block in Hessian file.")
    dim = int(entry.splitlines()[1])

    # Blocks are separated by column-index lines such as '  0  1  2  3 ...'; the
    # final block may list a single column.
    parts = re.split(r"^\s*\d+(?:\s+\d+)*\s*$", entry, flags=re.MULTILINE)
    if not len(parts) > 1:
        raise ParserError(f"Failed to parse blocks in hessian entry: {entry}")

    hessian: list[list[float]] = [[] for _ in range(dim)]
    for block in (part.strip() for part in parts[1:]):
        lines = block.splitlines()
        if len(lines) != dim:
            raise ParserError(
                f"Block line count {len(lines)} does not match dimension {dim}: {block}"
            )
        for i, line in enumerate(lines):
            hessian[i].extend(map(float, line.split()[1:]))
    return hessian


@codec.parser(
    file="directory",
    target="trajectory",
    calctypes=[CalcType.optimization, CalcType.transition_state],
)
def parse_trajectory(ctx: Context) -> list[ProgramOutput]:
    """Build an optimization trajectory from ``<basename>_trj.xyz`` and stdout."""
    if ctx.input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    if not ctx.stdout and ctx.failed:
        return []
    stdout = ctx.stdout or ""
    basename = parse_basename(stdout)
    try:
        file = ctx.path(f"{basename}_trj.xyz")
    except FileNotFoundError:
        if ctx.failed:
            return []
        raise ParserError(f"Trajectory file does not exist: {basename}_trj.xyz")

    # The trajectory file carries no charge/multiplicity; take them from the input.
    structures = Structure.open_multi(
        file,
        charge=ctx.input_data.structure.charge,
        multiplicity=ctx.input_data.structure.multiplicity,
    )

    initialization = re_search(
        r"^(.*?\*\*\*\*END\s+OF\s+INPUT\*\*\*\*\s*\n\s*=*)",
        stdout,
        flags=re.MULTILINE | re.VERBOSE | re.DOTALL,
    ).group(1)
    regex = (
        r"GEOMETRY\s*OPTIMIZATION\s*CYCLE\s*\d+.*?"  # cycle line
        r"\*+\n"  # line of stars
        r"(.*?)"  # body (captured)
        r"-+\s*\n\s*ORCA\s+GEOMETRY\s+RELAXATION\s+STEP"  # footer
    )
    per_step = re.findall(regex, stdout, flags=re.DOTALL)
    if not per_step:
        raise MatchNotFoundError(regex, stdout)

    trajectory: list[ProgramOutput] = []
    for structure, step_stdout in zip(structures, per_step):
        parsed = ctx.decode(CalcType.gradient, stdout=initialization + step_stdout)
        assert isinstance(parsed, SinglePointData)  # for mypy
        values = parsed.model_dump()
        values["energy"] = structure.extras[Structure._xyz_comment_key][-1]
        results = SinglePointData(**values)
        step_input = ProgramInput.model_validate(
            {
                **ctx.input_data.model_dump(),
                "calctype": CalcType.gradient
                if results.gradient is not None
                else CalcType.energy,
                "structure": structure,
            }
        )
        trajectory.append(
            ProgramOutput(
                input_data=step_input,
                success=True,
                results=results,
                execution=ExecutionInfo(scratch_dir=ctx.directory),
            )
        )
    return trajectory


@codec.parser(target=("provenance", "program_version"), required=False)
def parse_version(stdout: str) -> str:
    """Parse the version string from ORCA stdout."""
    return re_search(r"Program Version (\d+\.\d+\.\d+)", stdout).group(1)


@codec.parser(target="calcinfo_natoms", required=False)
def parse_natoms(stdout: str) -> int:
    """Parse the number of atoms from ORCA stdout."""
    return int(re_search(r"Number of atoms\s*...\s*(\d+)", stdout).group(1))


def parse_basename(stdout: str) -> str:
    """Parse the output file basename from ORCA stdout."""
    return Path(re_search(r"NAME\s+=\s+(.*)", stdout).group(1)).stem
