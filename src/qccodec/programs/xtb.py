"""xtb: the semiempirical GFN-xTB, GFN-FF, and g-xTB program, run on the command line.

The codec writes ``structure.xyz`` and runs::

    xtb structure.xyz --<runtype> --chrg <charge> --uhf <unpaired> <method flags> <keywords>

``model.method`` selects the Hamiltonian: ``gfn0``, ``gfn1``, ``gfn2``, ``gfnff``, or
``gxtb``. g-xTB needs the xtb build distributed with g-xTB
(https://github.com/grimme-lab/g-xtb); point qccompute at it with
``executables={"xtb": ...}`` or ``QCCOMPUTE_XTB_EXECUTABLE``. GFN0 reads its parameter
file from the directory named by ``XTBPATH``.

Keywords become command-line flags in xtb's own spelling: ``{"alpb": "water"}`` gives
``--alpb water``, ``{"etemp": 1000}`` gives ``--etemp 1000``, a list gives several
values (``{"alpb": ["water", "bar1M"]}``), ``True`` gives a bare flag, and ``False`` or
``None`` leaves the flag out. For optimizations, ``{"opt": "tight"}`` sets the
optimization level. Detailed input (``$constrain``, ``$fix``, ``$wall``, ...) goes in
an ``xcontrol`` file supplied through ``files`` and named with ``{"input":
"xcontrol"}``; such files are passed through verbatim, so atom indices in them are
1-based as xtb expects.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from qcdata import (
    CalcType,
    ExecutionInfo,
    ProgramInput,
    ProgramOutput,
    Provenance,
    SinglePointData,
    Structure,
)

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput
from ..utils import re_search
from ._common import invocation, parse_g98_freqs, parse_g98_normal_modes

STRUCTURE_FILE = "structure.xyz"
ENGRAD_FILE = "structure.engrad"

codec = Codec(
    "xtb",
    calctypes={
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
    },
)

METHODS: dict[str, tuple[str, ...]] = {
    "gfn0": ("--gfn", "0"),
    "gfn1": ("--gfn", "1"),
    "gfn2": ("--gfn", "2"),
    "gfnff": ("--gfnff",),
    "gxtb": ("--gxtb",),
}
RUNTYPES: dict[CalcType, str] = {
    CalcType.energy: "--sp",
    CalcType.gradient: "--grad",
    CalcType.hessian: "--hess",
    CalcType.optimization: "--opt",
}
# Flags the codec sets itself from the input, or that would change the run type.
RESERVED = {
    "chrg",
    "uhf",
    "gfn",
    "gfnff",
    "gxtb",
    "sp",
    "grad",
    "hess",
    "ohess",
    "bhess",
    "md",
    "omd",
    "metadyn",
    "siman",
    "modef",
    "path",
    "dock",
    "vip",
    "vea",
    "vipea",
    "vomega",
    "vfukui",
    "scc",
}


# ------------------------------------------------------------------ encoding


@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    """Write the structure and build the xtb command line.

    Raises:
        EncoderError: The model is missing or unknown, or a keyword names a flag the
            codec sets itself.
    """
    if inp.model is None:
        raise EncoderError("xtb requires a model, e.g. {'method': 'gfn2'}.")
    method = inp.model.method.lower().replace("-", "")
    if method not in METHODS:
        raise EncoderError(
            f"Unknown xtb method {inp.model.method!r}; use one of {sorted(METHODS)}."
        )
    keywords = dict(inp.keywords)
    reserved = sorted(RESERVED & {k.lower() for k in keywords})
    if reserved:
        raise EncoderError(
            f"Keywords {reserved} are set from the input's calctype, model, and "
            "structure; remove them."
        )

    runtype: list[str] = [RUNTYPES[inp.calctype]]
    level = keywords.pop("opt", None)
    if level is not None:
        if inp.calctype != CalcType.optimization:
            raise EncoderError("The 'opt' keyword sets the optimization level only.")
        runtype.append(str(level))

    arguments = (
        *runtype,
        "--chrg",
        str(inp.structure.charge),
        "--uhf",
        str(inp.structure.multiplicity - 1),
        *METHODS[method],
        *_flags(keywords),
    )
    return invocation(
        inp,
        {STRUCTURE_FILE: inp.structure.to_xyz()},
        STRUCTURE_FILE,
        arguments=arguments,
        environment={"OMP_NUM_THREADS": "1"},
    )


def _flags(keywords: dict[str, Any]) -> list[str]:
    """Turn keywords into ``--name value`` flags in insertion order."""
    flags: list[str] = []
    for name, value in keywords.items():
        if value is None or value is False:
            continue
        flags.append(f"--{name}")
        if value is True:
            continue
        if isinstance(value, Sequence) and not isinstance(value, str):
            flags.extend(str(v) for v in value)
        else:
            flags.append(str(value))
    return flags


# ------------------------------------------------------------------ decoding


@codec.failure
def failure(ctx: Context) -> str | None:
    """Xtb exits 0 when an optimization runs out of cycles; catch that and errors."""
    stdout = ctx.stdout or ""
    if "abnormal termination of xtb" in stdout:
        # The error block lists causes from outermost (-3-) to innermost (-1-).
        reasons = re.findall(r"^-\d+- (.+)$", stdout, flags=re.MULTILINE)
        return "xtb stopped: " + ("; ".join(reversed(reasons)) or "see logs")
    match = re.search(r"FAILED TO CONVERGE GEOMETRY OPTIMIZATION[^*\n]*", stdout)
    if match:
        return f"xtb: {match.group(0).strip().lower()}"
    return None


def _version(stdout: str) -> str | None:
    """Return the version number from stdout, or None."""
    match = re.search(r"xtb version (\d+\.\d+\.\d+)", stdout)
    return match.group(1) if match else None


@codec.parser(target=("provenance", "program_version"), required=False)
def parse_version(stdout: str) -> str:
    """Parse the version number, e.g. ``6.7.1``."""
    version = _version(stdout)
    if version is None:
        raise MatchNotFoundError(r"xtb version", stdout)
    return version


@codec.parser(
    target="energy", calctypes=[CalcType.energy, CalcType.gradient, CalcType.hessian]
)
def parse_energy(stdout: str) -> float:
    """Parse the total energy (Hartree) from the summary block."""
    return float(re_search(r"TOTAL ENERGY\s+(-?\d+\.\d+) Eh", stdout).group(1))


@codec.parser(file=ENGRAD_FILE, target="gradient", calctypes=[CalcType.gradient])
def parse_engrad_gradient(contents: str) -> list[list[float]]:
    """Parse the gradient (Hartree/Bohr) from xtb's ORCA-style ``.engrad`` file."""
    block = re_search(r"# The current gradient in Eh/bohr\n#\n([\s\S]*?)\n#", contents)
    values = [float(v) for v in block.group(1).split()]
    return [values[i : i + 3] for i in range(0, len(values), 3)]


@codec.parser(file="hessian", target="hessian", calctypes=[CalcType.hessian])
def parse_hessian(contents: str) -> list[list[float]]:
    """Parse the mass-unweighted Hessian (Hartree/Bohr^2) from the ``hessian`` file."""
    values = [float(v) for v in contents.split("$hessian", 1)[1].split()]
    n = int(round(len(values) ** 0.5))
    if n * n != len(values):
        raise ParserError(f"Expected a square Hessian, found {len(values)} values.")
    return [values[i * n : (i + 1) * n] for i in range(n)]


codec.parser(file="g98.out", target="freqs_wavenumber", calctypes=[CalcType.hessian])(
    parse_g98_freqs
)
codec.parser(
    file="g98.out", target="normal_modes_cartesian", calctypes=[CalcType.hessian]
)(parse_g98_normal_modes)


def _charges_file(ctx: Context) -> str:
    """GFN-FF writes ``gfnff_charges``; the xTB Hamiltonians write ``charges``."""
    model = ctx.input_data.model if ctx.input_data is not None else None
    if model is not None and model.method.lower().replace("-", "") == "gfnff":
        return "gfnff_charges"
    return "charges"


@codec.parser(
    file=_charges_file,
    target="atomic_charges",
    calctypes=[CalcType.energy, CalcType.gradient, CalcType.hessian],
    required=False,
)
def parse_charges(contents: str) -> list[float]:
    """Parse atomic partial charges, one per line."""
    return [float(v) for v in contents.split()]


@codec.parser(file="directory", target="trajectory", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[ProgramOutput]:
    """Build an optimization trajectory from ``xtbopt.log``.

    xtb records the structure and energy of each step but not its gradient, so steps
    are energy calculations.
    """
    if ctx.input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    try:
        log = ctx.read("xtbopt.log")
    except FileNotFoundError:
        if ctx.failed:
            return []
        raise
    structures = Structure.from_xyz_multi(
        log,
        charge=ctx.input_data.structure.charge,
        multiplicity=ctx.input_data.structure.multiplicity,
    )
    version = _version(ctx.stdout or "")
    provenance = Provenance(program="xtb", program_version=version)
    trajectory = []
    for structure in structures:
        comment = structure.extras[Structure._xyz_comment_key]
        energy = float(comment[comment.index("energy:") + 1])
        trajectory.append(
            ProgramOutput(
                input_data=ProgramInput.model_validate(
                    {
                        **ctx.input_data.model_dump(),
                        "calctype": CalcType.energy,
                        "structure": structure,
                    }
                ),
                success=True,
                results=SinglePointData(provenance=provenance, energy=energy),
                execution=ExecutionInfo(scratch_dir=ctx.directory),
            )
        )
    if not trajectory and not ctx.failed:
        raise ParserError("xtb optimization trajectory is empty.")
    return trajectory
