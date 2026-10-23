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

Hessians also fill frequencies and normal modes from ``g98.out``, where xtb writes
mass-weighted vectors rounded to two decimals; the codec converts them to unit-length
Cartesian displacements, as qcdata defines them.

One keyword is not an xtb flag: ``external_potentials``, point charges for QM/MM
electrostatic embedding, in the same form the Psi4 codec takes: ``[[charge, [x, y,
z]], ...]`` in bohr. An optional third entry, ``[charge, [x, y, z], atomic_number]``,
names the element whose chemical hardness xtb uses to damp that charge's interaction
at short range. The codec writes them to ``pcharge`` and adds an ``$embedding`` group to
the ``xcontrol`` file (appending to yours if you supply one); gradients then also fill
``point_charge_gradient`` from xtb's ``pcgrad``. Only GFN1- and GFN2-xTB apply point
charges: GFN0, GFN-FF, and g-xTB ignore them without a warning, so the codec rejects
them there.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from qcdata import (
    CalcType,
    ExecutionInfo,
    Frame,
    ProgramInput,
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
XCONTROL_FILE = "xcontrol"
POINT_CHARGE_FILE = "pcharge"
POINT_CHARGE_GRADIENT_FILE = "pcgrad"
# xtb 6.7.1 reads $embedding for these Hamiltonians and silently ignores it otherwise.
EMBEDDING_METHODS = {"gfn1", "gfn2"}
EMBEDDING_GROUP = (
    f"$embedding\n   input={POINT_CHARGE_FILE}\n"
    f"   gradient={POINT_CHARGE_GRADIENT_FILE}\n$end\n"
)

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
    point_charges = keywords.pop("external_potentials", None)
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

    generated: dict[str, str | bytes] = {
        STRUCTURE_FILE: inp.require_structure().to_xyz()
    }
    files = dict(inp.files)
    if point_charges:
        if method not in EMBEDDING_METHODS:
            raise EncoderError(
                f"xtb applies point charges only with {sorted(EMBEDDING_METHODS)}; "
                f"with {method} it ignores them without a warning."
            )
        xcontrol = str(keywords.get("input", XCONTROL_FILE))
        keywords["input"] = xcontrol
        generated[POINT_CHARGE_FILE] = point_charge_file(point_charges)
        user_xcontrol = files.pop(xcontrol, "")
        if isinstance(user_xcontrol, bytes):
            user_xcontrol = user_xcontrol.decode()
        generated[xcontrol] = user_xcontrol + EMBEDDING_GROUP
    inp = inp.model_copy(update={"files": files})

    arguments = (
        *runtype,
        "--chrg",
        str(inp.require_structure().charge),
        "--uhf",
        str(inp.require_structure().multiplicity - 1),
        *METHODS[method],
        *_flags(keywords),
    )
    return invocation(
        inp,
        generated,
        STRUCTURE_FILE,
        arguments=arguments,
        environment={"OMP_NUM_THREADS": "1"},
        threads=int(inp.keywords.get("parallel", 1) or 1),
    )


def point_charge_file(point_charges: Sequence[Sequence[Any]]) -> str:
    """Write ``external_potentials`` in xtb's point-charge file format.

    Args:
        point_charges: ``[charge, [x, y, z]]`` in bohr, or ``[charge, [x, y, z],
            atomic_number]``; xtb damps a charge's interaction at short range with
            the chemical hardness of that element.

    Raises:
        EncoderError: An entry does not have that shape.
    """
    lines = [str(len(point_charges))]
    for entry in point_charges:
        if len(entry) not in (2, 3) or len(entry[1]) != 3:
            raise EncoderError(
                "external_potentials entries are [charge, [x, y, z]] or [charge, "
                f"[x, y, z], atomic_number], not {entry!r}."
            )
        charge, (x, y, z) = float(entry[0]), (float(v) for v in entry[1])
        line = f"{charge:.10f} {x:.10f} {y:.10f} {z:.10f}"
        if len(entry) == 3:
            line += f" {int(entry[2])}"
        lines.append(line)
    return "\n".join(lines) + "\n"


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
    """Catch xtb errors, including those it reports with exit code 0.

    xtb warns about unknown command-line options and then runs without them, so a
    misspelled keyword, or ``gxtb`` on an xtb build without g-xTB, would otherwise
    silently produce a different calculation. It also exits 0 when an optimization
    runs out of cycles.
    """
    stdout = ctx.stdout or ""
    if "abnormal termination of xtb" in stdout:
        # The error block lists causes from outermost (-3-) to innermost (-1-).
        reasons = re.findall(r"^-\d+- (.+)$", stdout, flags=re.MULTILINE)
        return "xtb stopped: " + ("; ".join(reversed(reasons)) or "see logs")
    unknown = sorted(set(re.findall(r"Unknown option '([^']+)' provided", stdout)))
    if unknown:
        message = f"xtb ignored unknown options {unknown}."
        if "--gxtb" in unknown:
            message += (
                " This xtb build has no g-xTB; run the xtb build from"
                " https://github.com/grimme-lab/g-xtb (with qccompute, set"
                " QCCOMPUTE_XTB_EXECUTABLE or executables={'xtb': path})."
            )
        return message
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
    file="directory", target="normal_modes_cartesian", calctypes=[CalcType.hessian]
)(parse_g98_normal_modes)


@codec.parser(
    file=POINT_CHARGE_GRADIENT_FILE,
    target="point_charge_gradient",
    calctypes=[CalcType.gradient],
    required=False,
)
def parse_point_charge_gradient(contents: str) -> list[list[float]]:
    """Parse the gradient on the point charges (Hartree/bohr) from ``pcgrad``."""
    values = [float(v) for v in contents.split()]
    if len(values) % 3:
        raise ParserError(f"pcgrad holds {len(values)} values, not a multiple of 3.")
    return [values[i : i + 3] for i in range(0, len(values), 3)]


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


@codec.parser(file="directory", target="frames", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[Frame]:
    """Build an optimization trajectory from ``xtbopt.log``.

    xtb records the structure and energy of each step but not its gradient, so each
    frame holds an energy only.
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
        charge=ctx.input_data.require_structure().charge,
        multiplicity=ctx.input_data.require_structure().multiplicity,
    )
    version = _version(ctx.stdout or "")
    provenance = Provenance(program="xtb", program_version=version)
    trajectory = []
    for structure in structures:
        comment = structure.extras[Structure._xyz_comment_key]
        energy = float(comment[comment.index("energy:") + 1])
        trajectory.append(
            Frame(
                structure=structure,
                success=True,
                results=SinglePointData(provenance=provenance, energy=energy),
                execution=ExecutionInfo(scratch_dir=ctx.directory),
            )
        )
    if not trajectory and not ctx.failed:
        raise ParserError("xtb optimization trajectory is empty.")
    return trajectory
