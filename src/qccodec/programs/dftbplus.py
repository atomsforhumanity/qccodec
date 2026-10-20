"""DFTB+: density-functional tight binding (DFTB, SCC-DFTB, DFTB3) and xTB.

The codec writes ``dftb_in.hsd`` and ``geometry.xyz`` and runs ``dftb+`` in the
calculation directory; results come from DFTB+'s machine-readable ``results.tag``.
Install DFTB+ with ``conda install -c conda-forge dftbplus``; qccompute finds ``dftb+``
on ``PATH`` (or set ``QCCOMPUTE_DFTBPLUS_EXECUTABLE``). Tested with DFTB+ 25.1, and the
input pins HSD parser version 14.

Inputs map to DFTB+ as follows:

- ``model.method`` is the Hamiltonian: ``DFTB3`` (third order, with Hubbard
  derivatives and H-bond damping), ``SCC-DFTB`` (self-consistent charges), ``DFTB``
  (non-self-consistent), or ``GFN2-xTB``, ``GFN1-xTB``, ``IPEA1-xTB`` (DFTB+'s xTB
  implementation, which needs no parameter files).
- ``model.basis`` names the Slater-Koster parameter set for the DFTB methods
  (``3ob-3-1``, ``mio-1-1``) and is empty for xTB. The codec takes each element's
  maximum angular momentum and, for DFTB3, its Hubbard derivative and the H-bond
  damping exponent from ``qcref.dftb_parameter_sets``, as the set's documentation
  prescribes. Cite the references in the set's ``citation`` when you publish.
- The Slater-Koster files are not shipped with DFTB+. Download sets from
  https://github.com/dftbparams into one directory, each in its own subdirectory named
  like ``model.basis`` (as the archives unpack: ``<dir>/3ob-3-1/O-H.skf``), and name
  that directory with the ``slater_koster_dir`` keyword or the
  ``QCCODEC_DFTB_PARAMETERS`` environment variable, read when the input is encoded.
  (ASE's ``DFTB_PREFIX`` names a single set's directory instead, so it is not used.)
- ``structure.charge`` becomes ``Charge``. A multiplicity above 1 adds
  ``SpinPolarisation = Colinear { UnpairedElectrons = ... }``; DFTB methods then also
  need the set's ``SpinConstants`` in keywords.
- ``keywords`` are DFTB+ input in HSD form, as a nested dictionary merged into the
  generated input: ``{"Hamiltonian": {"SCCTolerance": 1e-8, "MaxSCCIterations":
  500}}``. A dictionary is an HSD block, ``True``/``False`` become ``Yes``/``No``,
  numbers are written as numbers, a list becomes space-separated values, and a string
  is written verbatim, so it carries HSD's own syntax: ``'"water"'`` for a quoted
  string, ``'Lbfgs {}'`` or a multi-line ``'DftD3 { ... }'`` for a typed block (HSD
  takes one assignment per line, so write such blocks across lines).
  ``Geometry`` and ``Hamiltonian/Charge`` come from ``structure`` and cannot
  be set. Atom indices in verbatim HSD (``MovedAtoms``, ``Atoms``) are 1-based, as
  DFTB+ counts them.
- ``cmdline_args`` are appended to the ``dftb+`` command.

Calculation types: ``energy``, ``gradient``, ``hessian`` (``Driver =
SecondDerivatives``: DFTB+ differentiates gradients at displaced geometries only, so
the result has the Hessian but no energy or gradient at the input geometry; run a
``gradient`` for those. The codec tightens ``SCCTolerance`` to 1e-8 for these runs,
since DFTB+'s default 1e-5 leaves noise of about 1e-4 in the Hessian), and ``optimization`` (``Driver = GeometryOptimisation`` with
LBFGS on all atoms; override any part through ``keywords["Driver"]``). Energies and
gradients also fill ``atomic_charges`` (Mulliken) and ``scf_dipole_moment``. The
optimization trajectory holds each step's structure and energy; the last step also
carries the gradient.

DFTB+ never moves or reorients the molecule, so results are in the input's frame.
DFTB+ stops on any keyword it does not recognize, and the codec reports its error
message.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
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
from qcref import dftb_parameter_sets

from ..codec import Codec, Context
from ..exceptions import EncoderError, MatchNotFoundError, ParserError
from ..native import SubprocessInput

INPUT_FILE = "dftb_in.hsd"
STRUCTURE_FILE = "geometry.xyz"
TRAJECTORY_PREFIX = "geo_end"
PARSER_VERSION = 14
PARAMETERS_VARIABLE = "QCCODEC_DFTB_PARAMETERS"

codec = Codec(
    "dftbplus",
    calctypes={
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
    },
)

DFTB_METHODS = {"dftb3": "DFTB3", "scc-dftb": "SCC-DFTB", "dftb": "DFTB"}
XTB_METHODS = {"gfn2-xtb": "GFN2-xTB", "gfn1-xtb": "GFN1-xTB", "ipea1-xtb": "IPEA1-xTB"}


@dataclass
class Typed:
    """An HSD node with a type: ``name = type { body }``."""

    type: str
    body: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ encoding


@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    """Write ``dftb_in.hsd`` and the geometry, and build the ``dftb+`` command.

    Raises:
        EncoderError: The method or parameter set is unknown or missing, an element is
            not covered by the set, the Slater-Koster directory is not given, keywords
            set structured input, or supplied files clash with generated ones.
    """
    if inp.model is None:
        raise EncoderError(
            "DFTB+ requires a model, e.g. {'method': 'DFTB3', 'basis': '3ob-3-1'}."
        )
    keywords = copy.deepcopy(dict(inp.keywords))
    skf_dir = keywords.pop("slater_koster_dir", None)
    hamiltonian = _hamiltonian(inp, skf_dir)
    document: dict[str, Any] = {
        "Geometry": Typed("xyzFormat", {"<<<": f'"{STRUCTURE_FILE}"'}),
        "Hamiltonian": hamiltonian,
        "Options": {"WriteResultsTag": True},
        "ParserOptions": {"ParserVersion": PARSER_VERSION},
    }
    if inp.calctype == CalcType.gradient:
        document["Analysis"] = {"PrintForces": True}
    elif inp.calctype == CalcType.hessian:
        document["Driver"] = Typed(
            "SecondDerivatives", {"Atoms": "1:-1", "Delta": 1e-4}
        )
        if hamiltonian.type == "DFTB" and hamiltonian.body["Scc"]:
            # The default 1e-5 leaves SCC noise of ~1e-4 in finite-difference Hessians.
            hamiltonian.body["SCCTolerance"] = 1e-8
    elif inp.calctype == CalcType.optimization:
        document["Driver"] = Typed(
            "GeometryOptimisation",
            {
                "Optimiser": "Lbfgs {}",
                "MovedAtoms": "1:-1",
                "MaxSteps": 200,
                "Convergence": {"GradElem": 1e-4},
                "AppendGeometries": True,
                "OutputPrefix": f'"{TRAJECTORY_PREFIX}"',
            },
        )
        document["Analysis"] = {"PrintForces": True}

    if "Geometry" in keywords:
        raise EncoderError("Geometry comes from structure; remove it from keywords.")
    if "Charge" in (keywords.get("Hamiltonian") or {}):
        raise EncoderError(
            "Charge comes from structure.charge; remove it from keywords."
        )
    _merge(document, keywords)

    generated: dict[str, str | bytes] = {
        INPUT_FILE: render(document),
        STRUCTURE_FILE: inp.structure.to_xyz(),
    }
    conflicts = generated.keys() & inp.files.keys()
    if conflicts:
        raise EncoderError(
            f"Supplied files conflict with generated inputs: {sorted(conflicts)}"
        )
    return SubprocessInput(
        command=("dftb+", *inp.cmdline_args),
        files={**inp.files, **generated},
        environment={"OMP_NUM_THREADS": "1"},
    )


def _hamiltonian(inp: ProgramInput, skf_dir: str | None) -> Typed:
    """Build the Hamiltonian block for the model."""
    assert inp.model is not None
    key = inp.model.method.lower().replace("_", "-")
    charge = inp.structure.charge
    spin: dict[str, Any] = {}
    if inp.structure.multiplicity > 1:
        unpaired = inp.structure.multiplicity - 1
        spin = {"SpinPolarisation": Typed("Colinear", {"UnpairedElectrons": unpaired})}

    if key in XTB_METHODS:
        if inp.model.basis:
            raise EncoderError(
                f"DFTB+'s xTB needs no parameter set; remove basis {inp.model.basis!r}."
            )
        return Typed(
            "xTB", {"Method": f'"{XTB_METHODS[key]}"', "Charge": charge, **spin}
        )
    if key not in DFTB_METHODS:
        raise EncoderError(
            f"Unknown DFTB+ method {inp.model.method!r}; use one of "
            f"{[*DFTB_METHODS.values(), *XTB_METHODS.values()]}."
        )
    method = DFTB_METHODS[key]
    if not inp.model.basis:
        raise EncoderError(
            f"{method} needs a Slater-Koster parameter set in model.basis, e.g. "
            f"'3ob-3-1' or 'mio-1-1'; known: {[s.name for s in dftb_parameter_sets.values()]}."
        )
    try:
        parameters = dftb_parameter_sets.by_name(inp.model.basis)
    except KeyError:
        raise EncoderError(
            f"Unknown DFTB parameter set {inp.model.basis!r}; known: "
            f"{[s.name for s in dftb_parameter_sets.values()]}."
        ) from None
    elements = list(dict.fromkeys(inp.structure.symbols))
    missing = [e for e in elements if e not in parameters.elements]
    if missing:
        raise EncoderError(
            f"Parameter set {parameters.name} has no parameters for {missing}; it "
            f"covers {sorted(parameters.elements)}."
        )
    body: dict[str, Any] = {"Scc": method != "DFTB", "Charge": charge}
    if method == "DFTB3":
        derivatives = {e: parameters.elements[e].hubbard_derivative for e in elements}
        undefined = [e for e, value in derivatives.items() if value is None]
        if undefined or parameters.h_damping_exponent is None:
            raise EncoderError(
                f"Parameter set {parameters.name} is a {parameters.model} set and "
                f"defines no DFTB3 Hubbard derivatives for {undefined or elements}; "
                "use SCC-DFTB, or a DFTB3 set such as 3ob-3-1."
            )
        body["ThirdOrderFull"] = True
        body["HubbardDerivs"] = {
            e: float(v) for e, v in derivatives.items() if v is not None
        }
        body["HCorrection"] = Typed(
            "Damping", {"Exponent": float(parameters.h_damping_exponent)}
        )
    body["MaxAngularMomentum"] = {
        e: f'"{parameters.elements[e].max_angular_momentum}"' for e in elements
    }
    body["SlaterKosterFiles"] = Typed(
        "Type2FileNames",
        {
            "Prefix": f'"{_skf_prefix(skf_dir, parameters.name)}"',
            "Separator": '"-"',
            "Suffix": '".skf"',
        },
    )
    body.update(spin)
    return Typed("DFTB", body)


def _skf_prefix(skf_dir: str | None, name: str) -> str:
    """Return ``<directory>/<parameter set>/``, where DFTB+ finds the ``.skf`` files.

    Raises:
        EncoderError: Neither ``slater_koster_dir`` nor ``QCCODEC_DFTB_PARAMETERS`` is
            set.
    """
    directory = skf_dir or os.environ.get(PARAMETERS_VARIABLE)
    if not directory:
        raise EncoderError(
            "Point DFTB+ at the Slater-Koster files with the 'slater_koster_dir' "
            f"keyword or the {PARAMETERS_VARIABLE} environment variable: a directory "
            "holding each parameter set as a subdirectory (download sets from "
            "https://github.com/dftbparams)."
        )
    return os.path.join(os.path.abspath(directory), name, "")


def _merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    """Merge keyword dictionaries into the generated document, in place.

    Dictionaries merge into blocks (including typed blocks' bodies); any other value
    replaces what was generated.
    """
    for key, value in updates.items():
        current = target.get(key)
        if isinstance(value, dict):
            if isinstance(current, Typed):
                _merge(current.body, value)
                continue
            if isinstance(current, dict):
                _merge(current, value)
                continue
        target[key] = value


def render(document: dict[str, Any], indent: int = 0) -> str:
    """Write a nested dictionary as HSD text."""
    pad = "  " * indent
    lines: list[str] = []
    for key, value in document.items():
        if isinstance(value, Typed):
            if value.body:
                lines.append(f"{pad}{key} = {value.type} {{")
                lines.append(render(value.body, indent + 1).rstrip("\n"))
                lines.append(f"{pad}}}")
            else:
                lines.append(f"{pad}{key} = {value.type} {{}}")
        elif isinstance(value, dict):
            lines.append(f"{pad}{key} {{")
            if value:
                lines.append(render(value, indent + 1).rstrip("\n"))
            lines.append(f"{pad}}}")
        elif key == "<<<":
            lines.append(f"{pad}<<< {value}")
        else:
            lines.append(f"{pad}{key} = {_scalar(value)}")
    return "\n".join(lines) + "\n"


def _scalar(value: Any) -> str:
    """Write one HSD value: booleans as Yes/No, lists space-separated, text verbatim."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple)):
        return " ".join(_scalar(v) for v in value)
    return str(value)


# ------------------------------------------------------------------ decoding


def parse_results_tag(contents: str) -> dict[str, list[float]]:
    """Read DFTB+'s ``results.tag``: ``name :type:rank:shape`` lines, then values.

    Returns:
        Each quantity's values in file order (Fortran column-major, so a 3 x N array
        of vectors reads atom by atom).
    """
    values: dict[str, list[float]] = {}
    name: str | None = None
    for line in contents.splitlines():
        header = re.match(r"^(\w+)\s*:(\w+):\d+:", line)
        if header:
            name = header.group(1) if header.group(2) in ("real", "integer") else None
            if name:
                values[name] = []
        elif name:
            values[name].extend(float(v.replace("D", "E")) for v in line.split())
    return values


def _tag(contents: str, name: str) -> list[float]:
    values = parse_results_tag(contents).get(name)
    if not values:
        raise MatchNotFoundError(name, contents)
    return values


def _vectors(values: list[float]) -> list[list[float]]:
    if len(values) % 3:
        raise ParserError(f"Expected xyz triples, found {len(values)} values.")
    return [values[i : i + 3] for i in range(0, len(values), 3)]


@codec.failure
def failure(ctx: Context) -> str | None:
    """Report DFTB+'s ``ERROR!`` message, or an optimization that did not converge."""
    stdout = ctx.stdout or ""
    if "ERROR!" in stdout:
        before, block = stdout.split("ERROR!", 1)
        reasons = re.findall(r"^-> (.+)$", block, flags=re.MULTILINE)
        path = re.search(r"^Path: (.+)$", block, flags=re.MULTILINE)
        message = "DFTB+ stopped: " + ("; ".join(reasons) or "see logs")
        if path:
            message += f" (at {path.group(1)})"
        if "ignored by the parser" in before:
            # DFTB+ lists the input it did not recognize in a warning just before.
            warning = before.split("ignored by the parser")[-1]
            ignored = re.findall(r"^Path: (.+)$", warning, flags=re.MULTILINE)
            message += f". Unrecognized input: {', '.join(ignored)}"
        return message
    if re.search(r"Geometry did NOT converge", stdout):
        return "DFTB+: geometry optimization did not converge within MaxSteps."
    return None


@codec.parser(target=("provenance", "program_version"), required=False)
def parse_version(stdout: str) -> str:
    """Parse ``release 25.1``, or ``dev-<commit>`` for development builds."""
    match = re.search(
        r"DFTB\+ (?:release (\S+)|development version \(commit: (\w+)\))", stdout
    )
    if match is None:
        raise MatchNotFoundError("DFTB+ version", stdout)
    return match.group(1) or f"dev-{match.group(2)}"


@codec.parser(
    file="results.tag", target="energy", calctypes=[CalcType.energy, CalcType.gradient]
)
def parse_energy(contents: str) -> float:
    """Parse the total (Mermin free) energy in Hartree."""
    return _tag(contents, "mermin_energy")[0]


@codec.parser(file="results.tag", target="gradient", calctypes=[CalcType.gradient])
def parse_gradient(contents: str) -> list[list[float]]:
    """Parse the gradient (Hartree/bohr), the negative of DFTB+'s forces."""
    return [[-v for v in vector] for vector in _vectors(_tag(contents, "forces"))]


@codec.parser(
    file="results.tag",
    target="atomic_charges",
    calctypes=[CalcType.energy, CalcType.gradient],
    required=False,
)
def parse_charges(contents: str) -> list[float]:
    """Parse Mulliken gross atomic charges (e)."""
    return _tag(contents, "gross_atomic_charges")


@codec.parser(
    file="results.tag",
    target="scf_dipole_moment",
    calctypes=[CalcType.energy, CalcType.gradient],
    required=False,
)
def parse_dipole(contents: str) -> list[float]:
    """Parse the dipole moment (e bohr)."""
    return _tag(contents, "dipole_moments")[:3]


@codec.parser(file="hessian.out", target="hessian", calctypes=[CalcType.hessian])
def parse_hessian(contents: str) -> list[list[float]]:
    """Parse the Hessian (Hartree/bohr^2) from ``hessian.out``, row by row."""
    values = [float(v) for v in contents.split()]
    n = int(round(len(values) ** 0.5))
    if n * n != len(values):
        raise ParserError(f"Expected a square Hessian, found {len(values)} values.")
    return [values[i * n : (i + 1) * n] for i in range(n)]


@codec.parser(file="directory", target="trajectory", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[ProgramOutput]:
    """Build the trajectory from ``geo_end.xyz`` and each step's energy in stdout.

    Every step carries its structure and energy; the last also carries the gradient
    from ``results.tag``.
    """
    if ctx.input_data is None:
        raise ParserError("Optimization trajectory parsing requires input_data.")
    try:
        frames = ctx.read(f"{TRAJECTORY_PREFIX}.xyz")
    except FileNotFoundError:
        if ctx.failed:
            return []
        raise
    energies = [
        float(m.group(1))
        for m in re.finditer(
            r"^Total Mermin free energy:\s+(-?\d+\.\d+) H",
            ctx.stdout or "",
            flags=re.MULTILINE,
        )
    ]
    structures = _xyz_frames(frames, ctx.input_data.structure)
    if len(energies) < len(structures):
        raise ParserError(
            f"{len(structures)} optimization steps but {len(energies)} energies."
        )
    gradient: list[list[float]] | None = None
    try:
        gradient = parse_gradient(ctx.read("results.tag"))
    except (FileNotFoundError, MatchNotFoundError):
        gradient = None
    version = None
    try:
        version = parse_version(ctx.stdout or "")
    except MatchNotFoundError:
        pass
    provenance = Provenance(program="dftbplus", program_version=version)
    trajectory = []
    for i, structure in enumerate(structures):
        last = i == len(structures) - 1
        calctype = CalcType.gradient if last and gradient else CalcType.energy
        trajectory.append(
            ProgramOutput(
                input_data=ProgramInput.model_validate(
                    {
                        **ctx.input_data.model_dump(),
                        "calctype": calctype,
                        "structure": structure,
                    }
                ),
                success=True,
                results=SinglePointData.model_validate(
                    {
                        "provenance": provenance,
                        "energy": energies[i],
                        "gradient": gradient if calctype == CalcType.gradient else None,
                    }
                ),
                execution=ExecutionInfo(scratch_dir=ctx.directory),
            )
        )
    return trajectory


def _xyz_frames(text: str, reference: Structure) -> list[Structure]:
    """Read DFTB+'s appended xyz frames, whose atom lines carry an extra charge column."""
    lines = text.splitlines()
    structures = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natoms = int(lines[i])
        atoms = [line.split()[:4] for line in lines[i + 2 : i + 2 + natoms]]
        xyz = f"{natoms}\n\n" + "\n".join(" ".join(atom) for atom in atoms) + "\n"
        structures.append(
            Structure.from_xyz(
                xyz, charge=reference.charge, multiplicity=reference.multiplicity
            )
        )
        i += 2 + natoms
    return structures
