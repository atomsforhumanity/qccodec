"""ORCA single point: energy, gradient, or Hessian.

Requires: ``pip install qccompute`` (it installs qccodec) and ORCA, free for academic
use from https://orcaforum.kofo.mpg.de. Put the ORCA directory first on PATH, or point
qccompute at it with QCCOMPUTE_ORCA_EXECUTABLE or ``executables={"orca": ...}``: on
many Linux desktops ``/usr/bin/orca`` is the GNOME screen reader, not ORCA. Parallel
runs (``pal``) also need the OpenMPI version ORCA was built with, and ORCA stops with
"ORCA has to be called with full pathname" unless it runs by its absolute path, which
qccompute does from the release after 0.15.0 (the one with ``executables=``).

How the codec talks to ORCA (manual: https://www.faccts.de/docs/orca/6.0/manual/):

- ``model.method`` and ``model.basis`` start the simple-input line, ``! method basis``.
  Further simple-input words go in ``model.method`` after the method name, e.g.
  "wB97X-D3 TightSCF" or "b3lyp D3BJ CPCM(water)". Leave ``basis`` out for composite
  methods such as "r2scan-3c".
- The calculation type adds ``! Energy``, ``! EnGrad``, or ``! Freq``.
- ``maxcore`` (MB per process) and ``pal`` (number of processes) become ``%maxcore``
  and ``%pal nprocs ... end``.
- A keyword whose value is a dictionary becomes an input block:
  ``{"scf": {"maxiter": 200}}`` writes ``%scf maxiter 200 end``.
- A keyword set to True adds a simple-input word: ``{"TightSCF": True}`` writes
  ``! tightscf`` (ORCA ignores case), and ``numgrad``/``numfreq`` request
  numerical gradients and Hessians. False or None leaves the word out.
- The molecule is passed in your coordinates and atom order; ORCA does not move it.
- ORCA stops on an unknown simple-input word ("INPUT ERROR ... UNRECOGNIZED OR
  DUPLICATED KEYWORD(S)"); the raised error carries ORCA's output.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy, CalcType.hessian

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="orca",
    calctype=calctype,
    structure=structure,
    # Simple-input words after the method; CPCM(water) is implicit solvent.
    model=Model(method="wB97X-D3 TightSCF CPCM(water)", basis="def2-SVP"),
    # e.g. Model(method="r2SCAN-3c") with no basis, or "uks" for open shells
    keywords={
        "maxcore": 1000,  # memory per process, MB
        # "pal": 4,  # parallel processes (needs OpenMPI)
        #
        # Any ORCA input block, as a dictionary:
        "scf": {"maxiter": 200},
        # "cpcm": {"epsilon": 80.4, "refrac": 1.33},  # a custom solvent, with "CPCM"
        #
        # Numerical derivatives, for methods without analytic ones:
        # "numgrad": True,
        # "numfreq": True,
    },
)

output = compute(
    prog_input,
    # timeout=3600,  # seconds before ORCA is killed
    # collect_files=True,  # keep every output file (orca.gbw, orca.hess, ...)
    # executables={"orca": "/opt/orca/orca"},  # or set QCCOMPUTE_ORCA_EXECUTABLE
)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Hessian (Hartree/bohr^2), shape:", results.hessian.shape)
print("ORCA version:", results.provenance.program_version)
