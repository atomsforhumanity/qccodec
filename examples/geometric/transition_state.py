"""geomeTRIC transition-state search driving another program's gradients.

Requires: ``pip install qccompute qccodec[geometric]`` (geomeTRIC itself) and the
gradient program on PATH. This example uses CREST for fast GFN2-xTB gradients:
https://github.com/crest-lab/crest/releases (official static build).

Notes for geomeTRIC transition-state searches (https://geometric.readthedocs.io/):

- ``CalcType.transition_state`` turns on geomeTRIC's saddle-point optimizer.
  geomeTRIC needs a starting Hessian, which it builds by finite differences of child
  gradients (two per Cartesian coordinate); ``"hessian": "first"`` is the default.
  ``results.trajectory`` holds every gradient calculation, so those displaced
  structures appear in it, and in ``results.energies``, before the optimization
  steps. The last entry is the transition state.
- Start close to the saddle point. This example starts from nearly planar ammonia;
  its inversion transition state is planar, with one imaginary frequency.
- Check the result with a Hessian (e.g. examples/crest/single_point.py with
  CalcType.hessian, as done at the end here): a transition state has exactly one
  imaginary (negative) frequency.
- The child program, coordinate systems, constraints, and QM/MM point charges work
  as in optimization.py; swap the ``ProgramSpec`` for xtb, Psi4, TeraChem, ...
- Transition-state steps use a small trust radius by default (``trust`` 0.01,
  ``tmax`` 0.03 angstrom), so expect more steps than for a minimum.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

# Ammonia near its planar inversion saddle point, coordinates in bohr.
structure = Structure(
    symbols=["N", "H", "H", "H"],
    geometry=[
        [0.0, 0.0, 0.01],
        [1.88, 0.0, 0.01],
        [-0.94, 1.62, 0.01],
        [-0.94, -1.62, 0.01],
    ],
    charge=0,
    multiplicity=1,
)

# The program that computes energies and gradients: CREST with GFN2-xTB.
gradient_program = ProgramSpec(
    program="crest",
    calctype=CalcType.gradient,
    model=Model(method="gfn2"),
    keywords={"threads": 1},  # add "calculation": {"level": [{"alpb": "water"}]}
)

# Psi4 instead (any gradient program works; see optimization.py for more):
# gradient_program = ProgramSpec(
#     program="psi4",
#     calctype=CalcType.gradient,
#     model=Model(method="hf", basis="sto-3g"),
#     keywords={"scf_type": "pk"},
#     cmdline_args=["-n", "1", "--memory", "2GB"],
# )

prog_input = ProgramInput(
    program="geometric",
    calctype=CalcType.transition_state,
    structure=structure,
    subprograms=[gradient_program],
    keywords={
        "coordsys": "tric",  # or "dlc", "prim", "cart"
        "maxiter": 100,
        "convergence_set": "gau",  # or "gau_tight", ...
        # "hessian": "first",  # or "each": exact Hessian every step (costly)
        # "trust": 0.01,  # starting trust radius (angstrom)
        # "tmax": 0.03,  # largest trust radius (angstrom)
    },
)

output = compute(
    prog_input,
    # propagate_wfn=True,  # TeraChem: start each step from the previous orbitals
    # timeout=3600,
)

results = output.results
print("Gradient calculations:", len(results.trajectory))
print("Transition-state energy (Hartree):", results.energies[-1])
print("Transition-state structure (bohr):\n", results.final_structure.geometry)

# Confirm the saddle point: exactly one imaginary (negative) frequency.
hessian = compute(
    ProgramInput(
        program="crest",
        calctype=CalcType.hessian,
        structure=results.final_structure,
        model=Model(method="gfn2"),
        keywords={"threads": 1},
    )
)
print("Frequencies (cm^-1):", hessian.results.freqs_wavenumber)
