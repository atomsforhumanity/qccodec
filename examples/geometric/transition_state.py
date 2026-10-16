"""geomeTRIC transition-state search driving another program's gradients.

Requires: ``pip install qccompute qccodec[geometric]`` (geomeTRIC itself) and the
gradient program on PATH. This example uses CREST for fast GFN2-xTB gradients:
https://github.com/crest-lab/crest/releases (official static build).

Notes for geomeTRIC transition-state searches (https://geometric.readthedocs.io/):

- ``CalcType.transition_state`` turns on geomeTRIC's saddle-point optimizer.
  geomeTRIC needs a starting Hessian, which it builds by finite differences of child
  gradients (two per Cartesian coordinate). Those displaced gradients are not
  optimizer steps, so ``results.trajectory`` leaves them out: it holds one frame per
  step, and each Hessian geomeTRIC computes is attached to the frame at its
  geometry (``frame.results.hessian``).
- ``"hessian"`` chooses when: ``"first"`` (the default) at the start,
  ``"first+last"`` also at the end, which is how this example checks its result,
  ``"each"`` at every step (costly).
- Start close to the saddle point. This example starts from nearly planar ammonia;
  its inversion transition state is planar, with one imaginary frequency.
- A transition state has exactly one negative Hessian eigenvalue (one imaginary
  frequency). The final frame's Hessian shows it, whatever the child program.
- The child program, coordinate systems, constraints, and QM/MM point charges work
  as in optimization.py; swap the ``ProgramSpec`` for xtb, Psi4, TeraChem, ...
- Transition-state steps use a small trust radius by default (``trust`` 0.01,
  ``tmax`` 0.03 angstrom), so expect more steps than for a minimum.
"""

import numpy as np
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
#     model=Model(method="wb97x-d", basis="def2-svp"),
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
        "hessian": "first+last",  # exact Hessians at the start and the end
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
print("Steps:", len(results.trajectory))
print("Transition-state energy (Hartree):", results.energies[-1])
print("Transition-state structure (bohr):\n", results.final_structure.geometry)

# Confirm the saddle point: exactly one negative eigenvalue of the final Hessian.
final_hessian = results.trajectory[-1].results.hessian
eigenvalues = np.linalg.eigvalsh(final_hessian)
print("Negative Hessian eigenvalues:", int((eigenvalues < -1e-3).sum()), "(1 for a TS)")
