"""ORCA geometry optimization with its built-in optimizer.

Requires: ``pip install qccompute`` (it installs qccodec) and ORCA on PATH, ahead of
``/usr/bin/orca`` (the GNOME screen reader on many Linux desktops); or set
QCCOMPUTE_ORCA_EXECUTABLE. See single_point.py for installing and running ORCA.

Notes for ORCA optimizations (manual: https://www.faccts.de/docs/orca/6.0/manual/,
"Geometry Optimization"):

- The codec adds ``! Opt``. Convergence presets are simple-input words that go in
  ``model.method``: "LooseOpt", "TightOpt", "VeryTightOpt".
- Optimizer settings go in the ``geom`` block: ``{"geom": {"maxiter": 100}}``.
  Constraints and scans also live there, with ORCA's 0-based atom indices, which
  match ``structure``.
- Every step in ``results.trajectory`` carries its structure, energy, and gradient,
  read from ORCA's ``orca_trj.xyz`` and its output.
- The first step is your input structure: ORCA does not move or reorient it.
- For transition states use ``CalcType.transition_state`` (transition_state.py).
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

# Coordinates are in bohr.
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="orca",
    calctype=CalcType.optimization,
    structure=structure,
    # Simple-input words after the method: convergence preset, solvent, ...
    model=Model(
        method="wB97X-D3 TightOpt", basis="def2-SVP"
    ),  # add "CPCM(water)" for solvent
    keywords={
        "maxcore": 1000,  # memory per process, MB
        # "pal": 4,  # parallel processes (needs OpenMPI)
        "geom": {
            "maxiter": 50,  # optimization steps
            # "calc_hess": True,  # exact Hessian at the first step
        },
    },
)

output = compute(prog_input)  # timeout=..., collect_files=..., executables=...

results = output.results
print("Steps:", len(results.trajectory))
print("Energies (Hartree):", results.energies)
print("Final structure (bohr):\n", results.final_structure.geometry)
# Save the trajectory, e.g. to view in a molecule viewer:
# results.save("optimization.json"); results.final_structure.save("final.xyz")
