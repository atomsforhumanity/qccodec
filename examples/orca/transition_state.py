"""ORCA transition-state search (OptTS).

Requires: ``pip install qccompute`` (it installs qccodec) and ORCA on PATH, ahead of
``/usr/bin/orca`` (the GNOME screen reader on many Linux desktops); or set
QCCOMPUTE_ORCA_EXECUTABLE. See single_point.py for installing and running ORCA.

Notes for ORCA transition-state searches (manual:
https://www.faccts.de/docs/orca/6.0/manual/, "Transition State Searches"):

- The codec adds ``! OptTS``. ORCA's eigenvector-following needs a good starting
  Hessian: ``{"geom": {"calc_hess": True}}`` computes the exact one at the first step
  (ORCA otherwise starts from a model Hessian and can miss the saddle point).
  ``recalc_hess`` recomputes it every n steps for difficult cases.
- Start close to the saddle point. This example starts from nearly planar ammonia;
  its inversion transition state is planar, with one imaginary frequency.
- Check the result with a Hessian (single_point.py with CalcType.hessian): a
  transition state has exactly one imaginary (negative) frequency.
- For a scan or NEB to find a starting guess, see ORCA's manual; the codec runs only
  the OptTS step itself.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

# Nearly planar ammonia, coordinates in bohr.
structure = Structure(
    symbols=["N", "H", "H", "H"],
    geometry=[
        [0.0, 0.0, 0.25],
        [1.77, 0.0, -0.10],
        [-0.885, 1.533, -0.10],
        [-0.885, -1.533, -0.10],
    ],
)

prog_input = ProgramInput(
    program="orca",
    calctype=CalcType.transition_state,
    structure=structure,
    model=Model(method="hf", basis="sto-3g"),
    keywords={
        "maxcore": 1000,  # memory per process, MB
        # "pal": 4,  # parallel processes (needs OpenMPI)
        "geom": {
            "calc_hess": True,  # exact Hessian at the first step
            "maxiter": 50,
            # "recalc_hess": 5,  # recompute the exact Hessian every 5 steps
        },
    },
)

output = compute(prog_input)  # timeout=..., collect_files=..., executables=...

results = output.results
print("Steps:", len(results.trajectory))
print("Transition-state energy (Hartree):", results.energies[-1])
print("Transition-state structure (bohr):\n", results.final_structure.geometry)
