"""Psi4 transition-state search with optking.

Requires: ``pip install qccompute`` (it installs qccodec) and Psi4 on PATH, e.g.
``conda install -c conda-forge psi4``.

Notes for Psi4 transition-state searches:

- The codec sets ``opt_type ts`` and, unless you set it, ``full_hess_every 0``: an
  exact Hessian at the first step, which optking needs to find a saddle point.
  Setting ``opt_type`` to anything else is an error for this calculation type.
- Start close to the saddle point. This example starts from nearly planar ammonia;
  its inversion transition state is planar, with one imaginary frequency.
- Check the result with a Hessian (single_point.py with CalcType.hessian): a
  transition state has exactly one imaginary (negative) frequency.
- Point charges are not supported here in Psi4 1.11; use geomeTRIC with Psi4
  gradients for QM/MM transition states.
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
    program="psi4",
    calctype=CalcType.transition_state,
    structure=structure,
    model=Model(method="hf", basis="sto-3g"),
    keywords={
        "scf_type": "pk",
        "geom_maxiter": 50,
        # "full_hess_every": 3,  # recompute the exact Hessian every 3 steps
        # "step_type": "rs_i_rfo",  # optking's default for transition states
    },
    cmdline_args=["-n", "1", "--memory", "2GB"],
)

output = compute(prog_input)

results = output.results
print("Steps:", len(results.trajectory))
print("Transition-state energy (Hartree):", results.energies[-1])
print("Transition-state structure (bohr):\n", results.final_structure.geometry)
