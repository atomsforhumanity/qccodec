"""Psi4 geometry optimization with its built-in optimizer, optking.

Requires: ``pip install qccompute`` (it installs qccodec) and Psi4 on PATH, e.g.
``conda install -c conda-forge psi4``.

Notes for Psi4 optimizations:

- optking options are ordinary Psi4 options: ``g_convergence``, ``geom_maxiter``,
  ``step_type``, ``opt_coordinates``, ``full_hess_every``, ... (Psi4 manual, "Geometry
  Optimization").
- Every step in ``results.frames`` carries its structure, energy, and gradient;
  the last step also carries the final charges, orbitals, and Psi4 variables.
- The first step is exactly your input structure: Psi4 does not move or reorient it.
- Point charges (``external_potentials``) do not work in Psi4 1.11 optimizations
  (it never leaves the starting geometry), so the codec rejects them. For QM/MM
  optimizations use geomeTRIC with Psi4 gradients; see examples/geometric/.
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
    program="psi4",
    calctype=CalcType.optimization,
    structure=structure,
    model=Model(method="wb97x-d", basis="def2-svp"),
    keywords={
        # optking settings:
        "g_convergence": "gau",  # or "qchem", "gau_tight", "gau_verytight", ...
        "geom_maxiter": 50,
        # "opt_coordinates": "both",  # redundant internals plus Cartesians
        # "full_hess_every": 0,  # exact Hessian at the first step
        #
        # Implicit solvent (ddX; needs pyddx next to Psi4):
        # "ddx": True,
        # "ddx_model": "pcm",
        # "ddx_solvent": "water",
        # "symmetry": "c1",  # keep an asymmetric path open from a symmetric start
    },
    cmdline_args=["-n", "1", "--memory", "2GB"],
)

output = compute(prog_input)  # timeout=..., collect_files=..., executables=...

results = output.results
print("Steps:", len(results.frames))
print("Energies (Hartree):", results.energies)
print("Final structure (bohr):\n", results.final_structure.geometry)
# Save the trajectory, e.g. to view in a molecule viewer:
# results.save("optimization.json"); results.final_structure.save("final.xyz")
