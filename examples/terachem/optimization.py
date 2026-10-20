"""TeraChem geometry optimization with its built-in optimizer.

Requires: ``pip install qccompute`` (it installs qccodec) and ``terachem`` on PATH.
TeraChem runs only on NVIDIA CUDA GPUs and needs a license file (from PetaChem).

Notes for TeraChem optimizations (manual: https://mtzweb.stanford.edu/terachem/):

- Only TeraChem's newer optimizer is supported: ``"new_minimizer": "yes"`` is
  required, and the codec raises an error without it.
- Optimizer settings are ordinary tc.in keywords: ``nstep``, ``min_coordinates``,
  ``min_converge_e``, ``min_converge_grms``, ``min_converge_gmax``, ...
- TeraChem does not stop on a misspelled keyword: it prints "Unrecognized or
  unnecessary keywords: These are being ignored." in its output and carries on.
  Check ``output.logs`` for that warning when a setting seems to have no effect.
- Every step in ``results.trajectory`` carries its structure, energy, and gradient,
  read from TeraChem's ``optim.xyz``/``optim_trajectory.xyz`` and its output.
- The first step is your input structure: TeraChem does not move or reorient it.
- ``constraints`` is not a tc.in keyword: it is the same dictionary the geomeTRIC
  codec takes, with 0-based atom indices and ``set`` distances in bohr. The codec
  writes TeraChem's ``$constraint_freeze``/``$constraint_set`` groups (1-based atoms,
  angstrom) for you. Scans are not supported.
- Transition-state searches are not supported by this codec; use geomeTRIC with
  TeraChem gradients (``program="geometric"``) instead.
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
    program="terachem",
    calctype=CalcType.optimization,
    structure=structure,
    model=Model(method="wb97xd3", basis="def2-svp"),
    keywords={
        "purify": "no",
        "new_minimizer": "yes",  # required
        "nstep": 50,  # maximum optimization steps
        # "min_coordinates": "cartesian",  # default: TRIC internal coordinates
        # Convergence criteria (TeraChem's defaults shown):
        # "min_converge_grms": 3.0e-4,  # RMS gradient (Hartree/bohr)
        # "min_converge_gmax": 4.5e-4,  # largest gradient component
        # "min_converge_e": 1.0e-6,  # energy change (Hartree)
        #
        # Implicit solvent (COSMO):
        # "pcm": "cosmo",
        # "epsilon": 78.39,
        #
        # Constraints: freeze atoms or internal coordinates, or set them to a value.
        # Types are "xyz", "distance", "angle", and "dihedral"; indices are 0-based.
        # "constraints": {
        #     "freeze": [{"type": "xyz", "indices": [0]}],
        #     "set": [{"type": "distance", "indices": [0, 1], "value": 1.85}],  # bohr
        # },
    },
)

output = compute(prog_input)  # timeout=..., collect_files=..., executables=...

results = output.results
print("Steps:", len(results.trajectory))
print("Energies (Hartree):", results.energies)
print("Final structure (bohr):\n", results.final_structure.geometry)
# Save the trajectory, e.g. to view in a molecule viewer:
# results.save("optimization.json"); results.final_structure.save("final.xyz")
