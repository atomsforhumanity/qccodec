"""geomeTRIC geometry optimization driving another program's gradients.

Requires: ``pip install qccompute qccodec[geometric]`` (geomeTRIC itself) and the
gradient program on PATH. This example uses CREST for fast GFN2-xTB gradients:
https://github.com/crest-lab/crest/releases (official static build).

How it works (https://geometric.readthedocs.io/):

- geomeTRIC runs in Python and asks for a gradient at every step. Each one is a full
  ``gradient`` calculation of the program in ``subprograms``, with that program's own
  model and keywords, so any program with gradients works: CREST, xtb, Psi4, ORCA,
  TeraChem, ... Swap the ``ProgramSpec`` below for one of the commented alternatives.
- ``keywords`` are geomeTRIC's optimizer parameters (``geometric.params.OptParams``,
  the long ``geometric-optimize --help`` names). geomeTRIC ignores names it does not
  know, so check spelling against its documentation.
- ``coordsys``: "tric" (default, translation-rotation internal coordinates, good for
  several molecules), "dlc", "prim", or "cart".
- ``constraints`` use geomeTRIC's dictionary format with 0-based atom indices,
  distances in bohr, and angles in degrees. Scans are not supported.
- Every step in ``results.trajectory`` is the child's full output (energy, gradient,
  charges, ...), whichever program it is.
- QM/MM: this is the way to optimize a QM region in point charges with Psi4, since
  Psi4 1.11's own optimizer cannot (see the Psi4 child below). The charges stay fixed.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

# Coordinates are in bohr.
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

# The program that computes energies and gradients: CREST with GFN2-xTB.
gradient_program = ProgramSpec(
    program="crest",
    calctype=CalcType.gradient,
    model=Model(method="gfn2"),
    keywords={"threads": 1, "calculation": {"level": [{"alpb": "water"}]}},
)

# xtb instead (examples/xtb/single_point.py lists its keywords):
# gradient_program = ProgramSpec(
#     program="xtb",
#     calctype=CalcType.gradient,
#     model=Model(method="gfn2"),
#     keywords={"alpb": "water"},
# )

# Psi4 instead, here with QM/MM point charges [charge, [x, y, z]] in bohr:
# gradient_program = ProgramSpec(
#     program="psi4",
#     calctype=CalcType.gradient,
#     model=Model(method="hf", basis="sto-3g"),
#     keywords={
#         "scf_type": "pk",
#         "external_potentials": [[-0.2, [0.0, 0.0, 8.0]], [0.2, [2.0, 2.0, 9.0]]],
#     },
#     cmdline_args=["-n", "1", "--memory", "2GB"],
# )

# TeraChem instead (GPU); pass propagate_wfn=True to compute() below:
# gradient_program = ProgramSpec(
#     program="terachem",
#     calctype=CalcType.gradient,
#     model=Model(method="b3lyp", basis="6-31g*"),
#     keywords={"dftd": "d3"},
# )

prog_input = ProgramInput(
    program="geometric",
    calctype=CalcType.optimization,
    structure=structure,
    subprograms=[gradient_program],
    keywords={
        "coordsys": "tric",  # or "dlc", "prim", "cart"
        "maxiter": 100,  # maximum optimization steps
        "convergence_set": "gau",  # or "gau_loose", "gau_tight", "gau_verytight", ...
        # Individual criteria override the set (Hartree, Hartree/bohr, angstrom):
        # "convergence_energy": 1e-6,
        # "convergence_grms": 3e-4,
        # "convergence_gmax": 4.5e-4,
        # "convergence_drms": 1.2e-3,
        # "convergence_dmax": 1.8e-3,
        # "trust": 0.1,  # starting trust radius (angstrom)
        # "tmax": 0.3,  # largest trust radius (angstrom)
        #
        # Constraints (0-based atom indices; distances in bohr, angles in degrees):
        # "constraints": {
        #     "freeze": [{"type": "xyz", "indices": [0]}],  # fix the oxygen in place
        #     "set": [{"type": "angle", "indices": [1, 0, 2], "value": 104.5}],
        # },
    },
)

output = compute(
    prog_input,
    # propagate_wfn=True,  # TeraChem: start each step from the previous orbitals
    # timeout=3600,  # seconds before the whole optimization is stopped
    # collect_files=True,  # keep each child's output files in its results.files
    # executables={"crest": "/path/to/crest"},  # the child's executable
)

results = output.results
print("Steps:", len(results.trajectory))
print("Energies (Hartree):", results.energies)
print("Final gradient (Hartree/bohr):\n", results.trajectory[-1].results.gradient)
print("Final structure (bohr):\n", results.final_structure.geometry)
print("Gradient program:", results.trajectory[-1].input_data.program)
# Save the trajectory, e.g. to view in a molecule viewer:
# results.save("optimization.json"); results.final_structure.save("final.xyz")
