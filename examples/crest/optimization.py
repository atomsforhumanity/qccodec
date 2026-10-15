"""CREST geometry optimization with its built-in ANCopt optimizer.

Requires: ``pip install qccompute`` (it installs qccodec) and CREST 3 on PATH, the
official static build from https://github.com/crest-lab/crest/releases (conda-forge's
CREST 3.0.2 crashes in its optimizer). No separate xtb is needed.

Notes for CREST optimizations (https://crest-lab.github.io/crest-docs/):

- Optimizer settings live in the ``calculation`` section of CREST's TOML input:
  ``optlev`` ("crude", "sloppy", "loose", "normal", "tight", "vtight", "extreme")
  and ``maxcycle``. Method settings (solvent, electronic temperature, ...) go in its
  ``level`` table, as in single_point.py. CREST stops on keys it does not know.
- Each step in ``results.trajectory`` has its structure and energy; the last step
  also has the gradient.
- For transition states, or an optimizer with more control, drive CREST gradients
  with geomeTRIC (examples/geometric/).
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
    program="crest",
    calctype=CalcType.optimization,
    structure=structure,
    model=Model(method="gfn2"),  # or "gfn1", "gfnff"
    keywords={
        "threads": 1,  # default: number of CPUs, at most 16
        "calculation": {
            # ANCopt settings:
            "optlev": "tight",  # convergence level
            "maxcycle": 100,  # maximum optimization steps
            # The method's settings:
            "level": [
                {
                    "alpb": "water",  # implicit solvent; or "gbsa": "water"
                    # "etemp": 300.0,  # electronic temperature (K)
                }
            ],
        },
    },
)

output = compute(
    prog_input,
    # timeout=600,  # seconds before CREST is killed
    # collect_files=True,  # keep every output file (crestopt.xyz, ...) in results.files
    # executables={"crest": "/path/to/crest"},  # or QCCOMPUTE_CREST_EXECUTABLE
)

results = output.results
print("Steps:", len(results.trajectory))
print("Energies (Hartree):", results.energies)
print("Final gradient (Hartree/bohr):\n", results.trajectory[-1].results.gradient)
print("Final structure (bohr):\n", results.final_structure.geometry)
# Save the trajectory, e.g. to view in a molecule viewer:
# results.save("optimization.json"); results.final_structure.save("final.xyz")
