"""xtb geometry optimization with its built-in optimizer, ANCopt.

Requires: ``pip install qccompute`` (it installs qccodec) and xtb on PATH, e.g.
``conda install -c conda-forge xtb`` (or the g-xTB build of xtb for ``method="gxtb"``,
https://github.com/grimme-lab/g-xtb).

Notes for xtb optimizations (https://xtb-docs.readthedocs.io/en/latest/optimization.html):

- ``{"opt": "tight"}`` sets the convergence level: "crude", "sloppy", "loose",
  "normal" (default), "tight", "verytight", or "extreme". ``cycles`` caps the steps.
- xtb exits normally when it runs out of cycles; the codec reports that as a failure
  ("failed to converge geometry optimization").
- Each step in ``results.frames`` has its structure and energy but no gradient:
  xtb's log records only those.
- Constraints go in an xcontrol file (``$fix``, ``$constrain``, ``$wall``; see
  https://xtb-docs.readthedocs.io/en/latest/xcontrol.html). Its atom indices are
  1-based, unlike the 0-based indices used elsewhere in qcdata.
- Other keywords become xtb command-line flags, as in single_point.py: solvent,
  accuracy, electronic temperature, threads.
- For transition states, or an optimizer that records gradients at every step, drive
  xtb gradients with geomeTRIC (examples/geometric/).
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
    program="xtb",
    calctype=CalcType.optimization,
    structure=structure,
    model=Model(method="gfn2"),  # or "gfn1", "gfnff", "gxtb" (g-xTB build)
    keywords={
        # ANCopt settings:
        "opt": "tight",  # convergence level
        "cycles": 100,  # maximum optimization steps
        #
        "parallel": 1,  # threads
        "alpb": "water",  # implicit solvent; or "gbsa": "h2o"
        # "etemp": 300,  # electronic temperature (K)
        #
        # Constraints from an xcontrol file (pass it in `files` below):
        # "input": "xcontrol",
    },
    # Keep the first O-H bond at its starting length and freeze the oxygen atom.
    # xcontrol atom indices are 1-based: atom 1 is structure.symbols[0].
    # files={
    #     "xcontrol": (
    #         "$constrain\n"
    #         "   force constant=1.0\n"
    #         "   distance: 1, 2, auto\n"
    #         "$fix\n"
    #         "   atoms: 1\n"
    #         "$end\n"
    #     )
    # },
)

output = compute(
    prog_input,
    # timeout=600,  # seconds before xtb is killed
    # collect_files=True,  # keep every output file (xtbopt.xyz, ...) in results.files
    # executables={"xtb": "/path/to/xtb"},  # or QCCOMPUTE_XTB_EXECUTABLE
)

results = output.results
print("Steps:", len(results.frames))
print("Energies (Hartree):", results.energies)
print("Final structure (bohr):\n", results.final_structure.geometry)
# Save the trajectory, e.g. to view in a molecule viewer:
# results.save("optimization.json"); results.final_structure.save("final.xyz")
