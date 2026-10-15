"""CREST single point: GFN-xTB or GFN-FF energy, gradient, or Hessian.

Requires: ``pip install qccompute`` (it installs qccodec) and CREST 3 on PATH, the
official static build from https://github.com/crest-lab/crest/releases (conda-forge's
CREST 3.0.2 crashes in its optimizer). CREST computes GFN-xTB and GFN-FF with its
built-in tblite and gfnff libraries, so no separate xtb is needed.

How the codec talks to CREST (https://crest-lab.github.io/crest-docs/page/documentation/inputfiles.html):

- ``keywords`` are CREST's TOML input file as a dictionary: top-level settings such
  as ``threads``, and sections such as ``calculation``. CREST stops on any key it
  does not know, so typos fail loudly.
- ``model.method`` ("gfn2", "gfn1", "gfn0", "gfnff") and the structure's charge and
  multiplicity are written into every ``[[calculation.level]]``; level settings
  (solvent, electronic temperature, accuracy) go in the one level shown below. Do
  not set ``charge`` or ``uhf`` in keywords.
- ``runtype`` comes from the calculation type ("singlepoint" or "numhess").
- The Hessian is numerical and comes with harmonic frequencies and normal modes.
- For xtb's full set of options (and g-xTB), use program "xtb" (examples/xtb/).
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy, CalcType.hessian

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,  # open shells: CREST gets multiplicity - 1 unpaired electrons
)

prog_input = ProgramInput(
    program="crest",
    calctype=calctype,
    structure=structure,
    model=Model(method="gfn2"),  # or "gfn1", "gfn0", "gfnff"
    keywords={
        # Top-level CREST settings:
        "threads": 1,  # default: number of CPUs, at most 16
        # Settings for the method itself, one [[calculation.level]] table:
        "calculation": {
            "level": [
                {
                    "alpb": "water",  # implicit solvent; or "gbsa": "water"
                    # "etemp": 300.0,  # electronic temperature (K)
                    # "accuracy": 1.0,  # SCC accuracy; lower is tighter
                    # "maxscc": 300,  # maximum SCC iterations
                }
            ]
        },
    },
)

output = compute(
    prog_input,
    # timeout=600,  # seconds before CREST is killed
    # collect_files=True,  # keep every output file in output.results.files
    # executables={"crest": "/path/to/crest"},  # or QCCOMPUTE_CREST_EXECUTABLE
)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Frequencies (cm^-1):", results.freqs_wavenumber)
print("CREST version:", results.provenance.program_version)
