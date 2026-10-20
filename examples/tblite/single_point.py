"""tblite single point: GFN2-xTB, GFN1-xTB, or IPEA1-xTB energy or gradient.

Requires: ``pip install qccompute tblite`` (or ``conda install -c conda-forge
tblite-python``). tblite is a Python library, so nothing has to be on PATH: the
calculation runs inside this Python process.

How the codec talks to tblite (https://tblite.readthedocs.io/):

- ``model.method`` picks the parametrization: "gfn2", "gfn1", or "ipea1". There is no
  basis set.
- Charge and multiplicity come from ``structure``; tblite gets multiplicity - 1
  unpaired electrons.
- ``keywords`` use tblite's own names. Settings go to ``Calculator.set`` and
  interactions (implicit solvent, electric field, spin polarization) to
  ``Calculator.add``; anything else is rejected, so a typo fails loudly.
- ``temperature`` is the electronic temperature in Hartree, not kelvin (multiply
  kelvin by 3.166808578545117e-6).
- tblite has no Hessian. Use ``program="xtb"`` for GFN2-xTB Hessians, and geomeTRIC
  with tblite gradients for optimizations (examples/geometric/).
- An SCF that does not converge raises with tblite's message.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="tblite",
    calctype=calctype,
    structure=structure,
    model=Model(method="gfn2"),  # or "gfn1", "ipea1"
    keywords={
        # Calculator settings:
        "accuracy": 1.0,  # SCC accuracy; lower is tighter
        "max-iter": 250,  # maximum SCC iterations
        # "temperature": 300 * 3.166808578545117e-6,  # electronic temperature (Hartree)
        # "verbosity": 0,  # silence the SCC printout
        #
        # Implicit solvent; pick one model. A solvent name, or [solvent, state]
        # with state "gsolv" (default), "bar1mol", or "reference":
        "alpb-solvation": "water",
        # "gbsa-solvation": "water",
        # "ddX-solvation": ["water", "cosmo"],
        #
        # "spin-polarization": 1.0,  # spin-polarized xTB for open shells
        # "electric-field": [0.0, 0.0, 0.001],  # uniform field (atomic units)
    },
)

output = compute(prog_input)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
print("Partial charges:", results.atomic_charges)
print("Dipole (e*bohr):", results.scf_dipole_moment)
print("tblite version:", results.provenance.program_version)
