"""Orb single point: energy or gradient from an Orb potential.

Requires: ``pip install qccompute orb-models``, in an environment without
``fairchem-core``. Orb is a Python library, so nothing has to be on PATH: the model
runs inside this Python process, on the GPU when PyTorch sees one. Weights download on
first use; later calculations in the same process reuse the loaded model.

How the codec talks to Orb (https://github.com/orbital-materials/orb-models):

- ``model.method`` is a name from ``orb_models.forcefield.pretrained
  .ORB_PRETRAINED_MODELS``: "orb-v3-conservative-omol" (OMol25, conservative forces;
  the usual choice for molecules), "orb-v3-direct-omol" (faster, non-conservative
  forces), "orbmol-v2", or the materials models. There is no basis set.
- OMol25 models read charge and multiplicity from ``structure``; the others describe
  neutral closed-shell systems only.
- ``precision`` is "float32-high" by default, or "float32-highest" or "float64".
- Orb has no Hessian. Use geomeTRIC with Orb gradients for optimizations and
  transition states (examples/geometric/).
- Energies come back in Hartree and gradients in Hartree/bohr, converted from Orb's
  eV and angstrom.
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
    program="orb",
    calctype=calctype,
    structure=structure,
    model=Model(method="orb-v3-conservative-omol"),  # or "orb-v3-direct-omol", ...
    keywords={
        "device": "cpu",  # or "cuda"; leave out to use a GPU when present
        # "precision": "float64",  # or "float32-highest"
    },
)

output = compute(prog_input)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
print("orb-models version:", results.provenance.program_version)
