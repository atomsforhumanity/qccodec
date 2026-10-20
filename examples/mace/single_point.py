"""MACE single point: energy, gradient, or Hessian from a MACE potential.

Requires: ``pip install qccompute mace-torch``, in an environment without
``fairchem-core`` (the two need different versions of e3nn). MACE is a Python library,
so nothing has to be on PATH: the model runs inside this Python process, on the GPU
when PyTorch sees one. Models download on first use; later calculations in the same
process reuse the loaded model.

How the codec talks to MACE (https://github.com/ACEsuit/mace):

- ``model.method`` is a foundation-model family, optionally with the model argument
  MACE's loader takes: "mace-off" or "mace-off-small"/"-medium"/"-large" (MACE-OFF23
  organic force field), "mace-mp" or "mace-mp-medium-mpa-0" etc. (materials), or
  "mace-omol" (MACE-OMOL-0, trained on OMol25). A path to a ``.model`` file runs your
  own model. There is no basis set.
- Only MACE-OMOL reads charge and multiplicity; MACE-OFF and MACE-MP describe neutral
  closed-shell molecules, and the codec refuses anything else for them.
- ``default_dtype`` is "float64" or "float32" (MACE's default otherwise: float64 for
  MACE-OFF and MACE-OMOL, float32 for MACE-MP).
- Hessians come from automatic differentiation.
- Energies come back in Hartree and gradients in Hartree/bohr, converted from MACE's
  eV and angstrom.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy, CalcType.hessian

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,  # other charges and multiplicities need "mace-omol"
    multiplicity=1,
)

prog_input = ProgramInput(
    program="mace",
    calctype=calctype,
    structure=structure,
    model=Model(method="mace-off-small"),  # or "mace-off", "mace-omol", "mace-mp", ...
    keywords={
        "device": "cpu",  # or "cuda"; leave out to use a GPU when present
        # "default_dtype": "float32",  # faster, less precise
    },
)

output = compute(prog_input)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Hessian (Hartree/bohr^2):", results.hessian.shape)
print("mace-torch version:", results.provenance.program_version)
