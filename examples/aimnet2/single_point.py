"""AIMNet2 single point: energy, gradient, or Hessian from a machine-learned potential.

Requires: ``pip install qccompute aimnet``. AIMNet2 is a Python library, so nothing has
to be on PATH: the model runs inside this Python process, on the GPU when PyTorch sees
one. Weights download on first use; later calculations in the same process reuse the
loaded model, so an optimization or path pays the loading cost once.

How the codec talks to AIMNet2 (https://github.com/isayevlab/aimnetcentral):

- ``model.method`` is a model from aimnet's registry: "aimnet2" (ωB97M-D3, 14
  elements, any charge), "aimnet2-rxn" (reactions, H C N O), "aimnet2-nse" (radicals
  and other open shells), "aimnet2-2025" (B97-3c), ...; or a path to a model file.
  There is no basis set.
- Charge comes from ``structure``. Only "aimnet2-nse" takes a multiplicity; the other
  models are closed-shell and the codec refuses open shells for them.
- aimnet refuses elements and charges a model was not trained on; set
  ``validate_species`` to False to override (results are then undefined).
- Hessians come from automatic differentiation, not finite differences.
- Energies come back in Hartree and gradients in Hartree/bohr, converted from
  AIMNet2's eV and angstrom.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy, CalcType.hessian

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,  # other than 1 needs "aimnet2-nse"
)

prog_input = ProgramInput(
    program="aimnet2",
    calctype=calctype,
    structure=structure,
    model=Model(method="aimnet2"),  # or "aimnet2-rxn", "aimnet2-nse", "aimnet2-2025"
    keywords={
        "device": "cpu",  # or "cuda", "cuda:1"; leave out to use a GPU when present
        # "validate_species": False,  # allow elements/charges outside the training set
    },
)

output = compute(prog_input)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Hessian (Hartree/bohr^2):", results.hessian.shape)
print("Partial charges:", results.atomic_charges)
print("aimnet version:", results.provenance.program_version)
