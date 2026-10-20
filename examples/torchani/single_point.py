"""TorchANI single point: energy, gradient, or Hessian from an ANI potential.

Requires: ``pip install qccompute torchani``. TorchANI is a Python library, so nothing
has to be on PATH: the model runs inside this Python process, on the GPU when PyTorch
sees one. Parameters download on first use; later calculations in the same process
reuse the loaded model.

How the codec talks to TorchANI (https://github.com/aiqm/torchani):

- ``model.method`` names a model in ``torchani.models``: "ANI2x" (ωB97X/6-31G(d);
  H C N O F S Cl), "ANI1x", "ANI1ccx" (CCSD(T)/CBS; H C N O), "ANI2xr", "ANI2dr",
  "ANImbis", "SnnANI2xr". Case and dashes are ignored. There is no basis set.
- ANI describes neutral closed-shell molecules only; the codec refuses anything else.
- ``dtype`` is "float64" by default, for smooth gradients and Hessians; "float32" is
  faster on GPUs.
- ANI models are ensembles; ``model_index`` picks one member instead of the average.
- Hessians come from automatic differentiation. Energies are in Hartree.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy, CalcType.hessian

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,  # ANI: neutral closed-shell only
    multiplicity=1,
)

prog_input = ProgramInput(
    program="torchani",
    calctype=calctype,
    structure=structure,
    model=Model(method="ANI2x"),  # or "ANI1ccx", "ANI1x", ...
    keywords={
        "device": "cpu",  # or "cuda"; leave out to use a GPU when present
        "dtype": "float64",  # or "float32"
        # "model_index": 0,  # one member of the ensemble instead of its average
    },
)

output = compute(prog_input)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Hessian (Hartree/bohr^2):", results.hessian.shape)
print("torchani version:", results.provenance.program_version)
