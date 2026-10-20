"""FAIR-Chem single point: energy, gradient, or Hessian from UMA or eSEN.

Requires: ``pip install qccompute fairchem-core``, and access to the gated weights:
request it at https://huggingface.co/facebook/UMA (or facebook/OMol25 for eSEN), then
run ``hf auth login``. FAIR-Chem is a Python library, so nothing has to be on PATH: the
model runs inside this Python process, on the GPU when PyTorch sees one. Later
calculations in the same process reuse the loaded model.

How the codec talks to FAIR-Chem (https://fair-chem.github.io/):

- ``model.method`` is a pretrained model ("uma-s-1p2p1", "uma-s-1p1", "uma-m-1p1",
  "esen-sm-conserving-all-omol", ...) or a path to a checkpoint. There is no basis
  set.
- ``task`` picks the UMA head: "omol" (molecules, the default) reads charge and
  multiplicity from ``structure``; "omat", "oc20", "odac", and "omc" are for
  materials and catalysis.
- ``inference_settings`` defaults to the uncompiled model, ready in seconds for any
  molecule. fairchem's own "default" preset compiles the model, which takes tens of
  seconds in every process and again for each new molecule size: worth it only for
  long runs on one system.
- Hessians come from automatic differentiation.
- Energies come back in Hartree and gradients in Hartree/bohr, converted from
  FAIR-Chem's eV and angstrom.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy, CalcType.hessian

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="fairchem",
    calctype=calctype,
    structure=structure,
    model=Model(method="uma-s-1p2p1"),  # or "uma-m-1p1", "esen-sm-conserving-all-omol"
    keywords={
        "task": "omol",
        "device": "cpu",  # or "cuda"; leave out to use a GPU when present
        # A fairchem preset by name, or InferenceSettings fields:
        # "inference_settings": "turbo",
        # "inference_settings": {"compile": False, "base_precision_dtype": "float64"},
    },
)

output = compute(prog_input)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Hessian (Hartree/bohr^2):", results.hessian.shape)
print("fairchem-core version:", results.provenance.program_version)
