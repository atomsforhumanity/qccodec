"""TeraChem frontend server single point: energy or gradient.

Requires: ``pip install "qccompute" "qccodec[tcpb]"`` (the tcpb client) and a running
TeraChem frontend; this machine needs no TeraChem install. TeraChem itself runs on a
machine with an NVIDIA CUDA GPU and a TeraChem license. The codec refuses tcpb 0.16.0,
which predates the current qcdata models; it needs a newer tcpb release.

The frontend is a TeraChem protocol-buffer server (``terachem -s 11111``) paired with
an HTTP file server (port 8080 by default) over the server's scratch directory, as in
the Docker images at https://hub.docker.com/r/mtzgroup/terachem. Compared with
``terachem-pbs`` (see examples/terachem-pbs/), the file server lets the client upload
starting orbitals and download TeraChem's output and scratch files.

How the codec talks to the server (tcpb docs: https://mtzgroup.github.io/tcpb-client/):

- The client connects to TCPB_HOST:TCPB_PORT (default 127.0.0.1:11111) and the file
  server at TCPB_FRONTEND_HOST:TCPB_FRONTEND_PORT (default 127.0.0.1:8080). Set these
  environment variables before running, or in the script before ``compute`` as below.
- ``model.method`` must be one of the methods the server protocol knows: "hf",
  "b3lyp", "pbe0", "wb97x", "wb97xd3", "camb3lyp", "blyp", "pbe", "svwn", ...
  Do not prefix "u" for unrestricted; set ``"restricted": False`` (and
  ``"closed_shell": False`` for open shells) in ``keywords`` instead.
- All other ``keywords`` are sent as tc.in options with ``str(value)``, so write
  TeraChem's own values: "yes"/"no", not True/False.
- Starting orbitals: put TeraChem's ``c0`` file (or ``ca0`` and ``cb0``) as bytes in
  ``ProgramInput.files``; the client uploads it and sets the ``guess`` keyword.
- Only energies and gradients are supported.
"""

import os

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy

# Where the TeraChem frontend listens; set before compute() imports the tcpb client.
os.environ.setdefault("TCPB_HOST", "127.0.0.1")
os.environ.setdefault("TCPB_PORT", "11111")
os.environ.setdefault("TCPB_FRONTEND_HOST", "127.0.0.1")
os.environ.setdefault("TCPB_FRONTEND_PORT", "8080")

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="terachem-fe",
    calctype=calctype,
    structure=structure,
    model=Model(method="wb97xd3", basis="def2-svp"),  # e.g. "b3lyp"/"6-31gss"
    keywords={
        # tc.in options, as strings TeraChem understands:
        "purify": "no",
        "convthre": "1e-6",  # SCF convergence
        "maxit": "100",  # SCF iterations
        # "pcm": "cosmo",  # implicit solvent
        # "epsilon": "78.39",
        #
        # Open shells (also set structure multiplicity):
        # "restricted": False,  # unrestricted (UHF/UKS)
        # "closed_shell": False,
    },
    # Starting orbitals from an earlier TeraChem run:
    # files={"c0": open("c0", "rb").read()},
)

output = compute(prog_input)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
print("Mulliken charges:", results.extras.get("charges"))
print("Files from the server:", sorted(results.files))
