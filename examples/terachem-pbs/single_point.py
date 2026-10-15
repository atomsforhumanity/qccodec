"""TeraChem protocol-buffer server single point: energy or gradient.

Requires: ``pip install "qccompute" "qccodec[tcpb]"`` (the tcpb client) and a running
TeraChem server; this machine needs no TeraChem install. TeraChem itself runs on a
machine with an NVIDIA CUDA GPU and a TeraChem license. The codec refuses tcpb 0.16.0,
which predates the current qcdata models; it needs a newer tcpb release.

Start the server on the GPU machine with ``terachem -s 11111`` (the port), or with the
Docker image at https://hub.docker.com/r/mtzgroup/terachem. The server keeps
TeraChem and its GPU state loaded between calculations, so repeated single points
(dynamics, scans, optimizers driving TeraChem) skip TeraChem's start-up cost.

How the codec talks to the server (tcpb docs: https://mtzgroup.github.io/tcpb-client/):

- The client connects to TCPB_HOST:TCPB_PORT (default 127.0.0.1:11111). Set these
  environment variables before running, or in the script before ``compute`` as below.
  An SSH tunnel (``ssh -L 11111:localhost:11111 gpu-host``) reaches a remote server.
- ``model.method`` must be one of the methods the server protocol knows: "hf",
  "b3lyp", "pbe0", "wb97x", "wb97xd3", "camb3lyp", "blyp", "pbe", "svwn", ...
  (``tcpb.terachem_server_pb2.JobInput.MethodType``). Do not prefix "u" for
  unrestricted; set ``"restricted": False`` (and ``"closed_shell": False`` for open
  shells) in ``keywords`` instead.
- All other ``keywords`` are sent as tc.in options with ``str(value)``, so write
  TeraChem's own values: "yes"/"no", not True/False.
- Only energies and gradients are supported; for Hessians and optimizations run
  TeraChem directly (``program="terachem"``) or drive this server with geomeTRIC.
"""

import os

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy

# Where the TeraChem server listens; set before compute() imports the tcpb client.
os.environ.setdefault("TCPB_HOST", "127.0.0.1")
os.environ.setdefault("TCPB_PORT", "11111")

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,
)

prog_input = ProgramInput(
    program="terachem-pbs",
    calctype=calctype,
    structure=structure,
    model=Model(method="hf", basis="sto-3g"),  # e.g. "wb97x"/"6-31gss"
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
        #
        # Extra results from the server:
        # "bond_order": True,  # Meyer bond orders in results.extras
        # "mo_output": True,  # MO coefficients and AO data in results.extras
    },
)

output = compute(prog_input)  # timeout=... does not apply: the call runs in Python

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
print("Mulliken charges:", results.extras.get("charges"))
