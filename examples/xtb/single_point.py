"""xtb single point: energy, gradient, or Hessian with GFN-xTB, GFN-FF, or g-xTB.

Requires: ``pip install qccompute`` (it installs qccodec) and xtb on PATH, e.g.
``conda install -c conda-forge xtb``. For g-xTB (``method="gxtb"``) use the xtb build
from https://github.com/grimme-lab/g-xtb instead; it also runs every other method.
GFN0-xTB reads its parameter file from the directory named by ``XTBPATH``.

How the codec talks to xtb (https://xtb-docs.readthedocs.io/):

- ``model.method`` picks the Hamiltonian: "gfn0", "gfn1", "gfn2", "gfnff", or "gxtb".
  There is no basis set.
- Charge and multiplicity come from ``structure`` (passed as ``--chrg`` and ``--uhf``).
- ``keywords`` become xtb command-line flags (``xtb --help``): ``{"alpb": "water"}``
  gives ``--alpb water``, ``True`` gives a bare flag, and ``False``/``None`` drop it.
  Flags the codec sets itself (``chrg``, ``uhf``, ``gfn``, ``sp``, ``hess``, ...)
  are rejected.
- xtb only warns about options it does not know and runs without them; the codec
  turns that warning into a failure, so a typo never silently changes the method.
- Detailed settings (``$scc``, ``$wall``, ``$constrain``, ...) go in an xcontrol file
  (``man 7 xcontrol``); see the commented example below. Atom indices in it are
  1-based, as xtb expects.
- g-xTB has no ALPB/GBSA parameters (xtb stops with "No ALPB/GBSA parameters found")
  and xtb 6.7.1 silently ignores ``cpcmx`` with g-xTB, so run g-xTB in the gas phase.
- xtb's Hessian is numerical (finite differences of analytic gradients) and comes
  with harmonic frequencies and normal modes. Do not pass ``cma``: it moves the
  molecule to its principal axes, so results would not match ``structure``.
"""

from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, Structure

calctype = CalcType.gradient  # or CalcType.energy, CalcType.hessian

# Coordinates are in bohr (Structure.from_xyz / Structure.open read angstrom files).
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
    charge=0,
    multiplicity=1,  # open shells: xtb gets multiplicity - 1 unpaired electrons
)

prog_input = ProgramInput(
    program="xtb",
    calctype=calctype,
    structure=structure,
    model=Model(method="gfn2"),  # or "gfn1", "gfn0", "gfnff", "gxtb" (g-xTB build)
    keywords={
        # Any xtb command-line flag, without the leading dashes:
        "parallel": 1,  # threads
        "acc": 1.0,  # SCC accuracy; lower is tighter (0.0001 to 1000)
        "iterations": 250,  # maximum SCC iterations
        # "etemp": 300,  # electronic temperature (K); raise it for difficult SCCs
        #
        # Implicit solvent; pick one model (GFN1/GFN2/GFN-FF only, not g-xTB):
        "alpb": "water",  # ALPB; ["water", "bar1M"] also sets the reference state
        # "gbsa": "h2o",  # GBSA (its own solvent names)
        # "cpcmx": "water",  # CPCM-X
        #
        # Extra output, found in the logs or with collect_files=True:
        # "molden": True,  # orbitals in molden.input
        # "wbo": True,  # Wiberg bond orders in wbo
        # "json": True,  # xtbout.json
        #
        # Detailed settings from an xcontrol file (pass it in `files` below):
        # "input": "xcontrol",
    },
    # files={"xcontrol": "$scc\n   temp=500\n$end\n"},
)

output = compute(
    prog_input,
    # timeout=600,  # seconds before xtb is killed
    # collect_files=True,  # keep every output file in output.results.files
    # executables={"xtb": "/path/to/g-xtb/bin/xtb"},  # or QCCOMPUTE_XTB_EXECUTABLE
)

results = output.results
print("Energy (Hartree):", results.energy)
if results.gradient is not None:
    print("Gradient (Hartree/bohr):\n", results.gradient)
if results.hessian is not None:
    print("Frequencies (cm^-1):", results.freqs_wavenumber)
    print("Normal modes (modes, atoms, xyz):", results.normal_modes_cartesian.shape)
print("Partial charges:", results.atomic_charges)
print("xtb version:", results.provenance.program_version)
