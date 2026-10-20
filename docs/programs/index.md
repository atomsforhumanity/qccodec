# Programs

Each program page explains how qccodec maps a `ProgramInput` onto that program, then
gives a complete example script for every calculation type it supports. The scripts
run as written (they are tested), show every common option, and document the
program's quirks for that kind of calculation.

## Conventions shared by every program

- **Keywords are the program's own.** `ProgramInput.keywords` pass through to the
  program in its native form, so the program's documentation applies. The program
  pages say where each codec puts them (an input file, command-line flags, a TOML
  document, ...).
- **Structured data stays structured.** The method and basis go in `model`; the
  geometry, charge, and multiplicity go in `structure`. Codecs reject keywords that
  duplicate them.
- **Atom indices are 0-based**, matching `structure`. Codecs convert them for
  programs that count from 1; files you supply verbatim (such as an xtb `xcontrol`)
  use the program's own convention.
- **Results describe your structure exactly.** Codecs stop programs from moving,
  reorienting, or symmetrizing the molecule, so gradients, Hessians, and trajectories
  line up with the input.
- **Coordinates are in bohr** and energies in Hartree, as everywhere in qcdata.
- **Errors carry the program's explanation.** When a program fails, the error
  message says why, and any partial results are attached.

## Finding the program

qccompute runs a program by name from `PATH`. To use a different build, or one that
isn't on `PATH`, pass `compute(..., executables={"orca": "/path/to/orca"})` or set
`QCCOMPUTE_ORCA_EXECUTABLE` (the program name in capitals).

## Programs that are Python libraries

tblite and the machine-learned potentials (AIMNet2, FAIR-Chem, MACE, Orb, TorchANI)
are Python libraries, not programs: qccompute calls them inside the calling Python
process, so install the library next to qccompute instead of putting anything on
`PATH`. They share a few conventions:

- **Models load once per process.** The first calculation downloads and loads the
  weights; later ones with the same model and device reuse them, so an optimization
  or a reaction path pays that cost once. Loaded models hold memory (GPU memory on a
  GPU): the `QCCODEC_MAX_MODELS` most recently used stay loaded (2 by default; raise it
  on large GPUs, or set 0 to load every time), and `qccodec.clear_model_cache()`
  unloads them all.
- **`device` picks the hardware**: `"cpu"`, `"cuda"`, or `"cuda:1"`. Without it, a GPU
  is used when PyTorch sees one.
- **Keywords are checked.** Each codec accepts a short list of settings and rejects
  anything else, so a misspelled option fails instead of being ignored.
- **Charge and spin** reach models that understand them (AIMNet2-NSE, FAIR-Chem's
  `omol` head, MACE-OMOL, Orb's OMol25 models). Codecs refuse ions and open shells for
  models trained only on neutral closed-shell molecules.
- **Optimizations and transition states** run under [geomeTRIC](geometric.md), with the
  library's gradients as the child calculation.
- Some libraries pin incompatible dependencies; `fairchem-core` cannot share an
  environment with `mace-torch` or `orb-models`.

## xtb, tblite, or CREST?

All three run the GFN-xTB methods and give the same GFN2-xTB energies and gradients.
Use `program="tblite"` for fast in-process energies and gradients (no program to
install, but no Hessians or optimizer of its own). Use `program="xtb"` for single
points, Hessians, and optimizations with xtb's full set of options, for point charges,
and for g-xTB. Use `program="crest"` for conformer searches and CREST's other workflows.

## Optimizations: the program's optimizer or geomeTRIC?

Programs with their own optimizer (CREST, ORCA, Psi4, TeraChem, xtb) run
`optimization` themselves. [geomeTRIC](geometric.md) can drive any program's gradients
instead, which is useful for a consistent optimizer across programs, constraints, and
QM/MM optimizations with Psi4 point charges.
