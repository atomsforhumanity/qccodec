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

## xtb or CREST?

Both run the GFN-xTB methods and give the same GFN2-xTB energies and gradients. Use
`program="xtb"` for single points, Hessians, and optimizations with xtb's full set of
options, and for g-xTB. Use `program="crest"` for conformer searches and CREST's other
workflows.

## Optimizations: the program's optimizer or geomeTRIC?

Programs with their own optimizer (CREST, ORCA, Psi4, TeraChem, xtb) run
`optimization` themselves. [geomeTRIC](geometric.md) can drive any program's gradients
instead, which is useful for a consistent optimizer across programs, constraints, and
QM/MM optimizations with Psi4 point charges.
