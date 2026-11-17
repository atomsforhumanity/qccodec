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
- **Every keyword must take effect.** A keyword the program does not know fails at
  once, before anything runs, naming the closest one (kind `input`, code
  `unknown_key`): the codec checks it against the program's settings where they can be
  listed (xtb, tblite, geomeTRIC, Sella, CREST's tables, the machine-learned
  potentials), and where they cannot (ORCA's words and blocks, Psi4's options, DFTB+'s
  input) the program refuses it itself before calculating and the codec reports that
  the same way. TeraChem's keywords are passed through unchecked.
- **Structured data stays structured.** The method and basis go in `model`; the
  geometry, charge, and multiplicity go in `structure`. Codecs reject keywords that
  duplicate them.
- **Atom indices are 0-based**, matching `structure`. Codecs convert them for
  programs that count from 1; files you supply verbatim (such as an xtb `xcontrol`)
  use the program's own convention.
- **Results describe your structure exactly.** Codecs stop programs from moving,
  reorienting, or symmetrizing the molecule, so gradients, Hessians, and trajectories
  line up with the input.
- **Normal modes mean the same thing everywhere:** unit-length Cartesian
  displacements (not mass-weighted) in the input's frame and atom order, one per
  frequency, whatever form the program writes them in. Their sign is the program's.
  `qccodec.testing.check_normal_modes` verifies a codec's modes against its Hessian.
- **Coordinates are in bohr** and energies in Hartree, as everywhere in qcdata.
- **Errors carry the program's explanation** and a kind (see [Errors](#errors)). When
  a program fails, the error message says why, and any partial results are attached.

## Finding the program

qccodec runs a program by name from `PATH`, always by its absolute path (recorded in
`execution.extras["executable"]`). To use a different build, or one that isn't on
`PATH`, set `QCCODEC_ORCA_EXECUTABLE` (the program name in capitals), or register
`qccodec.CodecProgram(codec, executable="/path/to/orca")` in your process.

## Threads

`keywords={"threads": n}` asks for `n` CPU threads, for every program; the codec
writes it in the program's own spelling (ORCA's `pal`, Psi4's `-n`, xtb's `parallel`,
CREST's `threads`) or sets `OMP_NUM_THREADS`. Giving both `threads` and the program's
own setting is an error. Without a request, a program plans what suits it, and the
runner grants at most that:

| program | plans without a request |
|---|---|
| Psi4, CREST | all the threads the runner has |
| geomeTRIC, Sella | what their gradient subprogram plans |
| ORCA | one (parallel ORCA needs OpenMPI's `mpirun`, so `pal` is asked for, never assumed) |
| xtb, tblite, DFTB+, TeraChem (its work is on its GPUs), the machine-learned potentials | one |

## Versions and numerics identities

Each program reports two strings, both computed once per process:

- **`version()`** is the program's own version: what its executable answers to
  `--version`, or the installed version of a Python library. It is the version
  provenance records (`results.provenance.program_version`).
- **`numerics(spec)`** identifies the code that produces the numbers. A store records
  it for every run (`identities`, `{program: numerics}`) and reuses a run only when the
  programs that would run now report the same strings. For an executable it is the
  version followed by the sha256 of the executable file, so two builds that print the
  same version (upstream xtb and the g-xTB build) stay apart, and so does a launcher
  script that stays the same while the program behind it changes (Psi4's `psi4`). For
  a Python library it is the library's version.

What they look like on one machine:

| program | `version()` | `numerics(spec)` |
|---|---|---|
| `xtb` (the g-xTB build) | `6.7.1` | `6.7.1 sha256:cab4190d6e72bcac483481f904051083b3730d417a43fb47d1fafe9f75b24e1a` |
| `orca` | `6.0.0` | `6.0.0 sha256:588ebd02b7d2653f2641381ac92c9666a00d5b8e02215ce2faeeb7337e4bd8f3` |
| `psi4` | `1.11` | `1.11 sha256:c0edf90d4f8a53c8e7c3e759cbd33e2efbf48f699ddb458c2f45444c35686ef0` |
| `tblite` | `0.7.0` | `0.7.0` |
| `geometric` | `1.1.1` | `1.1.1` |
| `files` (`FileInput`) | qccodec's | `sha256:...` of the command's executable |

A program that is not installed reports `not installed` for both, and an executable
whose answer to `--version` names no version reports `unknown`. To look a result up in
a store by hand, take the identities from `find_program(name).numerics(spec)` rather
than writing them out: `store.get(key, {"tblite": "0.7.0"})` matches a tblite run,
`{"xtb": "6.7.1"}` matches no xtb run.

## Errors

A failed calculation raises `qcdata.ProgramError` whose `error.kind` says why, in the
vocabulary every qc* program shares, and `error.code` refines it. Kinds decide which
stored failures a store reuses (only those the input determines), so each codec maps
every failure its program is known to report; anything else is `internal`. The kinds
and codes the codecs use:

| failure | kind | code |
|---|---|---|
| a keyword the program does not have (with the closest one named) | `input` | `unknown_key` |
| a keyword of another calculation type | `input` | `not_applicable` |
| a charge or multiplicity the model cannot describe | `input` | `not_modeled` |
| atoms on top of each other, a multiplicity the electrons cannot have, an unknown method, basis set or solvent, a value the program does not take | `input` | |
| a request the program does not implement | `unsupported` | |
| the SCF did not converge (with its last iterate's energy where the program prints it) | `convergence` | `scf_not_converged` |
| coupled-perturbed (CPHF/CPKS) equations did not converge, where the output tells | `convergence` | `response_not_converged` |
| an optimizer ran out of iterations | `convergence` | `max_iter` |
| an IRC started from a structure that is not a first-order saddle point | `input` | `not_first_order_saddle` |
| a subprogram failed (`error.cause` holds its error) | `program` | |
| no usable GPU | `backend` | |
| the program is not installed | `environment` | |
| the deadline passed | `timeout` | |

Each program page lists its own. A failure keeps everything recovered so far
(`exc.output.results`, an optimization's trajectory), the program's log
(`exc.output.logs`) and the traceback (`exc.output.traceback`). A program out of memory
(`resource`, `host_memory` or `device_memory`) gives `required_gib` and `available_gib`
in `error.details` where its output says what it needed, in the terms of its own memory
setting (ORCA's `%maxcore`, Psi4's `--memory`, TeraChem's `gpumem`): the setting with
which the calculation fits and the one it ran with.

A result the program accepted without meeting its convergence thresholds (an SCF it
carried on with: ORCA's `ConvIgnored`, Psi4's `fail_on_maxiter` off, DFTB+'s
`ConvergentSCCOnly = No`) succeeds with the warning `loosely_converged` in
`output.warnings`, whose message gives the residual reached and the rule.

## Hessians

`results.hessian` is the full Cartesian second derivative of the energy at the input
geometry, never projected, as qcdata defines it, so it serves away from stationary
points too (the start of a transition-state search, an IRC's updates). Every codec's
Hessian is checked against central differences of the program's own gradients on a
distorted water molecule. xtb projects translations and rotations out of the Hessian it
writes, so its codec reads the one xtb writes before projecting; Psi4 projects
rotations out of finite-difference Hessians only with `fd_project`, which its codec
refuses for Hessians. Projection belongs to the harmonic analysis
(`vibrational_analysis`), which records what it projected in `projection`.

## Programs that are Python libraries

tblite and the machine-learned potentials (AIMNet2, FAIR-Chem, MACE, Orb, TorchANI)
are Python libraries, not programs: qccompute calls them in a worker process of the
same Python, so install the library next to qccompute instead of putting anything on
`PATH`. They share a few conventions:

- **Models load once per process.** The first calculation downloads and loads the
  weights; later ones with the same model and device reuse them, so an optimization
  or a reaction path pays that cost once. Loaded models hold memory (GPU memory on a
  GPU): the `QCCODEC_MAX_MODELS` most recently used stay loaded (2 by default; raise it
  on large GPUs, or set 0 to load every time), and `qccodec.clear_model_cache()`
  unloads them all.
- **`device` picks the hardware**: `"cpu"`, `"cuda"`, or `"cuda:1"`. Without it, or with
  `"cuda"`, the calculation asks the runner for one GPU and runs on the one it is
  granted, or on the CPU when the runner has none to give; `resolved.settings` records
  the device it ran on. An explicit index (`"cuda:1"`) asks the runner for none.
- **One thread unless asked.** Each plans one CPU thread; `{"threads": n}` asks for
  more, which the worker's numerical libraries get.
- **Keywords are checked.** Each codec accepts a short list of settings and rejects
  anything else, so a misspelled option fails instead of being ignored.
- **Charge and spin** reach models that understand them (AIMNet2-NSE, FAIR-Chem's
  `omol` head, MACE-OMOL, Orb's OMol25 models). Codecs refuse ions and open shells for
  models trained only on neutral closed-shell molecules.
- **Optimizations and transition states** run under [geomeTRIC](geometric.md) or
  [Sella](sella.md), with the library's gradients (and, for Sella, optionally its
  Hessians) as the child calculations.
- Some libraries pin incompatible dependencies; `fairchem-core` cannot share an
  environment with `mace-torch` or `orb-models`.

geomeTRIC, also a Python library, runs in the caller's process instead: it writes only
in its calculation directory and logs through one logger, so several optimizations run
side by side in threads and their subprograms are found as any calculation in that
process finds them, including programs registered there with
`qccompute.register_program`. The others need a process of their own: tblite's Fortran
writes to the process's standard output and takes its threads from `OMP_NUM_THREADS`,
the machine-learned potentials set PyTorch's threads and CUDA devices for the whole
process, and importing Sella imports JAX and sets its environment variables.

## xtb, tblite, or CREST?

All three run the GFN-xTB methods and give the same GFN2-xTB energies and gradients.
Use `program="tblite"` for fast in-process energies and gradients (no program to
install, but no Hessians or optimizer of its own). Use `program="xtb"` for single
points, Hessians, and optimizations with xtb's full set of options, for point charges,
and for g-xTB. Use `program="crest"` for conformer searches and CREST's other workflows.

## Optimizations: the program's optimizer, geomeTRIC, or Sella?

Programs with their own optimizer (CREST, ORCA, Psi4, TeraChem, xtb) run
`optimization` themselves. [geomeTRIC](geometric.md) and [Sella](sella.md) can drive
any program's gradients instead, which is useful for a consistent optimizer across
programs, constraints, and QM/MM optimizations with Psi4 point charges. Sella can also
start a transition-state search from a Hessian program (analytic or cheap Hessians,
such as xtb's or a machine-learned potential's) instead of finite differences.
