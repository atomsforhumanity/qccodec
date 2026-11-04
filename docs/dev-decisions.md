# Design Decisions

Internal notes on why qccodec is shaped as it is; not part of the site.

## qccodec translates, qccompute executes

A codec encodes a `ProgramInput` into a native invocation (`SubprocessInput` or
`PythonInput`) and decodes what the program produced into qcdata results. It never
runs anything, so codecs can be tested from recorded runs without the programs, and
one runner (qccompute) owns scheduling, scratch directories, resources, and failure
handling for every program.

Decoding builds whatever the calculation type returns: `SinglePointData`, but also
optimization trajectories, IRC paths, and conformer ensembles as `Frame`s, each frame
a structure and the single-point results computed there, in the input's frame.

## Codecs are transparent

Keywords reach the program unchanged, in its units, with its own defaults, so the
program's documentation applies. What a codec may decide (accepted 2026-11-04):

- What the input means (calculation type to program switches, units, charge,
  coordinate frame): always the codec's job.
- Correctness (Psi4's `no_com`, IRC alignment, Sella's IRC end test, failing on
  non-convergence): always, documented in the codec's docstring as a deviation.
- Operational, the same numbers (compilation, logging, scratch files): allowed when
  the library default fits our runner badly; documented, and overridable by keyword.
  Decided from measurements on current versions (Orb and FAIR-Chem compile off).
- Scientific (thresholds, step sizes, TS Hessian strategy, NEB images): the program's
  default. Tuned values belong in the research-skills repository.
- Precision: the program's standard full precision by default (FP32 for
  machine-learned potentials); reduced-precision speed modes such as TF32 are opt-in.
  A codec must not leave process-wide numeric settings changed.
- Threads and GPUs: the user's and the scheduler's; a codec declares what it was
  asked to use (CREST's own default is 1 thread).

## Drivers compute children through the runner

geomeTRIC and Sella ask for energies and gradients. Each request is a child
calculation, computed by the runner's `Compute` (`PythonInput.compute_callback`) with
the call shape of `qccompute.compute`: one input, or a list computed together, which
lets a path optimization's images be batched or run side by side. Children go through
their own codecs, so any program with gradients can be driven, and failed children
are kept as frames. What drivers share lives in `programs/_driver.py`.

## Other parser packages to look at

- [iodata](https://github.com/theochem/iodata) and
  [cclib](https://cclib.github.io/contents.html) are commonly recommended.
  `iodata` was
  [published](https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.26468) in 2020.
