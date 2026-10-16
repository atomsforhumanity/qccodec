# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Added

- A runnable example script for every program and calculation type in `examples/<program>/` (`single_point.py`, `optimization.py`, `transition_state.py`, `conformer_search.py`), showing every common option (implicit solvent, parallelism, convergence, program extras) and each program's quirks. The integration suite runs every script, and `tests/test_examples.py` fails if a program lacks one.
- A documentation site at https://qccodec.docs.atomsforhumanity.org: a page per program (the codec module's docstring plus its example scripts), guides to decoding output and writing a codec, and an API reference. A Docs workflow builds it on pull requests and publishes it on release tags or by hand. The README now points there.
- A `psi4` codec: energy, gradient, hessian (with frequencies and normal modes), optimization, and transition_state (optking, with energies and gradients for every step; transition states set `opt_type ts` and, unless given, `full_hess_every 0`) for any Psi4 method. Psi4 runs a small driver script with its own interpreter (`psi4 input.py -k`), so the calling environment never imports Psi4, and results come back as JSON instead of parsed text. Keywords pass to `psi4.set_options` unchanged, except driver arguments (`external_potentials` for QM/MM point charges in bohr, `dertype`, `project_trans`, `project_rot`) and the molecule's `symmetry`. The molecule keeps the input's frame (`no_com`, `no_reorient`); Psi4 detects only symmetry that frame already has, and `symmetry: c1` turns it off. Results also fill `atomic_charges` (Mulliken), `point_charge_gradient`, orbital energies and occupations, the dipole, and every scalar Psi4 variable in `extras["psi4_variables"]`. Psi4 errors raise `CalculationError` with Psi4's own message. Point charges are rejected for hessian, optimization, and transition_state, which Psi4 1.11 gets wrong (Hessians fail or disagree with finite differences of its gradients; optimizations never leave the starting geometry); run geomeTRIC with Psi4 gradients for QM/MM optimizations.
- AGENTS.md and CONTRIBUTING.md require every codec to keep results in the input structure's frame.
- Recorded integration cases replace this machine's home directory and hostname in text files.
- An `xtb` codec that runs the xtb program on the command line: energies, gradients, Hessians (with frequencies and normal modes), and optimizations with GFN0-, GFN1-, GFN2-xTB, GFN-FF, and g-xTB. Keywords pass through as xtb's own command-line flags; partial charges fill `atomic_charges`. xtb errors, unknown command-line options (which xtb only warns about, for example `--gxtb` on a build without g-xTB), and optimizations that run out of cycles raise `CalculationError`, although xtb exits 0 for the last two.
- A "Supported programs" README section, including when to run xTB through `xtb` or `crest`.
- Integration tests in `tests/integration/` that run CREST, TeraChem, ORCA, and geomeTRIC through qccompute (a test-only dependency group). They are deselected by default (`pytest -m integration`), skip when a program is missing, and can save runs as fixture cases with `--record-cases`. A new Integration workflow runs them nightly on GitHub-hosted runners (CREST, geomeTRIC) and on a self-hosted runner for private and GPU programs (TeraChem, ORCA) once one is configured.

### Removed

- The xtb-python codec and the `xtb` extra. xtb-python has had no release since 2022 and has no wheels for current Python; `program="xtb"` now runs the xtb program instead.

### Changed

- Examples and documentation use ωB97X with dispersion and def2-SVP (Psi4 `wb97x-d`, ORCA `wB97X-D3`, TeraChem `wb97xd3`) instead of HF/STO-3G.
- When a program exits with an error and the codec's failure hook explains why, `Codec.decode` now raises `CalculationError` with that explanation and the recovered data, instead of returning partial data whose only trace of the failure was the exit status. `parse(..., failed=True)` still recovers quietly; its new `report_failure` argument raises the explanation on request.
- Require `qcdata>=0.20.1` for `SinglePointData.atomic_charges`.

### Fixed

- geomeTRIC trajectories included every finite-difference gradient behind a Hessian (2 x 3N per Hessian), so a transition-state search on ammonia reported 27 "steps". The trajectory now holds one frame per optimizer step, and each Hessian geomeTRIC computes is attached to the frame at its geometry (`hessian="first+last"` puts one on the final frame).
- geomeTRIC's `coordsys="hdlc"` ran TRIC; it now adds Cartesian coordinates as geomeTRIC's own interface does.
- CREST optimization trajectories were shifted to the center of mass; they are now in the input's frame.
- ORCA: `{"numgrad": False}` still requested numerical gradients, and scalar keywords such as `{"tightscf": True}` were dropped silently. `True` now adds a simple-input word, `False`/`None` leaves it out, and any other non-dictionary value is an error.
- TeraChem optimizations from newer builds (2026.10 confirmed), which write `optim_trajectory.xyz` with the energy inside a `frame N SCF Energy = ...` comment instead of `optim.xyz`.

## [0.13.1] - 2026-10-14

### Changed

- Development tooling: shared ruff (including Google docstring rules), mypy (fully typed library code), and pytest settings; pre-commit hooks that run the locked tools through `uv run`; one CI workflow; a tag-driven release workflow with `scripts/prepare_release.py` and an approval-gated PyPI deployment. Added CONTRIBUTING.md and AGENTS.md.
- Depend on `qcref` instead of the deprecated `qcconst`.

## [0.13.0] - 2026-10-14

- Require `qcdata>=0.20.0`.

### Changed

- One `Codec` object per program, defined in a single module under `qccodec/programs/`. A codec holds the program name, supported calculation types, declared decode `options`, an encoder, either decorated parsers or a `decode` function, and an optional `propagate_wfn` function. Hooks attach with `@codec.encoder`, `@codec.parser(...)`, `@codec.failure`, and `@codec.decoder`.
- `encode()` returns a complete `SubprocessInput` or `PythonInput`; nothing is executed by qccodec. Program-specific execution details (commands, environment defaults, restart translation, native Python bindings for xTB, geomeTRIC, and tcpb) live here, not in qccompute.
- Parsers read `"stdout"` (default), a named file or glob relative to the calculation directory, a file named by a function of the `Context`, or `"directory"`. Directory parsers receive a single `Context` with `stdout`, `directory`, `input_data`, `failed`, `path()`, `read()`, and `decode()` for nested decoding.
- The decode pipeline runs every applicable parser before raising, so partial data on errors is as complete as possible. A program failure hook recovers partial data and raises `CalculationError` with it attached, in both execution and standalone `decode()`.
- Added `qccodec.register(codec)` / `unregister(name)` so scripts can add or override codecs by program name for the current process, including as children of optimizers.
- `get_codec()` accepts a program name or an input object (`FileInput` resolves to the pass-through file codec).
- Decode options are declared per codec with defaults and passed per call: `decode(..., options={"collect_rotamers": True})`. Parsers read them from `Context.options`; unknown names raise `EncoderError`.
- `codec.propagate_wfn(output, input_data)` moves a previous output's wavefunction onto the next input (TeraChem: `c0`/`ca0`/`cb0` plus the `guess` keyword), replacing the `restart` hook.
- `qccodec.utils` provides `re_search` and `re_finditer`.
- Added `qccodec.testing`: fixture cases under `tests/data/<program>/<case>/` (`input.json`, `expected.json`, stdout, output files, optional `case.json` and `encoded/`), `cases()`/`run_case()` to check encoding, decoding, and each parser against them, `assert_data_equal()`, and `check_codec()` for the structural contract every codec must satisfy. The built-in parser tests now use this convention; the hand-maintained `answers/` modules are gone.
- Parsers can be limited to program versions with `@codec.parser(..., versions=">=6.0")` (PEP 440 specifiers). The version parser runs first and selects which parsers apply; parsers without `versions` run on every version, and same-target alternatives are allowed when each declares versions. `Codec(version_key=...)` adapts non-PEP 440 version strings (TeraChem compares builds by release date). `parse`/`decode` accept `version=` for logs that never print one. Adds a `packaging` dependency.
- Third-party packages can publish codecs as entry points in the `qccodec.codecs` group (`myprog = "mypackage.codec:codec"`); `get_codec` and `supported_programs` discover them.

### Removed

- The QCEngine/QCElemental compatibility codec and the `qccodec[qcengine]` extra. Programs without a codec raise `CodecNotFoundError`; add native codecs instead (Psi4 is the first candidate).
- `CodecDefinition`, `ParserSpec.explicit_calctypes`, `iter_files`, the per-program `*FileType` enums, `DataCollector`, and the `encoders/`, `parsers/`, and `codecs/` packages. The `Codec` protocol is replaced by the concrete `Codec` class.
- Per-program `make_codec` factories and `get_codec(name, **options)`; use decode `options` instead.
- geomeTRIC's `gradient_codec` option; register the gradient codec with `qccodec.register` instead.

### Breaking behavior changes

- `required=True` parsers always require their file or value; previously parsers registered for all calculation types tolerated a missing artifact.
- TeraChem standalone `decode(..., directory=...)` takes the run directory containing `scr*/`, not the `scr*` directory itself. A TeraChem log without a `Job finished:` line is reported as a `CalculationError` with the recoverable data attached.
- CREST rotamers are not parsed unless requested with `options={"collect_rotamers": True}`, for both execution and standalone decoding.


## [0.12.0] - 2026-09-16

### Changed

- Regenerate `uv.lock` before release commits and tags; restore it with the version and changelog if release preparation fails.
- Use qcdata’s shared data-type lookup and preserve partial scientific values on failed executions.
- Add `decode(..., failed=True)` and CLI `--failed` for recovering partial data; keep CREST energy-only trajectory snapshots successful when no gradient is available.

#### Breaking changes for qcdata 0.19.0

- `encode(input_data)` selects the encoder from `input_data.program`.
- Decoded scientific data now carries its own `provenance.program` and `provenance.program_version`, including partial data on parser errors. Unknown versions remain `None`.
- Trajectories use `ProgramOutput.results` and `execution`, preserving the input's requested executor separately from the parsed producer.
- CREST intermediate optimization records contain actual energies without fabricated gradients; the final record uses the parsed gradient when available.
- Leaf encoders reject a missing model with an intentional `EncoderError`.

## [0.11.2] - 2026-08-28

### Fixed

- `decode()` no longer fails on ORCA Hessian calculations. Analytic Hessian jobs don't print a `CARTESIAN GRADIENT` block in ORCA stdout, so gradient parsing for `hessian` calctype now uses a dedicated, non-required `parse_gradient_hessian` parser instead of the `gradient`-calctype parser.
- ORCA optimization trajectory parsing now passes `charge` and `multiplicity` from `input_data.structure` into each `Structure` parsed from the `_trj.xyz` file instead of silently defaulting to `(0, 1)`.
- ORCA `parse_hessian()` recognizes that the final Hessian block entry may only contain a single column. See `tests/data/orca/single_column.hess` for an example.

## [0.11.1] - 2026-08-26

- Updated GitHub references from `coltonbh` to `atomsforhumanity`.

## [0.11.0] - 2026-07-15

### Fixed

- `decode()` now raises a parser error when output artifacts required by parser specs for the requested calculation type are missing, while preserving any partial data parsed from available artifacts.

## [0.10.2] - 2026-06-10

- Marked ORCA `parse_natoms` as `required=False` to fix parsing of xTB log files.
- Fixed gradient regex to account for arbitrary number of stars in log file.

## [0.10.1] - 2026-05-02

### Fixed

- ORCA encoder no longer writes `None` if no basis (e.g., when no basis is needed for an XTB calculation). [#42](https://github.com/atomsforhumanity/qccodec/pull/42) by [TroyNSmith](https://github.com/TroyNSmith)

## [0.10.0] - 2026-03-26

### Changed

- Updated the codebase, tests, and documentation to use `qcdata` in place of `qcio`. [#41](https://github.com/atomsforhumanity/qccodec/pull/41)
- Replaced deprecated `Results` usage with `ProgramOutput` in parser trajectories and test fixtures. [#41](https://github.com/atomsforhumanity/qccodec/pull/41)

## [0.9.1] - 2025-11-14

### Added

- Parsers and encoders for ORCA (by [@avcopan](https://github.com/avcopan)) [#33](https://github.com/atomsforhumanity/qccodec/pull/33), [#36](https://github.com/atomsforhumanity/qccodec/pull/36), and [#40](https://github.com/atomsforhumanity/qccodec/pull/40).

## [0.9.0] - 2025-10-09

### Changed

- Updated to `qcio` 0.16.0 which reverts input data structure names.

## [0.8.1] - 2025-10-07

### Fixed

- Correctly propagate `Structure` charge and multiplicity through TeraChem optimization trajectory parsing.

## [0.8.0] - 2025-10-06

### Removed

- 🚨Python 3.9 support. Minimum supported version is now 3.10. [#37](https://github.com/atomsforhumanity/qccodec/pull/37).

### Changed

- 🚨 Updated to `qcio>=0.15.0` which incorporates new `qcio` nomenclature. [#37](https://github.com/atomsforhumanity/qccodec/pull/37)
- Changed TeraChem encoder `Enum` lookups to compare `Enums` directly instead of string value.
- Updated `prog_inp` pytest fixture to have no keywords so that it can be reused in other outside of TeraChem.

### Added

- Test matrix that runs off a "pip" installed version of `qccodec`.

## [0.7.8] - 2025-07-07

### Fixed

- `crest` parsers imported missing `qcio.constants` module. Updated to `qcconst.constants`.

### Changed

- Bumped minimum `qcconst` version from `0.1.2` to `0.2.1` to fix `mypy` issue.

## [0.7.7] - 2025-06-23

### Changed

- TeraChem trajectory parsing now splits the optimization stdout into sections for each gradient call and runs `decode(...)` on each chunk. This fixes the excited state parsing bug mentioned in `Fixed` and setups up parser to be better isolated for their calculation type (e.g., no more `parse_gradients` but rather just `parse_gradient` for all cases of a single point calculation, even within an optimization workflow).
- TeraChem logs for each gradient call in an optimization calculation are now added to each `ProgramOutput` in `OptimizationResults.trajectory`.

### Fixed

- Excited state parsing bug within an optimization calculation that would result in all excited states for the whole optimization calculation being added to each `ProgramOutput` in the `.trajectory`.

### Removed

- `terachem.parse_gradients` function now that that gradient parsing is done gradient call-by-gradient call for optimizations instead of in one batch.

## [0.7.6] - 2025-04-01

### Added

- Logging to `decode` clearly detail what parsing actions are being taken and why on various filetypes.

### Changed

- Renamed package from `qcparse` -> `qccodec`.
- Removed `poetry` in favor of `hatchling` build system and `uv` for project management.
- 🚨 Refactored top-level `parse` function to a new signature `decode(program: str, calctype: CalcType, *, stdout: Optional[str] = None, directory: Optional[Union[str, Path]] = None, input_data: Optional[StructuredInputs] = None) -> StructureResults`.
- Changed parsers to no longer set data on a data collector object but to rather be pure functions returning their parsed data.
- Renamed `@parser` decorator to `@register` and accept `target` kwarg that tells the registry where to place the parsed data on the data collector object.
- `DataCollector` now inherits from `dict` rather than being a `SimpleNamespace`.
- Switched classes from pydantic `BaseModel` to `dataclasses` since I'm not using any advanced validation logic.

### Removed

- 🚨 `get_file_contents` function since we no longer pass filenames to top-level `decode` (formerly `parse`) function.
- 🚨 `parse_results` function previously kept for backwards compatibility.
- `pydantic` dependency.

## [0.7.5] - 2025-03-24

### Added

- TeraChem `parse_excited_states` function.

## [0.7.4] - 2025-02-25

### Changed

- CREST encoder now selects `min(os.cpu_count(), 16)` for `threads` if not set by the user.

## [0.7.3] - 2025-02-08

### Fixed

- Bug in parsing CREST's `g98.out` file for more than three frequencies.

## [0.7.2] - 2025-02-05

### Added

- Parse CREST's `numhess` g98.out file for frequency and normal mode displacement vectors.

## [0.7.1] - 2025-01-15

### Changed

- More flexibly defined `qcio` dependency version from `^0.11.8` to `>=0.12.1` to account for the missing `eval_type_backport` package.

## [0.7.0] - 2024-12-20

### Changed

- 🚨 Dropped Python 3.8 support.

### Removed

- `black` and `isort` in favor of `ruff`.

## [0.6.4] - 2024-10-01

### Added

- Encoders and parsers for CREST to support `energy`, `gradient`, `hessian`, and `optimization` calculations.

## [0.6.3] - 2024-09-12

### Added

- `parse_optimization_dir(...) -> OptimizationResults` for TeraChem.

## [0.6.2] - 2024-08-13

### Added

- `CREST` encoder and directory parser for conformer search output directories.

## [0.6.1] - 2024-08-08

### Added

- Program version parser for `CREST` stdout.

## [0.6.0] - 2024-06-10

### Changed

- Updated to `qcio 0.10.0` to use new `Structure` rather than `Molecule` nomenclature.

## [0.5.3] - 2024-04-04

### Changed

- TeraChem `parse_git_commit` updated to `parse_version_control_details` to accommodate older versions of TeraChem compiled from the `Hg` days.

### Removed

- TeraChem `parse_version` parser. It was unused given the switch to only parsing `SinglePointResults` objects and not `Provenance` objects as well. The `parse_version_string` function is used by `qcop` to get the version of the program. We do not need to set the version of the program at `SinglePointResults.extras.program_version` anymore.

## [0.5.2] - 2023-09-27

### Removed

- All input parsing details from the library.

### Added

- `encode` top level function and encoder for TeraChem input files.

### Changed

- Added `FileType.stdout` as default `filetype` argument to `parse` decorator to reduce boilerplate in parsers.

## [0.5.1] - 2023-09-19

### Changed

- Dropped Python dependency from `^3.8.1` to `^3.8`. Can't remember what older dependency required `3.8.1` but it's not needed anymore.

## [0.5.0] - 2023-08-31

### Changed

- Updated `pydantic` from `v1` -> `v2`.

## [0.4.1] - 2023-07-19

### Changed

- Updated `qcio` from `0.3.0` -> `0.4.0`.

## [0.4.0] - 2023-07-17

### Changed

- Updated to used `qcio>=0.3.0` flattened models and the `SinglePointResults`object.

## [0.3.2] - 2023-06-29

### Fixed

- Updated package description in pyproject.toml from TeraChem specific parsing and MolSSI to all QC packages and qcio.

## [0.3.1] - 2023-06-29

### Added

- Git commit parsing for TeraChem as part of version parsing

### Changed

- `qcio` `>=0.1.0` -> `>=0.2.0`

## [0.3.0] - 2023-06-28

### Changed

- Dropped support for `QCSchema` models and changed to `qcio` (now [qcdata](https://github.com/atomsforhumanity/qcdata)) data models.
- `parse` function now raises `NotImplementedError` and the default use case is to use `parse_computed_prop` instead and ignore inputs and provenance data. This is the minimum spec since QC programs can be powered using structured inputs and [qcop](https://github.com/atomsforhumanity/qccompute). I may go back to parsing entire `SinglePointSuccess/FailedOutput` objects if use cases arise.

## [0.2.1] - 2023-03-25

### Changed

- Generalized `CUDA error:` regex to catch all CUDA errors.

## [0.2.0] - 2023-03-24

### Added

- `cli` interface for parsing TeraChem outputs from the command line.
- `parse_natoms`, `parse_nmo`, `parse_total_charge`, `parse_spin_multiplicity`

### Removed

- Removed Hessian matrix dimension checking from `parse_hessian`. Dimension validation is already done via `pydantic` on the `AtomicResult` object.

## [0.1.0] - 2023-03-23

### Added

- Basic parsers for energy, gradient, Hessian (frequencies) calculations.
- Can return either `AtomicResult` or `FailedOperation` objects depending on whether calculation succeeded or failed.
- Tests for all parsers and the main `parse` function.

[unreleased]: https://github.com/atomsforhumanity/qccodec/compare/0.13.1...HEAD
[0.13.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.13.1
[0.13.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.13.0
[0.12.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.12.0
[0.11.2]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.11.2
[0.11.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.11.1
[0.11.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.11.1
[0.11.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.11.0
[0.10.2]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.10.2
[0.10.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.10.1
[0.10.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.10.0
[0.9.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.9.1
[0.9.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.9.0
[0.8.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.8.1
[0.8.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.8.0
[0.7.8]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.7.8
[0.7.7]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.7.7
[0.7.6]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.7.6
[0.7.5]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.7.5
[0.7.4]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.7.4
[0.7.3]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.7.3
[0.7.2]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.7.2
[0.7.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.7.1
[0.7.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.7.0
[0.6.4]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.6.4
[0.6.3]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.6.3
[0.6.2]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.6.2
[0.6.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.6.1
[0.6.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.6.0
[0.5.3]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.5.3
[0.5.2]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.5.2
[0.5.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.5.1
[0.5.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.5.0
[0.4.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.4.1
[0.4.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.4.0
[0.3.2]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.3.2
[0.3.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.3.1
[0.3.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.3.0
[0.2.1]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.2.1
[0.2.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.2.0
[0.1.0]: https://github.com/atomsforhumanity/qccodec/releases/tag/0.1.0
