# Contributing to qccodec

Thanks for helping! This guide covers setup, the pre-commit workflow, tests, pull requests, and releases. Code conventions (typing, docstrings, units) are in [AGENTS.md](./AGENTS.md), which applies to people and AI agents alike.

## Development setup

You need [uv](https://docs.astral.sh/uv/). Then:

```sh
git clone https://github.com/atomsforhumanity/qccodec.git
cd qccodec
uv sync --all-extras --all-groups   # creates .venv with the package and all dev tools
uv run pre-commit install           # run the checks automatically on every commit
```

## Pre-commit workflow

Every `git commit` runs these checks on the files you staged. All of them use the tool versions locked in `uv.lock`, so they behave the same on your machine and in CI.

| Check | What it does |
|---|---|
| `ruff check` | Lints (pyflakes, pycodestyle, import order, bugbear, pyupgrade, Google docstrings) and fixes what it can |
| `ruff format` | Formats Python code |
| `mypy` | Type-checks the package; library functions must be fully annotated |
| `typos` | Spell-checks code and docs |
| `detect-secrets` | Blocks commits that contain credentials or tokens |
| `uv lock --check` | Fails if `uv.lock` is out of date with `pyproject.toml` |
| file checks | Valid YAML and TOML, no merge-conflict markers, no files over 250 KB outside `tests/data/` and `docs/` |

If a hook fixes files (ruff often does), the commit stops so you can review the changes. Stage them and commit again:

```sh
git add -u && git commit
```

Run every check on the whole repository, exactly as CI does:

```sh
uv run pre-commit run --all-files
```

You can run the tools directly too: `uv run ruff check --fix .`, `uv run ruff format .`, `uv run mypy`. Please don't skip hooks with `--no-verify`; CI runs the same checks and will fail.

## Running tests

```sh
uv run pytest                               # all tests
uv run pytest --cov --cov-report=term-missing   # with coverage
```

Parsers are tested through recorded program runs in `tests/data/<program>/<case>/`; see [Tests](#7-tests) under "Adding a program" below.

## Integration tests

Tests in `tests/integration/test_<program>.py` run real programs through
[qccompute](https://github.com/atomsforhumanity/qccompute), which is a test-only
dependency in the `integration` group. They are marked `integration` and deselected by
default, and each test skips when its program is not on `PATH`.

```sh
uv sync --group integration --extra geometric
uv run pytest -m integration -rs              # -rs lists what skipped and why
uv run pytest -m integration --record-cases   # also save runs as fixture cases
```

Assert against reference values with tolerances, use the smallest molecule that
exercises the feature, and keep each test to seconds. Only the latest release of each
program is tested.

`--record-cases` saves each run that names a case, `run(input, case="water_gradient")`,
as `tests/data/<program>/<case>_<version>/` with `input.json`, `expected.json`, `stdout`,
`encoded/`, and the output files, then replays it through `qccodec.testing.run_case`.
Existing cases are never overwritten. Before committing a recorded case, delete output
files no parser reads and check `stdout` for anything private; `execution` metadata
(local paths, timings) is already stripped. This is the easiest way to add a fixture
for a new program version.

The [Integration workflow](.github/workflows/integration.yaml) runs nightly, on demand,
and on pull requests that touch codecs or these tests. It is not a required check.

- **CPU job.** Runs on GitHub-hosted runners. It installs the official CREST release binary
  and the xtb build that ships g-xTB, which also runs every other xtb method, each
  pinned by version and checksum in the workflow. Add a CPU program with an
  install step there and add it to `QCCODEC_REQUIRE_PROGRAMS`.
- **Private/GPU job.** Runs licensed programs that cannot be installed on
  GitHub-hosted runners (ORCA, TeraChem) and GPU programs (TeraChem) on a
  self-hosted runner. It is skipped until the repository variable
  `QC_PRIVATE_RUNNER` names the runner's label, or an
  [Actions Runner Controller](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners-with-actions-runner-controller/about-actions-runner-controller)
  scale set. The runner must provide the programs on `PATH`, a CUDA driver, network
  access to the TeraChem license server, and `curl` and `git` for the setup actions.
  `QC_PRIVATE_PROGRAMS` (default `terachem`) lists the programs it must have. Images
  containing licensed programs stay on the runner host; never push them to a public
  registry.
- **Fork pull requests** never run on the self-hosted runner. Keep "Require approval
  for all outside collaborators" enabled under Settings > Actions as a second guard.

`QCCODEC_REQUIRE_PROGRAMS` (comma-separated) turns skips into failures for the listed
programs, so a broken runner image cannot pass by skipping everything.

## Pull requests

1. Branch from `master`.
2. Make your change with tests. Bug fixes should include a test that fails without the fix.
3. Add a line under `## [unreleased]` in [CHANGELOG.md](./CHANGELOG.md) for anything users would notice.
4. Open a pull request. CI runs the pre-commit checks and the test suite on the oldest and newest supported Python versions.

Never commit `[tool.uv.sources]` entries that point qc* packages at local paths or git branches; CI rejects them. When a change needs an unreleased version of another qc* package, release that package first.

## Releases (maintainers)

1. On a branch, prepare the release and open a pull request:

   ```sh
   git switch -c release/X.Y.Z
   uv run python scripts/prepare_release.py X.Y.Z   # sets the version, dates the changelog, refreshes uv.lock
   git commit -am "Release X.Y.Z" && git push -u origin release/X.Y.Z
   ```

2. After the pull request is merged, tag the merge commit on `master`:

   ```sh
   git switch master && git pull && git tag X.Y.Z && git push origin X.Y.Z
   ```

3. The tag starts the Release workflow. Approve the waiting deployment to the `pypi` environment in the Actions tab; the workflow then publishes to PyPI and creates the GitHub release from the changelog section. The workflow refuses a tag that doesn't match the version in `pyproject.toml`.

## Adding a program

Hey there 👋! Look at you wanting to contribute! Every program is one file,
`src/qccodec/programs/<name>.py`, that builds a single `Codec`. This page walks through
the whole thing; the built-in `terachem.py`, `orca.py`, and `crest.py` are complete
examples of file-based programs, and `xtb.py` of a Python API.

### 1. The codec

```python
from qcdata import CalcType, ProgramInput
from qccodec import Codec, Context, SubprocessInput, re_search
from qccodec.exceptions import EncoderError
from ._common import invocation

codec = Codec("myprog", calctypes={CalcType.energy, CalcType.gradient})
```

`program` is matched against `ProgramInput.program`; `calctypes` are the calculation
types the encoder can produce. Pass `producer="other"` if results are really produced by
another program (the tcpb clients report `terachem`).

### 2. The encoder

```python
@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    if inp.model is None:
        raise EncoderError("myprog requires a model.")
    text = f"method {inp.model.method}\ncharge {inp.structure.charge}\n"
    return invocation(inp, {"run.inp": text, "geom.xyz": inp.structure.to_xyz()}, "run.inp")
```

`invocation()` builds `myprog run.inp <cmdline_args>` and merges user-supplied
`inp.files`, rejecting name clashes. You can also return a `SubprocessInput` directly,
or a `PythonInput(function=..., kwargs=...)` for programs with a Python API.
Raise `EncoderError` for anything the program cannot run. Never execute here.

### 3. Parsers

Each parser is a small function from contents to one value:

```python
@codec.parser(target="energy", calctypes=[CalcType.energy, CalcType.gradient])
def parse_energy(stdout: str) -> float:
    return float(re_search(r"Total energy:\s+(-?\d+\.\d+)", stdout).group(1))

@codec.parser(file="run.grad", target="gradient", calctypes=[CalcType.gradient])
def parse_gradient(contents: str) -> list[list[float]]:
    values = [float(v) for v in contents.split()]
    return [values[i : i + 3] for i in range(0, len(values), 3)]

@codec.parser(target=("provenance", "program_version"), required=False)
def parse_version(stdout: str) -> str:
    return re_search(r"myprog version (\S+)", stdout).group(1)
```

Rules the decorator enforces and the pipeline relies on:

- `target` names the field on the qcdata result (`SinglePointData`, `OptimizationData`,
  ...). A tuple fills a nested field. Two parsers may not fill the same target for
  overlapping calculation types.
- `calctypes` omitted means "every calculation type".
- `file` is `"stdout"` (default), a file name or glob relative to the calculation
  directory (`"*.hess"`, `"scr*/optim.xyz"`), a function of the `Context` returning a
  name (`lambda ctx: f"{basename(ctx.stdout)}.hess"`), or `"directory"`.
- `required=True` (default) means a missing file or a `MatchNotFoundError` fails the
  decode. Use `required=False` for values that are not always present. Use
  `re_search`/`re_finditer` so "not found" is reported as `MatchNotFoundError`; raise
  `ParserError` for malformed data.

Parsers that need more than one file take a `Context`:

```python
@codec.parser(file="directory", target="trajectory", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[ProgramOutput]:
    structures = Structure.open_multi(ctx.path("traj.xyz"), charge=ctx.input_data.structure.charge)
    step = ctx.decode(CalcType.gradient, stdout=ctx.stdout)   # reuse this codec's parsers
    ...
```

`ctx` carries `stdout`, `directory`, `input_data`, `failed`, `path()`, `read()`, and
`decode()`. A directory parser may omit `target` and return a mapping that is merged
into the result.

#### Program versions

When a new release changes an output format, add a parser for the new range and narrow
the old one with `versions=`; both may target the same field because both declare
versions:

```python
@codec.parser(target="energy", versions="<4.0")
def parse_energy_v3(stdout: str) -> float: ...

@codec.parser(target="energy", versions=">=4.0")
def parse_energy(stdout: str) -> float: ...
```

Parsers without `versions` run on every version. The version parser (target
`("provenance", "program_version")`) always runs first and cannot itself be gated. If
the program's version string is not PEP 440, pass `Codec(..., version_key=fn)` that maps
it to one. Record a fixture case from each supported release so every range is tested.

### 4. Failure detection (optional)

```python
@codec.failure
def failure(ctx: Context) -> str | None:
    return "myprog reported an error." if "ERROR" in (ctx.stdout or "") else None
```

Many programs exit with status 0 after failing. When this hook returns a message,
decoding recovers every value it can and raises `CalculationError` with that data.
The hook also runs when the program exits with an error: `decode` then raises
`CalculationError` with the hook's explanation instead of leaving only an exit
status, so write messages that say what went wrong (quote the program's own error
when it prints one).

### 5. Python APIs instead of files

Skip parsers and give the codec a `decode` function:

```python
def invoke(**kwargs): ...                       # imports the library, returns a plain snapshot
def encode(inp) -> PythonInput: return PythonInput(function=invoke, kwargs={...})
def decode(inp, out) -> SinglePointData: ...    # maps out.value into qcdata fields

codec = Codec("mylib", calctypes={CalcType.energy}, encode=encode, decode=decode)
```

Import the library inside `invoke` so encoding and decoding work (and are testable)
without it installed. Set `PythonInput.environment` for settings the library reads at
import time, and `loggers` for Python loggers whose output should be captured.

### 6. Options, wavefunction propagation, registration

- Declare decode options with defaults on the codec, `Codec(..., options={"collect_rotamers": False})`,
  and read them in parsers as `ctx.options["collect_rotamers"]`. Callers override them
  with `qccompute.compute(inp, options={...})` or `qccodec.decode(..., options={...})`;
  unknown names are rejected.
- Set `codec.propagate_wfn = fn` if the program can reuse a previous wavefunction.
  `fn(output, input_data)` returns a new `ProgramInput` with the wavefunction files from
  `output.results.files` attached and the guess keyword set. The encoder's
  `wavefunction_files` glob patterns tell qccompute which files to keep.
- Add the program to `_BUILTIN` in `src/qccodec/registry.py` so `get_codec("myprog")`
  finds it. Users can also do this at runtime with `qccodec.register(codec)`.

### 7. Tests

Record a real run as a fixture case and let the shared harness do the rest:

```
tests/data/<program>/<case>/
    input.json      ProgramInput used for the run (program, calctype, structure, ...)
    expected.json   the qcdata *Data object the codec must produce
    stdout          captured program output, if any
    <files>         output files at their real relative paths (scr.geom/optim.xyz, crest.engrad)
    case.json       optional settings, see below
    encoded/        optional: command.txt (one argument per line) and the native input
                    files encode(input) must generate
```

`case.json` keys: `options` (decode options for the codec), `failed` (decode in recovery
mode), `raises` (exception class name the decode must raise; `expected.json` then holds
the partial data attached to it), `directory` (force passing the case directory to
parsers; by default it is passed only when the case holds output files).

`tests/test_cases.py` runs every case under `tests/data` through
`qccodec.testing.run_case`, which checks the encoded files, the full decode, and each
registered parser applied directly to its own file against the matching slice of
`expected.json`. It also runs `qccodec.testing.check_codec` on every built-in codec.
To add coverage, add a case directory. When the program is installed, record one with
an [integration test](#integration-tests) and `--record-cases`; otherwise run the codec
once (`qccodec.decode(...)`), review the result, and save it with
`data.model_dump_json(indent=2)`. Keep encoder-specific error paths (rejected keywords,
unsupported calculation types) as ordinary unit tests next to the program module's
helpers, as `tests/test_terachem_encoders.py` does.

### How decoding works

`Codec.parse(calctype, stdout=..., directory=..., input_data=..., failed=..., options=...)`:

1. Runs the failure hook; a message switches to recovery mode.
2. Runs every parser registered for the calculation type, handing each its stdout,
   file contents, or the `Context`, and stores values at their targets.
3. If a required file or value was missing and not in recovery mode, raises
   `ParserError`/`MatchNotFoundError` with the partial data attached.
4. If the failure hook fired, raises `CalculationError` with the data attached.
5. Otherwise validates and returns the qcdata model for the calculation type.

`Codec.decode(input_data, native_output)` is what qccompute calls; it uses the custom
`decode` function if there is one, otherwise the pipeline above with
`failed=native_output.failed`.
