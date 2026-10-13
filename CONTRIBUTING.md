# Adding a program to qccodec

Hey there 👋! Look at you wanting to contribute! Every program is one file,
`src/qccodec/programs/<name>.py`, that builds a single `Codec`. This page walks through
the whole thing; the built-in `terachem.py`, `orca.py`, and `crest.py` are complete
examples of file-based programs, and `xtb.py` of a Python API.

## 1. The codec

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

## 2. The encoder

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

## 3. Parsers

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

## 4. Failure detection (optional)

```python
@codec.failure
def failure(ctx: Context) -> str | None:
    return "myprog reported an error." if "ERROR" in (ctx.stdout or "") else None
```

Many programs exit with status 0 after failing. When this hook returns a message,
decoding recovers every value it can and raises `CalculationError` with that data.

## 5. Python APIs instead of files

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

## 6. Options, wavefunction propagation, registration

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

## 7. Tests

Put sample outputs under `tests/data/<name>/` and use the `ParserTestCase` harness
in `tests/conftest.py` (see `tests/test_crest_parsers.py`). Each case calls the parser
directly and runs it through `codec.parse()` so both paths are covered. Add
encoder tests asserting on the generated input text.

## How decoding works

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
