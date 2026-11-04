# Writing a codec

Everything qccodec knows about a program lives in one `Codec` object: its name,
the calculation types it supports, an encoder, and either a set of small parsers
(file-based programs) or a decode function (Python APIs). Built-ins live in
`qccodec/programs/<name>.py`, one file per program, and you can build your own in a
script the same way:

```python
from qcdata import CalcType, ProgramInput
from qccodec import Codec, SubprocessInput, re_search

codec = Codec("myprog", calctypes={CalcType.energy})

@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    return SubprocessInput(
        command=("myprog", "run.inp"),
        files={"run.inp": f"{inp.model.method}\n{inp.structure.to_xyz()}"},
    )

@codec.parser(target="energy", calctypes=[CalcType.energy])
def parse_energy(stdout: str) -> float:
    return float(re_search(r"Total energy:\s+(-?\d+\.\d+)", stdout).group(1))
```

If the program uses several CPU threads or a GPU, say so in the native input:
`SubprocessInput(..., threads=4)` or `SubprocessInput(..., gpus=1)`. qccompute runs
lists of calculations side by side and uses these to keep within the machine's cores
and give each GPU calculation its own device.

Then run it with qccompute, either explicitly or by name after registering it:

```python
import qccodec, qccompute

output = qccompute.compute(input_data, codec=codec)

qccodec.register(codec)              # now resolvable as program="myprog",
output = qccompute.compute(input_data)   # including as a geomeTRIC gradient child
```

`register()` also lets you override a built-in codec by name for the current process.

## Parsers

A parser is a plain function from file contents to a value. The decorator records
where the value goes and when the parser runs, and returns the function unchanged so
it stays easy to test directly.

```python
@codec.parser(target="energy", calctypes=[CalcType.energy, CalcType.gradient])
def parse_energy(stdout: str) -> float: ...                      # reads stdout (default)

@codec.parser(file="crest.engrad", target="gradient", calctypes=[CalcType.gradient])
def parse_gradient(contents: str) -> list[list[float]]: ...      # reads a named file or glob

@codec.parser(file="directory", target="frames", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[Frame]: ...           # needs several files
```

- `target` is the result field to fill, or a tuple for nested fields such as
  `("provenance", "program_version")`. Directory parsers may omit it and return a
  mapping that is merged into the result.
- `calctypes` restricts when the parser runs; omit it to run for every calculation type.
- `file` is `"stdout"` (default), `"directory"`, a file name or glob relative to the
  calculation directory, or a function of the `Context` returning such a name.
- `required=False` tolerates a missing file or `MatchNotFoundError`.
- `versions=">=6.0"` limits a parser to some program versions (a PEP 440 specifier);
  omit it to run on every version. See [Program versions](#program-versions) below.

Directory parsers receive a `Context` with `stdout`, `directory`, `input_data`,
`failed`, `options`, `path(pattern)`, `read(pattern)`, and `decode(calctype, stdout=...)`
for nested decoding with the same codec.

## Failure detection and Python-native programs

```python
@codec.failure
def failure(ctx: Context) -> str | None:
    return "CREST reported FAILED." if "FAILED" in (ctx.stdout or "") else None
```

Many programs exit 0 after failing (TeraChem prints `Job finished:` only on
success). When the hook returns a message, decoding recovers what it can and raises
`CalculationError` with that data attached. This also happens when the program
exits with an error, so the hook's explanation reaches the user rather than just an
exit status. `parse(..., failed=True)` stays quiet and returns the recovered data;
pass `report_failure=True` to raise the explanation there too.
Programs driven through a Python API skip parsers entirely:

```python
codec = Codec("mlp", calctypes={CalcType.energy, CalcType.gradient}, encode=encode, decode=decode)
```

where `decode(input_data, native_output)` maps `native_output.value` into a qcdata
object.

## Batches

Libraries that evaluate many structures in one pass, such as machine-learned
potentials on a GPU, can let qccompute batch a list of inputs. Give the codec two
functions:

```python
codec = Codec(
    "mlp",
    calctypes={CalcType.energy, CalcType.gradient},
    encode=encode,
    decode=decode,
    encode_batch=encode_batch,   # (inputs) -> PythonInput returning one value per input
    batch_key=batch_key,         # (input) -> key; equal keys may share a batch, None: alone
    max_batch=64,
)
```

`encode_batch` gets inputs that share a batch key and returns a `PythonInput` whose
function returns a list of native values, one per input in order. Each value is decoded
with the codec's ordinary decoder, `decode(input, PythonOutput(value=...))`, so every
structure still gets its own result. The key holds everything a batch must share (model,
keywords, calculation type) and nothing about the structure; return `None` for inputs
the library computes one at a time, such as Hessians. The built-in machine-learned
potentials use `batch_signature` from `qccodec.programs._python` as their key.

## Options and wavefunction propagation

A codec declares the decode options it accepts, with defaults. Callers override them
per calculation and parsers read them from the context:

```python
codec = Codec("crest", calctypes={...}, options={"collect_rotamers": False})

@codec.parser(file="directory", calctypes=[CalcType.conformer_search])
def parse_rotamers(ctx: Context) -> dict:
    if not ctx.options["collect_rotamers"]:
        return {}
    ...

decode("crest", "conformer_search", directory=d, input_data=inp, options={"collect_rotamers": True})
qccompute.compute(inp, options={"collect_rotamers": True})
```

Unknown option names raise `EncoderError`. For optimizers that chain calculations, a
codec may set `codec.propagate_wfn = fn` where `fn(output, input_data)` returns a new
input carrying the previous output's wavefunction as the initial guess; TeraChem copies
`c0`/`ca0`/`cb0` and sets `guess`. qccompute calls it when `propagate_wfn=True`.

## Driver codecs

Some programs drive others: an optimizer asks for an energy and gradient at every
step. In qc* each request is a child calculation, a full calculation of a program
named in `subprograms`, computed through that program's own codec. The driver's codec
returns a `PythonInput` that names `compute_callback`, and the runner injects its
`Compute` into that keyword:

```python
def invoke_myopt(*, structure, gradient_spec, keywords, compute):
    output = compute(ProgramInput.from_spec(gradient_spec, structure))  # one child
    outputs = compute([child_a, child_b])  # several at once: batched or side by side
```

`compute` takes what `qccompute.compute` takes, one input or a list, and returns
outputs without raising for a failed calculation. `previous=` passes the outputs whose
wavefunctions start the children (one per input for a list); runners use them when
the user asks for `propagate_wfn=True`. Outside a runner, such as in tests,
`qccodec.sequential(fn)` makes a `Compute` from a function that computes one input
at a time.

The built-in drivers (geomeTRIC, Sella) share their machinery in
`qccodec.programs._driver`; a new driver should use it too, so every driver reads its
inputs and reports its results the same way:

- `subprograms(inp, CalcType.gradient, [CalcType.hessian])` returns the subprograms
  by calculation type: the first required, the rest optional, any other refused.
- `check_keywords(program, keywords, owned=..., known=...)` refuses keywords the codec
  sets itself (from the calculation type), and, for programs that silently ignore
  names they do not read, such as geomeTRIC, any name not in `known`.
- `child_resources(inp, specs)` returns the most threads and GPUs any child declares.
  The driver declares them: it occupies what its children use.
- `Recorder(program, compute)` computes the children and keeps the frames:
  `compute(child)` for a step, `compute(child, step=False)` for a probe such as a
  finite-difference displacement (kept only if it fails), `record_at(geometry)` for
  drivers that learn afterwards which geometry was a step, `compute_images(children)`
  for the images of a chain, and `add_hessian`/`attach_hessians` to put Hessians on
  the frames at their geometries. Each child starts from the last successful output
  of the same program; each image from its own last output.
- The driver's function returns an `Outcome`: `outcome(version)` with `frames`,
  `ts_index`, `paths`, and `error` filled in. `driver_decoder(program)` turns it into
  `OptimizationData`, `PathData` (`irc`), or `PathOptimizationData`
  (`path_optimization`), and raises `CalculationError` with that data when `error`
  is set.

```python
import traceback

from qccodec.programs._driver import (
    Recorder, check_keywords, child_resources, driver_decoder, outcome, subprograms,
)

def encode(inp: ProgramInput) -> PythonInput:
    specs = subprograms(inp, CalcType.gradient)
    check_keywords("myopt", inp.keywords, owned={"order"})
    threads, gpus = child_resources(inp, specs.values())
    return PythonInput(
        function=invoke_myopt,
        kwargs={
            "structure": inp.require_structure(),
            "gradient_spec": specs[CalcType.gradient],
            "keywords": dict(inp.keywords),
        },
        compute_callback="compute",
        threads=threads,
        gpus=gpus,
    )

def invoke_myopt(*, structure, gradient_spec, keywords, compute):
    recorder = Recorder("myopt", compute)
    result = outcome(version="1.0")
    try:
        ...  # each step: recorder.compute(ProgramInput.from_spec(gradient_spec, step))
    except Exception:
        result["error"] = traceback.format_exc()  # keep the frames computed so far
    result["frames"] = recorder.frames
    return result

codec = Codec(
    "myopt",
    calctypes={CalcType.optimization},
    encode=encode,
    decode=driver_decoder("myopt"),
)
```

The built-in drivers also name `child_callback="compute_child"`, a one-input callback
that runners before qccompute 0.23 inject instead; a new driver does not need it.

## Program versions

When a program changes its output format, register a parser per version range on the
same target; the parser functions themselves stay version-free:

```python
@codec.parser(target="gradient", calctypes=[CalcType.gradient], versions="<6.0")
def parse_gradient_v5(stdout: str) -> list[list[float]]: ...

@codec.parser(target="gradient", calctypes=[CalcType.gradient], versions=">=6.0")
def parse_gradient(stdout: str) -> list[list[float]]: ...
```

The parser targeting `("provenance", "program_version")` runs first and its value
selects which parsers apply. Programs whose version strings are not PEP 440 give the
codec a `version_key` (TeraChem compares builds by release date, so `v1.9-2023.08-dev`
matches `versions=">=2023.8"`). When no version can be read, alternatives for a target
are tried in registration order and the first match wins. Pass `version=` to
`decode`/`parse` for logs that never print one.

## Codecs from other packages

A package can ship a codec by publishing it as an entry point; `get_codec` and
`qccompute.compute` then find it by program name with no registration call:

```toml
[project.entry-points."qccodec.codecs"]
myprog = "mypackage.codec:codec"
```

## Testing a codec

`qccodec.testing` gives every codec the same test harness. A recorded program run is a
directory holding the inputs, the outputs, and the data the codec must produce:

```
tests/data/myprog/water_energy/
    input.json      ProgramInput used for the run
    expected.json   the qcdata *Data object the codec must produce
    stdout          captured program output
    <files>         output files at their real relative paths, e.g. scr.geom/optim.xyz
    case.json       optional: {"options": {...}, "failed": true, "raises": "ParserError"}
    encoded/        optional: command.txt plus the native input files encode() must write
```

```python
import pytest
from qccodec import get_codec
from qccodec.testing import cases, check_codec, run_case

@pytest.mark.parametrize("case", cases("tests/data"), ids=str)
def test_case(case):
    run_case(get_codec(case.input_data.program), case)

def test_contract():
    check_codec(codec, inputs=[some_program_input])
```

`run_case` checks the encoded files, the full decode (or the expected exception and its
partial data), and every registered parser applied directly to its own file against the
matching slice of `expected.json`. `check_codec` verifies the structural contract: parser
targets name real result fields, decoding nothing yields typed empty data in recovery
mode and typed partial data on error, and encoded files stay inside the directory.

[CONTRIBUTING.md](https://github.com/atomsforhumanity/qccodec/blob/master/CONTRIBUTING.md) walks through adding a program to qccodec step by step, including the example scripts every program needs.
