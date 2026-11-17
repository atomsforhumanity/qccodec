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

If the input asks for several CPU threads or a GPU, say so in the native input:
`SubprocessInput(..., threads=4)` or `SubprocessInput(..., gpus=1)`. qccompute packs
lists of calculations by these and gives each GPU calculation its own device: a
subprocess sees it in `CUDA_VISIBLE_DEVICES`, and a Python library is handed it as this
process's CUDA device (`cuda:N`) in the keyword `PythonInput(device_keyword=...)` names
(`cpu` when the runner granted none; left out when called directly). Leave
`threads` out when the input asks for none: the program then plans one thread, or
all the runner has with `CodecProgram(codec, threads=None)` for a program that scales
with threads, and runs with the runner's grant, through `OMP_NUM_THREADS` or, when the
codec has `cap_threads`, written into the input. The shared `threads` keyword is taken
by the adapter before your encoder sees the input.

Refuse what the program would ignore. Every setting given must take effect, so an
encoder checks keyword names against the program's settings where they can be listed,
with `qccodec.reject_unknown(program, keywords, known)`, which raises
`EncoderError(..., code="unknown_key")` naming the closest known name. `EncoderError`
takes `kind` (`input`, or `unsupported` for a valid request the program does not
implement) and `code` (`unknown_key`, `not_applicable`, `no_effect`, `not_modeled`).
Give the same names to the program, `CodecProgram(codec, keywords=known)` (or a
function of the calctype), and its `settings_schema` lists them; without them it
lists only the shared `threads` and allows any other key, for a program that checks
its own keywords.

The shared keyword `accuracy` (an accuracy target, `{"energy": ..., "gradient": ...,
"hessian": ...}`) never reaches your encoder: `Codec.encode` reads it with qcdata's
`Accuracy.from_keywords`. A calculation with nothing to converge (a force field, an ML
potential) meets any target as it is: the adapter accepts it and records it in the
output's `record` as met, `{"accuracy": {"target": {...}, "thresholds": {}, "estimate":
{...}}}` with an estimate of `0.0` for each target, and its
`settings_schema` lists `accuracy`. A calculation with convergence settings refuses it
with kind `unsupported`, since no codec maps targets onto its program's thresholds yet.
Say which with `Codec(..., converges=False)`, or a function of the input
(`converges=lambda inp: inp.model.method != "gfnff"`); the default is `True`.

Then make it a program and run it with qccompute:

```python
import qccodec, qccompute

qccompute.register_program(qccodec.CodecProgram(codec))  # program="myprog"
output = qccompute.compute(input_data)   # also as a geomeTRIC gradient subprogram
```

A codec that calls a Python library (one that returns a `PythonInput`) takes over its
process's output streams and working directory while it runs: make it
`CodecProgram(codec, in_process=False)` and qccompute runs it in a worker process. A
library that writes only where it is told and reports through Python loggers can run
in the caller's process instead: name `PythonInput(directory_keyword="directory",
loggers=(...))`, write every file under the directory the function is given, and the
adapter calls it without changing the working directory, environment or output
streams (geomeTRIC does this). A registered program reaches a worker only
when it is defined at the top level of an importable module; publish it as an entry
point (below) to be found everywhere. `CodecProgram(codec, executable=...,
environment=...)` configures where the program is and what it runs with;
`QCCODEC_<PROGRAM>_EXECUTABLE` does the first without code.

A store reuses a result only while the program's numerics identity is unchanged
(`CodecProgram.numerics`): the executable's version and sha256, or, for a launcher
of a container, the image's digest (`CodecProgram(codec, image=...)` or
`QCCODEC_<PROGRAM>_IMAGE`). A codec that calls a library gives the library's version
and, for an ML potential, the digest of its model's weights file:
`CodecProgram(codec, library=..., weights=...)` takes a function of the input that
returns the file, or None when it is not on disk yet.

## Parsers

A parser is a plain function from file contents to a value. The decorator records
where the value goes and when the parser runs, and returns the function unchanged so
it stays easy to test directly.

```python
@codec.parser(target="energy", calctypes=[CalcType.energy, CalcType.gradient])
def parse_energy(stdout: str) -> float: ...                      # reads stdout (default)

@codec.parser(file="crest.engrad", target="gradient", calctypes=[CalcType.gradient])
def parse_gradient(contents: str) -> list[list[float]]: ...      # reads a named file or glob

@codec.parser(file="directory", target="trajectory", calctypes=[CalcType.optimization])
def parse_trajectory(ctx: Context) -> list[ProgramOutput]: ...   # needs several files
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

Give each failure the program is known to report a kind. A store reuses a stored
failure only when its kind says the input determines it, so an SCF that does not
converge must not look like a crash:

```python
codec = Codec(
    "myprog",
    calctypes={CalcType.energy},
    errors=[
        (r"SCF not converged", "convergence", "scf_not_converged"),
        (r"unknown keyword", "input", "unknown_key"),
        (r"atoms too close", "input", None),
    ],
)
```

The first rule whose regular expression is found in the `CalculationError`'s message
(the failure hook's explanation, or a decoder's) gives the failure its kind and code;
anything else is `internal`, which means a bug. A failure that knows a number returns
`(message, details)` from the hook instead of the message alone, and `details` becomes
the error's `ErrorInfo.details`: a program out of memory (kind `resource`, code
`host_memory` or `device_memory`) gives `required_gib` and `available_gib` there when
its output says what it needed, for the whole calculation (ORCA's per-process
`%maxcore` figures times its processes). Record a fixture case of each failure with
`"raises": "CalculationError", "kind": ..., "code": ...` (and `"details"`) in its
`case.json`.
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

## Decode options

A codec declares the decode options it accepts, with defaults. Callers of `decode` and
`parse` override them, and parsers read them from the context. A choice that should be
made per calculation belongs in the input's keywords instead (CREST's
`{"output": {"rotamers": True}}`), which parsers read from `ctx.input_data`:

```python
codec = Codec("crest", calctypes={...}, options={"collect_rotamers": False})

@codec.parser(file="directory", calctypes=[CalcType.conformer_search])
def parse_rotamers(ctx: Context) -> dict:
    if not ctx.options["collect_rotamers"]:
        return {}
    ...

decode("crest", "conformer_search", directory=d, input_data=inp, options={"collect_rotamers": True})
```

Unknown option names raise `EncoderError`.

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

A package can ship a codec by publishing it as entry points: the codec for
`get_codec` and `decode`, and its program for `qccompute.compute`, which then finds it
by name in every process with no registration call:

```toml
[project.entry-points."qccodec.codecs"]
myprog = "mypackage.codec:codec"

[project.entry-points."qccompute.programs"]
myprog = "mypackage.codec:program"   # program = qccodec.CodecProgram(codec)
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

[CONTRIBUTING.md](https://github.com/atomsforhumanity/qccodec/blob/next/CONTRIBUTING.md) walks through adding a program to qccodec step by step, including the example scripts every program needs.
