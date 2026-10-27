# Whisper encoder reference

**AI-generated code and documentation.**

This small Rust runner reconstructs the Whisper-base encoder with LibTorch.
It reads the existing checkpoint and input fixture, compares the final output,
and saves intermediate tensors for comparison with Elixir. It supports the
base model's six layers, 512 channels, eight heads, and 3,000 input frames.
It is a diagnostic tool, not an Edifice runtime dependency.

The `tch` dependency is pinned to 0.23.0, which uses LibTorch 2.10.0. Set
`LIBTORCH` to an installation containing `include/` and `lib/`. The existing
development environment has one at the path below. The build links directly
to those native libraries.

Inside `devenv shell`, from the Edifice root:

```sh
export LIBTORCH="$PWD/.devenv/state/venv/lib/python3.12/site-packages/torch"
export LIBTORCH_CXX11_ABI=1
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
cargo build --locked --manifest-path tools/whisper_reference/Cargo.toml
tools/whisper_reference/target/debug/whisper-reference \
  "$HOME/.cache/edifice/openai/whisper-base/model.safetensors" \
  test/fixtures/numerical/whisper_encoder_reference.safetensors \
  /tmp/whisper-sdpa.safetensors sdpa
mix run scripts/compare_whisper_layers.exs /tmp/whisper-sdpa.safetensors
```

Use `eager` instead of `sdpa` for ordinary matrix-product attention. With the
current fixture, `sdpa` reproduces the saved output exactly; `eager` differs
by at most 0.0000944. Both use the same model equations and weights. This
separates attention implementation differences from model loading errors.

`scripts/compare_whisper_precision.exs` compares ordinary and higher-precision
Axon execution. The shared softmax helper still computes probabilities in f32,
so its f64 case is a diagnostic, not an all-f64 reference calculation.

The layer comparison exposes intermediate nodes, which can change XLA fusion
and rounding. Its final error need not equal a prediction that returns only
the final output. Keep the original numerical test as the acceptance check.

The runner also saves attention outputs and the two dense projections around
each feed-forward activation. The Elixir script feeds the saved reference inputs
into each operation separately. Regenerate the reference file to get these
additional comparisons; older files still support the earlier comparisons.

On the CPU with Nx 1.0.0 and Axon 0.9.0, the isolated comparisons against
LibTorch eager attention found these largest absolute differences:

| Operation | Largest difference across six blocks |
| --- | ---: |
| Attention including its projections | 0.00001645 |
| First feed-forward dense layer | 0.00011444 |
| GELU activation | 0.00001526 |
| Second feed-forward dense layer | 0.00001526 |

The largest isolated dense difference occurs in block six. This calculation
combines 512 input values per output. Different accumulation orders can round
differently even with identical inputs and weights. Together with the earlier
normalization checks, this supports accumulated arithmetic differences as a
working explanation; it does not prove the whole discrepancy is harmless or
establish transcription accuracy. The original final-output test still fails
at roughly 0.0002594 against its unchanged 0.0001 limit. No production math or
test tolerance was changed by this diagnostic extension.

The full isolated run is recorded in
`bench/results/pretrained_2026-10-26/whisper_isolated_operations.log`.

### Dense precision experiment

Run `mix run scripts/compare_whisper_dense_precision.exs /tmp/whisper-eager.safetensors`
after regenerating the eager reference with the current Rust runner. The script
first compares isolated feed-forward dense operations, then runs five complete
encoder variants against the original saved output. It uses f64 arithmetic
inside selected dense layers, including their bias addition, and rounds each
output back to f32. Stored checkpoint weights remain f32. This changes both
product and accumulation precision; it is not an accumulation-only kernel.
Production code and the original acceptance threshold remain unchanged.

With Nx 1.0.0 and Axon 0.9.0 on EXLA CPU, the last block's isolated first dense
error fell from 0.00011444 to 0.00003815. Complete encoder results were:

| Dense layers using f64 internally | Maximum final error | Mean final error | Values at or above 0.0001 |
| --- | ---: | ---: | ---: |
| None (baseline) | 0.00025940 | 0.0000133890 | 115 |
| Last block's first feed-forward projection | 0.00026131 | 0.0000133852 | 110 |
| First feed-forward projection in all six blocks | 0.00020790 | 0.0000133703 | 110 |
| Both feed-forward projections in all six blocks | 0.00020123 | 0.0000134081 | 148 |
| All 36 dense layers, including attention projections | 0.00023556 | 0.0000133872 | 147 |

All five variants fail the unchanged 0.0001 maximum-error requirement. More
accurate dense arithmetic reduces one local discrepancy, but does not reliably
improve agreement across the complete output. Even the variant with the lowest
maximum error has more values exceeding the limit than the baseline. The
reference itself uses f32 arithmetic, so greater precision does not guarantee
closer agreement with it. These results do not justify a production f64 change.
They also do not establish the effect on transcription accuracy.

Full results: `bench/results/pretrained_2026-10-26/whisper_dense_precision.log`.

### Convolution, normalization, and reference replacement

Run `mix run scripts/compare_whisper_early_precision.exs /tmp/whisper-sdpa.safetensors`.
This script requires the Rust SDPA reference and checks that its final output
exactly matches the original fixture. It tests selected operations with f64
arithmetic and f32 outputs, then separately replaces intermediate outputs with
exact reference tensors. Both experiments keep checkpoint parameters unchanged.

Using Nx 1.0.0 and Axon 0.9.0 on EXLA CPU:

| Operations using f64 internally | Maximum final error | Mean final error |
| --- | ---: | ---: |
| None | 0.00025940 | 0.0000133890 |
| First convolution | 0.00018978 | 0.0000133655 |
| Second convolution | 0.00025177 | 0.0000133825 |
| Both convolutions | 0.00018978 | 0.0000133615 |
| Both convolutions and their GELU activations | 0.00019169 | 0.0000133471 |
| Position addition | 0.00025940 | 0.0000133890 |
| First normalization | 0.00019455 | 0.0000133542 |
| Final normalization | 0.00026131 | 0.0000133875 |
| All 13 normalizations | 0.00022078 | 0.0000133378 |
| Both convolutions and all normalizations | 0.00017929 | 0.0000133354 |

All variants still exceed 0.0001. On identical reference inputs, higher
convolution precision reduces the first convolution+GELU maximum difference
from 0.000003576 to 0.000001431, and the second from 0.000003576 to 0.000002384.
The GELU operation stays f32 in these isolated comparisons. Every isolated
normalization is below 0.0001 in both precisions. These local improvements do
not account for the complete encoder discrepancy.

Replacing intermediate outputs gives a more specific lead. Each row below
starts the remaining f32 network from an exact reference tensor:

| Reference output supplied after | Maximum final error | Mean final error |
| --- | ---: | ---: |
| First convolution+GELU | 0.00020218 | 0.0000133696 |
| Second convolution+GELU | 0.00019073 | 0.0000133587 |
| Position addition | 0.00019073 | 0.0000133587 |
| Block 1 | 0.00020599 | 0.0000133322 |
| Block 2 | 0.00020123 | 0.0000133136 |
| Block 3 | 0.00019550 | 0.0000132558 |
| Block 4 | 0.00020790 | 0.0000130375 |
| Block 5 attention and residual addition | 0.00020218 | 0.0000130385 |
| Block 5 feed-forward section and residual addition | 0.00005341 | 0.0000061788 |
| Block 6 | 0.000007629 | 0.0000000675 |

The last two replacements leave no final values at or above 0.0001. Exact
inputs before block five's feed-forward section still produce a failing final
output; exact outputs after that section bring the remaining network below the
limit. That section is now the strongest target for further isolation: its
normalization, first dense layer, GELU, second dense layer, and residual addition.
This does not prove which operation is responsible, or establish transcription
accuracy. Replacing graph nodes can also change compiler fusion, so these are
diagnostic interventions, not replacements for the original acceptance test.

No production implementation or test tolerance changed. Full results:
`bench/results/pretrained_2026-10-26/whisper_early_precision.log`.

### Block five operation isolation

Run `mix run scripts/compare_whisper_block5.exs /tmp/whisper-sdpa.safetensors`.
This keeps the block-five residual input fixed at its reference value and moves
the reference boundary through the feed-forward branch. Each replacement uses
a distinct graph input, so replacing the branch never overwrites the residual
input. The script separately compares each operation with exact reference inputs.

| Exact reference supplied through | Maximum final error | Mean final error |
| --- | ---: | ---: |
| Section input | 0.00020218 | 0.0000130385 |
| Normalization | 0.00019932 | 0.0000130357 |
| First dense layer | 0.00020504 | 0.0000130255 |
| GELU | 0.00005150 | 0.0000061576 |
| Second dense layer | 0.00005341 | 0.0000061788 |
| Residual addition | 0.00005341 | 0.0000061788 |

The sharp improvement occurs at GELU. Its isolated maximum difference is only
0.000007629, but later operations amplify it. The isolated residual addition
matches exactly. A concrete GELU example is input 59.02690505981445: the
reference returns the same value, while the ordinary Axon/EXLA calculation
returns 59.02689743041992. At such a large positive input GELU should round to
the input in f32.

Two independent arithmetic experiments support this finding. One evaluates
selected GELUs in f64 and rounds their outputs to f32. The other uses the
equivalent expression `0.5 * x * erfc(-x / sqrt(2))` in f32, instead of Axon's
`x * (1 + erf(x / sqrt(2))) / 2`. Both reduce the isolated block-five GELU maximum
difference to 0.000001192 and its mean from 0.000000927 to about 0.000000018.
These are diagnostic model copies; no production activation was changed.

Complete encoder results, without any reference replacements:

| GELU experiment | Maximum error | Mean error | Values at or above 0.0001 |
| --- | ---: | ---: | ---: |
| Baseline | 0.00025940 | 0.0000133890 | 115 |
| Block five f64 | 0.00024033 | 0.0000069716 | 8 |
| All six feed-forward GELUs f64 | 0.00020027 | 0.0000019297 | 7 |
| Block five f32 erfc expression | 0.00023842 | 0.0000069810 | 8 |
| All six feed-forward GELUs f32 erfc expression | 0.00024605 | 0.0000019982 | 10 |

GELU evaluation is therefore a substantial contributor, not the entire failure.
All complete variants still fail the original maximum-error limit. With exact
block-five entry and all six feed-forward GELUs changed, both alternatives give
a final maximum error of 0.00001335. The remaining full-run outliers require
further investigation upstream of that entry. The experiments do not distinguish
the error-function kernel from compiler fusion or other arithmetic ordering;
that needs a smaller Axon/Nx/EXLA reproduction. They do not measure transcript
accuracy or establish GPU behavior. The GPU was occupied, so these runs used
the existing isolated CPU build and did not rebuild native libraries.

Full results: `bench/results/pretrained_2026-10-26/whisper_block5.log`.

### Small erf reproduction and remaining outliers

`mix run scripts/reproduce_gelu_cpu.exs` needs no checkpoint. It compares the
ordinary GELU formula, the erfc formula, standalone erf, and separate GELU steps
with host transfers preventing cross-step fusion. Expected values use Erlang's
f64 math functions on the actual rounded f32 inputs, then round to f32.
Both a short vector and a tensor with 2,048 columns reproduce the discrepancy.
Setting `XLA_FLAGS=--xla_cpu_enable_fast_math=false` before starting the VM does
not change the results. When using devenv, set this variable after `devenv shell`
with `env` so the shell configuration does not override it.

The error is present in standalone erf: for large positive arguments, this
CPU build returns 0.9999998211860657 rather than 1.0. Host transfers between
erf and the remaining GELU arithmetic produce the same GELU error as the fused
calculation. Thus fusion across the GELU steps is not necessary to reproduce it.

This matches a known upstream bug. The installed `xla` package is 0.10.0; its
Makefile pins OpenXLA revision `bb760b047bdbfeff962f0366ad5cc782c98657e0`.
At that revision, the scalar erf lowering clamps its argument to approximately
[-3.743921, 3.743921], then evaluates a rational approximation. It never selects
an exact +1 or -1 for larger original arguments. The generated LLVM for our
small reproduction contains the same clamp, polynomial arithmetic, and division
without that final selection. The small endpoint error therefore persists even
for very large inputs, where it should round away.

Sources: [pinned implementation](https://github.com/openxla/xla/blob/bb760b047bdbfeff962f0366ad5cc782c98657e0/xla/codegen/emitters/transforms/expand_float_ops.cc#L57),
[upstream fix merged April 24, 2026](https://github.com/openxla/xla/pull/41202).
This is evidence for updating/backporting the underlying XLA fix, rather than
treating the observed error as unavoidable f32 rounding. An updated binary still
needs validation; no XLA rebuild or dependency update was performed here.

To inspect generated code, run the reproduction in a fresh VM with
`XLA_FLAGS="--xla_dump_to=/tmp/edifice-gelu-hlo --xla_dump_hlo_as_text=true"`.
The saved `gelu_erf.before_optimizations.txt` and `gelu_erf.ir-no-opt.ll` under
`bench/results/pretrained_2026-10-26/` capture the erf operation and its compiled
implementation. `gelu_cpu.log` and `gelu_cpu_strict.log` record both flag variants.

`mix run scripts/trace_whisper_gelu_outliers.exs /tmp/whisper-sdpa.safetensors`
continues the network investigation using the diagnostic erfc formula. It also
changes the two convolution-stem GELUs, which earlier feed-forward-only runs
left unchanged. The full encoder maximum drops to 0.00013065, with mean error
0.000001895 and seven values still at or above 0.0001. The largest is at frame
890, channel 323: -15.34093952 versus reference -15.34107018.

| Exact reference supplied through, with all GELUs using erfc | Maximum final error | Values at or above 0.0001 |
| --- | ---: | ---: |
| No replacement | 0.00013065 | 7 |
| First convolution+GELU | 0.00011206 | 4 |
| Second convolution+GELU / position addition | 0.00015354 | 6 |
| Block 1 attention | 0.00011921 | 6 |
| Block 1 feed-forward section | 0.00013351 | 1 |
| Block 2 attention | 0.00008965 | 0 |
| Block 2 feed-forward section | 0.00007248 | 0 |
| Block 3 feed-forward section | 0.00003815 | 0 |
| Block 5 attention | 0.00001335 | 0 |

Every tested later boundary also remains below the limit. The remaining outliers
therefore point to differences accumulated through the early stem and first two
attention blocks, rather than solely to block five. This is not proof of a new
architecture bug, and replacement boundaries can change compilation. The next
priority is validating an XLA build with the upstream erf fix, then repeating
the original full-encoder test before pursuing smaller arithmetic differences.
No production code, dependency, or tolerance was changed. Full trace:
`bench/results/pretrained_2026-10-26/whisper_gelu_outliers.log`.

### Prebuilt XLA 0.11 validation

The fix can be tested without building XLA from source. The official
[XLA 0.11.0 release](https://github.com/elixir-nx/xla/releases/tag/v0.11.0) provides
a 161,993,015-byte x86_64 Linux CPU archive, already cached on this machine.
Its SHA-256 matches the release asset digest:
`f58a65af94e7058a82af7783b657dd4c0b29dc710cb5b8e8e6a0c1dc8f5d3f39`.
The release pins OpenXLA `91888df6ce85102c30220e41d952065925e10886`, whose scalar
erf implementation includes the saturation selection.

Released EXLA 1.0.0 cannot compile unchanged against the new headers. This
experiment applies only the native `c_src` changes from upstream Nx commit
[`171dcad658ae06f726aaacf785df2989eca047e0`](https://github.com/elixir-nx/nx/commit/171dcad658ae06f726aaacf785df2989eca047e0)
to an isolated dependency copy. These adapt device IDs and module ownership
to the new API and update a test include. Clang 21 is used because the new
headers fail to compile with the available GCC. No other upstream changes are
included; this is a compatibility experiment, not validation of a full upgrade.

Dependencies live at `/tmp/edifice-xla011-deps`, with build output at
`/tmp/edifice-xla011-build`. The copied Hex `xla` application still reports
0.10.0: `XLA_ARCHIVE_PATH` explicitly substitutes the official 0.11.0 native
archive. Edifice's lockfile, normal dependency directory, and active GPU job
were not changed. Four BEAM schedulers limited the native make job count to two.
The expensive XLA compiler build was avoided; only EXLA's bridge was rebuilt.

To reproduce from the existing isolated baseline dependencies, first copy them
with `cp -a --reflink=auto /tmp/edifice-pretrained-deps /tmp/edifice-xla011-deps`
to a fresh destination. Download the linked upstream commit's `.patch` and apply
it from that new `exla` directory with
`git apply -p2 --include='c_src/*' /tmp/edifice-exla011-upstream.patch`.
Inside Edifice's `devenv shell`, set:

```sh
export MIX_DEPS_PATH=/tmp/edifice-xla011-deps
export MIX_BUILD_PATH=/tmp/edifice-xla011-build
export MIX_ENV=test
export XLA_TARGET=cpu
export EXLA_CPU_ONLY=1
export ERL_FLAGS='+S 4:4'
export XLA_ARCHIVE_PATH="$HOME/.cache/xla/0.11.0/download/xla_extension-0.11.0-x86_64-linux-gnu-cpu.tar.gz"
export CXX=/nix/store/7p4sqkgph7ww7v26nssmhapsd9240l25-clang-wrapper-21.1.8/bin/clang++
mix deps.compile
EXLA_FORCE_REBUILD=true mix deps.compile exla --force
mix run scripts/reproduce_gelu_cpu.exs
mix test test/edifice/pretrained/numerical_validation_test.exs --include external
```

Use an available Clang path on other machines. The forced bridge rebuild matters:
copied object files from XLA 0.10 may otherwise be relinked, producing undefined
symbols at load time. Do not run this rebuild against a shared dependency tree
used by a live process.

Results: the small erf reproduction now returns exactly ±1 at large arguments;
GELU at +59.026905 returns the input exactly. The original pretrained test file
reports **four passes and one failure**, with Whisper still failing its unchanged
0.0001 maximum-error requirement. Whisper's maximum is 0.00017929 and mean is
0.0000019369, versus 0.00025940 and 0.0000133890 in the earlier 0.10 run. This
confirms the erf fix while leaving the remaining outliers unresolved. ViT,
ConvNeXt, ResNet, and DETR continue to pass their reference comparisons.

The practical recommendation is to adopt the upstream EXLA/XLA compatibility
update as a separately tested dependency upgrade, without a full XLA source
build or a permanent Edifice GELU workaround. Continue outlier diagnosis using
this fixed CPU build; GPU and wider upgrade compatibility remain untested.
Saved evidence under `bench/results/pretrained_2026-10-26/`:
`gelu_cpu_xla011.log`, `pretrained_test_xla011.log`, and
`xla011_bridge_build.log`.

### Adopting the upstream dependency update

Edifice now pins both Nx and EXLA to upstream commit
`171dcad658ae06f726aaacf785df2989eca047e0`, which integrates XLA 0.11 support.
The matching Nx revision includes the ordering information that this EXLA
revision uses for host callbacks. EXLA remains optional, and
`EDIFICE_LOCAL_NX=1` still selects the sibling development checkout.
XLA itself comes from its released Hex package and official prebuilt archives;
the diagnostic `XLA_ARCHIVE_PATH` override is no longer needed.

The development shell and CI select Clang to compile EXLA's native bridge.
After updating dependencies, rebuild that bridge with
`EXLA_FORCE_REBUILD=true mix deps.compile exla --force` if an older bridge was
already compiled. Do this only after processes using that dependency tree have
stopped, or use separate `MIX_DEPS_PATH` and `MIX_BUILD_PATH` directories.

The optional-EXLA consumer check now understands sparse Git checkouts and
isolated dependency directories. The small `GeluBackendTest` guards the exact
large-input behavior that exposed the upstream bug, without downloading model
weights. The Whisper reference test retains its original tolerance and remains
a separate outstanding numerical issue.

Validation used the actual pinned Git sources and Hex XLA 0.11.0, with no archive
override or local patches, in `/tmp/edifice-upstream-deps` and
`/tmp/edifice-upstream-build`. The new GELU regression, pretrained mapping
regressions, Whisper unit tests, recurrent replay, and sharding tests passed:
39 tests, zero failures. Sharding used two virtual CPU devices. The optional
consumer compiled and ran on BinaryBackend without EXLA. Compilation with
warnings treated as errors, formatting, and the unused-lock-entry check passed.

The five external pretrained comparisons still produce four passes and the
known Whisper failure (maximum 0.00017929, mean 0.0000019369). The test was not
relaxed or skipped. GPU validation was deferred because another GPU job was
active; this commit does not update ExPhil's checkout or dependencies.
Evidence: `upstream_dependency_tests.log`, `upstream_dependency_pretrained.log`,
and `upstream_dependency_optional.log` under `bench/results/pretrained_2026-10-26/`.

### Remaining discrepancy after the dependency update

Run `scripts/trace_whisper_gelu_outliers.exs REFERENCE native` to trace the
unchanged model on the fixed backend. It now has just three values at or above
0.0001: frame/channel pairs 1355/323, 1486/145, and 1486/323. The largest
difference is 0.00017929. Exact reference replacements do not improve the maximum
monotonically, so threshold crossings alone cannot assign blame to one operation.

The Rust runner now also saves Q, K, V, and the attention result before output
projection. Regenerate the SDPA reference, then run
`mix run scripts/isolate_whisper_early_blocks.exs REFERENCE` to examine the first
two blocks. Residual inputs stay fixed when a branch is replaced. All isolated
operations differ by less than 0.000016, and residual additions match exactly.
Selective f64 convolution, normalization, and first-block projection/attention
experiments all still fail the original maximum threshold. The f64 attention
policy experiment retains the production softmax helper's f32 internal step;
it is not the fully f64 comparison described below.

To distinguish arithmetic differences from model equations, the Rust runner
supports `eager-f64`. It casts the same checkpoint weights and input to f64,
uses f64 attention including softmax, and saves only the final encoder output:

```sh
tools/whisper_reference/target/debug/whisper-reference \
  "$HOME/.cache/edifice/openai/whisper-base/model.safetensors" \
  test/fixtures/numerical/whisper_encoder_reference.safetensors \
  /tmp/whisper-f64.safetensors eager-f64
mix run scripts/compare_whisper_f64_reference.exs /tmp/whisper-f64.safetensors
```

Use the LibTorch library path shown earlier for the Rust command. The Elixir
diagnostic makes a copy of the encoder with f64 computation throughout. It
replaces only the attention calculation with equivalent f64 matrix products
and softmax, avoiding the production helper's deliberate f32 cast. This does
not change production behavior or the original saved fixture.

| Comparison | Maximum absolute difference | Mean absolute difference |
| --- | ---: | ---: |
| Edifice/EXLA f64 versus independent LibTorch f64 | 5.43e-13 | 5.36e-15 |
| LibTorch f64 versus original LibTorch f32 fixture | 0.0001322195 | 0.0000012836 |
| Edifice/EXLA f64 versus original LibTorch f32 fixture | 0.0001322195 | 0.0000012836 |

Both f64 results exceed the original fixture's 0.0001 limit at three values.
This provides strong evidence that the remaining mismatch on this input is
accumulated f32 arithmetic, rather than different encoder equations or weight
mapping. It does not establish correctness on every input, decoder parity,
transcription quality, or GPU numerical behavior.

At this stage the original strict test was still failing. The broader experiment
below establishes separate f64 and f32 checks rather than tuning operations to
match one f32 fixture's rounding behavior.

Evidence: `whisper_native_outliers.log`, `whisper_early_blocks.log`, and
`whisper_f64_parity.log` under `bench/results/pretrained_2026-10-26/`.

### Varied-input validation and regression bounds

**AI-generated update.** `scripts/validate_whisper_inputs.exs` now compares ten
synthetic inputs through EXLA f32, LibTorch SDPA f32, EXLA f64, and LibTorch eager
f64. Each comparison covers all 768,000 output values. The inputs include the
original fixture, two constants, a scaled original, smooth time/frequency
patterns, seeded noise, an abrupt step, quiet noise, reversed time, and a pulse.
These exercise numerical behavior; they are not recorded speech or a test of
transcription quality.

| Comparison, maximum across ten inputs | Worst absolute error | Worst mean absolute error |
| --- | ---: | ---: |
| EXLA f64 / LibTorch f64 | 4.68e-12 | 8.61e-15 |
| EXLA f32 / LibTorch SDPA f32 | 0.001112 | 2.45e-6 |
| EXLA f32 / LibTorch f64 | 0.000921 | 2.76e-6 |
| LibTorch SDPA f32 / LibTorch f64 | 0.000440 | 1.60e-6 |

The first trial used `atol=1e-4, rtol=1e-5`. It failed on several inputs, even
when comparing LibTorch's own f32 and f64 results. Constant and abrupt inputs
produce larger isolated errors while the average stays small. All f64
comparisons agree closely. This strengthens the evidence for accumulated f32
rounding rather than a different encoder equation or weight mapping.

The reusable stress runner therefore checks each f32 value against
`abs(actual-reference) <= 0.002 + 0.00001 * abs(reference)` and also requires
mean absolute error at most `5e-6`. The absolute allowance is roughly twice the
largest initial error; the mean limit prevents a widespread shift from passing.
These limits were fixed after the first seven cases. Quiet noise, reversed time,
and the pulse were then added as holdouts; all passed without changing the
limits. These are empirical CPU regression bounds, not guaranteed limits for
every possible input or backend.

The original fixture's ExUnit test remains much tighter: `atol=1e-4, rtol=1e-5`
and mean error below `5e-6`. No other architecture's tolerance changed. A new
external test checks the full saved independent f64 output with `atol=1e-9`,
`rtol=1e-10`, and mean error below `1e-11`. The existing direct f32 erf/GELU
regression still guards the upstream bug; f64 agreement alone cannot do that.
The f64 diagnostic uses an equivalent attention calculation to avoid the
production helper's intentional f32 softmax cast. Production code is unchanged.

Build the Rust tool as described above, then run from the repository root:

```sh
devenv shell -- env MIX_DEPS_PATH=/tmp/edifice-upstream-deps \
  MIX_BUILD_PATH=/tmp/edifice-upstream-build MIX_ENV=test \
  XLA_TARGET=cpu EXLA_CPU_ONLY=1 ERL_FLAGS='+S 4:4' \
  mix run scripts/validate_whisper_inputs.exs
```

Those dependency/build directories are the isolated checkout used for these
measurements; use your own matching build if needed. `LIBTORCH` can select a
LibTorch installation. Its default is the existing environment's native torch
directory; the runner never starts Python. `WHISPER_VALIDATION_DIR` sets the
output directory (default `/tmp/whisper-input-validation`). The Rust tool now
accepts input-only fixtures and an optional `final-only` argument to avoid
writing large intermediate files. It still computes the entire encoder.

The committed `whisper_encoder_f64_reference.safetensors` is the runner's
`original-eager-f64.safetensors`, generated with tch 0.23.0 / LibTorch 2.10.0,
four CPU threads, and the checkpoint used by the existing fixture. SHA-256:

```text
checkpoint: 07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b
original fixture: 6a030bc4f91363d197f2d3b2b78ddc47a3303f82d1b6e4c25707b9e8e3c99081
f64 reference: f73f80ae83f28384c39f87466bcbb352a35f022c315b235a6d5e98581cba18a1
```

Logs: `whisper_varied_inputs_initial.log` records the failed initial bounds;
`whisper_varied_inputs.log` records the final ten cases, under
`bench/results/pretrained_2026-10-26/`. Measurements used the adopted Nx/EXLA
revision and XLA 0.11.0 on CPU. Another process was using the GPU; these results
make no GPU or performance claim. Real-speech features, decoder output, and
transcription quality remain outside this experiment.

Validation: all six pretrained numerical tests (including the new f64 test) and
the direct erf/GELU regression passed: seven tests, zero failures. Elixir and
Rust formatting checks passed.

References: [tch-rs](https://github.com/LaurentMazare/tch-rs),
[Whisper implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/modeling_whisper.py).

### Recorded speech

**AI-generated update.** The Rust `whisper-audio` executable now converts audio
into a Whisper-base input. It uses FFmpeg to decode mono 16 kHz signed 16-bit
audio, pads to 30 seconds, and computes the log-mel features with LibTorch and
OpenAI's published mel filter bank. Longer recordings use only their first
30 seconds; this diagnostic is not a long-form transcription implementation.
The preprocessing follows [OpenAI's audio implementation](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/audio.py).
No Python process is used.

`scripts/validate_whisper_recordings.exs` downloads three public clips and the
filter bank from pinned revisions, verifies their SHA-256 hashes, and passes
their features through the existing f32/f64 encoder comparisons. Downloads and
generated tensors stay in `/tmp` by default. The input sources and checksums
are recorded in the script. No audio is added to the repository.

| Recording | Duration | EXLA/Torch f64 max error | EXLA/Torch f32 max error | EXLA/Torch f32 mean error |
| --- | ---: | ---: | ---: | ---: |
| OpenAI Whisper test `jfk.flac` | 11 s | 2.58e-12 | 0.001746 | 1.74e-6 |
| Narsil/asr_dummy `1.flac` | 10.435 s | 1.36e-12 | 0.000544 | 1.36e-6 |
| Narsil/asr_dummy `2.flac` | 3.275 s | 1.97e-12 | 0.000811 | 1.57e-6 |

All twelve comparisons passed the bounds fixed in the synthetic-input
experiment. No tolerance was changed. The larger JFK f32 error is still well
localized; f64 agreement remains close. Full results are in
`bench/results/pretrained_2026-10-26/whisper_recorded_speech.log`.

Build both Rust executables with the existing Cargo command, put FFmpeg on PATH
(or set `FFMPEG` to its executable), then run:

```sh
devenv shell -- env MIX_DEPS_PATH=/tmp/edifice-upstream-deps \
  MIX_BUILD_PATH=/tmp/edifice-upstream-build MIX_ENV=test \
  XLA_TARGET=cpu EXLA_CPU_ONLY=1 ERL_FLAGS='+S 4:4' \
  mix run scripts/validate_whisper_recordings.exs
```

Append local recording paths to use your own audio instead of the public clips.
`WHISPER_RECORDINGS_DIR` changes the download/feature directory. `LIBTORCH` has
the same meaning as in the synthetic runner. Audio decoding and inference run
locally. The default fixture downloads and model configuration fetch still use
the network. The run above used FFmpeg 8.1.2 on CPU.

A Rust unit test verifies that silent audio maps to Whisper's log floor (-1.5),
including padding and the output shape. Both encoders receive the same feature
array, so encoder agreement does **not** independently establish frontend parity.
These three clips also do not measure transcription accuracy, accents, noisy
microphones, or long recordings. A separate frontend reference comparison and
decoder/checkpoint validation are needed before treating Edifice as a dictation
engine. In particular, the current key map skips `proj_out.weight` while the
decoder builds a separate output projection. The notebook's demonstration
decode loop is not evidence of pretrained transcript compatibility.

**AI-generated poem**

> A small difference starts the trail;
> Each layer tells a little tale.
> Keep the test and trace the seam;
> Clear evidence outlasts a gleam.
