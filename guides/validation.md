# What has been validated?

**AI-generated report and architecture-index labels — October 28, 2026.**

Edifice's architectures do not all have the same level of evidence. Many have
build, shape, or local behavior tests. A smaller set has numerical comparisons
with an independent implementation. A passing test covers its configuration,
inputs, numerical precision and backend; it does not certify every option.

## Read the index labels

| Label | What it means | What it does not establish |
|---|---|---|
| Reference (limited scope) | Independent outputs, or recurrence equations, are compared numerically for the cases listed below | Complete upstream-model compatibility or policy quality |
| Checkpoint (limited scope) | Real pretrained weights were loaded and compared with independent outputs | Every checkpoint, task, dtype or backend |
| Behavior | Specific invariants, gradients, replay or cache behavior have regression coverage | Independent paper/reference fidelity |
| Unreviewed | No evidence assessment has been recorded in this table | That the model has no tests, or that it is known to fail |

These labels are not a single ranking. A gradient comparison answers a different
question from a checkpoint-loading test. A GPU benchmark measures speed, not
accuracy. ExPhil integration and a successful optimizer update do not prove
that a model learns a useful game policy.

## Independent reference comparisons

| Architecture | Recorded evidence | Important boundary |
|---|---|---|
| Mamba-2 Full (`:mamba2_full`) | Rust/LibTorch f64 mixer fixtures; f32 outputs, carry and all input/parameter gradients; grouped, short and convolution-width-one cases; sequential/chunked parity; backbone step/replay parity; host and CUDA checks | Full-SSM mixer with supported options, not every upstream Mamba-2 variant or a pretrained language model |
| GRU | ATen-reference carried outputs and input/parameter gradients, including nonzero candidate bias; compiled checks and rollback tests | The GRU evidence does not automatically cover LSTM, xLSTM, or every parameter layout |
| Mamba (`:mamba`) | Independent scan outputs, carry and gradients; nonzero initial state; long non-power-of-two sequence; random-weight backbone fixture | Existing Edifice variant has fixed decay, no learned D skip and clamped time steps; not canonical upstream Mamba-1 |
| Legacy Mamba-2/SSD (`:mamba_ssd`) | Scan comparison with upstream SSD chunk decomposition, plus step/replay parity | Certifies the scan cases, not a canonical Mamba-2 mixer; use `:mamba2_full` for that |
| MinGRU | Random-weight backbone fixture and independent signed-recurrence outputs/gradients; prefix, carry and rollback checks | Signed-candidate variant; not the positive-candidate log-space formulation |
| DeepResLSTM, GAT, GQA, DiT | Small random-weight forward comparisons | Not pretrained compatibility or full training/reference-gradient certification |
| DeltaNet | Random-weight forward and gradient comparisons | Does not certify GatedDeltaNet or other DeltaNet variants |

Evidence and exact shapes/tolerances:
[recurrent reference tests](https://github.com/blasphemetheus/edifice/blob/main/test/edifice/pretrained/recurrent_reference_test.exs),
[recurrent limits](https://github.com/blasphemetheus/edifice/blob/main/docs/recurrent_reference_validation.md),
[Mamba-2 tests](https://github.com/blasphemetheus/edifice/blob/main/test/edifice/ssm/mamba2_test.exs),
[Mamba-2 configuration and GPU results](https://github.com/blasphemetheus/edifice/blob/main/docs/mamba2.md),
and [random-weight comparisons](https://github.com/blasphemetheus/edifice/blob/main/test/edifice/pretrained/architecture_numerical_test.exs).

## Pretrained checkpoints

| Architecture | Recorded evidence | Important boundary |
|---|---|---|
| ResNet-50, ViT-base, ConvNeXt-tiny | Loaded-checkpoint forward outputs match their reference fixtures | Specific checkpoint/configuration on EXLA host; not every model in each family |
| DETR | Loaded-checkpoint class logits and boxes match reference fixtures | Specific ResNet-50-backed checkpoint/configuration |
| Whisper-base | Encoder comparison, strict independent f64 checks, varied/recorded speech, and bounded one-file decoder token agreement | Strongest recorded results use the upstream EXLA/XLA 0.11 profile; published EXLA 1.0 has the CPU erf/GELU defect below. No streaming-dictation, long-form decoding or Whisper-training certification |

See the [checkpoint tests](https://github.com/blasphemetheus/edifice/blob/main/test/edifice/pretrained/numerical_validation_test.exs)
and [Whisper experiments and limits](https://github.com/blasphemetheus/edifice/blob/main/tools/whisper_reference/README.md).
Those results are not a claim that every checkpoint was rerun on every release
dependency profile. Tests requiring external files/downloads are tagged `external`;
the default suite does not run them.

## Local behavior checks

The LSTM and causal GatedSSM paths have prefix/step and rollback checks.
GatedDeltaNet has step/full and gradient comparisons. These compare Edifice
paths with one another, rather than an independent implementation.
[Stateful tests](https://github.com/blasphemetheus/edifice/tree/main/test/edifice/stateful)
contain the exact cases.

Decoder-only generation has trained-weight cache/full prediction comparisons,
chunked prefill, offsets, rollback, overflow and seeded sampling checks.
See [cache tests](https://github.com/blasphemetheus/edifice/blob/main/test/edifice/serving/decoder_cache_test.exs).
This does not certify arbitrary imported language-model checkpoints.

NSA and MoBA have local sparse-execution regressions; RetNet has normalization
regressions; Reservoir has spectral-radius checks. They are not independently
certified against their authors' complete implementations. See the
[architecture audit](https://github.com/blasphemetheus/edifice/blob/main/docs/architecture_completion_audit.md)
for known approximation limits. Do not extend these claims to similarly named variants.

## Dependency and backend limits for 0.3.0

The publishable baseline is Nx 1.0 and optional EXLA 1.0 from Hex. EXLA 1.0 uses
XLA 0.10. On CPU, that XLA build has a confirmed f32 `erf` saturation error which
also affects GELU. This is a backend numerical defect, not a weight-loading
failure. Do not claim the corrected Whisper results on that baseline.

`EDIFICE_UPSTREAM_NX=1` selects the previously validated Nx/EXLA commit
`171dcad658ae06f726aaacf785df2989eca047e0`, which uses XLA 0.11 and fixes that
defect. `EDIFICE_LOCAL_NX=1` takes precedence for sibling-checkout development.
Use separate dependency/build directories and a separate checkout for upstream
work: switching profiles changes `mix.lock`. A dependency package cannot impose
Git dependencies on Hex consumers; downstream applications can explicitly
override both Nx and EXLA with that same Git revision.

For a downstream application's `deps/0`, keep the other dependencies and add:

```elixir
{:nx, github: "elixir-nx/nx", sparse: "nx", override: true,
 ref: "171dcad658ae06f726aaacf785df2989eca047e0"},
{:exla, github: "elixir-nx/nx", sparse: "exla", override: true,
 ref: "171dcad658ae06f726aaacf785df2989eca047e0"}
```

In that separate checkout:

```sh
export EDIFICE_UPSTREAM_NX=1
export MIX_DEPS_PATH=/tmp/edifice-validation-upstream-deps
export MIX_BUILD_PATH=/tmp/edifice-validation-upstream-build
export XLA_TARGET=cpu EXLA_CPU_ONLY=1 CC=clang CXX=clang++
devenv shell -- mix deps.update nx exla xla
devenv shell -- mix test test/edifice/pretrained/gelu_backend_test.exs --include upstream_fix
```

The exact erf/GELU regression is retained unchanged and tagged `upstream_fix`.
It is excluded from the baseline suite because that published backend is known
to fail it. Run it explicitly to verify the fixed profile; a passing baseline
suite therefore does **not** mean the upstream defect is fixed. No numerical
tolerance was relaxed. Native CUDA libraries must be rebuilt against the
selected XLA version in an unused dependency tree; never replace a library
loaded by a running job.

Release-preparation checks on October 28 passed on the published CPU profile:
42 recurrent/reference/replay tests including compiled cases, three native-call
API checks, twelve smoke cases, the real consumer build without EXLA, and the
Hex package build. Documentation generation passed with existing hidden-module
link warnings. The backend-only saturation regression was explicitly run and
failed on XLA 0.10, confirming the limitation; the same regression and three
native-call API checks passed on the pinned XLA 0.11 profile. This was not a
fresh full-suite run, GPU certification, or pretrained-checkpoint sweep.

## Keeping the labels honest

Update this evidence record when a new reference comparison is reviewed. Then
update the explicit module lists in `scripts/update_architecture_validation.exs`
and run it with `devenv shell -- elixir scripts/update_architecture_validation.exs`.
New or unclassified models stay **Unreviewed**. The script's `--check` mode
verifies that the index matches those lists; it does not run or certify tests.

**AI-generated poem**

A shape can fit while numbers stray;  
A checked result shows one clear way.  
We mark the ground our tests have seen,  
And leave the unknown space between.
