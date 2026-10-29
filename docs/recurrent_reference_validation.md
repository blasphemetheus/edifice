# Recurrent reference validation

**AI-generated code and report.** This pass separates recurrence agreement from
whole-architecture compatibility. A passing recurrence test does not certify
the surrounding projections, normalization, checkpoint loader, or native kernel.

## Results — October 27, 2026

**14 new reference tests pass.** The combined reference, existing random-weight,
Mamba/SSD model, and new-recurrence regression run passes **52 tests**. Execution
covers BinaryBackend and compiled EXLA host checks; no new native CUDA or GPU
certification is claimed. Another GPU job was active, so tests ran in a bare VM
using existing dependency beams, without rebuilding shared native libraries.

The first run failed two checks because `Common.selective_scan_fallback/5`
ignored the supplied A matrix and substituted the builder's fixed diagonal.
Maximum output error was about **0.0197975**. The fallback now uses A; output
and gradient checks pass without changing tolerances. This also restores
portable gradients with respect to A. Fixed-A model equations are unchanged.
The native custom backward still omits A gradients and remains appropriate only
for fixed-A training; its limitation is now explicit in the API documentation.

The earlier eight random-weight cross-framework cases continue to pass. They
remain useful full-model regressions, but do not upgrade these recurrence checks
into canonical Mamba/Mamba-2 checkpoint certification. That would require a
separate compatible model implementation or explicit variant option.

Raw before/after logs are in
[`bench/results/recurrent_reference_2026-10-27/`](../bench/results/recurrent_reference_2026-10-27/).

## Reference provenance

The Rust binary `recurrent-reference` in `tools/whisper_reference/` reuses the
existing pinned `tch = 0.23.0` dependency. It calls LibTorch 2.10.0 directly;
it does not launch Python. Fixtures use deterministic sine-valued inputs and
weights rather than depending on a random-number generator's version.
Reference calculations and gradients use f64. Elixir tests evaluate f32.

| Case | Independent reference | Scope |
| --- | --- | --- |
| GRU | Actual LibTorch/ATen GRU operator | One-layer sequence, nonzero initial state, all six input/parameter gradients; batches 1/2, lengths 1/9, hidden sizes 3/4 |
| SSD | Rust port of the authors' four-stage chunk decomposition | Production MambaSSD scan node, chunks 3/5/16, inference and training paths, full state sequence and input gradients |
| Mamba | Rust port of real-valued `selective_scan_ref` equations | Arbitrary channel/state decay matrix; fixed-A carried-state case with nonzero initial memory; 37-token parallel scan; dt inside and outside clamp bounds |
| MinGRU | Closed-form affine recurrence using a lower-triangular decay matrix | Signed candidates, saturated gate logits, forward and input gradients; positive-candidate variant saved separately to expose the semantic difference |

Mamba/SSD source revision:
[`e9594ce1c732d97440f0332fdc43170a2294dbfa`](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa).
Relevant files:
[SSD](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/ssd_minimal.py),
[selective scan](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/selective_scan_interface.py),
[Mamba-2 block](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba2_simple.py).
The SSD Rust code is a translation of that decomposition, not a call into the
authors' installed package. Selective-scan adaptation uses batch/time/channel
layout, f64 arithmetic, explicit initial memory, and Edifice's caller-side dt
clamp. Upstream does not impose that clamp. Neither port executes the upstream
CUDA/Triton implementation.

The independent positive MinGRU convention comes from
[`cheind/mingru` at `aa35d421f6d84cd6abb8850dc602df4748780ceb`](https://github.com/cheind/mingru/blob/aa35d421f6d84cd6abb8850dc602df4748780ceb/mingru/functional.py).
It is a separately maintained implementation, not claimed to be the paper
authors' official package. Upstream license texts accompany the Rust source.

All gradients differentiate a weighted sum of every output timestep. The
weights are stored in each fixture. State outputs are compared separately;
the loss does not add a separate final-state term. GRU checks both the actual
Edifice carried backbone and `step/3`, including recurrent candidate bias, plus
the portable `FusedScan.gru_scan` entry point. Native fused dispatch is forced
off in this suite so that these results cannot be mistaken for native coverage.

Forward/state bounds are `atol=1e-5, rtol=1e-4`; gradient bounds are
`atol=2e-5, rtol=1e-4`. These bounds were set before the first run and were not
changed to resolve the selective-scan failure.

## Architecture limits found by the audit

**MambaSSD is not canonical Mamba-2.** It retains Edifice's Mamba-style projections
and per-state diagonal decay. Canonical Mamba-2 has scalar decay per head, a joint
x/B/C convolution, learned A/D, and gated RMS normalization. For the algorithm
comparison, each Edifice `(channel, state)` coordinate is represented as a scalar
SSD head with unit B/C and the discretized input as X. This is an exact diagonal
recurrence embedding, not a mapping of a Mamba-2 checkpoint. The optional
`structured_mask: true` modification is outside this reference comparison.

**Mamba's builder is a constrained variant.** It fixes A to `-1..-N`, clips dt,
omits learned D, and has different projection biases/normalization. The generic
selective-scan API nevertheless accepts arbitrary A and must honor it.

**MinGRU has signed linear candidates.** The independent log-space implementation
applies `g(x) = x + 0.5` for nonnegative x and `sigmoid(x)` otherwise. Both retain
an affine recurrence, but they are different models. Switching the candidate
activation would change existing trained checkpoints; this validation does not
make that change.

## Reproduce

Use a LibTorch 2.10.0 installation with `include/` and `lib/`. A newer installed
torch directory is not interchangeable with the pinned binding. This pass uses
the official CPU archive:
`https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip`.

Inside Edifice's `devenv shell`:

```sh
export LIBTORCH=/tmp/libtorch-recurrent-2.10/libtorch
export LIBTORCH_CXX11_ABI=1
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
cargo run --locked --manifest-path tools/whisper_reference/Cargo.toml \
  --bin recurrent-reference -- test/fixtures/numerical/recurrent_reference.safetensors

mix test test/edifice/pretrained/recurrent_reference_test.exs --include exla_only
```

The committed fixture makes ordinary tests independent of LibTorch or network
access. On this machine, prefix Mix commands with
`MIX_DEPS_PATH=/tmp/edifice-upstream-deps MIX_BUILD_PATH=/tmp/edifice-upstream-build`
to use the existing upstream-pinned dependencies instead of the stale default
dependency directory. EXLA checks select the host client by default; use
`EDIFICE_TEST_CLIENT=cuda` for the CUDA run when GPU resources are available.

For a bare-VM run that does not rebuild dependencies:

```sh
devenv shell -- elixir -pa '/tmp/edifice-upstream-build/lib/*/ebin' -e '
  Application.ensure_all_started(:axon)
  Application.ensure_all_started(:exla)
  Code.compile_file("lib/edifice/ssm/common.ex")
  Code.require_file("test/test_helper.exs")
  ExUnit.configure(include: [:exla_only])
  Code.require_file("test/edifice/pretrained/recurrent_reference_test.exs")'
```

SHA-256 provenance:

| Artifact | SHA-256 |
| --- | --- |
| LibTorch CPU archive | `c5bf8efda9224a2d971b19d1ef6cf3ba6fee8ab53e69c49427db003d1d300496` |
| Recurrent fixture | `0969f568199cbfdd5182369175fe8b8cd5c184a3879424d712b3bd94d87f07ae` |
| Upstream `ssd_minimal.py` | `6442a9fe25a6792258df657392ab1fab4dd873ee777d9dd5783cc76030218e07` |
| Upstream `selective_scan_interface.py` | `a570f4f15a2eebabf62024a9ffb0d1dde04280300405e1632f43809f448bfb1a` |
| Upstream MinGRU `functional.py` | `bc740f5bc1fd67a9b9bb2945ae7f26986760eadcdd03fa14f6e086b33b6c9e03` |

**AI-generated poem**

> A familiar name is not a proof;
> Compare the sums beneath the roof.
