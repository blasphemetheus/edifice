# Canonical Mamba-2

**AI-generated implementation and report.**

Edifice and ExPhil use `:mamba2_full` for the standard
Mamba-2 mixer. Existing `:mamba_ssd` models keep their original equations and
checkpoint format. They cannot be renamed to `:mamba2_full` or loaded into it.
ExPhil's existing `:mamba_2` name remains the legacy SSD variant. The original
Edifice registry name `:mamba2` was renamed to `:mamba2_full` for consistency.

The new mixer learns a decay rate for each head. It convolves the projected
input and both state projections together. Its output includes a learned skip
connection, then a SiLU gate and grouped RMS normalization. Its time steps use
softplus without the older Edifice Mamba time-step clamp.

## Use it

```elixir
opts = [
  embed_dim: 287,
  hidden_size: 256,
  num_layers: 2,
  head_dim: 64,
  state_size: 128,
  num_groups: 1,
  conv_size: 4
]

model = Edifice.build(:mamba2_full, opts)
{init, predict} = Axon.build(model)
inputs = Nx.broadcast(0.0, {1, 8, 287})
params = init.(inputs, Axon.ModelState.empty())
output = predict.(params, inputs)

state = Edifice.init_state(:mamba2_full, params, opts ++ [batch_size: 1])
frame = Nx.take(inputs, 0, axis: 1)
{output, state} = Edifice.step(:mamba2_full, params, state, frame)
```

Use `output_mode: :all` to return every timestep. The default returns the last.
The backbone adds an input projection when necessary, pre-RMS residual blocks,
and a final RMS normalization. It is a frame-vector backbone, not a language
model with embeddings and a vocabulary head.

`Edifice.SSM.Mamba2.build_block(input, opts)` builds only the canonical mixer,
without outer normalization or a residual. This is the boundary used by the
reference tests. The expanded width is `hidden_size * expand_factor`, with an
expansion factor of 2 by default. `head_dim` must divide that width;
`num_groups` must divide the number of heads. Projection bias defaults to false;
convolution bias defaults to true.

## Supported reference configuration

The target is the main
[Mamba2 module at revision e9594ce](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba2.py),
with the full expanded width assigned to the state-space model, per-head D,
and grouped RMS normalization **after** gating. The default initialization uses
A magnitudes in [1, 16], initial time steps in [0.001, 0.1], and D/norm weights
of one. Projection and convolution weights use fan-in uniform initialization.
To match upstream training with weight decay, exclude A_log, dt_bias and D
from decay in the optimizer. The model builder does not configure an optimizer.

Partial-SSM/MLP branches, per-channel D, disabled RMS normalization, normalization
before gating, time-step limits and nonzero dropout are rejected. Packed
sequences, distributed tensor parallelism and custom initialization ranges are
not implemented. Do not pass upstream constructor options directly; use the
Edifice options listed above. The validated compute type is f32.

## Weights and carried state

`Mamba2.from_reference(weights)` converts one unprefixed upstream mixer state
dict. It transposes dense and depthwise-convolution kernels and preserves
A_log, dt_bias, D, norm weights and optional biases. It does not download a
checkpoint or load an entire upstream language model. Build a matching block
and replace that block's entry in the initialized `Axon.ModelState.data` map.
Use matching bias options. Optimizer state is not converted.

For standalone use, `block_state(params, batch, state_size)` creates zero memory;
`forward(params, sequence, state)` and `block_step(params, state, frame)` return
both output and updated memory. Nonzero initial memory is supported.

Backbone state contains `h` shaped `[batch, layers, heads, head_dim, state_size]`
and `conv` shaped `[batch, layers, conv_size - 1, convolution_channels]`.
For convolution width 1, `conv` is an unused one-slot placeholder. This cache
stores the previous k-1 projected x/B/C inputs; upstream stores k slots. The
cache layouts are therefore not directly interchangeable. State snapshots,
restoration and compiled `Edifice.Stateful.jit_replay/3` work with this layout.

## Validation and limits

Thirteen tests cover four full-mixer reference cases: lengths 1/5/7/9, batch 2,
groups 1/2, convolution widths 1/3/4, biases on/off, and zero/nonzero carried
memory. Nontrivial A, D and norm weights prevent their omission from passing.
The time steps exceed 0.1, so accidentally retaining the older clamp also fails.
Tests compare every output, final recurrent/convolution memory, and gradients
of all parameters, input frames and initial memory. The loss is a weighted sum
of sequence outputs; it does not add a separate final-state term.

The original combined regression run passed 42 tests. The expanded Mamba-2
suite also checks chunk sizes 1/3/16, partial chunks, every reference gradient,
and an 80-frame GPU sequence with state size 128. Validation uses bare VMs with
existing dependency beams, avoiding shared native-library rebuilds.
The updated Edifice CPU regression set passes 32 tests, and the 13-test Mamba-2
suite passes with CUDA validation enabled. ExPhil's focused integration/config
set passes 21 tests. Its separate flag-parity suite still reports a pre-existing
missing default for `:recurrent_state`; the new `:num_groups` flag has a default.

The reference generator is Rust using `tch = 0.23.0` and LibTorch 2.10 CPU.
It ports the pinned upstream block equations and uses LibTorch's convolution,
linear algebra and automatic differentiation. It is **not** a run of the
upstream Python package or fused CUDA kernels. It uses f64 throughout, including
A, to provide a higher precision numerical comparison. Edifice checks f32 with
output/state tolerances `atol=2e-6, rtol=2e-5` and gradient tolerances
`atol=1e-5, rtol=2e-4`. BinaryBackend, compiled EXLA host, and CUDA are covered.

Additional checks cover the Axon block with converted weights, a two-layer
backbone with input projection, snapshot restoration, registry dispatch,
unsupported options, and compiled chunk replay. No pretrained language model,
reduced precision, or automatic migration of old ExPhil policies is certified.

Sequence execution defaults to chunked SSD (`chunk_size: 32`). Matrix products
handle time within each chunk; state passes between chunks. B/C remain grouped
through the matrix products instead of being duplicated per head. Stable masked
segment sums avoid overflowing the unused upper triangle. XLA compiles this Nx
graph into GPU matrix operations and fused elementwise/reduction kernels. No
upstream Triton runtime or new handwritten CUDA custom-call library is required.
Use `execution: :sequential` for the reference path. Stepping always uses the
length-one recurrence, with constant-size memory.

## ExPhil integration and training measurements

Use `--backbone mamba2_full` in ExPhil, including the Fox windowed streaming
driver. Shape settings, including `--num-groups`, survive policy export and
reload. The Agent supports carried stepping, cold-start replay, resync,
snapshots and per-bot resets. The integration test trains a real imitation step,
checks that A_log changes, exports the policy and loads it into a stateful Agent.
This does not add carried-state BPTT training to ExPhil's GRU-only BPTT pipeline.

Measurements use the shape from `coh_fox_mamba_v3s906_ep3/model_config.json`:
batch 128, 80 frames, 264 input features, hidden width 512, two layers. The
canonical state size is 128 (the legacy checkpoint used 16). These runs use
synthetic frames and newly initialized parameters, not that checkpoint's
incompatible weights. The loss is a weighted sum of the last backbone output;
controller heads, data loading and optimizer updates are excluded.

RTX 5090, f32, EXLA `precision: :highest`, 3 warmups and 10 synchronized samples:

| Execution | Forward median | Forward + backward median | Backward scratch memory |
| --- | ---: | ---: | ---: |
| Sequential reference, seed 71 | 9.196 ms | 5182.845 ms | 2.232 GB |
| Initial chunked SSD | 4.843 ms | 18.081 ms | 2.040 GB |
| Group-preserving SSD, seed 71 | 3.901 ms | 15.924 ms | 1.692 GB |

The primary reference/final comparison uses identical parameter seed 71 and
input seed 71. Earlier pilot runs used Axon's time-based parameter seed; they
measured 3.74–3.95 ms forward and 15.69–15.80 ms forward/backward for the final
implementation. The fixed-seed final executable contains 91 forward fusion
calls and 297 forward/backward fusion calls. These are compiler graph counts,
not a claim that the whole mixer is one kernel. Compilation takes 7.47 s for
forward and 12.88 s for forward/backward, separately from the warm timings.

The initial bottleneck is Nx's current reverse-mode handling of `while`: it
recomputes earlier states, taking quadratic time in sequence length. Chunked
SSD removes that loop from the training graph. Preserving shared B/C then cuts
roughly another 12–13% from forward/backward time and about 17% from scratch
memory versus the initial chunked implementation. The fixed-seed final path is
325× faster forward/backward than the sequential reference, with 24% less
scratch memory. These gains are relative to this reference
implementation, not to upstream's optimized Mamba-2 kernels.

Memory numbers come from the compiled executable's buffer assignment and exclude
arguments, outputs, optimizer state and allocator reservation. No other training
job ran during timing; desktop GPU processes remained. An isolated scan probe
is archived as well, but its fully expanded independent B/C inputs and exposed
state output differ from the model graph; its timings are not additive profile
fractions. See `bench/results/mamba2_full/` for raw measurements.

Reproduce one case per fresh GPU-enabled VM:

```sh
BENCH_CONFIG=/path/to/exphil/checkpoints/coh_fox_mamba_v3s906_ep3/model_config.json \
BENCH_EXECUTION=chunked BENCH_OUT=/tmp/mamba2_full.json \
devenv shell -- mix run bench/mamba2_training.exs
```

Use `BENCH_EXECUTION=sequential` for the old reference. `BENCH_BATCH` and
`BENCH_STATE` override the batch/state sizes. The runner separates compilation
from warm execution and reports compiler-generated fusion counts.

Run the tests in the project environment:

```sh
devenv shell -- mix test test/edifice/ssm/mamba2_test.exs --include exla_only
```

Regenerate the fixture with an isolated LibTorch 2.10 CPU installation:

```sh
export LIBTORCH=/path/to/libtorch
export LIBTORCH_CXX11_ABI=1
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
devenv shell -- cargo run --locked \
  --manifest-path tools/whisper_reference/Cargo.toml \
  --bin mamba2-reference -- test/fixtures/numerical/mamba2_reference.safetensors
```

The upstream Apache-2.0 license is preserved in
`tools/whisper_reference/licenses/mamba-Apache-2.0.txt`.

**AI-generated poem**

A frame arrives; the old state stays,  
Its learned decay weighs passing days.  
The reference sets the course to keep;  
We check each step before the leap.
