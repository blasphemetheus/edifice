# Canonical Mamba-2

**AI-generated implementation and report.**

Edifice now provides `:mamba2`, a separate implementation of the standard
Mamba-2 mixer. Existing `:mamba_ssd` models keep their original equations and
checkpoint format. They cannot be renamed to `:mamba2` or loaded into it.

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

model = Edifice.build(:mamba2, opts)
{init, predict} = Axon.build(model)
inputs = Nx.broadcast(0.0, {1, 8, 287})
params = init.(inputs, Axon.ModelState.empty())
output = predict.(params, inputs)

state = Edifice.init_state(:mamba2, params, opts ++ [batch_size: 1])
frame = Nx.take(inputs, 0, axis: 1)
{output, state} = Edifice.step(:mamba2, params, state, frame)
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

Nine tests cover four full-mixer reference cases: lengths 1/5/7/9, batch 2,
groups 1/2, convolution widths 1/3/4, biases on/off, and zero/nonzero carried
memory. Nontrivial A, D and norm weights prevent their omission from passing.
The time steps exceed 0.1, so accidentally retaining the older clamp also fails.
Tests compare every output, final recurrent/convolution memory, and gradients
of all parameters, input frames and initial memory. The loss is a weighted sum
of sequence outputs; it does not add a separate final-state term.

The combined Mamba-2, existing SSD, recurrent reference and compiled replay run
passes **42 tests**. Validation used a bare VM with existing dependency beams
to avoid rebuilding shared native libraries during other work.

The reference generator is Rust using `tch = 0.23.0` and LibTorch 2.10 CPU.
It ports the pinned upstream block equations and uses LibTorch's convolution,
linear algebra and automatic differentiation. It is **not** a run of the
upstream Python package or fused CUDA kernels. It uses f64 throughout, including
A, to provide a higher precision numerical comparison. Edifice checks f32 with
output/state tolerances `atol=2e-6, rtol=2e-5` and gradient tolerances
`atol=1e-5, rtol=2e-4`. BinaryBackend and compiled EXLA host are covered.

Additional checks cover the Axon block with converted weights, a two-layer
backbone with input projection, snapshot restoration, registry dispatch,
unsupported options, and compiled chunk replay. No pretrained language model,
reduced precision, GPU kernel, or ExPhil policy migration is certified here.

Sequence execution uses whole-sequence projection/convolution followed by a
compiled sequential recurrence. It implements the canonical math but does not
have the upstream fused SSD training algorithm's throughput. Stepping carries
constant-size memory and does work independent of history length. GPU timings
and a canonical ExPhil adapter remain separate work.

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
