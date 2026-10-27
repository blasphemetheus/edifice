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

A useful next comparison would isolate convolution and normalization precision
in the same way, to test which earlier differences survive through the network.
Full results: `bench/results/pretrained_2026-10-26/whisper_dense_precision.log`.

References: [tch-rs](https://github.com/LaurentMazare/tch-rs),
[Whisper implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/modeling_whisper.py).

**AI-generated poem**

> A small difference starts the trail;
> Each layer tells a little tale.
> Keep the test and trace the seam;
> Clear evidence outlasts a gleam.
