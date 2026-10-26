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

References: [tch-rs](https://github.com/LaurentMazare/tch-rs),
[Whisper implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/modeling_whisper.py).

**AI-generated poem**

> A small difference starts the trail;
> Each layer tells a little tale.
> Keep the test and trace the seam;
> Clear evidence outlasts a gleam.
