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

References: [tch-rs](https://github.com/LaurentMazare/tch-rs),
[Whisper implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/whisper/modeling_whisper.py).

**AI-generated poem**

> A small difference starts the trail;
> Each layer tells a little tale.
> Keep the test and trace the seam;
> Clear evidence outlasts a gleam.
