# Pretrained numerical validation: execution results

**AI-generated report and accompanying code changes — October 26, 2026.**

Four pretrained models now pass. Whisper's loading and architecture failures
are fixed, but its strict numerical check still fails. The expected tensors
and the `1e-4` maximum absolute error limit have not changed.

Validation uses the existing reference fixtures, EXLA's host backend, and the
locked Nx 1.0.0 / Axon 0.9.0 / EXLA 1.0.0 dependencies. The shared checkout had
older installed dependencies, so the locked packages were fetched and built in
separate `/tmp/edifice-pretrained-deps` and `/tmp/edifice-pretrained-build`
directories. The lock file and shared dependency installation were not changed.

```sh
devenv shell -- mix test test/edifice/pretrained/numerical_validation_test.exs --include external
devenv shell -- mix test test/edifice/pretrained/architecture_numerical_test.exs --include external
```

Pretrained suite: **5 tests, 1 failure, no skips**.

| Model | Result | Remaining work |
|---|---|---|
| ResNet-50 | Pass at existing `1e-4` absolute tolerance | Keep regression coverage |
| ViT-base | Pass at `1e-4` | Correct patch-weight flattening, separate attention heads, distinct learned position vectors, checkpoint normalization epsilon, and no extra CLS/position biases |
| Whisper-base encoder | Maximum error `2.593994e-4`; mean `1.338897e-5` | Trace remaining floating-point differences; this model is not numerically certified at `1e-4` |
| DETR | Class logits and boxes pass at `1e-4` | Correct positional encoding, add positions before Q/K projection, use symmetric backbone padding, and load query vectors into the parameter actually used by the model |
| ConvNeXt | Pass at `1e-4`; skip removed | Match checkpoint normalization epsilon; use the pinned Safetensors conversion below |

ConvNeXt uses the model repository's
[conversion PR](https://huggingface.co/facebook/convnext-tiny-224/discussions/4),
pinned to `754ee4efa2810dce51a80eff8b21737c8da9a92f`. Passing the original
fixture verifies this conversion against the previously generated outputs.
DETR's positional and projection order follows the
[reference implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/detr/modeling_detr.py).

Whisper now converts convolution kernels correctly, pads the stride-two
convolution symmetrically, interprets the configured mel-input length correctly,
loads the exact frozen position table, and omits the nonexistent encoder key
bias. Its encoder can predict directly from the loaded model state. Recomputing
the position table had amplified the output error to roughly 0.048; loading
the stored values removes that discrepancy.

## Remaining Whisper investigation

The [Rust reference runner](../../tools/whisper_reference/README.md) links to
LibTorch 2.10.0 through `tch` and reproduces the existing fixture exactly with
fused attention. Ordinary LibTorch attention differs by at most `9.4414e-5`.
The runner saves intermediate tensors; an Elixir comparison script locates the
first differences without regenerating or replacing the expected fixture.

Against that ordinary-attention reference, convolution activations differ by
about `3.6e-6` and `3.3e-6`. Differences grow through later encoder blocks. The
sixth block's normalized FFN input differs by up to `7.3e-4`, and its residual
output by `0.00102`. The final normalization reduces the discrepancy again.
These measurements suggest accumulated floating-point differences, but do not
establish that every remaining difference is harmless.

Feeding identical reference inputs into each normalization reduces the local
maximum errors to `2.4e-6`–`3.1e-5`. The late normalization therefore amplifies
differences already present in its input; its own arithmetic error is much
smaller than the accumulated error.

Returning intermediate tensors changes XLA fusion and rounding: that diagnostic
has final maximum error `1.96e-4` against ordinary LibTorch attention. The
acceptance test still returns only the final output and compares against the
original fused-attention fixture. Higher-precision execution is also available
as a diagnostic; it is not enabled by default and does not satisfy the strict
limit either. The next step is to compare the attention and FFN operations
using identical reference inputs, then identify which computation needs more
accurate accumulation without imposing whole-model f64 inference.

No decoder parity claim is made for Whisper. No numerical tolerance was raised,
no failing backend was skipped, and no new runtime LibTorch dependency was added.

The random-weight suite passes **8 tests**: MinGRU, DeepResLSTM, GAT, GQA,
DeltaNet forward and gradients, Mamba, and DiT. These small fixtures do not certify
all variants or pretrained checkpoint compatibility.

Final targeted run on locked dependencies: **151 tests, 1 failure, 2 existing
exclusions**. The only failure is Whisper's strict external numerical test.
The two exclusions are existing slow architecture tests. Focused regressions,
model shape tests, and all eight random-weight reference tests pass.
Raw test and diagnostic output is preserved under
[`bench/results/pretrained_2026-10-26`](../../bench/results/pretrained_2026-10-26).

The tests now pass `Axon.ModelState` directly. ResNet and DETR retain initialization
and merging for missing zero convolution biases. Focused regressions cover patch
layout, independent position vectors, separate attention heads, rectangular DETR
positions, query-vector loading, and Whisper convolution layout.

To reproduce the isolated CPU build in this CUDA-equipped development shell:

```sh
export MIX_DEPS_PATH=/tmp/edifice-pretrained-deps
export MIX_BUILD_PATH=/tmp/edifice-pretrained-build
export XLA_TARGET=cpu
export EXLA_CPU_ONLY=1
mix deps.get
mix test test/edifice/pretrained/numerical_validation_test.exs --include external
```

`EXLA_CPU_ONLY=1` is needed as well as `XLA_TARGET=cpu` when `nvcc` is on PATH.
Otherwise EXLA can compile CUDA support into a CPU build and fail to load it.

Do not loosen tolerances to mask these failures. See the
[architecture audit](../architecture_completion_audit.md) for bot-focused priorities.

**AI-generated poem**

> Four doors now open, one stays tight;
> We keep its measured fault in sight.
> Each small comparison lights the way;
> No softened test can make it day.
