# Pretrained numerical validation: execution results

**AI-generated release clarification, October 28:** the resolved results below
use the fixed upstream Nx/EXLA/XLA profile. The 0.3.0 release baseline uses
published EXLA 1.0 / XLA 0.10, which still has the CPU erf/GELU defect. See
[current dependency and validation limits](../../guides/validation.md).

**Current status, October 26:** the initial failure below is resolved. The
upstream Nx/EXLA/XLA update fixes the erf bug; independent f64 and varied/recorded
input checks justify a Whisper-specific f32 comparison. All six pretrained
numerical tests now pass. One-file English transcription also matches an
independent decoder's token sequence. See the [current evidence and limits](../../tools/whisper_reference/README.md).
The remainder of this file preserves the initial investigation results.

Validated October 24, 2026 using the existing PyTorch fixtures and downloaded
HuggingFace checkpoints. Test setup now explicitly selects EXLA's host backend
and compiler; setting `EXLA=1` alone did not do this.

```sh
devenv shell -- mix test test/edifice/pretrained/numerical_validation_test.exs --include external
devenv shell -- mix test test/edifice/pretrained/architecture_numerical_test.exs --include external
```

Pretrained suite: **5 tests, 3 failures, 1 skipped**. Tolerances were not changed.

| Model | Result | Remaining work |
|---|---|---|
| ResNet-50 | Pass at existing `1e-4` absolute tolerance | Keep regression coverage |
| ViT-base | Patch projection shape failure: `{1,196,3,16,16}` vs `{768}` | Convert convolution patch weights to the dense patch layout; also correct attention head splitting and positional embeddings |
| Whisper-base encoder | Convolution shape failure: `{1,3000,3}` vs `{1,1,512}` | Convert PyTorch rank-three convolution kernels to Axon layout, then rerun |
| DETR | Max absolute difference `10.1878109`, mean `2.9366438` | Compare backbone, position encoding, encoder and decoder intermediates; root cause is not isolated |
| ConvNeXt | Existing skip: checkpoint source lacks safetensors | Supply a compatible converted checkpoint and execute parity validation |

The random-weight suite passes **8 tests**: MinGRU, DeepResLSTM, GAT, GQA,
DeltaNet forward and gradients, Mamba, and DiT. These small fixtures do not certify
all variants or pretrained checkpoint compatibility.

Earlier changes corrected Axon build return order, input names, and initialization
plus checkpoint merging for missing parameters. The old claim that passing
`Axon.ModelState` is inherently unsupported is obsolete: current Axon warns that
plain parameter maps are deprecated. The tests still use those maps; that warning
is distinct from the three numerical/shape failures above.

Do not loosen tolerances to mask these failures. See the
[architecture audit](../history/architecture_completion_audit.md) for bot-focused priorities.
