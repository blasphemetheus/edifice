# Retention normalization and reservoir initialization

RetNet and RetNetV2 now normalize retention outputs independently per head
and per token. The former LayerNorm pooled all heads together. The replacement
retains the existing `*_group_norm` parameter names and channel-wise `gamma`
and `beta` shapes, but changes numerical behavior. Existing weights still load;
their predictions change. No normalization aggregates over time or batch rows.

This implements mean-centered GroupNorm as described by these Edifice models.
It is not a claim of complete checkpoint parity with current TorchScale, whose
[retention implementation](https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiscale_retention.py)
uses non-affine per-head RMSNorm. RetNetV2's documentation now accurately describes
its parallel gated variant; earlier claims of implemented chunkwise execution
and cross-chunk attention were incorrect.

Reservoir initialization now scales by the largest eigenvalue magnitude of the
actual sparse nonsymmetric matrix. The old Frobenius heuristic did not enforce
the requested radius. A portable host double-precision Hessenberg reduction and
implicit double-shift QR handle real and complex eigenvalues. The solver raises
on nonconvergence; it never silently substitutes a matrix norm. Acyclic support
graphs are recognized as nilpotent before numerical decomposition. A positive
target on a zero-radius draw raises with guidance to change the seed/sparsity;
a zero target explicitly removes recurrent connections.

Reservoir constants change for the same seed compared with earlier versions.
Old readout checkpoints need their original frozen reservoir to reproduce their
predictions. Radius below one is a practical initialization heuristic, not a
general guarantee of the driven nonlinear echo-state property; see
[Yildiz, Jaeger and Kiebel (2012)](https://www.ai.rug.nl/minds/uploads/2519_Yildizetal12.pdf).
Leak rate additionally changes the linearized spectrum to
`(1 - leak) + leak * lambda`.

## Validation

- 60 targeted tests pass (two existing excluded cases): per-head normalization,
  affine behavior, independence from future tokens and other heads, numerical
  input gradients, full RetNet/RetNetV2 prefix causality, existing model tests,
  frozen/trainable reservoir parameters and deterministic predictions.
- Checked-in NumPy `eigvals` references cover 12 dense/sparse matrices of sizes
  3, 8, 32 and 64. Analytic tests also cover complex rotations, a cyclic
  permutation, negative eigenvalues, nonnormal triangular matrices and nilpotence.
- A 500-neuron, 90%-sparse f64 fixture gave radius `7.170191043759526`, versus
  NumPy `7.170191043759543` (absolute difference `1.69e-14`). Portable host
  decomposition took 61.8 seconds under concurrent system load. This is a
  one-time model-build cost, with no recurrent inference overhead; large
  reservoirs remain expensive to initialize with the portable solver.

Regenerate small fixtures with
`devenv shell -- python scripts/generate_reservoir_spectrum_fixtures.py`.
Export the large validation matrix with
`devenv shell -- mix run scripts/validate_reservoir_spectrum.exs /tmp/spectrum.json`,
then compare `radius` with `max(abs(numpy.linalg.eigvals(matrix)))`.

These corrections address the two identified normalization/initialization
shortcuts. They do not complete the separate pretrained-validation backlog or
add missing RetNet chunkwise execution.
