# NSA / MoBA sparse execution — October 25, 2026

Corrected portable Nx implementations, RTX 5090 and Ryzen 9 9950X3D.
This compares gathered token attention with **dense execution of the same corrected
routing mask**, not the old incorrect implementations or FlashAttention.

Resident f32 Q/K/V, batch 1, four heads, dimension 32, block size 16, two selected
blocks, local window 32, compression 8. EXLA highest precision, five warmups and
30 timed samples. Timing includes routing, attention, a compiled output sum and
scalar host synchronization; initialization, compilation and input transfer are
excluded. Full outputs are compared separately at atol=rtol=1e-5. These are
attention-core measurements, not full models or ExPhil policy timings.

Benchmarks and CUDA tests used ExPhil's compiled local Nx/EXLA 1.0.0 development
runtime (OTP 28), loading the edited Edifice modules directly without rebuilding
ExPhil. Standard `mix test` also passed in Edifice's own dependency environment.
The machine is shared; CPU work and scheduling noise affect small differences.
No training or other GPU workload ran during the final CUDA benchmark.

## Results

Median milliseconds; dense → gathered. Raw JSON includes p95.

| Architecture | Tokens | CUDA | CPU |
|---|---:|---:|---:|
| moba | 64 | 0.085 → 0.097 | 0.076 → 0.258 |
| nsa | 64 | 0.097 → 0.093 | 0.137 → 0.121 |
| moba | 256 | 0.100 → 0.094 | 0.605 → 0.739 |
| nsa | 256 | 0.129 → 0.120 | 1.076 → 0.386 |
| moba | 1024 | 0.709 → 0.695 | 6.189 → 2.975 |
| nsa | 1024 | 1.014 → 0.716 | 10.331 → 1.872 |
| moba | 4096 | 3.615 → 2.751 | 105.135 → 15.774 |
| nsa | 4096 | 4.135 → 2.836 | 174.676 → 19.948 |

Short contexts can be slower: routing and gathers have overhead. This does not
justify replacing ExPhil's recurrent backbones for its short windows. Longer
contexts benefit, especially on CPU. Fused sparse CUDA kernels remain future work.

## Correctness and scope

- MoBA routes only to completed prior blocks and always includes the current
  block in its budget, as in the [official reference](https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_naive.py).
  Its current-block mean cannot leak future keys into selection. Exact top-k
  indices prevent ties from expanding the budget; token masks exclude future and
  padded positions. Short sequences and partial final blocks are supported.
- NSA's previously unused indices now drive actual cross-block retrieval.
  Selection scores aggregate causal compressed-attention probabilities by block
  overlap. Local attention gathers its window. Compression uses original queries
  and exposes only fully completed pooling windows. A trailing incomplete pool
  remains invisible, preserving prefix invariance; its tokens are still available
  in selected/local attention. Empty compressed rows produce zero output and
  finite gradients. Mixing-gate softmax is stable for large logits.
- NSA remains a documented **variant** of the [paper](https://arxiv.org/abs/2502.11089):
  non-overlapping average pooling, independent head selection, mandatory current
  block and scalar softmax gates. Learned overlapping compression, grouped-query
  selection, query-dependent sigmoid gates and specialized kernels are absent.
- Parameter names and shapes are unchanged. Outputs intentionally differ from
  older checkpoints; revalidate their quality before using corrected inference.
  Hard routing is nondifferentiable; gradients reach selected Q/K/V and the
  compressed/local paths. Near-tied routing may differ across numerical backends.

Selected-token scores have shape `[B,H,T,K*block_size]`, local scores
`[B,H,T,window]`, rather than `[B,H,T,T]`. Gathered K/V also carry a head-dimension
axis; this is not a peak-memory measurement or a guarantee of lower total memory
for every shape. MoBA routing and NSA compression still require respectively
O(T²/block_size) and O(T²/compression_ratio) scores. Probability aggregation uses
bounded overlapping-window gathers, avoiding an additional dense matrix product.

## Validation and reproduction

35 targeted tests pass on both the normal test backend and CUDA. Coverage includes
prefix invariance at every cut, exact tie budgets, distant retrieval, changing
NSA's selection budget, nonaligned pooling, partial lengths, dense-reference
outputs and Q/K/V gradients for full and sparse selection, local-window gradients,
and finite NSA gradients with empty compressed rows and large gate logits.

```sh
devenv shell -- mix test test/edifice/attention/{moba,nsa,sparse_attention}_test.exs
devenv shell -- mix format --check-formatted
devenv shell -- mix credo
```

To benchmark using the current Edifice dependencies (versions may differ from the
recorded run), set `SPARSE_CLIENT=host` or `cuda` and an output path:

```sh
SPARSE_CLIENT=cuda SPARSE_OUT=/tmp/sparse-cuda.json devenv shell -- mix run -e '
client = String.to_existing_atom(System.fetch_env!("SPARSE_CLIENT"))
Application.put_env(:exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, memory_fraction: 0.2, preallocate: false])
Nx.global_default_backend({EXLA.Backend, client: client})
Nx.Defn.global_default_options(compiler: EXLA, client: client, precision: :highest)
Code.require_file("scripts/bench_sparse_attention.exs")'
```
