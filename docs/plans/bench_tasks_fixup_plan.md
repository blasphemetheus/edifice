# bench/tasks/ Status

## Results (all 5 task benches run successfully)

### Sequence Classification — Cumsum Sign (binary, >50% = learning)
| Architecture | Category | Eval Acc | Status |
|---|---|---|---|
| min_gru | recurrent | **60.9%** | Learning |
| mamba | ssm | 45.3% | ~random |
| gqa | attention | 42.2% | ~random |
| retnet | attention | 42.2% | ~random |
| lstm | recurrent | — | FAIL: Axon rnn_state bug |

### Copy/Recall — Template Memory (4-class, >25% = learning)
| Architecture | Category | Eval Acc | Status |
|---|---|---|---|
| mamba | ssm | 26.6% | Barely learning |
| min_gru | recurrent | 26.6% | Barely learning |
| retnet | attention | 26.6% | Barely learning |
| delta_net | recurrent | 26.6% | Barely learning |
| lstm | recurrent | — | FAIL: Axon rnn_state bug |
| titans | recurrent | — | FAIL: arithmetic error |

### Image Classification — Quadrant Brightness (4-class, >25% = learning)
| Architecture | Category | Eval Acc | Status |
|---|---|---|---|
| mlp_mixer | mixer | **34.4%** | Learning |
| vit | attention | 21.9% | ~random |
| convnext | conv | 21.9% | ~random |
| efficient_vit | attention | 21.9% | ~random |
| resnet | conv | — | FAIL: conv + value_and_grad |

### Autoregressive — Next Token (8-class, >12.5% = learning)
| Architecture | Category | Eval Acc | Status |
|---|---|---|---|
| decoder_only | transformer | **100.0%** | Perfect |
| rwkv | attention | **100.0%** | Perfect |
| min_gru | recurrent | **100.0%** | Perfect |
| mamba | ssm | **93.8%** | Learning |

### Graph Classification — Edge Density (4-class, >25% = learning)
| Architecture | Category | Eval Acc | Status |
|---|---|---|---|
| gin_v2 | isomorphism | **50.0%** | Learning |
| gin | isomorphism | **32.8%** | Learning |
| gcn | spectral | 20.3% | ~random |
| gat | attention | 18.8% | ~random |

## Resolved: `value_and_grad` + EXLA.Backend Fix

**Fixed in `TaskHelpers.train/4`**: All captured tensors (inputs, targets, model state) are `Nx.backend_copy`'d to `Nx.BinaryBackend` before entering the `value_and_grad` closure. This avoids the EXLA.Backend vs Nx.Defn.Expr incompatibility.

Additionally, `Axon.build(model, mode: :inference)` is used everywhere.

## Known Failures (architecture-level, not fixable in bench code)

1. **LSTM**: `Axon.rnn_state/7` calls `Nx.Random.split` inside defn tracing — key type (u32) gets misinterpreted. Axon bug.
2. **ResNet**: Conv layers aren't defn-traceable through `value_and_grad`. Same issue noted in existing `training_throughput.exs`.
3. **Titans**: "bad argument in arithmetic expression" — model-specific issue.

## Remaining Work

- [ ] Copy/recall task barely learns — consider increasing epochs, adjusting LR, or making templates more distinct
- [ ] Image classification: most models barely learn — may need larger patch_size or different data generation
- [ ] Apply same `backend_copy` fix to existing `bench/training_throughput.exs`
