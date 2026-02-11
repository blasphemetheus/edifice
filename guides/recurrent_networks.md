# Recurrent Networks
> From classic gated RNNs through the "RNN is dead" era to a modern renaissance of linear recurrences, test-time learning, and surprise-gated memory.

## Overview

Recurrent neural networks maintain a hidden state that evolves as each token in a sequence is processed. This makes them natural for tasks where the model must accumulate context over time -- language modeling, time-series prediction, real-time control. The Recurrent family in Edifice spans four decades of ideas, from the 1991 LSTM through 2025's Titans architecture.

The history is non-linear. Classic LSTMs and GRUs dominated sequence modeling from 2014-2017, were largely displaced by transformers (2017-2023), and are now experiencing a revival through architectures that combine recurrent structure with modern training techniques. The key insight driving this renaissance: recurrent models offer O(1) memory per token at inference time, versus O(n) for transformers, making them compelling for long-context and real-time applications.

Edifice provides eight recurrent modules that cover the full spectrum: proven classics (LSTM, GRU), modern scaled variants (xLSTM), minimal parallel-scannable designs (MinGRU, MinLSTM), novel memory paradigms (DeltaNet, TTT, Titans), and the training-free reservoir computing baseline (Reservoir). Each module exposes a `build/1` function returning an Axon model, with consistent options for hidden size, layer stacking, and dropout.

## Conceptual Foundation

All gated recurrent architectures share a common principle: use learned gates to control information flow through time. The fundamental recurrence is:

    h_t = f(h_{t-1}, x_t)

where h is the hidden state and x is the input. The challenge is making f learnable without gradients vanishing or exploding over long sequences. The LSTM's solution -- a cell state modulated by forget, input, and output gates -- remains the conceptual foundation that every architecture in this family either extends, simplifies, or replaces.

The key equation for understanding the design space is the gated linear recurrence:

    c_t = f_t * c_{t-1} + i_t * v_t

where f_t is the forget gate, i_t is the input gate, and v_t is the new value. When f + i = 1, this becomes a convex interpolation (MinLSTM). When f and i use exponential gating, the dynamic range expands dramatically (xLSTM). When v_t is replaced by an error-correcting outer product, the recurrence becomes an associative memory (DeltaNet, TTT, Titans).

## Architecture Evolution

```
1991  LSTM (Hochreiter & Schmidhuber)
  |     - Cell state + 3 gates solves vanishing gradients
  |
1997  Bidirectional RNNs (Schuster & Paliwal)
  |
2001  Echo State Networks (Jaeger)
  |     - Fixed random reservoir + trained readout
  |     [Reservoir]
  |
2014  GRU (Cho et al.)
  |     - Simplified LSTM: 2 gates, merge cell/hidden
  |     [Recurrent: :lstm, :gru]
  |
2017  Transformers eclipse RNNs
  |     "Attention is all you need" era begins
  :
  :     (RNN "dark ages" -- still used in RL, robotics, edge)
  :
2021  Linear Attention / Delta Rule revival
  |     - DeltaNet: error-correcting associative memory
  |     [DeltaNet]
  |
2024  "Were RNNs All We Needed?" (Feng et al.)
  |     - MinGRU: single gate, parallel-scannable
  |     - MinLSTM: normalized gates, parallel-scannable
  |     [MinGRU, MinLSTM]
  |
2024  xLSTM (Beck et al., NeurIPS)
  |     - sLSTM: exponential gating + normalizer
  |     - mLSTM: matrix memory + key-value storage
  |     [XLSTM]
  |
2024  TTT (Sun et al.)
  |     - Hidden state IS a model, updated by gradient descent
  |     [TTT]
  |
2025  Titans (Behrouz et al.)
        - Surprise-gated long-term memory with momentum
        [Titans]
```

## When to Use What

| Scenario | Module | Rationale |
|----------|--------|-----------|
| Proven baseline, small model | `Recurrent` (LSTM/GRU) | Well-understood, stable, Axon-native cells |
| Scaling to transformer-competitive | `XLSTM` | Exponential gating + matrix memory; designed for large-scale |
| Parallel training is critical | `MinGRU` or `MinLSTM` | Simplified gates admit parallel prefix scan during training |
| Fastest minimal recurrence | `MinGRU` | Single gate, fewest parameters per layer |
| Error-correcting memory | `DeltaNet` | Delta rule subtracts current retrieval before update; better recall than linear attention |
| Adapting to distribution shift at inference | `TTT` | Hidden state self-improves via gradient steps; adapts to test data |
| Long-horizon memory with novelty detection | `Titans` | Surprise-gated updates + momentum; remembers more when confused |
| Fast prototyping, no training of dynamics | `Reservoir` | Fixed random reservoir, train only readout; seconds to fit |
| Real-time control (frame-by-frame) | `Recurrent` (stateful) | `build_stateful/1` + `initial_hidden/2` for online inference |
| Hybrid temporal + nonlinear | `Recurrent` (hybrid) | `build_hybrid/1` stacks RNN + MLP for richer representations |

### Parallel Training Capability

```
Sequential only:     LSTM, GRU, sLSTM, Reservoir
                     (must process token-by-token)

Parallel-scannable:  MinGRU, MinLSTM, mLSTM
                     (can use parallel prefix scan for O(log n) depth)

Matrix-memory:       DeltaNet, TTT, Titans
                     (sequential due to matrix state, but chunkable)
```

## Key Concepts

### Gating Mechanisms and Vanishing Gradients

The vanishing gradient problem occurs because multiplying many values less than 1 through time drives gradients to zero. The LSTM solves this with an additive cell state update: the cell state c_t is modified by adding new information (gated by i_t) and optionally forgetting old information (gated by f_t), but the gradient can flow through the forget gate path without repeated multiplication by small values.

The GRU simplifies this by merging the cell state and hidden state, using a single update gate z_t that interpolates between the old hidden state and a candidate:

    h_t = (1 - z_t) * h_{t-1} + z_t * candidate_t

Both architectures fundamentally use sigmoid gates (output in [0, 1]) to control information flow. This works but limits the dynamic range of gating decisions.

### Exponential Gating: xLSTM's Innovation

xLSTM replaces sigmoid gates with exponential gates:

    i_t = exp(W_i * x_t + ...)
    f_t = exp(W_f * x_t + ...)

Exponential gates can produce arbitrarily large values, allowing the network to make much stronger "remember this" or "forget this" decisions. To prevent numerical overflow, xLSTM tracks a normalizer state:

    n_t = f_t * n_{t-1} + i_t

and normalizes the output: h_t = o_t * (c_t / n_t).

The mLSTM variant goes further by replacing the scalar cell state with a matrix memory C that stores key-value associations, similar to attention but with a recurrent update rule. This gives mLSTM the ability to memorize associations (like attention) while maintaining O(d^2) state size regardless of sequence length.

### Minimal Recurrences: Stripping Gates to the Essentials

MinGRU and MinLSTM ask: what is the minimum gating structure that still works? The answer is surprisingly simple.

MinGRU uses a single gate that depends only on the input (not the hidden state):

    z_t = sigmoid(W_z * x_t)
    h_t = (1 - z_t) * h_{t-1} + z_t * W_h * x_t

MinLSTM uses normalized forget and input gates that sum to 1:

    f'_t = f_t / (f_t + i_t),   i'_t = i_t / (f_t + i_t)
    c_t = f'_t * c_{t-1} + i'_t * candidate_t

The critical insight is that removing the hidden-state dependency from gates makes the recurrence expressible as a linear scan, which can be parallelized during training using a parallel prefix sum algorithm. This gives O(log n) parallel depth instead of O(n) sequential depth -- a massive training speedup.

### Novel Paradigms: TTT, Titans, DeltaNet

These three architectures reimagine what a recurrent hidden state can be.

**DeltaNet** maintains an associative memory matrix S and updates it using the delta rule -- a learning rule from classical neural networks. At each step, it retrieves the current prediction (S * k_t), computes the error (v_t - prediction), and updates S to correct the error. This error-correcting property means DeltaNet does not blindly accumulate associations like linear attention; it actively fixes incorrect memories.

**TTT** (Test-Time Training) takes a radical approach: the hidden state is literally the weight matrix of a small inner model. At each token, this inner model does a gradient descent step on a self-supervised reconstruction loss. The hidden state improves itself by learning from the input stream. When the inner model is linear, TTT reduces to linear attention with the delta rule.

**Titans** extends TTT with a surprise-based gating mechanism. When the memory's prediction is accurate (low surprise), updates are small. When prediction error is high (the model is surprised), updates are large. This creates an attention-like effect where novel or important tokens get preferentially stored. Titans also uses gradient momentum for smoother memory evolution.

```
  Classical RNN          DeltaNet              TTT                 Titans
  +-----------+         +-----------+         +-----------+       +-----------+
  | h: vector |         | S: matrix |         | W: matrix |       | M: matrix |
  |           |         |           |         | (a model) |       | mom: matrix|
  | h = f(h,x)|        | S += beta *|        | W -= eta * |      | surprise = |
  |           |         |  (v-Sk)k^T|         |  dL/dW    |       |  ||Mk-v||^2|
  +-----------+         +-----------+         +-----------+       | M -= g*mom |
  State: O(d)           State: O(d^2)         State: O(d^2)       +-----------+
  Update: O(d)          Update: O(d^2)        Update: O(d^2)      State: O(2*d^2)
```

### Reservoir Computing: The No-Training Baseline

Echo State Networks take the opposite approach to all other architectures: the recurrent dynamics are fixed at random initialization and never trained. Only a linear readout layer is optimized (typically via ridge regression, though Edifice uses gradient descent through Axon).

The echo state property requires that the spectral radius of the reservoir weight matrix is less than 1, ensuring that the reservoir's state asymptotically depends only on the input history, not on initial conditions. The sparsity of connections (default 90% zeros) creates diverse temporal dynamics across reservoir neurons.

Reservoir computing is useful as a baseline (if it works, you may not need a trainable RNN), for extremely fast experimentation (training takes seconds), and for edge deployment where model size is less important than training cost.

## Complexity Comparison

| Module | State Size | Per-Step Compute | Training Mode | Learned Params (4-layer, d=256) |
|--------|-----------|-----------------|---------------|-------------------------------|
| `Recurrent` (LSTM) | O(d) | O(d^2) | Sequential BPTT | ~2.1M (4 * 4 * d^2 per layer) |
| `Recurrent` (GRU) | O(d) | O(d^2) | Sequential BPTT | ~1.6M (3 * 4 * d^2 per layer) |
| `XLSTM` (sLSTM) | O(d) | O(d^2) | Sequential | ~2.6M (gates + FFN per block) |
| `XLSTM` (mLSTM) | O(d^2) | O(d^2) | Parallelizable | ~3.5M (gates + KVQ + FFN) |
| `MinGRU` | O(d) | O(d^2) | Parallel scan | ~1.0M (2 projections per layer) |
| `MinLSTM` | O(d) | O(d^2) | Parallel scan | ~1.5M (3 projections per layer) |
| `DeltaNet` | O(d^2) | O(d^2) | Sequential (chunkable) | ~2.0M (QKVB + out per layer) |
| `TTT` | O(d_inner^2) | O(d * d_inner) | Sequential (chunkable) | ~1.2M (QKV + eta + out per layer) |
| `Titans` | O(2 * d_mem^2) | O(d * d_mem) | Sequential | ~1.5M (QKV + gate + FFN per layer) |
| `Reservoir` | O(d_res) | O(d_res^2) | Readout only | ~d_res * d_out (readout only) |

Where d = hidden_size, d_inner = TTT inner dimension, d_mem = Titans memory dimension, d_res = reservoir size.

## Module Reference

- `Edifice.Recurrent` -- Classic LSTM and GRU with multi-layer stacking, truncated BPTT, stateful inference, and hybrid RNN+MLP mode
- `Edifice.Recurrent.XLSTM` -- Extended LSTM with exponential gating; sLSTM for state-tracking, mLSTM for memorization, mixed for both
- `Edifice.Recurrent.MinGRU` -- Single-gate GRU stripped to its parallel-scannable core; fewest parameters per layer
- `Edifice.Recurrent.MinLSTM` -- Normalized forget/input gates (f + i = 1) enabling parallel prefix scan training
- `Edifice.Recurrent.DeltaNet` -- Linear attention with delta rule error correction; maintains O(d^2) associative memory matrix
- `Edifice.Recurrent.TTT` -- Test-Time Training where the hidden state is a model updated by self-supervised gradient steps
- `Edifice.Recurrent.Titans` -- Neural long-term memory with surprise-based gating and gradient momentum
- `Edifice.Recurrent.Reservoir` -- Echo State Network with fixed random reservoir and trainable linear readout

## Cross-References

- **state_space_models.md** -- SSMs (S4, Mamba) can be viewed as continuous-time linear RNNs; Mamba's selective mechanism parallels gated recurrences
- **attention_mechanisms.md** -- HGRN and Griffin blend gated recurrence with attention; linear attention is a special case of the DeltaNet/TTT memory formulation
- **building_blocks.md** -- RMSNorm and SwiGLU appear in xLSTM and Titans blocks as normalization and feed-forward components

## Further Reading

1. Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997) -- The foundational gated RNN that solved vanishing gradients.
2. Beck et al., "xLSTM: Extended Long Short-Term Memory" (NeurIPS 2024) -- arxiv.org/abs/2405.04517. Exponential gating and matrix memory for scaling LSTMs.
3. Feng et al., "Were RNNs All We Needed?" (2024) -- arxiv.org/abs/2410.01201. MinGRU and MinLSTM demonstrate that minimal gating suffices when parallel scan is available.
4. Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" (2024) -- arxiv.org/abs/2407.04620. TTT layers where the hidden state self-improves via gradient descent.
5. Behrouz et al., "Titans: Learning to Memorize at Test Time" (2025) -- arxiv.org/abs/2501.00663. Surprise-gated memory with momentum for adaptive long-term retention.
