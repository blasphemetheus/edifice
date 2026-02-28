defmodule Edifice.Recurrent.MIRAS do
  @moduledoc """
  MIRAS: Memory as Iterative Reasoning over Associative Structures.

  Implements three concrete variants from the MIRAS framework (Google Research,
  2025) which unifies sequence models as online optimization over associative
  memory. Each variant uses a different attentional bias (loss function) and
  retention strategy for the memory update rule.

  ## Variants

  ### Moneta — High-fidelity recall
  - Attentional bias: Generalized p-norm `||M@k - v||_p^p`
  - Retention: Weight decay `alpha * M`
  - Best for: Noisy or hard-retrieval tasks

  ### Yaad — Robust to outliers
  - Attentional bias: Huber loss with input-dependent threshold
  - Retention: Adaptive momentum
  - Best for: Messy or inconsistent data

  ### Memora — Memory stability
  - Attentional bias: KL-divergence `KL(softmax(v) || softmax(M@k))`
  - Retention: Softmax normalization + periodic weight resets
  - Best for: Strict probability mapping, controlled updates

  ## Memory Update Rule

  All variants share the core update:
  ```
  A_t = alpha_t * A_{t-1} - eta_t * grad(loss(M@k_t, v_t))
  M_t = A_t / ||A_t||
  ```

  Where `alpha_t` is a data-dependent forgetting rate and `eta_t` is a
  data-dependent learning rate, both produced by gating networks.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  [Input Projection] -> hidden_size
        |
  +----------------------------------+
  |      MIRAS Layer                 |
  |  Project to Q, K, V              |
  |  For each timestep:              |
  |    pred = M @ k                  |
  |    loss = variant_loss(pred, v)  |
  |    grad = d_loss/d_M             |
  |    A -= eta * grad               |
  |    A *= alpha (retention)        |
  |    M = normalize(A)              |
  |    output = M @ q                |
  +----------------------------------+
        | (repeat num_layers)
        v
  [Layer Norm] -> [Last Timestep]
        |
  Output [batch, hidden_size]
  ```

  ## Usage

      # Moneta (high-fidelity recall)
      model = MIRAS.build(
        embed_dim: 287,
        hidden_size: 256,
        variant: :moneta,
        num_layers: 4
      )

      # Yaad (robust to outliers)
      model = MIRAS.build(
        embed_dim: 287,
        hidden_size: 256,
        variant: :yaad,
        num_layers: 4
      )

      # Memora (stable probability mapping)
      model = MIRAS.build(
        embed_dim: 287,
        hidden_size: 256,
        variant: :memora,
        num_layers: 4
      )

  ## References

  - Paper: "It's All Connected: A Journey Through Test-Time Memorization,
    Attentional Bias, Retention, and Online Optimization"
  - Authors: Google Research (April 2025)
  - ArXiv: 2504.13173
  - Related: Titans (Behrouz et al., 2025) — MIRAS generalizes Titans
  """

  @default_hidden_size 256
  @default_memory_size 64
  @default_num_layers 4
  @default_dropout 0.1
  @default_momentum 0.9
  @default_p_norm 2.0

  @doc """
  Build a MIRAS model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:memory_size` - Memory key/value dimension (default: 64)
    - `:num_layers` - Number of MIRAS layers (default: 4)
    - `:dropout` - Dropout rate between layers (default: 0.1)
    - `:momentum` - Momentum coefficient for updates (default: 0.9)
    - `:variant` - MIRAS variant: `:moneta`, `:yaad`, or `:memora` (default: `:moneta`)
    - `:p_norm` - P-norm exponent for Moneta variant (default: 2.0)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model that processes sequences and outputs the last hidden state.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:memory_size, pos_integer()}
          | {:momentum, float()}
          | {:num_layers, pos_integer()}
          | {:p_norm, float()}
          | {:seq_len, pos_integer()}
          | {:variant, :moneta | :yaad | :memora}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    memory_size = Keyword.get(opts, :memory_size, @default_memory_size)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden dimension if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack MIRAS layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer = build_miras_layer(acc, hidden_size, memory_size, opts, "miras_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(layer, rate: dropout, name: "dropout_#{layer_idx}")
        else
          layer
        end
      end)

    # Final layer norm
    output = Axon.layer_norm(output, name: "final_norm")

    # Extract last timestep
    Axon.nx(
      output,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # MIRAS Layer
  # ============================================================================

  defp build_miras_layer(input, hidden_size, memory_size, opts, name) do
    variant = Keyword.get(opts, :variant, :moneta)
    momentum = Keyword.get(opts, :momentum, @default_momentum)
    p_norm = Keyword.get(opts, :p_norm, @default_p_norm)

    # Pre-norm for stability
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Project to Q, K, V, alpha gate, eta gate
    q_proj = Axon.dense(normed, memory_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(normed, memory_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(normed, memory_size, name: "#{name}_v_proj")

    # Data-dependent forgetting rate (alpha) and learning rate (eta)
    alpha_proj = Axon.dense(normed, memory_size, name: "#{name}_alpha_proj")
    eta_proj = Axon.dense(normed, memory_size, name: "#{name}_eta_proj")

    # Concatenate all projections for scan
    recurrence_input =
      Axon.concatenate([q_proj, k_proj, v_proj, alpha_proj, eta_proj],
        axis: 2,
        name: "#{name}_cat"
      )

    # Apply MIRAS recurrence
    recurrence_output =
      Axon.nx(
        recurrence_input,
        fn combined ->
          miras_scan(combined, memory_size, momentum, variant, p_norm)
        end,
        name: "#{name}_recurrence"
      )

    # Project back to hidden_size
    output = Axon.dense(recurrence_output, hidden_size, name: "#{name}_out_proj")

    # FFN branch
    ff_normed =
      Axon.layer_norm(Axon.add(input, output, name: "#{name}_mid_residual"),
        name: "#{name}_ff_norm"
      )

    ff_inner = Axon.dense(ff_normed, hidden_size * 2, name: "#{name}_ff_up")
    ff_inner = Axon.activation(ff_inner, :gelu, name: "#{name}_ff_gelu")
    ff_out = Axon.dense(ff_inner, hidden_size, name: "#{name}_ff_down")

    Axon.add(Axon.add(input, output, name: "#{name}_residual_1"), ff_out,
      name: "#{name}_residual_2"
    )
  end

  # ============================================================================
  # MIRAS Scan
  # ============================================================================

  defp miras_scan(combined, memory_size, momentum, variant, p_norm) do
    batch_size = Nx.axis_size(combined, 0)
    seq_len = Nx.axis_size(combined, 1)

    # Split projections
    q_all = Nx.slice_along_axis(combined, 0, memory_size, axis: 2)
    k_all = Nx.slice_along_axis(combined, memory_size, memory_size, axis: 2)
    v_all = Nx.slice_along_axis(combined, memory_size * 2, memory_size, axis: 2)
    alpha_all = Nx.slice_along_axis(combined, memory_size * 3, memory_size, axis: 2)
    eta_all = Nx.slice_along_axis(combined, memory_size * 4, memory_size, axis: 2)

    # Initialize memory and momentum accumulator
    m_init = Nx.broadcast(0.0, {batch_size, memory_size, memory_size})
    mom_init = Nx.broadcast(0.0, {batch_size, memory_size, memory_size})

    {_, _, output_list} =
      Enum.reduce(0..(seq_len - 1), {m_init, mom_init, []}, fn t, {m_prev, mom_prev, acc} ->
        q_t = Nx.slice_along_axis(q_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        alpha_raw = Nx.slice_along_axis(alpha_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        eta_raw = Nx.slice_along_axis(eta_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Data-dependent gates
        alpha_t = Nx.sigmoid(alpha_raw)
        eta_t = Nx.sigmoid(eta_raw)

        # Memory read: pred = M @ k
        pred = Nx.dot(m_prev, [2], [0], Nx.new_axis(k_t, 2), [1], [0])
        pred = Nx.squeeze(pred, axes: [2])

        # Compute gradient based on variant
        grad = compute_gradient(variant, pred, v_t, k_t, m_prev, memory_size, p_norm)

        # Momentum update
        mom_t = Nx.add(Nx.multiply(momentum, mom_prev), grad)

        # Memory update with data-dependent gates
        # alpha: forgetting (retention) — applied to memory
        # eta: learning rate — scales gradient
        alpha_expanded = Nx.new_axis(alpha_t, 2)
        eta_expanded = Nx.new_axis(eta_t, 2)
        m_t = Nx.subtract(Nx.multiply(alpha_expanded, m_prev), Nx.multiply(eta_expanded, mom_t))

        # Variant-specific normalization
        m_t = normalize_memory(variant, m_t)

        # Output: M_t @ q_t
        o_t = Nx.dot(m_t, [2], [0], Nx.new_axis(q_t, 2), [1], [0])
        o_t = Nx.squeeze(o_t, axes: [2])

        {m_t, mom_t, [o_t | acc]}
      end)

    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Variant-Specific Loss Gradients
  # ============================================================================

  # Moneta: Generalized p-norm loss ||pred - v||_p^p
  # Gradient: p * sign(error) * |error|^(p-1) * k^T
  defp compute_gradient(:moneta, pred, v_t, k_t, _m_prev, _memory_size, p_norm) do
    error = Nx.subtract(pred, v_t)

    grad_coeff =
      if p_norm == 2.0 do
        # Special case: standard MSE gradient = 2 * error
        Nx.multiply(2.0, error)
      else
        # General p-norm: p * sign(error) * |error|^(p-1)
        abs_error = Nx.abs(error)
        sign_error = Nx.sign(error)

        Nx.multiply(
          p_norm,
          Nx.multiply(sign_error, Nx.pow(Nx.add(abs_error, 1.0e-8), p_norm - 1.0))
        )
      end

    # Outer product: grad_coeff * k^T -> [batch, memory_size, memory_size]
    Nx.dot(
      Nx.new_axis(grad_coeff, 2),
      [2],
      [0],
      Nx.new_axis(k_t, 1),
      [1],
      [0]
    )
  end

  # Yaad: Huber loss with adaptive threshold
  # Gradient: error when |error| < delta, else delta * sign(error)
  defp compute_gradient(:yaad, pred, v_t, k_t, _m_prev, _memory_size, _p_norm) do
    error = Nx.subtract(pred, v_t)
    abs_error = Nx.abs(error)

    # Adaptive threshold: mean of absolute errors per sample
    delta = Nx.mean(abs_error, axes: [1], keep_axes: true)
    delta = Nx.max(delta, Nx.tensor(0.01))

    # Huber gradient: error if |error| < delta, else delta * sign(error)
    within_delta = Nx.less(abs_error, delta)

    grad_coeff =
      Nx.select(
        within_delta,
        error,
        Nx.multiply(delta, Nx.sign(error))
      )

    Nx.dot(
      Nx.new_axis(grad_coeff, 2),
      [2],
      [0],
      Nx.new_axis(k_t, 1),
      [1],
      [0]
    )
  end

  # Memora: KL-divergence loss KL(softmax(v) || softmax(pred))
  # Gradient: (softmax(pred) - softmax(v)) * k^T
  defp compute_gradient(:memora, pred, v_t, k_t, _m_prev, _memory_size, _p_norm) do
    # Softmax over feature dimension
    pred_probs = softmax_stable(pred)
    target_probs = softmax_stable(v_t)

    # KL gradient: softmax(pred) - softmax(v)
    grad_coeff = Nx.subtract(pred_probs, target_probs)

    Nx.dot(
      Nx.new_axis(grad_coeff, 2),
      [2],
      [0],
      Nx.new_axis(k_t, 1),
      [1],
      [0]
    )
  end

  # ============================================================================
  # Variant-Specific Memory Normalization
  # ============================================================================

  # Moneta: L2 normalization (keeps memory bounded)
  defp normalize_memory(:moneta, m) do
    norm = Nx.sqrt(Nx.sum(Nx.pow(m, 2), axes: [2], keep_axes: true))
    norm = Nx.max(norm, Nx.tensor(1.0e-6))
    Nx.divide(m, norm)
  end

  # Yaad: No explicit normalization (momentum handles stability)
  defp normalize_memory(:yaad, m), do: m

  # Memora: Row-wise softmax normalization (memory stores distributions)
  defp normalize_memory(:memora, m) do
    max_m = Nx.reduce_max(m, axes: [2], keep_axes: true)
    exp_m = Nx.exp(Nx.subtract(m, max_m))
    Nx.divide(exp_m, Nx.add(Nx.sum(exp_m, axes: [2], keep_axes: true), 1.0e-6))
  end

  defp softmax_stable(x) do
    max_x = Nx.reduce_max(x, axes: [1], keep_axes: true)
    exp_x = Nx.exp(Nx.subtract(x, max_x))
    Nx.divide(exp_x, Nx.add(Nx.sum(exp_x, axes: [1], keep_axes: true), 1.0e-6))
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a MIRAS model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
