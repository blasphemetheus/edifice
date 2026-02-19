defmodule Edifice.Recurrent.TTT do
  @moduledoc """
  Test-Time Training (TTT) Layers.

  Implements TTT layers from "Learning to (Learn at Test Time): RNNs with
  Expressive Hidden States" (Sun et al., 2024). In TTT, the hidden state
  is itself a model (a linear layer or small MLP) that is updated via a
  self-supervised gradient step at each token.

  ## Key Innovations

  - **Hidden state IS a model**: Instead of a vector, the hidden state is
    the weight matrix of a small inner model
  - **Self-supervised updates**: At each step, the inner model does a gradient
    step on a reconstruction loss
  - **Equivalent to linear attention**: TTT-Linear is mathematically equivalent
    to linear attention with the delta rule when the inner model is linear

  ## Paper-Faithful Implementation

  Follows the official TTT-Linear implementation (ttt-lm-pytorch) with these
  key stability mechanisms:

  1. **W_0 ~ N(0, 0.02)**: Small initialization keeps early predictions near zero,
     preventing gradient explosion in the first steps.
  2. **eta / head_dim scaling**: Inner learning rate is scaled by 1/d (d=inner_size),
     keeping weight updates small. Without this, eta in [0,1] is ~64x too large.
  3. **Inner LayerNorm**: Learnable LayerNorm on inner model predictions before
     computing reconstruction error. Prevents prediction magnitudes from drifting.
  4. **Output gating**: Sigmoid gate on output (like SwiGLU) for smoother gradients.

  ## Equations (TTT-Linear)

  ```
  # Project inputs
  q_t = W_q x_t                          # Query
  k_t = W_k x_t                          # Key
  v_t = W_v x_t                          # Value (reconstruction target)
  eta_t = sigmoid(W_eta x_t) / d         # Learning rate gate (scaled by 1/head_dim)

  # Inner model forward + LayerNorm
  pred_t = LayerNorm(W_{t-1} @ k_t)

  # Self-supervised gradient update
  error_t = pred_t - v_t
  grad_W = error_t @ k_t^T
  W_t = W_{t-1} - eta_t * grad_W

  # Gated output using updated model
  o_t = W_t @ q_t * sigmoid(gate_t)
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  [Input Projection] -> hidden_size
        |
        v
  +--------------------------------------+
  |        TTT Layer                     |
  |  Project to Q, K, V, eta, gate       |
  |  For each timestep:                  |
  |    pred = LayerNorm(W @ k)           |
  |    error = pred - v                  |
  |    W -= (eta/d) * error * k^T        |
  |    output = (W @ q) * sigmoid(gate)  |
  +--------------------------------------+
        | (repeat num_layers)
        v
  [Layer Norm] -> [Last Timestep]
        |
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = TTT.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        inner_size: 64,
        dropout: 0.1
      )

  ## References
  - Paper: https://arxiv.org/abs/2407.04620
  """

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default inner model dimension (key/value size)"
  @spec default_inner_size() :: pos_integer()
  def default_inner_size, do: 64

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.1

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a TTT model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:inner_size` - Inner model key/value dimension (default: 64)
    - `:num_layers` - Number of TTT layers (default: 4)
    - `:dropout` - Dropout rate between layers (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)
    - `:variant` - Inner model variant: `:linear` (default) or `:mlp`.
      The `:mlp` variant applies SiLU activation to keys and queries before
      the inner model, making the hidden state a 2-layer MLP instead of
      a single linear layer.
    - `:output_gate` - Apply sigmoid output gate (default: true). Provides
      smoother gradients by gating the TTT output before the residual.

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:inner_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:output_gate, boolean()}
          | {:seq_len, pos_integer()}
          | {:variant, :linear | :mlp}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    inner_size = Keyword.get(opts, :inner_size, default_inner_size())

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden dimension if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    output_gate = Keyword.get(opts, :output_gate, true)
    variant = Keyword.get(opts, :variant, :linear)

    # Stack TTT layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer =
          build_ttt_layer(
            acc,
            hidden_size,
            inner_size,
            "ttt_#{layer_idx}",
            output_gate,
            variant
          )

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(layer, rate: dropout, name: "dropout_#{layer_idx}")
        else
          layer
        end
      end)

    # Final layer norm
    output = Axon.layer_norm(output, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
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
  # TTT Layer
  # ============================================================================

  defp build_ttt_layer(input, hidden_size, inner_size, name, output_gate, variant) do
    # Pre-norm for stability
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Project to Q, K, V (inner_size), and eta (inner_size)
    q_proj = Axon.dense(normed, inner_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(normed, inner_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(normed, inner_size, name: "#{name}_v_proj")
    eta_proj = Axon.dense(normed, inner_size, name: "#{name}_eta_proj")

    # Output gate projection (sigmoid gate for smoother gradients)
    gate_proj =
      if output_gate do
        Axon.dense(normed, inner_size, name: "#{name}_gate_proj")
      else
        nil
      end

    # Concatenate projections: Q, K, V, eta, [gate]
    cat_inputs =
      if gate_proj do
        [q_proj, k_proj, v_proj, eta_proj, gate_proj]
      else
        [q_proj, k_proj, v_proj, eta_proj]
      end

    recurrence_input =
      Axon.concatenate(cat_inputs, axis: 2, name: "#{name}_cat")

    # Fix 1: W_0 ~ N(0, 0.02) per paper Section 4.1
    # Small init keeps early predictions near zero, preventing gradient explosion
    w0_param =
      Axon.param("#{name}_w0", {inner_size, inner_size},
        initializer: Axon.Initializers.normal(scale: 0.02)
      )

    # Fix 3: Learnable LayerNorm on inner model predictions (before loss)
    # Prevents prediction magnitudes from drifting as W accumulates updates
    ln_gamma =
      Axon.param("#{name}_inner_ln_gamma", {inner_size}, initializer: :ones)

    ln_beta =
      Axon.param("#{name}_inner_ln_beta", {inner_size}, initializer: :zeros)

    # Apply TTT recurrence with all stability params
    recurrence_output =
      Axon.layer(
        &ttt_scan_impl/5,
        [recurrence_input, w0_param, ln_gamma, ln_beta],
        name: "#{name}_recurrence",
        inner_size: inner_size,
        output_gate: output_gate,
        variant: variant,
        op_name: :ttt_scan
      )

    # Project back to hidden_size
    output = Axon.dense(recurrence_output, hidden_size, name: "#{name}_out_proj")

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  # Wrapper for Axon.layer callback — arity = num_inputs + 1 (for opts map)
  defp ttt_scan_impl(combined, w0, ln_gamma, ln_beta, opts) do
    inner_size = opts[:inner_size]
    output_gate = opts[:output_gate] || false
    variant = opts[:variant] || :linear
    ttt_scan(combined, w0, ln_gamma, ln_beta, inner_size, output_gate, variant)
  end

  defp ttt_scan(combined, w0, ln_gamma, ln_beta, inner_size, output_gate, variant) do
    # combined: [batch, seq_len, inner_size * N] where N=4 (no gate) or 5 (with gate)
    batch_size = Nx.axis_size(combined, 0)
    seq_len = Nx.axis_size(combined, 1)

    # Split into Q, K, V, eta, [gate]
    q_all = Nx.slice_along_axis(combined, 0, inner_size, axis: 2)
    k_all = Nx.slice_along_axis(combined, inner_size, inner_size, axis: 2)
    v_all = Nx.slice_along_axis(combined, inner_size * 2, inner_size, axis: 2)
    eta_pre = Nx.slice_along_axis(combined, inner_size * 3, inner_size, axis: 2)

    gate_all =
      if output_gate do
        Nx.slice_along_axis(combined, inner_size * 4, inner_size, axis: 2)
      else
        nil
      end

    # Fix 2: eta / inner_size scaling (CRITICAL for stability)
    # Paper: eta = sigmoid(W_eta @ x) / head_dim
    # Without this, the inner learning rate is ~64x too large
    eta = Nx.divide(Nx.sigmoid(eta_pre), inner_size)

    # Initialize inner model weights from learnable W_0
    # w0: [inner_size, inner_size] -> broadcast to [batch, inner_size, inner_size]
    w_init = Nx.broadcast(w0, {batch_size, inner_size, inner_size})

    # Sequential scan with TTT update
    {_, output_list} =
      Enum.reduce(0..(seq_len - 1), {w_init, []}, fn t, {w_prev, acc} ->
        q_t = Nx.slice_along_axis(q_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        eta_t = Nx.slice_along_axis(eta, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        gate_t =
          if output_gate do
            Nx.slice_along_axis(gate_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          else
            nil
          end

        # Inner model forward and update depend on variant
        {w_t, o_t} =
          case variant do
            :mlp ->
              # MLP variant: 2-layer inner model (k -> silu -> W @ k)
              k_act = Nx.multiply(k_t, Nx.sigmoid(k_t))

              pred = Nx.dot(w_prev, [2], [0], Nx.new_axis(k_act, 2), [1], [0])
              pred = Nx.squeeze(pred, axes: [2])

              # Fix 3: LayerNorm on prediction before computing error
              pred_normed = manual_layer_norm(pred, ln_gamma, ln_beta)
              error = Nx.subtract(pred_normed, v_t)

              grad =
                Nx.dot(
                  Nx.new_axis(Nx.multiply(eta_t, error), 2),
                  [2],
                  [0],
                  Nx.new_axis(k_act, 1),
                  [1],
                  [0]
                )

              w_new = Nx.subtract(w_prev, grad)
              q_act = Nx.multiply(q_t, Nx.sigmoid(q_t))
              out = Nx.dot(w_new, [2], [0], Nx.new_axis(q_act, 2), [1], [0])
              {w_new, Nx.squeeze(out, axes: [2])}

            _linear ->
              # Linear variant (default): pred = W @ k
              pred = Nx.dot(w_prev, [2], [0], Nx.new_axis(k_t, 2), [1], [0])
              pred = Nx.squeeze(pred, axes: [2])

              # Fix 3: LayerNorm on prediction before computing error
              pred_normed = manual_layer_norm(pred, ln_gamma, ln_beta)
              error = Nx.subtract(pred_normed, v_t)

              grad =
                Nx.dot(
                  Nx.new_axis(Nx.multiply(eta_t, error), 2),
                  [2],
                  [0],
                  Nx.new_axis(k_t, 1),
                  [1],
                  [0]
                )

              w_new = Nx.subtract(w_prev, grad)
              out = Nx.dot(w_new, [2], [0], Nx.new_axis(q_t, 2), [1], [0])
              {w_new, Nx.squeeze(out, axes: [2])}
          end

        # Fix 4: Output gating (sigmoid gate for smoother gradients)
        o_t =
          if output_gate do
            Nx.multiply(o_t, Nx.sigmoid(gate_t))
          else
            o_t
          end

        {w_t, [o_t | acc]}
      end)

    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # Manual LayerNorm for use inside the TTT scan (can't use Axon.layer_norm in raw Nx)
  defp manual_layer_norm(x, gamma, beta) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    var = Nx.variance(x, axes: [-1], keep_axes: true)
    normalized = Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(var, 1.0e-6)))
    Nx.add(Nx.multiply(normalized, gamma), beta)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a TTT model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end
end
