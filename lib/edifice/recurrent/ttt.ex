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

  ## Simplified Implementation

  For Axon compatibility, we approximate the per-token gradient update with
  a linear outer-product update rule. This captures the key insight (the hidden
  state adapts to the input distribution) while remaining graph-compilable.

  ## Equations (TTT-Linear)

  ```
  # Project inputs
  q_t = W_q x_t                          # Query
  k_t = W_k x_t                          # Key
  v_t = W_v x_t                          # Value (reconstruction target)
  eta_t = sigmoid(W_eta x_t)             # Learning rate gate

  # Inner model forward: prediction = W_t @ k_t
  pred_t = W_{t-1} @ k_t

  # Self-supervised loss gradient (MSE on reconstruction)
  # grad_W = (pred_t - v_t) @ k_t^T
  # W_t = W_{t-1} - eta_t * grad_W

  # Output using updated model
  o_t = W_t @ q_t
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  [Input Projection] -> hidden_size
        |
        v
  +----------------------------------+
  |        TTT Layer                 |
  |  Project to Q, K, V, eta         |
  |  For each timestep:              |
  |    pred = W @ k                  |
  |    error = pred - v              |
  |    W -= eta * error * k^T        |
  |    output = W @ q                |
  +----------------------------------+
        | (repeat num_layers)
        v
  [Layer Norm] -> [Last Timestep]
        |
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = TTT.build(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 4,
        inner_size: 64,
        dropout: 0.1
      )

  ## References
  - Paper: https://arxiv.org/abs/2407.04620
  """

  require Axon

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  def default_hidden_size, do: 256

  @doc "Default inner model dimension (key/value size)"
  def default_inner_size, do: 64

  @doc "Default number of layers"
  def default_num_layers, do: 4

  @doc "Default dropout rate"
  def default_dropout, do: 0.1

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a TTT model for sequence processing.

  ## Options
    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:inner_size` - Inner model key/value dimension (default: 64)
    - `:num_layers` - Number of TTT layers (default: 4)
    - `:dropout` - Dropout rate between layers (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    inner_size = Keyword.get(opts, :inner_size, default_inner_size())

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project to hidden dimension if needed
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack TTT layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer = build_ttt_layer(acc, hidden_size, inner_size, "ttt_#{layer_idx}")

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

  defp build_ttt_layer(input, hidden_size, inner_size, name) do
    # Pre-norm for stability
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Project to Q, K, V (inner_size), and eta (inner_size)
    # Q and output projection use hidden_size, K/V/eta use inner_size
    q_proj = Axon.dense(normed, inner_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(normed, inner_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(normed, inner_size, name: "#{name}_v_proj")
    eta_proj = Axon.dense(normed, inner_size, name: "#{name}_eta_proj")

    # Concatenate all projections
    recurrence_input =
      Axon.concatenate([q_proj, k_proj, v_proj, eta_proj],
        axis: 2,
        name: "#{name}_cat"
      )

    # Learnable W_0: initial inner model weights (paper uses learnable, not zeros)
    w0_param =
      Axon.param("#{name}_w0", {inner_size, inner_size},
        initializer: fn {n, m}, _type, _key ->
          # Small random init (scaled identity-like)
          Nx.multiply(Nx.eye({n, m}, type: :f32), 0.01)
        end
      )

    # Apply TTT recurrence with learnable W_0
    recurrence_output =
      Axon.layer(
        &ttt_scan_impl/3,
        [recurrence_input, w0_param],
        name: "#{name}_recurrence",
        inner_size: inner_size,
        op_name: :ttt_scan
      )

    # Project back to hidden_size
    output = Axon.dense(recurrence_output, hidden_size, name: "#{name}_out_proj")

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  # Wrapper for Axon.layer callback
  defp ttt_scan_impl(combined, w0, opts) do
    inner_size = opts[:inner_size]
    ttt_scan(combined, w0, inner_size)
  end

  defp ttt_scan(combined, w0, inner_size) do
    # combined: [batch, seq_len, inner_size * 4]
    batch_size = Nx.axis_size(combined, 0)
    seq_len = Nx.axis_size(combined, 1)

    # Split into Q, K, V, eta
    q_all = Nx.slice_along_axis(combined, 0, inner_size, axis: 2)
    k_all = Nx.slice_along_axis(combined, inner_size, inner_size, axis: 2)
    v_all = Nx.slice_along_axis(combined, inner_size * 2, inner_size, axis: 2)
    eta_pre = Nx.slice_along_axis(combined, inner_size * 3, inner_size, axis: 2)

    # Learning rate gate
    eta = Nx.sigmoid(eta_pre)

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

        # Inner model forward: pred = W @ k
        # w_prev: [batch, inner, inner], k_t: [batch, inner]
        pred = Nx.dot(w_prev, [2], [0], Nx.new_axis(k_t, 2), [1], [0])
        pred = Nx.squeeze(pred, axes: [2])

        # Self-supervised loss gradient: error = pred - v
        error = Nx.subtract(pred, v_t)

        # Weight update: W -= eta * error * k^T (outer product)
        # eta_t * error: [batch, inner], k_t: [batch, inner]
        grad =
          Nx.dot(
            Nx.new_axis(Nx.multiply(eta_t, error), 2),
            [2],
            [0],
            Nx.new_axis(k_t, 1),
            [1],
            [0]
          )

        w_t = Nx.subtract(w_prev, grad)

        # Output: W_t @ q_t with per-token normalization for stability
        o_t = Nx.dot(w_t, [2], [0], Nx.new_axis(q_t, 2), [1], [0])
        o_t = Nx.squeeze(o_t, axes: [2])

        # RMS normalization on output (stabilizes varying W scales)
        rms = Nx.sqrt(Nx.mean(Nx.pow(o_t, 2), axes: [-1], keep_axes: true) |> Nx.add(1.0e-6))
        o_t = Nx.divide(o_t, rms)

        {w_t, [o_t | acc]}
      end)

    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
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
