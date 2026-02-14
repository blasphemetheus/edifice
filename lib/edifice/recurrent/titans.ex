defmodule Edifice.Recurrent.Titans do
  @moduledoc """
  Titans - Neural Long-Term Memory with Surprise-Gated Updates.

  Implements the Titans architecture from "Titans: Learning to Memorize
  at Test Time" (Behrouz et al., 2025). Titans extend TTT-style test-time
  learning with a surprise-based gating mechanism: the memory is updated
  more aggressively when the model encounters surprising (high-error) inputs.

  ## Key Innovations

  - **Surprise-gated memory**: Memory update magnitude scales with prediction error
  - **Long-term memory module**: Persistent memory that adapts to data distribution
  - **Momentum-based updates**: Uses gradient momentum for smoother memory evolution
  - **Covariance-aware**: Optional second-order information for better updates

  ## Equations

  ```
  # Project inputs
  q_t = W_q x_t                          # Query
  k_t = W_k x_t                          # Key
  v_t = W_v x_t                          # Value

  # Memory read: retrieve current prediction
  pred_t = M_{t-1} @ k_t

  # Surprise = ||pred_t - v_t||^2 (prediction error)
  surprise_t = ||pred_t - v_t||^2

  # Surprise gate: higher surprise -> larger update
  gate_t = sigmoid(W_g * [x_t, surprise_t])

  # Memory update with surprise gating
  grad_t = (pred_t - v_t) @ k_t^T
  momentum_t = alpha * momentum_{t-1} + grad_t
  M_t = M_{t-1} - gate_t * eta * momentum_t

  # Output from updated memory
  o_t = M_t @ q_t
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  [Input Projection] -> hidden_size
        |
        v
  +----------------------------------+
  |      Titans Layer                |
  |  Project to Q, K, V              |
  |  For each timestep:              |
  |    pred = M @ k                  |
  |    surprise = ||pred - v||^2     |
  |    gate = f(x, surprise)         |
  |    M -= gate * eta * grad        |
  |    output = M @ q                |
  +----------------------------------+
        | (repeat num_layers)
        v
  [Layer Norm] -> [Last Timestep]
        |
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = Titans.build(
        embed_dim: 287,
        hidden_size: 256,
        memory_size: 64,
        num_layers: 4,
        dropout: 0.1
      )

  ## References
  - Paper: https://arxiv.org/abs/2501.00663
  """

  require Axon

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  def default_hidden_size, do: 256

  @doc "Default memory key/value dimension"
  def default_memory_size, do: 64

  @doc "Default number of layers"
  def default_num_layers, do: 4

  @doc "Default dropout rate"
  def default_dropout, do: 0.1

  @doc "Default momentum coefficient"
  def default_momentum, do: 0.9

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Titans model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:memory_size` - Memory key/value dimension (default: 64)
    - `:num_layers` - Number of Titans layers (default: 4)
    - `:dropout` - Dropout rate between layers (default: 0.1)
    - `:momentum` - Momentum coefficient for memory updates (default: 0.9)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    memory_size = Keyword.get(opts, :memory_size, default_memory_size())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

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

    # Stack Titans layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer_opts =
          Keyword.merge(opts,
            memory_size: memory_size,
            momentum: Keyword.get(opts, :momentum, default_momentum())
          )

        layer = build_titans_layer(acc, hidden_size, layer_opts, "titans_#{layer_idx}")

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
  # Titans Layer
  # ============================================================================

  defp build_titans_layer(input, hidden_size, opts, name) do
    memory_size = Keyword.get(opts, :memory_size, default_memory_size())
    momentum = Keyword.get(opts, :momentum, default_momentum())

    # Pre-norm for stability
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Project to Q, K, V (memory_size each) and surprise gate input
    q_proj = Axon.dense(normed, memory_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(normed, memory_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(normed, memory_size, name: "#{name}_v_proj")

    # Surprise gate projection: takes input features and will incorporate surprise
    gate_proj = Axon.dense(normed, memory_size, name: "#{name}_gate_proj")

    # Concatenate all projections
    recurrence_input =
      Axon.concatenate([q_proj, k_proj, v_proj, gate_proj],
        axis: 2,
        name: "#{name}_cat"
      )

    # Apply Titans recurrence
    recurrence_output =
      Axon.nx(
        recurrence_input,
        fn combined ->
          titans_scan(combined, memory_size, momentum)
        end,
        name: "#{name}_recurrence"
      )

    # Project back to hidden_size
    output = Axon.dense(recurrence_output, hidden_size, name: "#{name}_out_proj")

    # Feedforward branch for additional expressivity
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

  defp titans_scan(combined, memory_size, momentum) do
    # combined: [batch, seq_len, memory_size * 4]
    batch_size = Nx.axis_size(combined, 0)
    seq_len = Nx.axis_size(combined, 1)

    # Split into Q, K, V, gate_input
    q_all = Nx.slice_along_axis(combined, 0, memory_size, axis: 2)
    k_all = Nx.slice_along_axis(combined, memory_size, memory_size, axis: 2)
    v_all = Nx.slice_along_axis(combined, memory_size * 2, memory_size, axis: 2)
    gate_input = Nx.slice_along_axis(combined, memory_size * 3, memory_size, axis: 2)

    # Initialize memory: [batch, memory_size, memory_size]
    m_init = Nx.broadcast(0.0, {batch_size, memory_size, memory_size})

    # Initialize momentum accumulator
    mom_init = Nx.broadcast(0.0, {batch_size, memory_size, memory_size})

    # Sequential scan with surprise-gated updates
    {_, _, output_list} =
      Enum.reduce(0..(seq_len - 1), {m_init, mom_init, []}, fn t, {m_prev, mom_prev, acc} ->
        q_t = Nx.slice_along_axis(q_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        k_t = Nx.slice_along_axis(k_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v_t = Nx.slice_along_axis(v_all, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        g_input = Nx.slice_along_axis(gate_input, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Memory read: pred = M @ k
        pred = Nx.dot(m_prev, [2], [0], Nx.new_axis(k_t, 2), [1], [0])
        pred = Nx.squeeze(pred, axes: [2])

        # Surprise: ||pred - v||^2, reduced to scalar per batch per dim
        error = Nx.subtract(pred, v_t)
        surprise = Nx.mean(Nx.pow(error, 2), axes: [1], keep_axes: true)

        # Surprise gate: sigmoid(gate_input + log(surprise + eps))
        # Adding surprise as a bias to the gate makes updates larger when
        # the memory is inaccurate
        surprise_log = Nx.log(Nx.add(surprise, 1.0e-6))
        gate = Nx.sigmoid(Nx.add(g_input, surprise_log))

        # Gradient: error * k^T (outer product)
        grad =
          Nx.dot(
            Nx.new_axis(error, 2),
            [2],
            [0],
            Nx.new_axis(k_t, 1),
            [1],
            [0]
          )

        # Momentum update
        mom_t = Nx.add(Nx.multiply(momentum, mom_prev), grad)

        # Surprise-gated memory update: M -= gate * mom
        # Broadcast gate [batch, memory_size] -> [batch, memory_size, 1]
        gate_expanded = Nx.new_axis(gate, 2)
        m_t = Nx.subtract(m_prev, Nx.multiply(gate_expanded, mom_t))

        # Output: M_t @ q_t
        o_t = Nx.dot(m_t, [2], [0], Nx.new_axis(q_t, 2), [1], [0])
        o_t = Nx.squeeze(o_t, axes: [2])

        {m_t, mom_t, [o_t | acc]}
      end)

    output_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Titans model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end
end
