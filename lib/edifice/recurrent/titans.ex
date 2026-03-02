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

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default memory key/value dimension"
  @spec default_memory_size() :: pos_integer()
  def default_memory_size, do: 64

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.1

  @doc "Default momentum coefficient"
  @spec default_momentum() :: float()
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
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:memory_size, pos_integer()}
          | {:momentum, float()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
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

    # Apply Titans recurrence (dispatches through 3-tier CUDA pipeline)
    recurrence_output =
      Axon.nx(
        recurrence_input,
        fn combined ->
          Edifice.CUDA.FusedScan.titans_scan(combined,
            memory_size: memory_size, momentum: momentum)
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

  # titans_scan/3 removed — now dispatched through Edifice.CUDA.FusedScan.titans_scan/2

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
