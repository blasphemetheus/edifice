defmodule Edifice.SSM.GSS do
  @moduledoc """
  GSS: Gated State Space Model.

  Implements the Gated State Space model from "Long Range Language Modeling
  via Gated State Spaces" (Mehta et al., 2023). GSS simplifies S4 by using
  fixed (learned but not input-dependent) A, B, C matrices combined with
  multiplicative gating for non-linearity.

  ## Key Innovation: Fixed SSM + Multiplicative Gating

  Unlike Mamba (where B, C, dt are input-dependent), GSS uses:
  - **Fixed** diagonal A, B, C matrices (learned via `Axon.param`)
  - **Gating** for input-dependent non-linearity: `gate = sigmoid(W_g * x)`
  - Result: simpler than Mamba, more expressive than vanilla S4

  ## Equations

  ```
  # SSM with fixed parameters:
  h_t = A * h_{t-1} + B * x_t      # A, B are learned parameters (not input-dependent)
  y_t = C * h_t                      # C is a learned parameter

  # Gating for non-linearity:
  gate_t = sigmoid(W_g * x_t + b_g)
  output_t = gate_t * y_t
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  |         GSS Block                    |
  |  LayerNorm -> [SSM path, Gate path]  |
  |    SSM: linear -> scan(A,B) -> C*h   |
  |    Gate: linear -> sigmoid           |
  |  output = SSM * Gate                 |
  |  -> project -> residual              |
  |  LayerNorm -> FFN -> residual        |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Output [batch, hidden_size]
  ```

  ## Compared to Other SSMs

  | Model | A,B,C | Gating | Complexity |
  |-------|-------|--------|------------|
  | S4    | Fixed (HiPPO) | None | O(L log L) |
  | GSS   | Fixed (learned) | Multiplicative | O(L) |
  | Mamba | Input-dependent | SiLU | O(L) |

  ## Usage

      model = GSS.build(
        embed_dim: 287,
        hidden_size: 256,
        state_size: 16,
        num_layers: 4
      )

  ## References

  - Mehta et al., "Long Range Language Modeling via Gated State Spaces" (2023)
  - https://arxiv.org/abs/2206.13947
  """

  alias Edifice.Blocks.FFN

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:state_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default SSM state dimension"
  @spec default_state_size() :: pos_integer()
  def default_state_size, do: 16

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a GSS model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:state_size` - SSM state dimension (default: 16)
    - `:num_layers` - Number of GSS blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block = build_gss_block(acc, Keyword.merge(opts, layer_idx: layer_idx))

        residual = Axon.add(acc, block, name: "residual_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(residual, rate: dropout, name: "dropout_#{layer_idx}")
        else
          residual
        end
      end)

    output = Axon.layer_norm(output, name: "final_norm")

    Axon.nx(
      output,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # GSS Block
  # ============================================================================

  defp build_gss_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    state_size = Keyword.get(opts, :state_size, default_state_size())
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = "gss_block_#{layer_idx}"

    # 1. Gated SSM sub-layer
    normed = Axon.layer_norm(input, name: "#{name}_ssm_norm")

    # SSM path: linear -> scan with fixed A,B,C
    ssm_input = Axon.dense(normed, hidden_size, name: "#{name}_ssm_in")

    # Fixed SSM parameters (learned but not input-dependent)
    a_log = Axon.param("#{name}_a_log", {hidden_size, state_size}, initializer: &init_a_log/2)
    b_param = Axon.param("#{name}_b", {hidden_size, state_size}, initializer: :glorot_uniform)
    c_param = Axon.param("#{name}_c", {hidden_size, state_size}, initializer: :glorot_uniform)

    ssm_out =
      Axon.layer(
        &gss_ssm_impl/5,
        [ssm_input, a_log, b_param, c_param],
        name: "#{name}_ssm",
        hidden_size: hidden_size,
        state_size: state_size,
        op_name: :gss_ssm
      )

    # Gate path: sigmoid gating
    gate =
      normed
      |> Axon.dense(hidden_size, name: "#{name}_gate_proj")
      |> Axon.activation(:sigmoid, name: "#{name}_gate_sigmoid")

    # Gated output
    gated = Axon.multiply(ssm_out, gate, name: "#{name}_gated")

    # Project back
    ssm_block_out = Axon.dense(gated, hidden_size, name: "#{name}_out_proj")

    after_ssm = Axon.add(input, ssm_block_out, name: "#{name}_ssm_residual")

    # 2. FFN sub-layer
    ffn_normed = Axon.layer_norm(after_ssm, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        expansion_factor: 4,
        name: "#{name}_ffn"
      )

    Axon.add(after_ssm, ffn_out, name: "#{name}_ffn_residual")
  end

  # ============================================================================
  # GSS SSM Implementation
  # ============================================================================

  # SSM with fixed diagonal A, learned B, C
  defp gss_ssm_impl(x, a_log, b_param, c_param, _opts) do
    # x: [batch, seq_len, hidden_size]
    # a_log: [hidden_size, state_size] (log-space for stability)
    # b_param: [hidden_size, state_size]
    # c_param: [hidden_size, state_size]
    batch_size = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)
    hidden_size = Nx.axis_size(x, 2)
    state_size = Nx.axis_size(a_log, 1)

    # A = -softplus(a_log) for stability (negative diagonal)
    a_diag = Nx.negate(Nx.log1p(Nx.exp(a_log)))

    # Discretize: A_bar = exp(A) per timestep (fixed dt=1)
    a_bar = Nx.exp(a_diag)

    # Initialize hidden state
    h = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(x)), {batch_size, hidden_size, state_size})

    # Sequential scan: h_t = A_bar * h_{t-1} + B * x_t
    {_, outputs} =
      Enum.reduce(0..(seq_len - 1), {h, []}, fn t, {h_prev, acc} ->
        # x_t: [batch, hidden_size]
        x_t = Nx.slice_along_axis(x, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # h_t = A_bar * h_{t-1} + B * x_t
        # a_bar: [hidden_size, state_size] broadcast to [batch, hidden_size, state_size]
        # x_t: [batch, hidden_size] -> [batch, hidden_size, 1]
        x_expanded = Nx.new_axis(x_t, 2)
        bx = Nx.multiply(b_param, x_expanded)
        h_new = Nx.add(Nx.multiply(a_bar, h_prev), bx)

        # y_t = sum(C * h_t, axis=state_size) -> [batch, hidden_size]
        y_t = Nx.sum(Nx.multiply(c_param, h_new), axes: [2])

        {h_new, [y_t | acc]}
      end)

    # Stack: [batch, seq_len, hidden_size]
    outputs
    |> Enum.reverse()
    |> Nx.stack(axis: 1)
  end

  # Initialize A in log-space so that exp(A) is in a good decay range
  defp init_a_log(shape, _opts) do
    # Initialize so that A_bar = exp(-softplus(a_log)) is in [0.9, 0.999]
    # softplus(a_log) should be in [0.001, 0.105]
    # a_log ~ uniform(-5, -2) gives softplus in roughly this range
    key = Nx.Random.key(System.system_time(:nanosecond))
    {values, _} = Nx.Random.uniform(key, -5.0, -2.0, shape: shape)
    values
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the output size of a GSS model."
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      state_size: 16,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
