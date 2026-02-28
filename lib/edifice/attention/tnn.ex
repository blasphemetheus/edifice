defmodule Edifice.Attention.TNN do
  @moduledoc """
  Toeplitz Neural Network (TNN): Sequence modeling via learned Toeplitz convolutions.

  Implements the TNN architecture from "Toeplitz Neural Network for Sequence
  Modeling" (Qin et al., ICLR 2023 Spotlight). TNN replaces attention with
  learned position-based Toeplitz convolutions, achieving O(n log n) token
  mixing with excellent length extrapolation.

  ## Key Innovation

  A Toeplitz matrix T[i,j] = t[i-j] captures relative-position interactions
  with only O(n) parameters. TNN learns these coefficients via a Relative
  Position Encoder (RPE) — a small MLP mapping position indices to filter
  weights — applied as a causal long convolution.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Input projection to hidden_size
        |
  +--------------------------------------+
  |   TNN Block (x num_layers)           |
  |                                      |
  |   LayerNorm -> GTU (token mixing)    |
  |     u = SiLU(W_u * x)   (gate)      |
  |     v = SiLU(W_v * x)   (value)     |
  |     k = RPE(positions)   (kernel)    |
  |     k = k * gamma^pos    (decay)     |
  |     v = CausalConv(v, k) (TNO)      |
  |     out = W_out(RMSNorm(u * v))      |
  |   -> Residual                        |
  |                                      |
  |   LayerNorm -> GLU (channel mixing)  |
  |     g = SiLU(W_g * x)               |
  |     o = W_v * x                      |
  |     out = W_out(g * o)               |
  |   -> Residual                        |
  +--------------------------------------+
        |
  Final LayerNorm
        |
  Last timestep -> [batch, hidden_size]
  ```

  ## Complexity

  | Mechanism | Training | Inference |
  |-----------|----------|-----------|
  | Softmax Attention | O(n^2 d) | O(n^2 d) |
  | TNN (Toeplitz) | O(n d log n) | O(n d log n) |
  | TNN + ETSC | O(n d log n) | O(d) per step |

  ## Usage

      model = TNN.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        expand_ratio: 3,
        rpe_layers: 3
      )

  ## References
  - "Toeplitz Neural Network for Sequence Modeling" (Qin et al., ICLR 2023)
  - arXiv: https://arxiv.org/abs/2305.04749
  - "Accelerating TNN with Constant-time Inference" (EMNLP 2023)
  """

  @default_hidden_size 256
  @default_num_layers 4
  @default_expand_ratio 3
  @default_rpe_dim 32
  @default_rpe_layers 3
  @default_gamma 0.99
  @default_dropout 0.0
  @default_window_size 60

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:expand_ratio, pos_integer()}
          | {:rpe_dim, pos_integer()}
          | {:rpe_layers, pos_integer()}
          | {:rpe_activation, :relu | :silu}
          | {:activation, :relu | :silu}
          | {:causal, boolean()}
          | {:use_decay, boolean()}
          | {:gamma, float()}
          | {:dropout, float()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Toeplitz Neural Network for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of TNN blocks (default: 4)
    - `:expand_ratio` - GTU expansion factor (default: 3)
    - `:rpe_dim` - RPE MLP hidden dimension (default: max(hidden_size/8, 32))
    - `:rpe_layers` - Number of RPE hidden layers (default: 3)
    - `:rpe_activation` - RPE activation function (default: :relu)
    - `:activation` - GTU/GLU gate activation (default: :silu)
    - `:causal` - Use causal convolution (default: true)
    - `:use_decay` - Apply exponential decay to kernel (default: true)
    - `:gamma` - Decay rate (default: 0.99)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.

  ## Examples

      iex> model = Edifice.Attention.TNN.build(embed_dim: 32, hidden_size: 16, num_layers: 1, rpe_layers: 1)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input = Axon.input("state_sequence", shape: {nil, seq_len, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    block_opts = [
      hidden_size: hidden_size,
      expand_ratio: Keyword.get(opts, :expand_ratio, @default_expand_ratio),
      rpe_dim: Keyword.get(opts, :rpe_dim, max(div(hidden_size, 8), @default_rpe_dim)),
      rpe_layers: Keyword.get(opts, :rpe_layers, @default_rpe_layers),
      rpe_activation: Keyword.get(opts, :rpe_activation, :relu),
      activation: Keyword.get(opts, :activation, :silu),
      causal: Keyword.get(opts, :causal, true),
      use_decay: Keyword.get(opts, :use_decay, true),
      gamma: Keyword.get(opts, :gamma, @default_gamma),
      dropout: dropout,
      seq_len: seq_len
    ]

    x =
      Enum.reduce(1..num_layers, x, fn i, acc ->
        build_tnn_block(acc, Keyword.put(block_opts, :name, "tnn_#{i}"))
      end)

    x = Axon.layer_norm(x, name: "final_norm")

    Axon.nx(
      x,
      fn tensor ->
        s = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, s - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # TNN Block: GTU (token mixing) + GLU (channel mixing)
  # ============================================================================

  defp build_tnn_block(input, opts) do
    hidden_size = opts[:hidden_size]
    dropout = opts[:dropout]
    activation = opts[:activation]
    name = opts[:name]

    # Token mixing: Gated Toeplitz Unit
    normed = Axon.layer_norm(input, name: "#{name}_gtu_norm")
    gtu_out = build_gtu(normed, opts)
    gtu_out = maybe_dropout(gtu_out, dropout, "#{name}_gtu_drop")
    x = Axon.add(input, gtu_out, name: "#{name}_gtu_res")

    # Channel mixing: Gated Linear Unit
    normed2 = Axon.layer_norm(x, name: "#{name}_glu_norm")
    glu_out = build_glu(normed2, hidden_size, activation, "#{name}_glu")
    glu_out = maybe_dropout(glu_out, dropout, "#{name}_glu_drop")
    Axon.add(x, glu_out, name: "#{name}_glu_res")
  end

  # ============================================================================
  # GTU: Gated Toeplitz Unit (token mixing via learned convolution)
  # ============================================================================

  defp build_gtu(input, opts) do
    hidden_size = opts[:hidden_size]
    expand_ratio = opts[:expand_ratio]
    seq_len = opts[:seq_len]
    gamma = opts[:gamma]
    use_decay = opts[:use_decay]
    rpe_dim = opts[:rpe_dim]
    rpe_layers_count = opts[:rpe_layers]
    rpe_activation = opts[:rpe_activation]
    activation = opts[:activation]
    name = opts[:name]

    expand_dim = hidden_size * expand_ratio

    # Gate and value projections: [batch, seq, hidden] -> [batch, seq, expand]
    u =
      input
      |> Axon.dense(expand_dim, name: "#{name}_u")
      |> Axon.activation(activation, name: "#{name}_u_act")

    v =
      input
      |> Axon.dense(expand_dim, name: "#{name}_v")
      |> Axon.activation(activation, name: "#{name}_v_act")

    # RPE generates Toeplitz kernel from normalized positions [0, 1]
    positions =
      Nx.divide(
        Nx.iota({1, seq_len, 1}, axis: 1, type: :f32),
        max(seq_len - 1, 1)
      )

    pos_node = Axon.constant(positions)

    kernel =
      build_rpe(
        pos_node,
        expand_dim,
        rpe_dim,
        rpe_layers_count,
        rpe_activation,
        "#{name}_rpe"
      )

    # Exponential decay biases kernel toward local interactions
    kernel =
      if use_decay do
        decay =
          Nx.pow(gamma, Nx.iota({1, seq_len, 1}, axis: 1, type: :f32))

        Axon.multiply(kernel, Axon.constant(decay), name: "#{name}_decay")
      else
        kernel
      end

    # Causal Toeplitz convolution on value branch
    v_conv =
      Axon.layer(
        &causal_long_conv_impl/3,
        [v, kernel],
        name: "#{name}_tno",
        op_name: :toeplitz_conv
      )

    # Gate: u * TNO(v), then SimpleRMSNorm + output projection
    Axon.multiply(u, v_conv, name: "#{name}_gate")
    |> Axon.nx(&simple_rms_norm/1, name: "#{name}_srms")
    |> Axon.dense(hidden_size, name: "#{name}_out")
  end

  # ============================================================================
  # RPE: Relative Position Encoder (small MLP generating kernel coefficients)
  # ============================================================================

  defp build_rpe(positions, output_dim, rpe_dim, num_layers, activation, name) do
    # Input projection: [1, seq_len, 1] -> [1, seq_len, rpe_dim]
    x = Axon.dense(positions, rpe_dim, name: "#{name}_in")

    # Hidden layers: Norm -> Act -> Dense
    x =
      Enum.reduce(1..num_layers, x, fn i, acc ->
        acc
        |> Axon.nx(&simple_rms_norm/1, name: "#{name}_norm_#{i}")
        |> Axon.activation(activation, name: "#{name}_act_#{i}")
        |> Axon.dense(rpe_dim, name: "#{name}_h_#{i}")
      end)

    # Output projection: -> [1, seq_len, output_dim]
    x
    |> Axon.nx(&simple_rms_norm/1, name: "#{name}_norm_out")
    |> Axon.activation(activation, name: "#{name}_act_out")
    |> Axon.dense(output_dim, name: "#{name}_out")
  end

  # ============================================================================
  # SimpleRMSNorm: x / sqrt(mean(x^2) + eps) — no learnable parameters
  # ============================================================================

  defp simple_rms_norm(x) do
    variance = Nx.mean(Nx.multiply(x, x), axes: [-1], keep_axes: true)
    Nx.divide(x, Nx.sqrt(Nx.add(variance, 1.0e-6)))
  end

  # ============================================================================
  # Causal long convolution via depthwise Nx.conv
  # ============================================================================

  # signal: [batch, seq_len, channels]
  # filter: [1, seq_len, channels] (RPE-generated kernel, batch=1)
  defp causal_long_conv_impl(signal, filter, _opts) do
    channels = Nx.axis_size(signal, 2)
    seq_len = Nx.axis_size(signal, 1)

    # Transpose to channel-first for Nx.conv: [batch, channels, seq_len]
    signal_t = Nx.transpose(signal, axes: [0, 2, 1])

    # Squeeze filter batch dim: [seq_len, channels]
    h = Nx.squeeze(filter, axes: [0])

    # Reverse for convolution (Nx.conv does cross-correlation, flip for true conv)
    h_rev = Nx.reverse(h, axes: [0])

    # Reshape to depthwise kernel: [channels, 1, seq_len]
    kernel =
      h_rev
      |> Nx.transpose(axes: [1, 0])
      |> Nx.reshape({channels, 1, seq_len})

    # Causal depthwise conv: left-pad by (seq_len - 1)
    result =
      Nx.conv(signal_t, kernel,
        padding: [{seq_len - 1, 0}],
        feature_group_size: channels
      )

    # Transpose back: [batch, channels, seq_len] -> [batch, seq_len, channels]
    Nx.transpose(result, axes: [0, 2, 1])
  end

  # ============================================================================
  # GLU: Gated Linear Unit (channel mixing, no sequence interaction)
  # ============================================================================

  defp build_glu(input, hidden_size, activation, name) do
    gate =
      input
      |> Axon.dense(hidden_size, name: "#{name}_g")
      |> Axon.activation(activation, name: "#{name}_g_act")

    value = Axon.dense(input, hidden_size, name: "#{name}_v")

    Axon.multiply(gate, value, name: "#{name}_mul")
    |> Axon.dense(hidden_size, name: "#{name}_out")
  end

  defp maybe_dropout(x, dropout, name) do
    if dropout > 0, do: Axon.dropout(x, rate: dropout, name: name), else: x
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a TNN model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a TNN model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    expand_ratio = Keyword.get(opts, :expand_ratio, @default_expand_ratio)
    rpe_dim = Keyword.get(opts, :rpe_dim, max(div(hidden_size, 8), @default_rpe_dim))
    rpe_layers_count = Keyword.get(opts, :rpe_layers, @default_rpe_layers)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    expand_dim = hidden_size * expand_ratio

    # GTU: u_proj + v_proj + RPE + out_proj
    rpe_params =
      1 * rpe_dim + rpe_dim * rpe_dim * rpe_layers_count + rpe_dim * expand_dim

    gtu = hidden_size * expand_dim * 2 + rpe_params + expand_dim * hidden_size

    # GLU: gate + value + out (all hidden -> hidden)
    glu = hidden_size * hidden_size * 3

    per_layer = gtu + glu
    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0
    input_proj + per_layer * num_layers
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      expand_ratio: 3,
      rpe_dim: 32,
      rpe_layers: 3,
      rpe_activation: :relu,
      activation: :silu,
      causal: true,
      use_decay: true,
      gamma: 0.99,
      window_size: 60,
      dropout: 0.0
    ]
  end
end
