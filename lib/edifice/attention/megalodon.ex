defmodule Edifice.Attention.Megalodon do
  @moduledoc """
  MEGALODON: Mega-scale Model with Complex EMA and Timestep Normalization.

  Extends the Mega architecture with improvements for unlimited context length,
  from "MEGALODON: Efficient LLM Pretraining and Inference with Unlimited
  Context Length" (Meta, 2024).

  ## Key Improvements over Mega

  1. **Complex EMA (CEMA)**: Uses complex-valued exponential moving averages
     instead of real-valued. Complex values model oscillatory patterns naturally:
     `h_t = (alpha_r + i*alpha_i) * h_{t-1} + (1-|alpha|) * x_t`
     The real and imaginary parts capture phase-shifted temporal patterns.

  2. **Timestep normalization**: Normalizes hidden states by the effective
     number of timesteps contributing to them, enabling length generalization.
     Without this, EMA outputs scale differently for different sequence lengths.

  3. **Normalized attention with 2-hop residual**: Adds a second residual
     connection that spans two sub-layers (EMA + attention), providing a
     "shortcut" for gradient flow in deep models.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-------------------------------------+
  | MEGALODON Block                      |
  |  LayerNorm -> CEMA -> residual       |
  |  LayerNorm -> NormAttn -> residual   |
  |  + 2-hop residual (skip EMA+Attn)   |
  |  LayerNorm -> FFN -> residual        |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = Megalodon.build(
        embed_dim: 287,
        hidden_size: 256,
        ema_dim: 16,
        num_layers: 4
      )

  ## References

  - Ma et al., "MEGALODON: Efficient LLM Pretraining and Inference with
    Unlimited Context Length" (Meta, 2024)
  - https://arxiv.org/abs/2404.08801
  """

  alias Edifice.Blocks.FFN

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:ema_dim, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default EMA expansion dimension"
  @spec default_ema_dim() :: pos_integer()
  def default_ema_dim, do: 16

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
  Build a MEGALODON model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:ema_dim` - EMA expansion dimension (default: 16)
    - `:num_layers` - Number of MEGALODON blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that outputs [batch, hidden_size] from the last position.
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

    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_megalodon_block(acc,
          hidden_size: hidden_size,
          ema_dim: Keyword.get(opts, :ema_dim, default_ema_dim()),
          dropout: dropout,
          name: "megalodon_block_#{layer_idx}"
        )
      end)

    x = Axon.layer_norm(x, name: "final_norm")

    Axon.nx(
      x,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # MEGALODON Block
  # ============================================================================

  defp build_megalodon_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    ema_dim = Keyword.get(opts, :ema_dim, default_ema_dim())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    name = Keyword.get(opts, :name, "megalodon_block")

    # Save input for 2-hop residual
    two_hop_input = input

    # 1. Complex EMA sub-layer
    ema_normed = Axon.layer_norm(input, name: "#{name}_ema_norm")

    # CEMA parameters
    proj_w =
      Axon.param("#{name}_cema_proj_w", {hidden_size, ema_dim}, initializer: :glorot_uniform)

    proj_b = Axon.param("#{name}_cema_proj_b", {ema_dim}, initializer: :zeros)
    # Complex alpha: real and imaginary parts
    alpha_real = Axon.param("#{name}_cema_alpha_r", {ema_dim}, initializer: :zeros)
    alpha_imag = Axon.param("#{name}_cema_alpha_i", {ema_dim}, initializer: :zeros)
    out_w = Axon.param("#{name}_cema_out_w", {ema_dim, hidden_size}, initializer: :glorot_uniform)
    out_b = Axon.param("#{name}_cema_out_b", {hidden_size}, initializer: :zeros)

    ema_out =
      Axon.layer(
        &cema_impl/8,
        [ema_normed, proj_w, proj_b, alpha_real, alpha_imag, out_w, out_b],
        name: "#{name}_cema",
        op_name: :cema
      )

    ema_out = maybe_dropout(ema_out, dropout, "#{name}_ema_drop")
    x = Axon.add(input, ema_out, name: "#{name}_ema_residual")

    # 2. Normalized gated attention sub-layer
    attn_normed = Axon.layer_norm(x, name: "#{name}_attn_norm")

    attn_out =
      build_normalized_attention(attn_normed,
        hidden_size: hidden_size,
        name: "#{name}_nattn"
      )

    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(x, attn_out, name: "#{name}_attn_residual")

    # 2-hop residual: adds the input from before both EMA and attention
    x = Axon.add(x, two_hop_input, name: "#{name}_2hop_residual")

    # 3. FFN sub-layer
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.gated_layer(ffn_normed,
        hidden_size: hidden_size,
        activation: :silu,
        dropout: dropout,
        name: "#{name}_ffn"
      )

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # ============================================================================
  # Complex EMA (CEMA) with Timestep Normalization
  # ============================================================================

  defp cema_impl(input, proj_w, proj_b, alpha_real, alpha_imag, out_w, out_b, _opts) do
    # Project input to EMA space
    projected = Nx.dot(input, [2], proj_w, [0]) |> Nx.add(proj_b)

    # Complex alpha: alpha = sigmoid(alpha_r) * exp(i * alpha_i)
    # We represent complex EMA as two coupled real EMAs
    alpha_r = Nx.sigmoid(alpha_real)
    alpha_i = alpha_imag

    # Precompute cos/sin for complex rotation
    cos_alpha = Nx.multiply(alpha_r, Nx.cos(alpha_i))
    sin_alpha = Nx.multiply(alpha_r, Nx.sin(alpha_i))

    batch_size = Nx.axis_size(input, 0)
    ema_dim = Nx.axis_size(proj_w, 1)
    seq_len = Nx.axis_size(input, 1)

    # Complex magnitude for input scaling: 1 - |alpha|
    alpha_mag = alpha_r
    one_minus_mag = Nx.subtract(1.0, alpha_mag)

    # Initialize complex hidden state (real + imaginary)
    h_real = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(input)), {batch_size, ema_dim})
    h_imag = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(input)), {batch_size, ema_dim})
    # Timestep count for normalization
    count = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(input)), {batch_size, ema_dim})

    {ema_outputs, _} =
      Enum.reduce(0..(seq_len - 1), {[], {h_real, h_imag, count}}, fn t,
                                                                      {outputs, {hr, hi, cnt}} ->
        x_t = Nx.slice_along_axis(projected, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Complex EMA update:
        # h_t = alpha * h_{t-1} + (1-|alpha|) * x_t
        # where alpha multiplication is complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        new_hr =
          Nx.add(
            Nx.subtract(Nx.multiply(cos_alpha, hr), Nx.multiply(sin_alpha, hi)),
            Nx.multiply(one_minus_mag, x_t)
          )

        new_hi =
          Nx.add(
            Nx.add(Nx.multiply(sin_alpha, hr), Nx.multiply(cos_alpha, hi)),
            Nx.multiply(one_minus_mag, x_t)
          )

        # Timestep normalization: count effective timesteps
        new_count = Nx.add(Nx.multiply(alpha_mag, cnt), 1.0)

        # Normalize by effective count
        safe_count = Nx.max(new_count, 1.0)
        # Take real part, normalized
        y_t = Nx.divide(new_hr, safe_count)

        {[Nx.new_axis(y_t, 1) | outputs], {new_hr, new_hi, new_count}}
      end)

    ema_seq = ema_outputs |> Enum.reverse() |> Nx.concatenate(axis: 1)

    # Project back to hidden_size
    Nx.dot(ema_seq, [2], out_w, [0]) |> Nx.add(out_b)
  end

  # ============================================================================
  # Normalized Gated Attention
  # ============================================================================

  defp build_normalized_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    name = Keyword.get(opts, :name, "nattn")

    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    gate = Axon.dense(input, hidden_size, name: "#{name}_gate")
    gate = Axon.sigmoid(gate, name: "#{name}_gate_sigmoid")

    attn_out =
      Axon.layer(
        &normalized_attention_impl/5,
        [q, k, v, gate],
        name: "#{name}_compute",
        op_name: :normalized_attention
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out")
  end

  defp normalized_attention_impl(q, k, v, gate, _opts) do
    d_k = Nx.axis_size(k, 2)
    scale = Nx.sqrt(Nx.tensor(d_k, type: Nx.type(q)))
    scores = Nx.dot(q, [2], [0], k, [2], [0]) |> Nx.divide(scale)

    # Causal mask
    seq_len = Nx.axis_size(q, 1)
    batch_size = Nx.axis_size(q, 0)
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.greater_equal(rows, cols)
    mask = mask |> Nx.new_axis(0) |> Nx.broadcast({batch_size, seq_len, seq_len})

    scores = Nx.select(mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))

    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.add(Nx.sum(weights, axes: [-1], keep_axes: true), 1.0e-8))

    attn_out = Nx.dot(weights, [2], [0], v, [1], [0])

    # Gated output
    Nx.multiply(gate, attn_out)
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc "Get the output size of a MEGALODON model."
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      ema_dim: 16,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
