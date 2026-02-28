defmodule Edifice.Generative.TarFlow do
  @moduledoc """
  TarFlow: Transformer-based Autoregressive Normalizing Flow.

  Uses masked self-attention on flattened image patches (or sequence tokens)
  to define an autoregressive flow. Each flow step applies a transformer
  block with causal masking to predict affine coupling parameters (scale
  and shift) for each position conditioned on all previous positions.

  ```
  TarFlow: z ~ N(0,I)  -->  [flow_1]  -->  [flow_2]  -->  ...  -->  x
                               |               |
                         Transformer       Transformer
                         (masked attn)     (masked attn)
  ```

  ## Key Innovation

  Reuses standard transformer architecture (attention + FFN) as flow coupling
  layers. The autoregressive masking pattern on the sequence of patches defines
  the triangular Jacobian needed for tractable likelihood computation. Unlike
  RealNVP-style flows that split dimensions, TarFlow treats each patch position
  as a conditioning variable for subsequent positions.

  ## Architecture

  ```
  Input [batch, num_patches, patch_dim]
        |
  +------------------------------------------+
  |  TarFlow Block (x num_flows)            |
  |                                          |
  |  Masked Transformer (causal attention)   |
  |  -> Predicts (scale, shift) per position |
  |  -> Affine transform: y = x * exp(s) + t|
  |  -> Accumulate log_det += sum(s)         |
  +------------------------------------------+
        |
  Output: {transformed, log_det_sum}
  ```

  Returns `{forward_model, inverse_model}` tuple where forward maps
  data to latent space and inverse maps latent to data.

  ## Usage

      {encoder, decoder} = TarFlow.build(
        input_size: 64,
        num_flows: 4,
        hidden_size: 128,
        num_heads: 4
      )

  ## Reference

  - Zhai et al., "Autoregressive Image Generation without Vector
    Quantization" (Apple, ICML 2025)
  """

  @default_input_size 64
  @default_num_flows 4
  @default_hidden_size 128
  @default_num_heads 4
  @default_dropout 0.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:num_flows, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:dropout, float()}

  @doc """
  Build a TarFlow model.

  Returns `{encoder, decoder}` where:
  - `encoder` maps data to latent space (forward pass, produces logdet)
  - `decoder` maps latent to data space (generation)

  ## Options

    - `:input_size` - Dimension of each input token/patch (required, default: 64)
    - `:num_flows` - Number of flow steps (default: 4)
    - `:hidden_size` - Transformer hidden dimension (default: 128)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:dropout` - Dropout rate (default: 0.0)

  ## Returns

    `{encoder_model, decoder_model}` tuple of Axon models.
  """
  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    input_size = Keyword.get(opts, :input_size, @default_input_size)
    num_flows = Keyword.get(opts, :num_flows, @default_num_flows)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    encoder = build_encoder(input_size, num_flows, hidden_size, num_heads, dropout)
    decoder = build_decoder(input_size, num_flows, hidden_size, num_heads, dropout)
    {encoder, decoder}
  end

  defp build_encoder(input_size, num_flows, hidden_size, num_heads, dropout) do
    input = Axon.input("input", shape: {nil, nil, input_size})

    # Project input to hidden dimension
    x = Axon.dense(input, hidden_size, name: "enc_input_proj")

    # Apply flow steps sequentially
    # Each step uses a masked transformer to predict scale and shift
    {x, log_det} =
      Enum.reduce(0..(num_flows - 1), {x, nil}, fn i, {curr, acc_logdet} ->
        # Transformer block with causal masking predicts affine params
        # Output is 2 * input_size: [scale, shift]
        params_pred =
          build_flow_transformer(curr, hidden_size, num_heads, dropout, "enc_flow_#{i}")

        affine_params = Axon.dense(params_pred, input_size * 2, name: "enc_affine_#{i}")

        # Apply affine transform
        {transformed, step_logdet} =
          build_affine_forward(input, affine_params, input_size, "enc_aff_fwd_#{i}")

        # Re-project for next step
        next_input = Axon.dense(transformed, hidden_size, name: "enc_reproj_#{i}")

        # Accumulate log determinant
        new_logdet =
          if acc_logdet do
            Axon.add(acc_logdet, step_logdet)
          else
            step_logdet
          end

        {next_input, new_logdet}
      end)

    # Final projection back to input space
    output = Axon.dense(x, input_size, name: "enc_output_proj")

    # Return container with output and log determinant
    Axon.container(%{output: output, log_det: log_det})
  end

  defp build_decoder(input_size, num_flows, hidden_size, num_heads, dropout) do
    input = Axon.input("latent", shape: {nil, nil, input_size})

    # Project to hidden dimension
    x = Axon.dense(input, hidden_size, name: "dec_input_proj")

    # Apply inverse flow steps in reverse order
    x =
      Enum.reduce((num_flows - 1)..0//-1, x, fn i, curr ->
        params_pred =
          build_flow_transformer(curr, hidden_size, num_heads, dropout, "dec_flow_#{i}")

        affine_params = Axon.dense(params_pred, input_size * 2, name: "dec_affine_#{i}")

        # Apply inverse affine transform
        inv = build_affine_inverse(input, affine_params, input_size, "dec_aff_inv_#{i}")

        # Re-project for next step
        Axon.dense(inv, hidden_size, name: "dec_reproj_#{i}")
      end)

    Axon.dense(x, input_size, name: "dec_output_proj")
  end

  # Build a single transformer block with causal masking for flow parameter prediction
  defp build_flow_transformer(input, hidden_size, num_heads, dropout, name) do
    head_dim = div(hidden_size, num_heads)

    # Self-attention with causal mask
    q = Axon.dense(input, hidden_size, name: "#{name}_q")
    k = Axon.dense(input, hidden_size, name: "#{name}_k")
    v = Axon.dense(input, hidden_size, name: "#{name}_v")

    attn_out =
      Axon.layer(
        &causal_attention_impl/4,
        [q, k, v],
        name: "#{name}_attn",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :causal_attention
      )

    attn_out = Axon.dense(attn_out, hidden_size, name: "#{name}_attn_proj")

    # Residual + LayerNorm
    x = Axon.add(input, attn_out)
    x = Axon.layer_norm(x, name: "#{name}_ln1")

    # FFN
    ffn =
      x
      |> Axon.dense(hidden_size * 4, name: "#{name}_ffn1")
      |> Axon.activation(:silu)
      |> Axon.dense(hidden_size, name: "#{name}_ffn2")

    ffn = if dropout > 0, do: Axon.dropout(ffn, rate: dropout), else: ffn

    # Residual + LayerNorm
    x = Axon.add(x, ffn)
    Axon.layer_norm(x, name: "#{name}_ln2")
  end

  # Causal self-attention
  defp causal_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    {batch, seq_len, _} = Nx.shape(q)

    q = reshape_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_heads(v, batch, seq_len, num_heads, head_dim)

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.greater_equal(rows, cols)

    mask =
      mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores =
      Nx.select(
        mask,
        scores,
        Nx.broadcast(Nx.tensor(-1.0e9, type: Nx.type(scores)), Nx.shape(scores))
      )

    max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scores, max_s))
    weights = Nx.divide(exp_s, Nx.sum(exp_s, axes: [-1], keep_axes: true))

    out = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    out
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp reshape_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Forward affine transform: y = x * exp(s) + t
  # Returns {transformed, log_det}
  defp build_affine_forward(input, affine_params, input_size, name) do
    transformed =
      Axon.layer(
        fn x, params, _opts ->
          {s, t} = split_affine_params(params, input_size)
          # Clamp scale for stability
          s = Nx.clip(s, -5.0, 5.0)
          Nx.add(Nx.multiply(x, Nx.exp(s)), t)
        end,
        [input, affine_params],
        name: "#{name}_transform",
        op_name: :affine_forward
      )

    log_det =
      Axon.layer(
        fn _x, params, _opts ->
          {s, _t} = split_affine_params(params, input_size)
          s = Nx.clip(s, -5.0, 5.0)
          # Sum log-det over input dims
          Nx.sum(s, axes: [-1])
        end,
        [input, affine_params],
        name: "#{name}_logdet",
        op_name: :affine_logdet
      )

    {transformed, log_det}
  end

  # Inverse affine transform: x = (y - t) * exp(-s)
  defp build_affine_inverse(input, affine_params, input_size, name) do
    Axon.layer(
      fn y, params, _opts ->
        {s, t} = split_affine_params(params, input_size)
        s = Nx.clip(s, -5.0, 5.0)
        Nx.multiply(Nx.subtract(y, t), Nx.exp(Nx.negate(s)))
      end,
      [input, affine_params],
      name: name,
      op_name: :affine_inverse
    )
  end

  # Split affine parameters into scale and shift
  defp split_affine_params(params, input_size) do
    s = Nx.slice_along_axis(params, 0, input_size, axis: -1)
    t = Nx.slice_along_axis(params, input_size, input_size, axis: -1)
    {s, t}
  end

  @doc """
  Get the output dimension for a model configuration.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :input_size, @default_input_size)
  end

  @doc """
  Recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      input_size: 64,
      num_flows: 4,
      hidden_size: 128,
      num_heads: 4,
      dropout: 0.0
    ]
  end
end
