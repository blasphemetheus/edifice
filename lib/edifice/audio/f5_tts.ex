defmodule Edifice.Audio.F5TTS do
  @moduledoc """
  F5-TTS — Fairytaler that Fakes Fluent and Faithful speech with Flow matching.

  <!-- verified: true, date: 2026-02-28 -->

  A non-autoregressive text-to-speech system based on flow matching and a
  DiT (Diffusion Transformer) backbone. Predicts a velocity field that maps
  Gaussian noise to mel spectrograms via ODE integration.

  ## Architecture

  ```
  Text tokens [batch, text_len]
        |
  TextEncoder: Embedding + SinPosEmbed + ConvNeXtV2 x conv_layers
        |
  text_embed [batch, seq_len, text_dim]

  noisy_mel [batch, seq_len, mel_dim]   cond_mel [batch, seq_len, mel_dim]
        |                                      |
        +--------------------------------------+-------- text_embed
        |
  InputEmbedding: Concat -> Dense(dim) + ConvPositionEmbedding
        |
  x [batch, seq_len, dim]

  timestep [batch]
        |
  TimestepMLP: SinPosEmb -> Dense -> SiLU -> Dense
        |
  t_embed [batch, dim]

  DiT Block x depth:
    AdaLN (shift/scale/gate x 2 from t_embed)
    Self-Attention with RoPE
    FFN (GELU)
        |
  Final AdaLN + Dense(mel_dim)
        |
  velocity [batch, seq_len, mel_dim]
  ```

  ## Inputs

    - `"noisy_mel"` — `[batch, seq_len, mel_dim]` interpolated mel
    - `"cond_mel"` — `[batch, seq_len, mel_dim]` reference mel (zeros where generating)
    - `"text"` — `[batch, seq_len]` character token IDs (0 = filler)
    - `"timestep"` — `[batch]` flow step t in [0, 1]

  ## Output

    Velocity field `[batch, seq_len, mel_dim]`.

  ## References

  - Yushen Chen et al., "F5-TTS: A Fairytaler that Fakes Fluent and
    Faithful Speech with Flow Matching" (2024)
  - https://arxiv.org/abs/2410.06885
  """

  alias Edifice.Blocks.{AdaptiveNorm, RoPE, SinusoidalPE}

  @default_mel_dim 100
  @default_dim 1024
  @default_depth 22
  @default_heads 16
  @default_ff_mult 2
  @default_dropout 0.1
  @default_text_dim 512
  @default_text_num_embeds 256
  @default_conv_layers 4
  @default_conv_mult 2
  @default_conv_pos_kernel 31
  @default_conv_pos_groups 16

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:conv_layers, pos_integer()}
          | {:conv_mult, pos_integer()}
          | {:conv_pos_groups, pos_integer()}
          | {:conv_pos_kernel, pos_integer()}
          | {:depth, pos_integer()}
          | {:dim, pos_integer()}
          | {:dropout, float()}
          | {:ff_mult, pos_integer()}
          | {:heads, pos_integer()}
          | {:mel_dim, pos_integer()}
          | {:text_dim, pos_integer()}
          | {:text_num_embeds, pos_integer()}

  @doc """
  Build the F5-TTS velocity prediction network.

  ## Options

    - `:mel_dim` - Mel spectrogram channels (default: 100)
    - `:dim` - Hidden dimension (default: 1024)
    - `:depth` - Number of DiT blocks (default: 22)
    - `:heads` - Attention heads (default: 16)
    - `:ff_mult` - FFN dimension multiplier (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:text_dim` - Text embedding dimension (default: 512)
    - `:text_num_embeds` - Vocabulary size (default: 256)
    - `:conv_layers` - ConvNeXt V2 blocks for text (default: 4)
    - `:conv_mult` - ConvNeXt intermediate multiplier (default: 2)
    - `:conv_pos_kernel` - Conv position embedding kernel (default: 31)
    - `:conv_pos_groups` - Conv position embedding groups (default: 16)
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    mel_dim = Keyword.get(opts, :mel_dim, @default_mel_dim)
    dim = Keyword.get(opts, :dim, @default_dim)
    depth = Keyword.get(opts, :depth, @default_depth)
    heads = Keyword.get(opts, :heads, @default_heads)
    head_dim = div(dim, heads)
    ff_mult = Keyword.get(opts, :ff_mult, @default_ff_mult)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    text_dim = Keyword.get(opts, :text_dim, @default_text_dim)
    text_num_embeds = Keyword.get(opts, :text_num_embeds, @default_text_num_embeds)
    conv_layers = Keyword.get(opts, :conv_layers, @default_conv_layers)
    conv_mult = Keyword.get(opts, :conv_mult, @default_conv_mult)
    conv_pos_kernel = Keyword.get(opts, :conv_pos_kernel, @default_conv_pos_kernel)
    conv_pos_groups = Keyword.get(opts, :conv_pos_groups, @default_conv_pos_groups)

    # Inputs
    noisy_mel = Axon.input("noisy_mel", shape: {nil, nil, mel_dim})
    cond_mel = Axon.input("cond_mel", shape: {nil, nil, mel_dim})
    text = Axon.input("text", shape: {nil, nil})
    timestep = Axon.input("timestep", shape: {nil})

    # --- Text Encoder ---
    # Embedding (tokens shifted +1 in paper; we use num_embeds+1 to accommodate filler at 0)
    text_embed =
      text
      |> Axon.embedding(text_num_embeds + 1, text_dim, name: "text_embed")

    # Sinusoidal position embedding for text
    text_embed = add_sinusoidal_pe(text_embed, text_dim, "text_pe")

    # ConvNeXt V2 blocks
    text_embed =
      Enum.reduce(0..(conv_layers - 1), text_embed, fn i, acc ->
        convnext_v2_block(acc, text_dim, conv_mult, "text_conv_#{i}")
      end)

    # --- Timestep Embedding ---
    t_embed =
      SinusoidalPE.timestep_layer(timestep,
        hidden_size: div(dim, 4),
        num_steps: 1000,
        name: "timestep_pe"
      )
      |> Axon.dense(dim, name: "t_mlp_1")
      |> Axon.activation(:silu, name: "t_mlp_silu")
      |> Axon.dense(dim, name: "t_mlp_2")

    # --- Input Embedding ---
    # Concat noisy_mel + cond_mel + text_embed along feature dim
    input_cat =
      Axon.layer(
        fn mel, cond, txt, _opts ->
          Nx.concatenate([mel, cond, txt], axis: -1)
        end,
        [noisy_mel, cond_mel, text_embed],
        name: "input_cat",
        op_name: :concatenate
      )

    x =
      input_cat
      |> Axon.dense(dim, name: "input_proj")
      |> conv_position_embedding(dim, conv_pos_kernel, conv_pos_groups, "conv_pos")

    # --- DiT Blocks ---
    x =
      Enum.reduce(0..(depth - 1), x, fn i, acc ->
        dit_block(acc, t_embed, dim, heads, head_dim, ff_mult, dropout, "dit_#{i}")
      end)

    # --- Final AdaLN + Output projection ---
    # Final norm: SiLU(t) -> Dense(dim * 2) -> shift, scale
    final_params =
      t_embed
      |> Axon.activation(:silu, name: "final_adaln_silu")
      |> Axon.dense(dim * 2, name: "final_adaln_proj")

    x =
      x
      |> Axon.layer_norm(name: "final_ln")
      |> AdaptiveNorm.modulate(final_params, hidden_size: dim, offset: 0, name: "final_mod")
      |> Axon.dense(mel_dim, name: "output_proj")

    x
  end

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :mel_dim, @default_mel_dim)
  end

  # ===========================================================================
  # DiT Block: AdaLN(6-param) -> Self-Attention(RoPE) -> FFN(GELU)
  # ===========================================================================

  defp dit_block(x, t_embed, dim, heads, head_dim, ff_mult, dropout, name) do
    # Project timestep to 6 parameters: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    adaln_params =
      t_embed
      |> Axon.activation(:silu, name: "#{name}_adaln_silu")
      |> Axon.dense(dim * 6, name: "#{name}_adaln_proj")

    # --- Self-Attention sub-layer ---
    normed =
      x
      |> Axon.layer_norm(name: "#{name}_ln1")
      |> AdaptiveNorm.modulate(adaln_params,
        hidden_size: dim,
        offset: 0,
        name: "#{name}_mod_attn"
      )

    attn = self_attention_rope(normed, dim, heads, head_dim, dropout, name)

    attn =
      AdaptiveNorm.gate(attn, adaln_params,
        hidden_size: dim,
        gate_index: 2,
        name: "#{name}_gate_attn"
      )

    x = Axon.add(x, attn, name: "#{name}_res_attn")

    # --- FFN sub-layer ---
    normed2 =
      x
      |> Axon.layer_norm(name: "#{name}_ln2")
      |> AdaptiveNorm.modulate(adaln_params,
        hidden_size: dim,
        offset: 3,
        name: "#{name}_mod_ffn"
      )

    ff_dim = dim * ff_mult

    ff =
      normed2
      |> Axon.dense(ff_dim, name: "#{name}_ff1")
      |> Axon.activation(:gelu, name: "#{name}_ff_gelu")
      |> Axon.dropout(rate: dropout, name: "#{name}_ff_drop")
      |> Axon.dense(dim, name: "#{name}_ff2")

    ff =
      AdaptiveNorm.gate(ff, adaln_params,
        hidden_size: dim,
        gate_index: 5,
        name: "#{name}_gate_ffn"
      )

    Axon.add(x, ff, name: "#{name}_res_ffn")
  end

  # ===========================================================================
  # Self-Attention with RoPE
  # ===========================================================================

  defp self_attention_rope(x, dim, heads, head_dim, dropout, name) do
    q = Axon.dense(x, dim, name: "#{name}_q")
    k = Axon.dense(x, dim, name: "#{name}_k")
    v = Axon.dense(x, dim, name: "#{name}_v")

    # Apply RoPE to Q and K
    {q, k} = apply_rope_pair(q, k, heads, head_dim, "#{name}_rope")

    # Scaled dot-product attention (already in multi-head shape after RoPE)
    attn =
      Axon.layer(
        &sdpa_4d/4,
        [q, k, v],
        name: "#{name}_sdpa",
        heads: heads,
        head_dim: head_dim,
        op_name: :attention
      )

    attn
    |> Axon.dense(dim, name: "#{name}_out_proj")
    |> Axon.dropout(rate: dropout, name: "#{name}_attn_drop")
  end

  # Apply RoPE to Q and K, returning 4D tensors [batch, heads, seq, head_dim]
  defp apply_rope_pair(q, k, heads, head_dim, name) do
    qk =
      Axon.layer(
        fn q_t, k_t, _opts ->
          {batch, seq, _} = Nx.shape(q_t)

          # Reshape to [batch, heads, seq, head_dim]
          q_4d =
            q_t |> Nx.reshape({batch, seq, heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

          k_4d =
            k_t |> Nx.reshape({batch, seq, heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

          # Apply RoPE
          {q_rot, k_rot} = RoPE.apply_rotary_4d(q_4d, k_4d)

          # Stack along a new axis to return as single tensor
          Nx.stack([q_rot, k_rot])
        end,
        [q, k],
        name: name,
        op_name: :rope
      )

    # Split back into separate Q and K
    q_out =
      Axon.nx(qk, fn t -> Nx.squeeze(Nx.slice_along_axis(t, 0, 1, axis: 0), axes: [0]) end,
        name: "#{name}_q"
      )

    k_out =
      Axon.nx(qk, fn t -> Nx.squeeze(Nx.slice_along_axis(t, 1, 1, axis: 0), axes: [0]) end,
        name: "#{name}_k"
      )

    {q_out, k_out}
  end

  # SDPA for 4D Q/K (already [batch, heads, seq, head_dim]) and 3D V
  defp sdpa_4d(q, k, v, opts) do
    heads = opts[:heads]
    head_dim = opts[:head_dim]
    {batch, _heads, _seq, _hd} = Nx.shape(q)
    {_, seq, _} = Nx.shape(v)

    # Reshape V to 4D
    v_4d = v |> Nx.reshape({batch, seq, heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)
    weights = Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    out = Nx.dot(weights, [3], [0, 1], v_4d, [2], [0, 1])
    # [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
    out |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({batch, seq, heads * head_dim})
  end

  # ===========================================================================
  # Convolutional Position Embedding
  # ===========================================================================

  # Conv1d(dim, dim, kernel, groups) -> Mish -> Conv1d -> Mish, residual add
  defp conv_position_embedding(x, dim, kernel, groups, name) do
    h =
      x
      |> Axon.conv(dim,
        kernel_size: kernel,
        padding: :same,
        feature_group_size: groups,
        name: "#{name}_conv1"
      )
      |> Axon.activation(:mish, name: "#{name}_mish1")
      |> Axon.conv(dim,
        kernel_size: kernel,
        padding: :same,
        feature_group_size: groups,
        name: "#{name}_conv2"
      )
      |> Axon.activation(:mish, name: "#{name}_mish2")

    Axon.add(x, h, name: "#{name}_res")
  end

  # ===========================================================================
  # ConvNeXt V2 Block (text encoder)
  # ===========================================================================

  # Depthwise Conv -> LN -> Dense(intermediate) -> GELU -> GRN -> Dense(dim) + residual
  defp convnext_v2_block(x, dim, mult, name) do
    intermediate = dim * mult

    h =
      x
      |> Axon.conv(dim,
        kernel_size: 7,
        padding: :same,
        feature_group_size: dim,
        name: "#{name}_dw_conv"
      )
      |> Axon.layer_norm(name: "#{name}_ln")
      |> Axon.dense(intermediate, name: "#{name}_pw1")
      |> Axon.activation(:gelu, name: "#{name}_gelu")
      |> grn_layer(intermediate, "#{name}_grn")
      |> Axon.dense(dim, name: "#{name}_pw2")

    Axon.add(x, h, name: "#{name}_res")
  end

  # ===========================================================================
  # Global Response Normalization (GRN)
  # ===========================================================================

  # GRN: normalize by L2 norm along sequence dimension
  defp grn_layer(x, dim, name) do
    Axon.layer(
      fn t, gamma, beta, _opts ->
        # t: [batch, seq, dim]
        # L2 norm along seq dim
        gx = Nx.sqrt(Nx.sum(Nx.pow(t, 2), axes: [1], keep_axes: true))
        # Normalize: Gx / mean(Gx)
        nx_val = Nx.divide(gx, Nx.add(Nx.mean(gx, axes: [-1], keep_axes: true), 1.0e-6))
        Nx.add(Nx.add(Nx.multiply(gamma, Nx.multiply(t, nx_val)), beta), t)
      end,
      [x, Axon.param("gamma", {1, 1, dim}), Axon.param("beta", {1, 1, dim})],
      name: name,
      op_name: :grn
    )
  end

  # ===========================================================================
  # Sinusoidal Position Embedding for text
  # ===========================================================================

  defp add_sinusoidal_pe(x, _dim, name) do
    Axon.layer(
      fn t, _opts ->
        {_batch, seq, d} = Nx.shape(t)
        half = div(d, 2)

        positions = Nx.iota({seq}, type: :f32)

        freqs =
          Nx.exp(
            Nx.multiply(
              Nx.iota({half}, type: :f32),
              Nx.negate(Nx.divide(:math.log(10_000.0), half))
            )
          )

        angles = Nx.outer(positions, freqs)
        pe = Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: -1)
        # Broadcast: [1, seq, dim]
        Nx.add(t, Nx.new_axis(pe, 0))
      end,
      [x],
      name: name,
      op_name: :sinusoidal_pe
    )
  end
end
