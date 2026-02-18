defmodule Edifice.SSM.HybridBuilder do
  @moduledoc """
  Flexible hybrid architecture builder for combining different layer types.

  This module provides a declarative way to construct hybrid models by specifying
  a sequence of layer types. Unlike Jamba/Zamba which have fixed patterns, this
  allows arbitrary combinations:

  ## Supported Layer Types

  | Type | Module | Complexity | Best For |
  |------|--------|------------|----------|
  | `:mamba` | GatedSSM | O(L) | Long sequences |
  | `:attention` | Attention | O(L^2) | Global context |
  | `:gla` | GLA | O(L) | Fast linear attention |
  | `:rwkv` | RWKV | O(L) | Linear RNN |
  | `:ffn` | Dense+GELU | O(1) | Feature transform |
  | `:kan` | KAN | O(1) | Learnable activations |

  ## Usage

      # Custom hybrid: [Mamba, Mamba, Attention, Mamba, GLA, FFN]
      pattern = [:mamba, :mamba, :attention, :mamba, :gla, :ffn]
      model = HybridBuilder.build(pattern, embed_dim: 287, hidden_size: 256)

      # With shared layers (like Zamba)
      model = HybridBuilder.build(
        [:mamba, :mamba, :mamba, :mamba, :mamba, :mamba],
        embed_dim: 287,
        shared_layers: %{attention: [2, 4, 6]}  # Apply shared attention after these
      )

  ## Predefined Patterns

      HybridBuilder.pattern(:jamba_like, 6)   # [M, A, M, A, M, A]
      HybridBuilder.pattern(:zamba_like, 6)   # [M, M, M, M, M, M] + shared attn
      HybridBuilder.pattern(:mamba_gla, 6)    # [M, M, GLA, M, M, GLA]
      HybridBuilder.pattern(:full_hybrid, 6)  # [M, A, GLA, RWKV, M, A]
  """

  alias Edifice.Attention.{GLA, RWKV}
  alias Edifice.Attention.MultiHead, as: Attention
  alias Edifice.Blocks.FFN
  alias Edifice.Feedforward.KAN
  alias Edifice.SSM.GatedSSM

  @type layer_type :: :mamba | :attention | :gla | :rwkv | :ffn | :kan
  @type pattern :: [layer_type()]

  @default_hidden_size 256
  @default_state_size 16
  @default_num_heads 4
  @default_head_dim 64
  @default_window_size 60
  @default_dropout 0.1

  @doc """
  Build a hybrid model from a layer pattern.

  ## Options

  **Required:**
    - `:embed_dim` - Input embedding dimension

  **Architecture:**
    - `:hidden_size` - Internal dimension (default: 256)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:shared_layers` - Map of layer_type => positions for shared weights

  **Layer-specific options** (prefixed by layer type):
    - `:mamba_state_size` - SSM state dimension (default: 16)
    - `:mamba_expand_factor` - Expansion factor (default: 2)
    - `:attention_num_heads` - Number of attention heads (default: 4)
    - `:attention_window_size` - Sliding window size (default: 60)
    - `:gla_num_heads` - GLA heads (default: 4)
    - `:rwkv_head_size` - RWKV head size (default: 64)
    - `:kan_grid_size` - KAN grid size (default: 5)

  ## Returns

    An Axon model outputting [batch, hidden_size].
  """
  @spec build(pattern(), keyword()) :: Axon.t()
  def build(pattern, opts) when is_list(pattern) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :attention_window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    shared_layers = Keyword.get(opts, :shared_layers, %{})

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, seq_len, embed_dim})

    # Project to hidden dimension
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Add positional encoding
    x = Attention.add_positional_encoding(x, name: "pos_encoding")

    # Build each layer in the pattern
    x =
      pattern
      |> Enum.with_index(1)
      |> Enum.reduce(x, fn {layer_type, idx}, acc ->
        layer_name = build_layer_name(layer_type, idx, shared_layers)

        build_layer(
          acc,
          layer_type,
          Keyword.merge(opts,
            hidden_size: hidden_size,
            dropout: dropout,
            name: layer_name,
            seq_len: seq_len
          )
        )
      end)

    # Apply shared layers at specified positions
    x = apply_shared_layers(x, pattern, shared_layers, opts)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep
    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Get a predefined layer pattern.

  ## Available Patterns

    - `:jamba_like` - Interleaved Mamba + Attention
    - `:zamba_like` - All Mamba (use with shared_layers)
    - `:mamba_gla` - Mamba + Gated Linear Attention
    - `:rwkv_attention` - RWKV + Sparse Attention
    - `:full_hybrid` - Mix of all layer types
    - `:ssm_stack` - Pure SSM (Mamba only)
  """
  @spec pattern(atom(), pos_integer()) :: pattern()
  def pattern(:jamba_like, num_layers) do
    # Alternate Mamba and Attention
    Enum.map(1..num_layers, fn idx ->
      if rem(idx, 2) == 0, do: :attention, else: :mamba
    end)
  end

  def pattern(:zamba_like, num_layers) do
    # All Mamba (shared attention applied separately)
    List.duplicate(:mamba, num_layers)
  end

  def pattern(:mamba_gla, num_layers) do
    # Mamba with periodic GLA
    Enum.map(1..num_layers, fn idx ->
      if rem(idx, 3) == 0, do: :gla, else: :mamba
    end)
  end

  def pattern(:rwkv_attention, num_layers) do
    # RWKV with sparse attention
    Enum.map(1..num_layers, fn idx ->
      if rem(idx, 4) == 0, do: :attention, else: :rwkv
    end)
  end

  def pattern(:full_hybrid, num_layers) do
    # Diverse mix for exploration
    layer_cycle = [:mamba, :attention, :gla, :rwkv, :mamba, :ffn]

    Enum.map(1..num_layers, fn idx ->
      Enum.at(layer_cycle, rem(idx - 1, length(layer_cycle)))
    end)
  end

  def pattern(:ssm_stack, num_layers) do
    List.duplicate(:mamba, num_layers)
  end

  @doc """
  Build with a named pattern.

  Convenience function combining pattern/2 and build/2.

      HybridBuilder.build_pattern(:jamba_like, 6, embed_dim: 287)
  """
  @spec build_pattern(atom(), pos_integer(), keyword()) :: Axon.t()
  def build_pattern(pattern_name, num_layers, opts) do
    build(pattern(pattern_name, num_layers), opts)
  end

  @doc """
  Estimate parameter count for a hybrid model.
  """
  @spec param_count(pattern(), keyword()) :: non_neg_integer()
  def param_count(pattern, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    state_size = Keyword.get(opts, :mamba_state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :mamba_expand_factor, 2)
    num_heads = Keyword.get(opts, :attention_num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :attention_head_dim, @default_head_dim)

    inner_size = hidden_size * expand_factor
    dt_rank = max(div(hidden_size, 16), 1)
    attn_hidden = num_heads * head_dim
    ffn_dim = hidden_size * 4

    # Per-layer costs
    layer_costs = %{
      mamba:
        hidden_size * (2 * inner_size) +
          inner_size * 4 +
          inner_size * (2 * state_size) +
          inner_size * dt_rank +
          dt_rank * inner_size +
          inner_size * hidden_size,
      attention:
        attn_hidden * 3 * attn_hidden +
          attn_hidden * attn_hidden +
          hidden_size * ffn_dim +
          ffn_dim * hidden_size,
      gla:
        hidden_size * (3 * num_heads * head_dim) +
          hidden_size * hidden_size,
      rwkv:
        hidden_size * hidden_size * 5 +
          hidden_size * ffn_dim +
          ffn_dim * hidden_size,
      ffn:
        hidden_size * ffn_dim +
          ffn_dim * hidden_size,
      kan: hidden_size * hidden_size * 10
    }

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    total_layers =
      Enum.reduce(pattern, 0, fn layer_type, acc ->
        acc + Map.get(layer_costs, layer_type, 0)
      end)

    input_proj + total_layers
  end

  @doc """
  Visualize a layer pattern as a string diagram.
  """
  @spec visualize(pattern()) :: String.t()
  def visualize(pattern) do
    type_chars = %{
      mamba: "M",
      attention: "A",
      gla: "G",
      rwkv: "R",
      ffn: "F",
      kan: "K"
    }

    header = "Layer pattern: "
    layers = Enum.map_join(pattern, " → ", &Map.get(type_chars, &1, "?"))

    type_legend =
      pattern
      |> Enum.uniq()
      |> Enum.map_join(", ", fn type ->
        "#{Map.get(type_chars, type, "?")}=#{type}"
      end)

    "#{header}#{layers}\nLegend: #{type_legend}"
  end

  # ============================================================================
  # Private Layer Builders
  # ============================================================================

  defp build_layer_name(layer_type, idx, shared_layers) do
    # Check if this position uses a shared layer
    positions = Map.get(shared_layers, layer_type, [])

    if idx in positions do
      "shared_#{layer_type}"
    else
      "#{layer_type}_#{idx}"
    end
  end

  defp apply_shared_layers(x, _pattern, shared_layers, _opts)
       when map_size(shared_layers) == 0 do
    x
  end

  defp apply_shared_layers(x, _pattern, _shared_layers, _opts) do
    # Shared layers are already applied via naming in build_layer
    x
  end

  defp build_layer(input, :mamba, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    state_size = Keyword.get(opts, :mamba_state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :mamba_expand_factor, 2)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.fetch!(opts, :name)

    # Pre-LayerNorm
    normalized = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # Mamba block
    block =
      GatedSSM.build_mamba_block(
        normalized,
        hidden_size: hidden_size,
        state_size: state_size,
        expand_factor: expand_factor,
        conv_size: 4,
        name: name
      )

    # Dropout + residual
    block = maybe_dropout(block, dropout, "#{name}_dropout")
    Axon.add(input, block, name: "#{name}_residual")
  end

  defp build_layer(input, :attention, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :attention_num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :attention_head_dim, @default_head_dim)
    window_size = Keyword.get(opts, :attention_window_size, @default_window_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len)
    name = Keyword.fetch!(opts, :name)

    attn_hidden = num_heads * head_dim

    # Pre-LayerNorm
    attn_input = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # Project to attention dimension if needed
    attn_input =
      if hidden_size != attn_hidden do
        Axon.dense(attn_input, attn_hidden, name: "#{name}_proj_in")
      else
        attn_input
      end

    # Precompute mask
    mask =
      if seq_len do
        Attention.window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
      else
        nil
      end

    # Sliding window attention
    attended =
      Attention.sliding_window_attention(attn_input,
        window_size: window_size,
        num_heads: num_heads,
        head_dim: head_dim,
        mask: mask,
        qk_layernorm: true,
        name: name
      )

    # Project back to hidden_size if needed
    attended =
      if hidden_size != attn_hidden do
        Axon.dense(attended, hidden_size, name: "#{name}_proj_out")
      else
        attended
      end

    attended = maybe_dropout(attended, dropout, "#{name}_dropout")
    x = Axon.add(input, attended, name: "#{name}_residual")

    # FFN sub-block
    build_ffn_sublayer(x, hidden_size, dropout, name)
  end

  defp build_layer(input, :gla, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :gla_num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :gla_head_dim, @default_head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.fetch!(opts, :name)

    # Pre-LayerNorm
    normalized = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # GLA block
    gla_out =
      GLA.build_gla_block(
        normalized,
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        name: name
      )

    gla_out = maybe_dropout(gla_out, dropout, "#{name}_dropout")
    x = Axon.add(input, gla_out, name: "#{name}_residual")

    # FFN sub-block
    build_ffn_sublayer(x, hidden_size, dropout, name)
  end

  defp build_layer(input, :rwkv, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    head_size = Keyword.get(opts, :rwkv_head_size, 64)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.fetch!(opts, :name)

    # Pre-LayerNorm
    normalized = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # RWKV time-mixing block
    time_out =
      RWKV.build_time_mixing(
        normalized,
        hidden_size: hidden_size,
        head_size: head_size,
        name: "#{name}_time"
      )

    time_out = maybe_dropout(time_out, dropout, "#{name}_time_dropout")
    x = Axon.add(input, time_out, name: "#{name}_time_residual")

    # RWKV channel-mixing block
    channel_input = Axon.layer_norm(x, name: "#{name}_channel_pre_norm")

    channel_out =
      RWKV.build_channel_mixing(
        channel_input,
        hidden_size: hidden_size,
        name: "#{name}_channel"
      )

    channel_out = maybe_dropout(channel_out, dropout, "#{name}_channel_dropout")
    Axon.add(x, channel_out, name: "#{name}_channel_residual")
  end

  defp build_layer(input, :ffn, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.fetch!(opts, :name)

    build_ffn_sublayer(input, hidden_size, dropout, name)
  end

  defp build_layer(input, :kan, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    grid_size = Keyword.get(opts, :kan_grid_size, 5)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.fetch!(opts, :name)

    # Pre-LayerNorm
    normalized = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # KAN layer
    kan_out =
      KAN.build_kan_layer(
        normalized,
        hidden_size,
        grid_size: grid_size,
        name: name
      )

    kan_out = maybe_dropout(kan_out, dropout, "#{name}_dropout")
    Axon.add(input, kan_out, name: "#{name}_residual")
  end

  defp build_layer(_input, unknown_type, _opts) do
    raise ArgumentError, "Unknown layer type: #{inspect(unknown_type)}"
  end

  defp build_ffn_sublayer(input, hidden_size, dropout, name) do
    ffn_input = Axon.layer_norm(input, name: "#{name}_ffn_pre_norm")

    ffn = FFN.layer(ffn_input, hidden_size: hidden_size, name: "#{name}_ffn")

    ffn = maybe_dropout(ffn, dropout, "#{name}_ffn_dropout")
    Axon.add(input, ffn, name: "#{name}_ffn_residual")
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)
end
