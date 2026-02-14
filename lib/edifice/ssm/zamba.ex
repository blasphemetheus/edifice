defmodule Edifice.SSM.Zamba do
  @moduledoc """
  Zamba: Mamba with Single Shared Attention layer.

  Named after Zyphra's Zamba architecture, this module implements a more efficient
  hybrid than Jamba by using a single shared attention layer that is applied at
  regular intervals throughout the Mamba stack.

  ## Key Difference from Jamba

  | Aspect | Jamba | Zamba |
  |--------|-------|-------|
  | Attention layers | Multiple (interleaved) | One (shared weights, reused) |
  | KV cache | O(L x N_attn) | O(L) - 10x reduction |
  | Parameters | Higher | Lower |
  | Pattern | [M, M, A, M, M, A] | [M, M, M, M, M, M] -> A (shared) |

  ## Architecture Pattern

  ```
  Input [batch, seq_len, embed_dim]
        │
        ▼
  ┌─────────────────────────────────────┐
  │  Mamba Block 1                       │
  ├─────────────────────────────────────┤
  │  Mamba Block 2                       │
  ├─────────────────────────────────────┤
  │  ...                                 │
  ├─────────────────────────────────────┤
  │  Mamba Block N                       │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │  Shared Attention (applied every K) │ <- Same weights, reused
  └──────────────┬──────────────────────┘
                 │
                 ▼
  [batch, hidden_size]
  ```

  ## Why Single Shared Attention Works

  The insight from Zamba: attention layers primarily serve to propagate
  information globally, not to learn diverse patterns. A single layer
  with shared weights can achieve similar global information flow at
  a fraction of the parameter cost.

  ## Usage

      # Default Zamba (6 Mamba layers, 1 shared attention applied every 3)
      model = Zamba.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 6,
        attention_every: 3
      )

      # Minimal attention variant (attention only at the end)
      model = Zamba.build(
        embed_dim: 287,
        num_layers: 6,
        attention_every: 6  # Only applied after final layer
      )

  ## Reference

  - Paper: "Zamba: A Compact 7B SSM Hybrid Model" (arXiv:2405.16712)
  - Key insight: Single shared attention achieves 10x KV cache reduction
  """

  require Axon

  alias Edifice.Attention.MultiHead, as: Attention
  alias Edifice.SSM.GatedSSM

  # Default hyperparameters
  @default_hidden_size 256
  @default_state_size 16
  @default_expand_factor 2
  @default_conv_size 4
  @default_num_layers 6
  # Apply shared attention every 3 Mamba layers
  @default_attention_every 3
  @default_num_heads 4
  @default_head_dim 64
  @default_window_size 60
  @default_dropout 0.1

  @doc """
  Build a Zamba model (Mamba + Single Shared Attention).

  ## Options

  **Architecture:**
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of Mamba layers (default: 6)
    - `:attention_every` - Apply shared attention every N Mamba layers (default: 3)

  **Mamba-specific:**
    - `:state_size` - SSM state dimension (default: 16)
    - `:expand_factor` - Mamba expansion factor (default: 2)
    - `:conv_size` - Causal conv kernel size (default: 4)

  **Shared Attention:**
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per attention head (default: 64)
    - `:window_size` - Attention window size (default: 60)

  **General:**
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Fixed sequence length for JIT optimization (default: window_size)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.

  ## Example

      model = Zamba.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 6,
        attention_every: 3  # Shared attention applied 2x total
      )
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    attention_every = Keyword.get(opts, :attention_every, @default_attention_every)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Mamba options
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)

    # Shared attention options
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Pre-compute attention mask for sliding window
    precomputed_mask =
      if seq_len do
        Attention.window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
      else
        nil
      end

    attn_hidden_dim = num_heads * head_dim

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project input to hidden dimension if different
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Add positional encoding (important for attention to know positions)
    x = Attention.add_positional_encoding(x, name: "pos_encoding")

    # Build Mamba layers with shared attention applied periodically
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        # Mamba layer
        mamba_out =
          build_mamba_layer(
            acc,
            hidden_size: hidden_size,
            state_size: state_size,
            expand_factor: expand_factor,
            conv_size: conv_size,
            dropout: dropout,
            name: "mamba_layer_#{layer_idx}"
          )

        # Apply shared attention at intervals (using SAME weights for all applications)
        if rem(layer_idx, attention_every) == 0 do
          apply_shared_attention(
            mamba_out,
            hidden_size: hidden_size,
            attn_hidden_dim: attn_hidden_dim,
            num_heads: num_heads,
            head_dim: head_dim,
            dropout: dropout,
            window_size: window_size,
            precomputed_mask: precomputed_mask,
            # Same name = shared weights
            name: "shared_attention"
          )
        else
          mamba_out
        end
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
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
  Build a Mamba layer with residual connection.

  Uses Pre-LayerNorm for stability.
  """
  @spec build_mamba_layer(Axon.t(), keyword()) :: Axon.t()
  def build_mamba_layer(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "mamba_layer")

    # Pre-LayerNorm
    normalized_input = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # Mamba block
    block =
      GatedSSM.build_mamba_block(
        normalized_input,
        hidden_size: hidden_size,
        state_size: state_size,
        expand_factor: expand_factor,
        conv_size: conv_size,
        name: name
      )

    # Apply dropout to block output
    block =
      if dropout > 0 do
        Axon.dropout(block, rate: dropout, name: "#{name}_dropout")
      else
        block
      end

    # Residual connection with original input
    Axon.add(input, block, name: "#{name}_residual")
  end

  @doc """
  Apply the single shared attention layer.

  The key insight: by using the same layer name ("shared_attention"),
  Axon will reuse the same parameters for all applications of this layer.
  This is what makes Zamba different from Jamba - one set of attention
  weights, applied multiple times.
  """
  @spec apply_shared_attention(Axon.t(), keyword()) :: Axon.t()
  def apply_shared_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    attn_hidden_dim = Keyword.get(opts, :attn_hidden_dim, hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    precomputed_mask = Keyword.get(opts, :precomputed_mask, nil)
    # CRITICAL: Same name = shared weights
    name = Keyword.get(opts, :name, "shared_attention")

    # Pre-LayerNorm
    attn_input = Axon.layer_norm(input, name: "#{name}_pre_norm")

    # Project to attention dimension if needed
    attn_input =
      if hidden_size != attn_hidden_dim do
        Axon.dense(attn_input, attn_hidden_dim, name: "#{name}_proj_in")
      else
        attn_input
      end

    # Sliding window attention
    attended =
      Attention.sliding_window_attention(attn_input,
        window_size: window_size,
        num_heads: num_heads,
        head_dim: head_dim,
        mask: precomputed_mask,
        qk_layernorm: true,
        name: name
      )

    # Project back to hidden_size if needed
    attended =
      if hidden_size != attn_hidden_dim do
        Axon.dense(attended, hidden_size, name: "#{name}_proj_out")
      else
        attended
      end

    # Dropout
    attended =
      if dropout > 0 do
        Axon.dropout(attended, rate: dropout, name: "#{name}_dropout")
      else
        attended
      end

    # Residual connection
    x = Axon.add(input, attended, name: "#{name}_residual")

    # FFN sub-block (also shared)
    ffn_input = Axon.layer_norm(x, name: "#{name}_ffn_pre_norm")
    ffn_dim = hidden_size * 4

    ffn =
      ffn_input
      |> Axon.dense(ffn_dim, name: "#{name}_ffn1")
      |> Axon.gelu()
      |> Axon.dense(hidden_size, name: "#{name}_ffn2")

    ffn =
      if dropout > 0 do
        Axon.dropout(ffn, rate: dropout, name: "#{name}_ffn_dropout")
      else
        ffn
      end

    Axon.add(x, ffn, name: "#{name}_ffn_residual")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Zamba model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a Zamba model.

  Note: Zamba has fewer parameters than Jamba because attention weights
  are shared (counted once, not per layer).
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)

    inner_size = hidden_size * expand_factor
    attn_hidden_dim = num_heads * head_dim
    dt_rank = max(div(hidden_size, 16), 1)

    # Per Mamba layer
    mamba_per_layer =
      hidden_size * (2 * inner_size) +
        inner_size * 4 +
        inner_size * (2 * state_size) +
        inner_size * dt_rank + dt_rank * inner_size +
        inner_size * hidden_size

    # Shared attention (counted ONCE, not per application)
    ffn_dim = hidden_size * 4

    # QKV
    # output proj
    # FFN1
    # FFN2
    shared_attn_params =
      attn_hidden_dim * 3 * attn_hidden_dim +
        attn_hidden_dim * attn_hidden_dim +
        hidden_size * ffn_dim +
        ffn_dim * hidden_size

    # Input projection
    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    # Total: all Mamba layers + ONE shared attention
    input_proj + mamba_per_layer * num_layers + shared_attn_params
  end

  @doc """
  Get recommended defaults for real-time sequence processing.

  Optimized for:
  - Real-time inference
  - 1-second context window
  - Minimal memory footprint (shared attention)
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4,
      num_layers: 6,
      # Shared attention every 3 Mamba layers (applied 2x total)
      attention_every: 3,
      num_heads: 4,
      head_dim: 64,
      window_size: 60,
      dropout: 0.1
    ]
  end

  @doc """
  Get the layer pattern for a given configuration.

  Returns a list describing each layer type for debugging/visualization.
  The shared attention is marked with "(shared)" to distinguish from Jamba.

  ## Example

      iex> Zamba.layer_pattern(num_layers: 6, attention_every: 3)
      [:mamba, :mamba, :mamba_attention, :mamba, :mamba, :mamba_attention]
  """
  @spec layer_pattern(keyword()) :: [atom()]
  def layer_pattern(opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    attention_every = Keyword.get(opts, :attention_every, @default_attention_every)

    Enum.map(1..num_layers, fn idx ->
      if rem(idx, attention_every) == 0, do: :mamba_attention, else: :mamba
    end)
  end

  @doc """
  Compare Zamba vs Jamba parameter counts.

  Shows the parameter savings from using shared attention.

  ## Example

      iex> Zamba.compare_to_jamba(embed_dim: 287, num_layers: 6)
      %{
        zamba_params: 450_000,
        jamba_params: 600_000,
        savings: 150_000,
        savings_percent: 25.0
      }
  """
  @spec compare_to_jamba(keyword()) :: map()
  def compare_to_jamba(opts) do
    zamba_params = param_count(opts)

    # Calculate equivalent Jamba params
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    attention_every = Keyword.get(opts, :attention_every, @default_attention_every)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)

    attn_hidden_dim = num_heads * head_dim
    ffn_dim = hidden_size * 4

    # Jamba has multiple independent attention layers
    num_attn_layers = div(num_layers, attention_every)

    attn_per_layer =
      attn_hidden_dim * 3 * attn_hidden_dim +
        attn_hidden_dim * attn_hidden_dim +
        hidden_size * ffn_dim +
        ffn_dim * hidden_size

    # Jamba = Zamba base + (num_attn_layers - 1) extra attention layers
    jamba_extra = attn_per_layer * (num_attn_layers - 1)
    jamba_params = zamba_params + jamba_extra

    savings = jamba_params - zamba_params
    savings_percent = if jamba_params > 0, do: savings / jamba_params * 100, else: 0.0

    %{
      zamba_params: zamba_params,
      jamba_params: jamba_params,
      savings: savings,
      savings_percent: Float.round(savings_percent, 1)
    }
  end
end
