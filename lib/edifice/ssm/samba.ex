defmodule Edifice.SSM.Samba do
  @moduledoc """
  Samba: Hybrid Mamba + Sliding Window Attention + SwiGLU MLP.

  Implements the Samba architecture from "Samba: Simple Hybrid State Space
  Models for Efficient Unlimited Context Language Modeling" (Ren et al.,
  ICLR 2025).

  ## Key Innovation

  Layer-wise interleaving of Mamba (recurrent compression), Sliding Window
  Attention (precise short-range retrieval), and SwiGLU MLP (factual recall).
  Scales to 256K context with perfect memory recall (trained on 4K).

  ## Architecture

  Each Samba block is a fixed 3-sublayer unit with pre-norm residuals:

  ```
  Input [batch, seq_len, hidden_size]
        |
  +==========================================+
  | Samba Block (repeated num_blocks times)  |
  |                                          |
  |  x = x + Mamba(LayerNorm(x))            |  <- recurrent compression
  |  x = x + SWA(LayerNorm(x))             |  <- precise local retrieval
  |  x = x + SwiGLU(LayerNorm(x))          |  <- factual recall
  |                                          |
  +==========================================+
        |
  [batch, hidden_size]  (last timestep)
  ```

  ## Usage

      model = Samba.build(
        embed_dim: 287,
        hidden_size: 256,
        num_blocks: 4,
        num_heads: 8,
        window_size: 60
      )

  ## References
  - Paper: https://arxiv.org/abs/2406.07522
  - Code: https://github.com/microsoft/Samba
  """

  alias Edifice.SSM.{Common, Mamba}
  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.FFN

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_blocks, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:window_size, pos_integer()}
          | {:mlp_dim, pos_integer()}
          | {:state_size, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:conv_size, pos_integer()}
          | {:dropout, float()}

  @doc """
  Build a Samba model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_blocks` - Number of Mamba+SWA+MLP blocks (default: 4)
    - `:num_heads` - SWA query heads (default: 8)
    - `:head_dim` - SWA per-head dimension (default: hidden_size / num_heads)
    - `:window_size` - SWA window size (default: 60)
    - `:mlp_dim` - SwiGLU intermediate dimension (default: 4 * hidden_size)
    - `:state_size` - Mamba SSM state dimension (default: 16)
    - `:expand_factor` - Mamba expansion factor (default: 2)
    - `:conv_size` - Mamba conv kernel size (default: 4)
    - `:dropout` - Dropout rate (default: 0.0)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.

  ## Examples

      iex> model = Edifice.SSM.Samba.build(embed_dim: 32, hidden_size: 16, num_blocks: 1, num_heads: 4, window_size: 8)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_blocks = Keyword.get(opts, :num_blocks, 4)
    dropout = Keyword.get(opts, :dropout, 0.0)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden_size if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack Samba blocks (each block = Mamba + SWA + MLP)
    output =
      Enum.reduce(1..num_blocks, x, fn block_idx, acc ->
        block = build_samba_block(acc, block_idx, opts)

        # Residual is already handled within build_samba_block for each sublayer.
        # Add inter-block dropout if needed.
        if dropout > 0 and block_idx < num_blocks do
          Axon.dropout(block, rate: dropout, name: "block_dropout_#{block_idx}")
        else
          block
        end
      end)

    # Extract last timestep
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

  # Build one Samba block: Mamba sublayer + SWA sublayer + SwiGLU sublayer
  defp build_samba_block(x, block_idx, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 8)
    head_dim = Keyword.get(opts, :head_dim, div(hidden_size, num_heads))
    window_size = Keyword.get(opts, :window_size, 60)
    mlp_dim = Keyword.get(opts, :mlp_dim, hidden_size * 4)
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    expand_factor = Keyword.get(opts, :expand_factor, Common.default_expand_factor())
    conv_size = Keyword.get(opts, :conv_size, Common.default_conv_size())
    dropout = Keyword.get(opts, :dropout, 0.0)
    prefix = "samba_#{block_idx}"

    # 1. Mamba sublayer: x = x + Mamba(Norm(x))
    mamba_opts = [
      hidden_size: hidden_size,
      state_size: state_size,
      expand_factor: expand_factor,
      conv_size: conv_size,
      name: "#{prefix}_mamba",
      layer_idx: block_idx
    ]

    mamba_norm = Axon.layer_norm(x, name: "#{prefix}_mamba_norm")
    mamba_out = Mamba.build_mamba_block(mamba_norm, mamba_opts)
    x = Axon.add(x, mamba_out, name: "#{prefix}_mamba_residual")

    # 2. SWA sublayer: x = x + Proj(SWA(Norm(x)))
    # sliding_window_attention returns [batch, seq, num_heads * head_dim]
    # Need output projection if num_heads * head_dim != hidden_size
    attn_dim = num_heads * head_dim
    swa_norm = Axon.layer_norm(x, name: "#{prefix}_swa_norm")

    swa_out =
      MultiHead.sliding_window_attention(swa_norm,
        num_heads: num_heads,
        head_dim: head_dim,
        window_size: window_size,
        name: "#{prefix}_swa"
      )

    swa_out =
      if attn_dim != hidden_size do
        Axon.dense(swa_out, hidden_size, name: "#{prefix}_swa_proj")
      else
        swa_out
      end

    x = Axon.add(x, swa_out, name: "#{prefix}_swa_residual")

    # 3. SwiGLU MLP sublayer: x = x + SwiGLU(Norm(x))
    mlp_norm = Axon.layer_norm(x, name: "#{prefix}_mlp_norm")

    mlp_out =
      FFN.gated_layer(mlp_norm,
        hidden_size: hidden_size,
        inner_size: mlp_dim,
        dropout: dropout,
        name: "#{prefix}_mlp"
      )

    Axon.add(x, mlp_out, name: "#{prefix}_mlp_residual")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Samba model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, 256)
  end

  @doc """
  Calculate approximate parameter count for a Samba model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_blocks = Keyword.get(opts, :num_blocks, 4)
    state_size = Keyword.get(opts, :state_size, 16)
    expand_factor = Keyword.get(opts, :expand_factor, 2)
    conv_size = Keyword.get(opts, :conv_size, 4)
    mlp_dim = Keyword.get(opts, :mlp_dim, hidden_size * 4)

    inner_size = hidden_size * expand_factor
    dt_rank = max(div(hidden_size, 16), 1)

    # Mamba sublayer per block
    mamba_params =
      hidden_size * (2 * inner_size) +
        conv_size * inner_size +
        inner_size * (2 * state_size) +
        inner_size * dt_rank + dt_rank * inner_size +
        inner_size * hidden_size

    # SWA sublayer per block (Q, K, V, output projections)
    swa_params = 4 * hidden_size * hidden_size

    # SwiGLU MLP per block (gate + up + down)
    mlp_params = 3 * hidden_size * mlp_dim

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj + (mamba_params + swa_params + mlp_params) * num_blocks
  end
end
