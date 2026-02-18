defmodule Edifice.Feedforward.KAT do
  @moduledoc """
  KAT: KAN-Attention Transformer — attention blocks with KAN replacing FFN.

  Combines standard multi-head self-attention with Kolmogorov-Arnold Network
  (KAN) layers as the feed-forward sublayer, replacing the typical MLP FFN.
  KAN layers use learnable activation functions on edges (basis functions like
  B-splines, sine, Chebyshev) instead of fixed activations on nodes.

  ## Architecture

  ```
  Input [batch, seq, embed_dim]
        |
        v
  +-------------------------------------+
  |  TransformerBlock (per layer):       |
  |    norm -> MultiHead Attention       |
  |    norm -> KAN Layer (replaces FFN)  |
  +-------------------------------------+
        | (repeat num_layers)
        v
  Final Norm -> Last Timestep
  Output [batch, hidden_size]
  ```

  ## Why KAN Instead of FFN?

  | Aspect | Standard FFN | KAN FFN |
  |--------|-------------|---------|
  | Activation | Fixed (ReLU/GELU) on nodes | Learnable on edges |
  | Expressiveness | Requires width for accuracy | Learns optimal activation |
  | Interpretability | Low | Higher (visualizable) |
  | Parameters | O(n^2) | O(n^2 * grid_size) |

  ## Usage

      model = KAT.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        grid_size: 8,
        basis: :bspline,
        num_layers: 4
      )

  ## References
  - Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
  - Vaswani et al., "Attention Is All You Need" (2017)
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}
  alias Edifice.Feedforward.KAN

  @doc """
  Build a KAT (KAN-Attention Transformer) model.

  ## Options
    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:grid_size` - Number of KAN basis functions per edge (default: 8)
    - `:basis` - KAN basis function: `:bspline`, `:sine`, `:chebyshev`, `:fourier`, `:rbf` (default: :bspline)
    - `:num_layers` - Number of transformer layers (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Sequence length (default: 60)

  ## Returns
    An Axon model outputting `[batch, hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:basis, :bspline | :sine | :chebyshev | :fourier | :rbf}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:grid_size, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    grid_size = Keyword.get(opts, :grid_size, 8)
    basis = Keyword.get(opts, :basis, :bspline)
    num_layers = Keyword.get(opts, :num_layers, 4)
    dropout = Keyword.get(opts, :dropout, 0.1)
    window_size = Keyword.get(opts, :window_size, 60)

    ModelBuilder.build_sequence_model(
      embed_dim: embed_dim,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      block_builder: fn input, block_opts ->
        layer_idx = block_opts[:layer_idx]
        name = "kat_block_#{layer_idx}"

        TransformerBlock.layer(input,
          attention_fn: fn x, attn_name ->
            MultiHead.self_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              dropout: dropout,
              causal: true,
              name: attn_name
            )
          end,
          custom_ffn: fn x, ffn_name ->
            KAN.build_kan_layer(x, hidden_size,
              grid_size: grid_size,
              basis: basis,
              name: ffn_name
            )
          end,
          hidden_size: hidden_size,
          dropout: dropout,
          name: name
        )
      end
    )
  end

  @doc """
  Get the output size of a KAT model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, 256)
  end

  @doc """
  Get recommended defaults for KAT.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 4,
      grid_size: 8,
      basis: :bspline,
      num_layers: 4,
      dropout: 0.1,
      window_size: 60
    ]
  end
end
