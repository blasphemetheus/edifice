defmodule Edifice.Meta.MixtureOfDepths do
  @moduledoc """
  Mixture of Depths: per-token routing where only top-C% tokens are processed.

  Implements the Mixture-of-Depths approach from "Mixture-of-Depths: Dynamically
  allocating compute in transformer-based language models" (Raposo et al., 2024).
  A learned router scores each token and only the top-C% (by capacity ratio) are
  processed through the full transformer block; the rest skip via residual.

  ## Architecture

  ```
  Input [batch, seq, hidden]
        |
        v
  +-----------------------------+
  | Per Layer:                  |
  |   Router: dense -> sigmoid  |
  |   -> soft gate per token    |
  |   Transformer block on all  |
  |   output = gate*block +     |
  |            (1-gate)*input   |
  +-----------------------------+
        | (repeat num_layers)
        v
  Final Norm -> Last Timestep
  Output [batch, hidden_size]
  ```

  ## How It Works

  For each layer, a router network produces a per-token score in [0, 1].
  A top-C selection mechanism identifies which tokens should receive full
  processing. In this Axon-compatible implementation, all tokens pass through
  the transformer block, but the router gate controls how much of the block
  output vs. the residual input each token uses:

      output_t = gate_t * block(input_t) + (1 - gate_t) * input_t

  Tokens with low router scores effectively skip the block via residual.

  ## Usage

      model = MixtureOfDepths.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        capacity_ratio: 0.5,
        num_layers: 4
      )

  ## References
  - Raposo et al., "Mixture-of-Depths" (2024)
  - https://arxiv.org/abs/2404.02258
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @doc """
  Build a Mixture of Depths model.

  ## Options
    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:capacity_ratio` - Fraction of tokens to process (default: 0.5)
    - `:num_layers` - Number of transformer layers (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Sequence length (default: 60)

  ## Returns
    An Axon model outputting `[batch, hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:capacity_ratio, float()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 4)
    capacity_ratio = Keyword.get(opts, :capacity_ratio, 0.5)
    dropout = Keyword.get(opts, :dropout, 0.1)
    num_layers = Keyword.get(opts, :num_layers, 4)
    window_size = Keyword.get(opts, :window_size, 60)

    ModelBuilder.build_sequence_model(
      embed_dim: embed_dim,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      window_size: window_size,
      block_builder: fn input, block_opts ->
        layer_idx = block_opts[:layer_idx]
        name = "mod_block_#{layer_idx}"

        # Router: produces per-token gate [batch, seq, 1] -> sigmoid -> [0, 1]
        gate =
          input
          |> Axon.dense(1, name: "#{name}_router")
          |> Axon.sigmoid(name: "#{name}_router_sigmoid")

        # Apply top-C sparsification: keep top capacity_ratio fraction, zero out the rest
        gate =
          Axon.nx(
            gate,
            fn g ->
              # g: [batch, seq, 1]
              g_squeezed = Nx.squeeze(g, axes: [2])
              seq_len = Nx.axis_size(g_squeezed, 1)
              c = max(1, round(capacity_ratio * seq_len))

              # Get top-C values and create a threshold mask
              {top_vals, _top_indices} = Nx.top_k(g_squeezed, k: c)
              # Threshold is the minimum of the top-C values
              threshold = Nx.reduce_min(top_vals, axes: [1], keep_axes: true)
              mask = Nx.greater_equal(g_squeezed, threshold)

              # Apply mask: zero out gates below threshold
              masked = Nx.select(mask, g_squeezed, Nx.broadcast(0.0, Nx.shape(g_squeezed)))
              Nx.new_axis(masked, 2)
            end,
            name: "#{name}_top_c"
          )

        # Transformer block processes all tokens
        block_output =
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
            hidden_size: hidden_size,
            dropout: dropout,
            name: name
          )

        # Gated output: gate * block_output + (1 - gate) * input
        # gate: [batch, seq, 1], block_output: [batch, seq, hidden], input: [batch, seq, hidden]
        Axon.layer(
          fn block_out, inp, g, _opts ->
            one_minus_g = Nx.subtract(1.0, g)
            Nx.add(Nx.multiply(g, block_out), Nx.multiply(one_minus_g, inp))
          end,
          [block_output, input, gate],
          name: "#{name}_gated_output",
          op_name: :mod_gate
        )
      end
    )
  end

  @doc """
  Get the output size of a MixtureOfDepths model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, 256)
  end

  @doc """
  Get recommended defaults for MixtureOfDepths.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 4,
      capacity_ratio: 0.5,
      num_layers: 4,
      dropout: 0.1,
      window_size: 60
    ]
  end
end
