defmodule Edifice.Meta.Coconut do
  @moduledoc """
  Coconut: Continuous Chain of Thought via latent reasoning.

  <!-- verified: true, date: 2026-02-28 -->

  A latent reasoning architecture where a transformer feeds its own hidden
  states back as input embeddings rather than generating discrete text tokens.
  This enables reasoning in continuous latent space, where dense vectors can
  encode multiple candidate reasoning paths simultaneously (implicit
  breadth-first search).

  ## Architecture

  ```
  Input [batch, window_size, embed_dim]
        |
  Projection -> [batch, window_size, hidden_size]
        |
  Initial TransformerBlock (unique params)
        |
  +--- Thought Loop (num_thoughts iterations) ---+
  |  Shared TransformerBlock (weight-tied)        |
  |        |                                      |
  |  Extract last-position hidden state           |
  |        |                                      |
  |  Write into next thought slot position        |
  +-----------------------------------------------+
        |
  Final TransformerBlock (unique params)
        |
  Final Norm -> Last Position
  Output [batch, hidden_size]
  ```

  ## Continuous Latent Reasoning

  At each thought step, the shared transformer processes the full sequence
  (input tokens + prior thought vectors). The last-position hidden state
  becomes a continuous "thought" that encodes the model's intermediate
  reasoning. This thought vector is inserted into the next sequence position,
  so subsequent steps can attend to all prior thoughts.

  Unlike discrete Chain-of-Thought (which generates text tokens), continuous
  thoughts are dense vectors that can represent superpositions of partial
  solutions — enabling implicit breadth-first search through the reasoning
  space.

  ## Weight Sharing

  The first and last transformer layers have unique parameters (boundary
  layers). All intermediate thought steps share the same transformer weights,
  applied once per thought. This follows the middle-cycle weight sharing
  pattern (same as MoR).

  ## References

  - Hao et al., "Training Large Language Models to Reason in a Continuous
    Latent Space" (Meta, ICLR 2025)
  - https://arxiv.org/abs/2412.06769
  """

  alias Edifice.Attention.MultiHead
  alias Edifice.Blocks.TransformerBlock

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_thoughts 3
  @default_num_layers 2
  @default_dropout 0.1
  @default_window_size 60

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_thoughts, pos_integer()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Coconut continuous chain of thought model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Transformer hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_thoughts` - Number of continuous thought steps (default: 3)
    - `:num_layers` - Transformer layers per thought step (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Input sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_thoughts = Keyword.get(opts, :num_thoughts, @default_num_thoughts)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)

    # Input: [batch, window_size, embed_dim]
    input = Axon.input("input", shape: {nil, window_size, embed_dim})

    # Project to hidden_size if needed
    x =
      if embed_dim == hidden_size do
        input
      else
        Axon.dense(input, hidden_size, name: "input_proj")
      end

    # Initial transformer layer (unique params — boundary layer)
    x = build_transformer_layers(x, hidden_size, num_heads, dropout, num_layers, "first_block")

    # Thought loop: each iteration runs shared transformer + extracts thought
    x =
      Enum.reduce(1..num_thoughts, x, fn thought_idx, acc ->
        # Shared transformer block (same name = weight sharing across thoughts)
        processed =
          build_transformer_layers(
            acc,
            hidden_size,
            num_heads,
            dropout,
            num_layers,
            "shared_thought_block"
          )

        # Extract last-position hidden state as the continuous thought
        thought =
          Axon.nx(processed, fn t ->
            seq_len = Nx.axis_size(t, 1)
            t[[.., seq_len - 1, ..]]
          end)

        # Write thought into the next slot: shift sequence left by 1, append thought
        # This makes room for the new thought while preserving most context
        Axon.layer(
          fn seq, thought_vec, _opts ->
            # seq: [batch, seq_len, hidden_size], thought_vec: [batch, hidden_size]
            # Drop the first position and append the thought as the new last position
            shifted = seq[[.., 1..-1//1, ..]]
            appended = Nx.new_axis(thought_vec, 1)
            Nx.concatenate([shifted, appended], axis: 1)
          end,
          [processed, thought],
          name: "thought_#{thought_idx}_insert",
          op_name: :thought_insert
        )
      end)

    # Final transformer layer (unique params — boundary layer)
    x = build_transformer_layers(x, hidden_size, num_heads, dropout, num_layers, "last_block")

    # Final norm and extract last position (the final thought)
    x
    |> Axon.layer_norm(name: "final_norm")
    |> Axon.nx(fn t ->
      seq_len = Nx.axis_size(t, 1)
      t[[.., seq_len - 1, ..]]
    end)
  end

  @doc "Get the output size of a Coconut model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  # Build one or more transformer layers with a given name prefix.
  # When num_layers > 1, each layer gets a unique name (name_0, name_1, ...).
  # When num_layers == 1, just use the name directly.
  defp build_transformer_layers(x, hidden_size, num_heads, dropout, num_layers, name) do
    Enum.reduce(0..(num_layers - 1), x, fn i, acc ->
      layer_name = if num_layers == 1, do: name, else: "#{name}_#{i}"

      TransformerBlock.layer(acc,
        attention_fn: fn inp, attn_name ->
          MultiHead.self_attention(inp,
            hidden_size: hidden_size,
            num_heads: num_heads,
            dropout: dropout,
            causal: true,
            name: attn_name
          )
        end,
        hidden_size: hidden_size,
        dropout: dropout,
        name: layer_name
      )
    end)
  end
end
