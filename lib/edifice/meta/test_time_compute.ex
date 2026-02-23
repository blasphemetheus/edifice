defmodule Edifice.Meta.TestTimeCompute do
  @moduledoc """
  Test-Time Compute — backbone + scoring network for inference-time scaling.

  Implements the "best-of-N" approach to test-time compute scaling: a backbone
  generates hidden representations, while a parallel scorer network evaluates
  each position. At inference time, multiple completions are generated and
  the scorer selects the best one.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Backbone (GQA + RoPE + SwiGLU, all timesteps)
        |
  [batch, seq_len, hidden_size]
        |
  +-- Scorer: dense(scorer_hidden) -> silu -> dense(1)
        |
  Axon.container(%{backbone: [batch, seq_len, hidden_size],
                    scores: [batch, seq_len, 1]})
  ```

  ## Static Utility

  `select_best_of_n/2` — given scores `[N, batch]`, returns argmax per batch
  element, selecting the highest-scoring completion.

  ## Usage

      model = TestTimeCompute.build(
        embed_dim: 256,
        hidden_size: 256,
        num_layers: 4,
        scorer_hidden: 128
      )

      # At inference: generate N completions, score each, pick best
      best_indices = TestTimeCompute.select_best_of_n(scores)

  ## References

  - "Scaling LLM Test-Time Compute Optimally" (Snell et al., 2024)
  """

  alias Edifice.Blocks.TransformerBlock
  alias Edifice.Attention.GQA

  @default_hidden_size 256
  @default_num_layers 4
  @default_num_heads 4
  @default_num_kv_heads 2
  @default_scorer_hidden 128
  @default_dropout 0.1

  @doc """
  Build a Test-Time Compute model with backbone and scorer.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Backbone hidden dimension (default: 256)
    - `:num_layers` - Number of backbone transformer layers (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_kv_heads` - Number of key/value heads for GQA (default: 2)
    - `:scorer_hidden` - Hidden dimension for scorer MLP (default: 128)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    An `Axon.container` with keys `:backbone` and `:scores`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_kv_heads, pos_integer()}
          | {:scorer_hidden, pos_integer()}
          | {:dropout, float()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, @default_num_kv_heads)
    scorer_hidden = Keyword.get(opts, :scorer_hidden, @default_scorer_hidden)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, Keyword.get(opts, :window_size, 60))

    # Build backbone (all timesteps)
    backbone = build_backbone(
      embed_dim: embed_dim,
      hidden_size: hidden_size,
      num_layers: num_layers,
      num_heads: num_heads,
      num_kv_heads: num_kv_heads,
      dropout: dropout,
      seq_len: seq_len
    )

    # Scorer MLP: dense(scorer_hidden) -> silu -> dense(1)
    scores =
      backbone
      |> Axon.dense(scorer_hidden, name: "scorer_dense1")
      |> Axon.activation(:silu, name: "scorer_act")
      |> maybe_dropout(dropout, "scorer_dropout")
      |> Axon.dense(1, name: "scorer_dense2")

    Axon.container(%{backbone: backbone, scores: scores})
  end

  defp build_backbone(opts) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_layers = Keyword.fetch!(opts, :num_layers)
    num_heads = Keyword.fetch!(opts, :num_heads)
    num_kv_heads = Keyword.fetch!(opts, :num_kv_heads)
    dropout = Keyword.fetch!(opts, :dropout)
    seq_len = Keyword.get(opts, :seq_len, 60)

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
        name = "ttc_block_#{layer_idx}"

        attn_fn = fn x_in, attn_name ->
          GQA.build_gqa_attention(x_in,
            hidden_size: hidden_size,
            num_heads: num_heads,
            num_kv_heads: num_kv_heads,
            rope: true,
            name: attn_name
          )
        end

        TransformerBlock.layer(acc,
          attention_fn: attn_fn,
          hidden_size: hidden_size,
          norm: :rms_norm,
          ffn_type: :gated,
          dropout: dropout,
          name: name
        )
      end)

    Axon.layer_norm(x, name: "final_norm")
  end

  @doc """
  Select the best completion from N candidates using scores.

  Given scores of shape `[N, batch]`, returns the index of the
  highest-scoring candidate per batch element.

  ## Parameters

    - `scores` - Tensor of shape `[N, batch]` with scalar scores per candidate
    - `opts` - Options (unused, reserved for future use)

  ## Returns

    Tensor of shape `[batch]` with the argmax index per batch element.
  """
  @spec select_best_of_n(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def select_best_of_n(scores, _opts \\ []) do
    Nx.argmax(scores, axis: 0)
  end

  @doc "Get the output size of the model (hidden_size of backbone)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      num_heads: 4,
      num_kv_heads: 2,
      scorer_hidden: 128,
      dropout: 0.1
    ]
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)
end
