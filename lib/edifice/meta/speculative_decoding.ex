defmodule Edifice.Meta.SpeculativeDecoding do
  @moduledoc """
  Speculative Decoding — accelerate autoregressive generation with draft+verify.

  Coordinates a small "draft" model and a large "verifier" model to speed up
  inference. The draft model generates K candidate tokens cheaply, then the
  verifier checks them in a single forward pass. Accepted tokens skip the
  expensive verifier's autoregressive steps.

  ## How It Works

  1. Draft model generates K candidate tokens autoregressively (fast)
  2. Verifier scores all K tokens in one forward pass (parallel)
  3. Accept longest prefix where draft and verifier agree
  4. Continue from first disagreement

  ## Architecture

  ```
  build/1 returns {draft_model, verifier_model}

  Draft Model (small, fast):
    [batch, seq_len, embed_dim] → DecoderOnly (few layers) → [batch, hidden]

  Verifier Model (large, accurate):
    [batch, seq_len, embed_dim] → DecoderOnly (many layers) → [batch, hidden]
  ```

  ## Usage

      {draft, verifier} = SpeculativeDecoding.build(
        embed_dim: 256,
        draft_model_opts: [hidden_size: 128, num_layers: 2],
        verifier_model_opts: [hidden_size: 512, num_layers: 8]
      )

      # At inference time:
      accepted = SpeculativeDecoding.accept_reject(draft_tokens, verifier_tokens)

  ## References

  - "Fast Inference from Transformers via Speculative Decoding"
    (Leviathan et al., 2023) — https://arxiv.org/abs/2211.17192
  """

  @default_draft_hidden 128
  @default_draft_layers 2
  @default_verifier_hidden 256
  @default_verifier_layers 6

  @doc """
  Build draft and verifier models for speculative decoding.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:draft_model_type` - Draft architecture (default: `:decoder_only`)
    - `:draft_model_opts` - Options for draft model (default: small config)
    - `:verifier_model_type` - Verifier architecture (default: `:decoder_only`)
    - `:verifier_model_opts` - Options for verifier model (default: larger config)

  ## Returns

    `{draft_model, verifier_model}` — a 2-tuple of Axon models.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:draft_model_type, atom()}
          | {:draft_model_opts, keyword()}
          | {:verifier_model_type, atom()}
          | {:verifier_model_opts, keyword()}

  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    draft_type = Keyword.get(opts, :draft_model_type, :decoder_only)
    verifier_type = Keyword.get(opts, :verifier_model_type, :decoder_only)

    draft_opts =
      Keyword.get(opts, :draft_model_opts, [])
      |> Keyword.put_new(:hidden_size, @default_draft_hidden)
      |> Keyword.put_new(:num_layers, @default_draft_layers)
      |> Keyword.put_new(:embed_dim, embed_dim)

    verifier_opts =
      Keyword.get(opts, :verifier_model_opts, [])
      |> Keyword.put_new(:hidden_size, @default_verifier_hidden)
      |> Keyword.put_new(:num_layers, @default_verifier_layers)
      |> Keyword.put_new(:embed_dim, embed_dim)

    draft = Edifice.build(draft_type, draft_opts)
    verifier = Edifice.build(verifier_type, verifier_opts)

    {draft, verifier}
  end

  @doc """
  Accept-reject step: compare draft tokens to verifier tokens.

  Finds the longest accepted prefix where draft and verifier tokens match.
  Returns the number of accepted tokens (0 means first token mismatched).

  ## Parameters

    - `draft_tokens` - Tensor of draft token IDs `[K]` or `[batch, K]`
    - `verifier_tokens` - Tensor of verifier token IDs `[K]` or `[batch, K]`

  ## Returns

    Integer tensor with the number of accepted tokens per batch element.
  """
  @spec accept_reject(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def accept_reject(draft_tokens, verifier_tokens) do
    # Element-wise comparison
    matches = Nx.equal(draft_tokens, verifier_tokens)

    # Count consecutive matches from the start using cumulative product
    # matches: [1, 1, 0, 1] -> cum_prod: [1, 1, 0, 0] -> sum = 2
    matches
    |> Nx.cumulative_product(axis: -1)
    |> Nx.sum(axes: [-1])
  end

  @doc "Get the output size (delegates to verifier model's hidden_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    verifier_opts = Keyword.get(opts, :verifier_model_opts, [])
    Keyword.get(verifier_opts, :hidden_size, @default_verifier_hidden)
  end
end
