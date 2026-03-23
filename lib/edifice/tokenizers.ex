defmodule Edifice.Tokenizers do
  @moduledoc """
  Optional tokenizer integration for text preprocessing.

  Wraps the `tokenizers` library for loading HuggingFace tokenizers
  and converting text to token ID tensors for Edifice language models.

  ## Usage

      tokenizer = Edifice.Tokenizers.from_pretrained("meta-llama/Llama-3-8B")
      token_ids = Edifice.Tokenizers.encode(tokenizer, "Hello, world!")
      text = Edifice.Tokenizers.decode(tokenizer, token_ids)

  ## Requirements

  Add `{:tokenizers, "~> 0.5", optional: true}` to your deps.
  """

  # Use apply/3 for all Tokenizers calls to avoid compile-time warnings
  # when the optional dependency is not installed.
  defp tokenizer_mod, do: Tokenizers.Tokenizer
  defp encoding_mod, do: Tokenizers.Encoding

  @doc """
  Load a pretrained tokenizer from HuggingFace Hub.
  """
  def from_pretrained(repo_id) do
    ensure_tokenizers!()
    {:ok, tokenizer} = apply(tokenizer_mod(), :from_pretrained, [repo_id])
    tokenizer
  end

  @doc """
  Load a tokenizer from a local file.
  """
  def from_file(path) do
    ensure_tokenizers!()
    {:ok, tokenizer} = apply(tokenizer_mod(), :from_file, [path])
    tokenizer
  end

  @doc """
  Encode text to token ID tensor.

  ## Options

    * `:return_type` - `:tensor` (default) or `:list`
    * `:add_special_tokens` - Whether to add BOS/EOS tokens (default: `true`)

  ## Returns

    `Nx.Tensor` of shape `{1, seq_len}` with token IDs (s32).
  """
  def encode(tokenizer, text, opts \\ []) do
    ensure_tokenizers!()
    return_type = Keyword.get(opts, :return_type, :tensor)
    add_special = Keyword.get(opts, :add_special_tokens, true)

    {:ok, encoding} = apply(tokenizer_mod(), :encode, [tokenizer, text, [add_special_tokens: add_special]])
    ids = apply(encoding_mod(), :get_ids, [encoding])

    case return_type do
      :tensor -> Nx.tensor([ids], type: :s32)
      :list -> ids
    end
  end

  @doc """
  Encode a batch of texts.

  ## Returns

    `Nx.Tensor` of shape `{batch, max_len}` (padded with 0s).
  """
  def encode_batch(tokenizer, texts, opts \\ []) do
    ensure_tokenizers!()
    add_special = Keyword.get(opts, :add_special_tokens, true)

    encodings =
      Enum.map(texts, fn text ->
        {:ok, encoding} = apply(tokenizer_mod(), :encode, [tokenizer, text, [add_special_tokens: add_special]])
        apply(encoding_mod(), :get_ids, [encoding])
      end)

    max_len = encodings |> Enum.map(&length/1) |> Enum.max()

    padded =
      Enum.map(encodings, fn ids ->
        ids ++ List.duplicate(0, max_len - length(ids))
      end)

    Nx.tensor(padded, type: :s32)
  end

  @doc """
  Decode token IDs back to text.
  """
  def decode(tokenizer, token_ids) do
    ensure_tokenizers!()

    ids =
      case token_ids do
        %Nx.Tensor{} -> Nx.to_flat_list(token_ids)
        list when is_list(list) -> list
      end

    {:ok, text} = apply(tokenizer_mod(), :decode, [tokenizer, ids])
    text
  end

  @doc """
  Get the vocabulary size of a tokenizer.
  """
  def vocab_size(tokenizer) do
    ensure_tokenizers!()
    apply(tokenizer_mod(), :get_vocab_size, [tokenizer])
  end

  defp ensure_tokenizers! do
    unless Code.ensure_loaded?(Tokenizers.Tokenizer) do
      raise RuntimeError,
            "Edifice.Tokenizers requires the :tokenizers dependency. " <>
              "Add {:tokenizers, \"~> 0.5\", optional: true} to your deps."
    end
  end
end
