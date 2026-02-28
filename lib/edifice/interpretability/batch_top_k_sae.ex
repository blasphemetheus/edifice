defmodule Edifice.Interpretability.BatchTopKSAE do
  @moduledoc """
  BatchTopK Sparse Autoencoder for mechanistic interpretability.

  Instead of enforcing a fixed top-k per sample, applies top-k globally across
  the entire batch. Each sample can have a variable number of active features
  as long as the total across the batch meets the budget.

  ## Architecture

  ```
  Input [batch, input_size]
        |
  Encoder: dense(dict_size) + ReLU
        |
  Batch top-k: flatten → global top-k → threshold → mask
        |
  [batch, dict_size]  (sparse activations, variable sparsity per sample)
        |
  Decoder: dense(input_size)
        |
  Output [batch, input_size]  (reconstruction)
  ```

  ## Comparison to Standard TopK

  Standard TopK forces exactly k active features per sample. This means a rare
  but important feature must compete equally in every sample, even where it's
  irrelevant. BatchTopK allows rare features to "borrow" activation budget from
  samples where they're strongly present, while other samples use fewer features.

  ## Usage

      # Total budget of 128 active features across the batch.
      # With batch_size=4, this averages ~32 per sample.
      model = BatchTopKSAE.build(
        input_size: 256,
        dict_size: 4096,
        batch_k: 128
      )

  ## References

  - Bussmann et al., "BatchTopK Sparse Autoencoders" (ICLR 2025)
  """

  import Nx.Defn

  alias Edifice.Interpretability.SparseAutoencoder

  @default_dict_size 4096
  @default_batch_k 128

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:dict_size, pos_integer()}
          | {:batch_k, pos_integer()}

  @doc """
  Build a BatchTopK sparse autoencoder.

  ## Options

    - `:input_size` - Dimension of input activations (required)
    - `:dict_size` - Number of dictionary features (default: #{@default_dict_size})
    - `:batch_k` - Total number of active features across the entire batch
      (default: #{@default_batch_k}). Set to `batch_size * desired_per_sample_k`
      for your training batch size.

  ## Returns

    An Axon model mapping `[batch, input_size]` to `[batch, input_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    batch_k = Keyword.get(opts, :batch_k, @default_batch_k)

    input = Axon.input("batch_topk_sae_input", shape: {nil, input_size})

    # Encoder
    hidden = Axon.dense(input, dict_size, name: "batch_topk_sae_encoder")
    hidden = Axon.activation(hidden, :relu, name: "batch_topk_sae_encoder_act")

    # Batch-global top-k sparsify
    hidden =
      Axon.layer(
        fn acts, _opts -> batch_top_k_sparsify(acts, batch_k) end,
        [hidden],
        name: "batch_topk_sae_sparsify",
        op_name: :batch_top_k_sparsify
      )

    # Decoder
    Axon.dense(hidden, input_size, name: "batch_topk_sae_decoder")
  end

  @doc """
  Build the encoder portion only (for extracting sparse activations).

  Returns sparse hidden activations `[batch, dict_size]`.
  """
  @spec build_encoder([build_opt()]) :: Axon.t()
  def build_encoder(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    dict_size = Keyword.get(opts, :dict_size, @default_dict_size)
    batch_k = Keyword.get(opts, :batch_k, @default_batch_k)

    input = Axon.input("batch_topk_sae_input", shape: {nil, input_size})

    hidden = Axon.dense(input, dict_size, name: "batch_topk_sae_encoder")
    hidden = Axon.activation(hidden, :relu, name: "batch_topk_sae_encoder_act")

    Axon.layer(
      fn acts, _opts -> batch_top_k_sparsify(acts, batch_k) end,
      [hidden],
      name: "batch_topk_sae_sparsify",
      op_name: :batch_top_k_sparsify
    )
  end

  @doc """
  Compute BatchTopK SAE training loss: reconstruction MSE + L1 sparsity penalty.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn loss(input, reconstruction, hidden_acts, opts \\ []) do
    SparseAutoencoder.loss(input, reconstruction, hidden_acts, opts)
  end

  @doc """
  Batch-global top-k sparsification.

  Flattens activations across the batch dimension, selects the top `batch_k`
  values globally, then masks the original activations.
  """
  @spec batch_top_k_sparsify(Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  defn batch_top_k_sparsify(activations, batch_k) do
    flat = Nx.flatten(activations)

    # Global top-k across the entire batch
    {top_values, _top_indices} = Nx.top_k(flat, k: batch_k)

    # Threshold: the batch_k-th largest value globally
    threshold = Nx.slice_along_axis(top_values, batch_k - 1, 1, axis: 0)

    # Apply mask to original (unflatten by broadcasting)
    mask = Nx.greater_equal(activations, Nx.reshape(threshold, {}))
    Nx.select(mask, activations, Nx.tensor(0.0, type: Nx.type(activations)))
  end

  @doc "Get the output size of the BatchTopK SAE (same as input_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :input_size)
  end
end
